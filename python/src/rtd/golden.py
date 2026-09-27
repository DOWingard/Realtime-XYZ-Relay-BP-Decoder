"""Decode syndromes with IBM's relay_bp and write its outputs as golden files for the C++ decoder.

Every algorithm parameter must be given on the command line: relay_bp's Python defaults differ
from the Relay-BP paper, so nothing is left to them. Each shot is decoded by a freshly
constructed relay_bp decoder. RelayDecoder keeps per-leg bookkeeping that is not reset between
decodes, so a reused decoder would log stale leg data. Its leg log (relay_logging.out, written to
the current directory) is read from a private temporary directory per process.

Syndromes come either from rtd-sample shots (--shots, --first, --count) or from single columns of
H (--single-columns, --seed): row 0 is the zero syndrome and row i >= 1 is column columns[i - 1]
of H, for distinct columns drawn with numpy.random.default_rng(seed).choice(n, K, replace=False).

Output directory contents (S = syndromes, m = detectors, n = columns, L = relay legs run):
    detectors.npy         uint8   [S, m]   the syndromes decoded, in order
    decoding.npy          uint8   [S, n]   relay_bp's error estimate e
    success.npy           uint8   [S]      1 if H e = syndrome (mod 2)
    iterations.npy        int64   [S]      BP iterations, summed over all relay legs run
    weight.npy            float64 [S]      sum of ln((1 - p_j) / p_j) over e_j = 1, +inf if not converged
    posterior.npy         float64 [K, n]   posterior log-likelihood ratios of the first K shots
    columns.npy           int64   [K]      single-column source only: column of H behind row i + 1
    gammas.npy            float64 [T, n]   relay only: memory strengths; leg r >= 1 uses row r mod T
    legs_ptr.npy          int64   [S + 1]  relay only: legs of shot s are rows legs_ptr[s]..legs_ptr[s+1]-1
    leg_iterations.npy    int64   [L]      relay only: iterations of each leg, leg 0 first
    leg_converged.npy     uint8   [L]      relay only: 1 if the leg's hard decision matched the syndrome
    leg_unique_best.npy   uint8   [L]      relay only: relay_bp's flag for the returned leg when no
                                           other converged leg reached the same weight
    manifest.json         configuration, input checksums, versions, timing, summary, output checksums
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import stim

from rtd import log
from rtd.export import _relay_bp_source, _sha256

logger = logging.getLogger("rtd.golden")

GOLDEN_FORMAT_VERSION = 1
RELAY_LOG_NAME = "relay_logging.out"
STOPPING_CRITERIA = ("pre_iter", "nconv", "all")
FLOAT_TYPES = ("f32", "f64")
DECODERS = ("min_sum", "relay")
_CLASS_NAMES = {
    ("min_sum", "f32"): "MinSumBPDecoderF32",
    ("min_sum", "f64"): "MinSumBPDecoderF64",
    ("relay", "f32"): "RelayDecoderF32",
    ("relay", "f64"): "RelayDecoderF64",
}
_ARTIFACT_FILES = ("H_indptr.npy", "H_indices.npy", "priors.npy")
# Every file this tool can write; on --overwrite, those a run does not produce are removed so a
# directory never mixes outputs of two runs.
OUTPUT_FILES = (
    "detectors.npy",
    "decoding.npy",
    "success.npy",
    "iterations.npy",
    "weight.npy",
    "posterior.npy",
    "columns.npy",
    "gammas.npy",
    "legs_ptr.npy",
    "leg_iterations.npy",
    "leg_converged.npy",
    "leg_unique_best.npy",
    "manifest.json",
)

# Algorithm flags, by argparse dest. None of them has a default.
_COMMON_FLAGS = {"alpha": "--alpha", "alpha_scaling": "--alpha-scaling", "gamma0": "--gamma0"}
_MIN_SUM_FLAGS = {"max_iter": "--max-iter"}
_RELAY_FLAGS = {
    "pre_iter": "--pre-iter",
    "num_sets": "--num-sets",
    "set_max_iter": "--set-max-iter",
    "stopping": "--stopping",
    "stop_nconv": "--stop-nconv",
    "gamma_rows": "--gamma-rows",
    "gamma_seed": "--gamma-seed",
    "gamma_interval": "--gamma-interval",
}
_SHOTS_FLAGS = {"first": "--first", "count": "--count"}
_SINGLE_COLUMN_FLAGS = {"seed": "--seed"}


# ---------------------------------------------------------------------------------------------
# relay_bp leg log


@dataclass(frozen=True)
class RelayLegs:
    """Per-leg records of one relay decode, in execution order (leg 0 = the initial Mem-BP run)."""

    iterations: np.ndarray  # int64 [L]
    converged: np.ndarray  # uint8 [L]
    unique_best: np.ndarray  # uint8 [L]


# relay_bp formats each record as "{set - 1}, {iterations}, {converged}, {unique_best}".
_LEG_LINE = re.compile(r"(-?\d+), (\d+), ([01]), ([01])")


def parse_relay_log(text: str) -> RelayLegs:
    """Parse relay_logging.out as written by one freshly constructed decoder after one decode.

    The file starts with '#' header lines (written by the constructor), followed by one line per
    executed leg. Leg k is printed with set_idx k - 1, so leg 0 prints -1 and the records must read
    -1, 0, 1, ... in order. A second run of -1 means the file holds more than one decode.
    """
    header_lines = 0
    records: list[tuple[int, int, int]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            if records:
                raise ValueError(f"relay log line {lineno}: header line after leg records: {line!r}")
            header_lines += 1
            continue
        match = _LEG_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"relay log line {lineno}: expected 'set_idx, num_iter, converged, unique_best', got {line!r}")
        set_idx, iterations, converged, unique_best = (int(g) for g in match.groups())
        expected = len(records) - 1
        if set_idx != expected:
            if set_idx == -1:
                raise ValueError(
                    f"relay log line {lineno}: a second decode starts here; the log must come from a "
                    "freshly constructed decoder that decoded exactly one syndrome"
                )
            raise ValueError(f"relay log line {lineno}: set_idx {set_idx}, expected {expected}")
        records.append((iterations, converged, unique_best))
    if header_lines == 0:
        raise ValueError("relay log has no '#' header; it was not written by a freshly constructed decoder")
    if not records:
        raise ValueError("relay log has no leg records")
    columns = list(zip(*records))
    return RelayLegs(
        iterations=np.array(columns[0], dtype=np.int64),
        converged=np.array(columns[1], dtype=np.uint8),
        unique_best=np.array(columns[2], dtype=np.uint8),
    )


def check_relay_legs(legs: RelayLegs, success: bool, iterations: int, params: dict[str, Any]) -> None:
    """Check one shot's leg records against relay_bp's control flow; raise on any inconsistency."""
    pre_iter, num_sets, set_max_iter = params["pre_iter"], params["num_sets"], params["set_max_iter"]
    stopping, stop_nconv = params["stopping_criterion"], params["stop_nconv"]
    count = legs.iterations.size
    conv = legs.converged.astype(bool)
    problems = []
    if int(legs.iterations.sum()) != iterations:
        problems.append(f"leg iterations sum to {int(legs.iterations.sum())}, decoder reported {iterations}")
    if bool(conv.any()) != success:
        problems.append(f"success is {success} but converged legs are {np.flatnonzero(conv).tolist()}")
    if count > num_sets + 1:
        problems.append(f"{count} legs logged, at most num_sets + 1 = {num_sets + 1} can run")
    limits = np.full(count, set_max_iter, dtype=np.int64)
    limits[0] = pre_iter
    if np.any(legs.iterations > limits) or np.any(legs.iterations[~conv] != limits[~conv]):
        problems.append(f"leg iterations {legs.iterations.tolist()} inconsistent with limits and convergence")
    if stopping == "all" and count != num_sets + 1:
        problems.append(f"stopping 'all' ran {count} legs, expected {num_sets + 1}")
    if stopping == "pre_iter" and count != (1 if conv[0] else num_sets + 1):
        problems.append(f"stopping 'pre_iter' ran {count} legs with leg 0 converged={bool(conv[0])}")
    if stopping == "nconv":
        n_conv = int(conv.sum())
        if n_conv > stop_nconv or (n_conv == stop_nconv and not conv[-1]):
            problems.append(f"stopping 'nconv' ({stop_nconv}) ran on after {n_conv} converged legs")
        if count < num_sets + 1 and n_conv != stop_nconv:
            problems.append(f"stopping 'nconv' ended after {count} legs with only {n_conv} converged")
    best = np.flatnonzero(legs.unique_best)
    if best.size > 1 or (best.size == 1 and not conv[best[0]]):
        problems.append(f"unique-best flags {best.tolist()} not on a single converged leg")
    if problems:
        raise RuntimeError("relay leg log inconsistent: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------
# Decoding (runs in worker processes, or in-process for --workers 1)


@dataclass(frozen=True)
class Problem:
    """Everything a worker needs to build one relay_bp decoder: the check matrix and priors, the
    relay_bp class, and every keyword passed to its constructor."""

    class_name: str
    indptr: np.ndarray
    indices: np.ndarray
    shape: tuple[int, int]
    priors: np.ndarray
    kwargs: dict[str, Any]

    @property
    def is_relay(self) -> bool:
        return self.class_name.startswith("Relay")

    def check_matrix(self) -> sp.csr_matrix:
        # relay_bp accepts only the class named "csr_matrix" (not csr_array), with sorted indices.
        h = sp.csr_matrix(
            (np.ones(self.indices.size, dtype=np.uint8), self.indices, self.indptr), shape=self.shape
        )
        if type(h).__name__ != "csr_matrix" or h.dtype != np.uint8 or not h.has_sorted_indices:
            raise TypeError(f"check matrix is {type(h).__name__} {h.dtype}, sorted={h.has_sorted_indices}")
        return h


@dataclass
class ShotResult:
    index: int
    decoding: np.ndarray
    success: bool
    iterations: int
    posterior: np.ndarray | None
    legs: RelayLegs | None
    construct_seconds: float
    decode_seconds: float


def _construct_capturing_stderr(factory: Callable[[], Any]) -> tuple[Any, str]:
    """Call factory() with file descriptor 2 redirected, returning what native code printed there.

    relay_bp's Rust constructor prints warnings with eprintln!, which bypasses Python logging.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with tempfile.TemporaryFile() as capture:
            os.dup2(capture.fileno(), 2)
            try:
                obj = factory()
            finally:
                os.dup2(saved, 2)
            capture.seek(0)
            text = capture.read().decode(errors="replace")
    finally:
        os.close(saved)
    return obj, text


class ShotDecoder:
    """Decodes single syndromes, each with a newly constructed relay_bp decoder."""

    def __init__(self, problem: Problem, workdir: Path):
        import relay_bp

        self._problem = problem
        self._cls = getattr(relay_bp, problem.class_name)
        self._h = problem.check_matrix()
        self._priors = np.ascontiguousarray(problem.priors, dtype=np.float64)
        self._workdir = workdir.resolve()
        self._log_path = self._workdir / RELAY_LOG_NAME
        self._relay_params = problem.kwargs if problem.is_relay else None

    def construct(self) -> tuple[Any, str]:
        if self._problem.is_relay and Path.cwd().resolve() != self._workdir:
            raise RuntimeError(f"relay decoders must be built in {self._workdir}, cwd is {Path.cwd()}")
        self._log_path.unlink(missing_ok=True)
        return _construct_capturing_stderr(
            lambda: self._cls(check_matrix=self._h, error_priors=self._priors, **self._problem.kwargs)
        )

    def decode(self, index: int, syndrome: np.ndarray, want_posterior: bool) -> ShotResult:
        t0 = time.perf_counter()
        decoder, stderr_text = self.construct()
        t1 = time.perf_counter()
        result = decoder.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8))
        t2 = time.perf_counter()
        if stderr_text:
            logger.debug("relay_bp stderr during construction", extra={"shot": index, "stderr": stderr_text.strip()})

        decoding = np.asarray(result.decoding)
        n = self._problem.shape[1]
        if decoding.dtype != np.uint8 or decoding.shape != (n,):
            raise RuntimeError(f"shot {index}: decoding has dtype {decoding.dtype} shape {decoding.shape}, expected uint8 ({n},)")
        success, iterations = bool(result.success), int(result.iterations)
        posterior = None
        if want_posterior:
            posterior = np.array(result.posterior_ratios, dtype=np.float64)
            if posterior.shape != (n,):
                raise RuntimeError(f"shot {index}: posterior has shape {posterior.shape}, expected ({n},)")

        legs = None
        if self._relay_params is not None:
            legs = parse_relay_log(self._log_path.read_text())
            check_relay_legs(legs, success, iterations, self._relay_params)
        else:
            max_iter = self._problem.kwargs["max_iter"]
            if not 1 <= iterations <= max_iter or (not success and iterations != max_iter):
                raise RuntimeError(f"shot {index}: {iterations} iterations with success={success}, max_iter={max_iter}")

        logger.debug(
            "shot decoded",
            extra={
                "shot": index,
                "success": success,
                "iterations": iterations,
                "legs": None if legs is None else int(legs.iterations.size),
                "decode_seconds": t2 - t1,
            },
        )
        return ShotResult(index, decoding.copy(), success, iterations, posterior, legs, t1 - t0, t2 - t1)


_WORKER: ShotDecoder | None = None


def _init_worker(problem: Problem, parent_dir: str, level: int) -> None:
    global _WORKER
    log.configure(level)
    try:
        workdir = Path(tempfile.mkdtemp(prefix=f"worker-{os.getpid()}-", dir=parent_dir))
        os.chdir(workdir)
        _WORKER = ShotDecoder(problem, workdir)
    except Exception:
        logger.exception("worker initialisation failed", extra={"pid": os.getpid(), "parent_dir": parent_dir})
        raise
    logger.debug("worker ready", extra={"pid": os.getpid(), "workdir": str(workdir)})


def _run_in_worker(index: int, syndrome: np.ndarray, want_posterior: bool) -> ShotResult:
    if _WORKER is None:
        raise RuntimeError("worker process was not initialised")
    try:
        return _WORKER.decode(index, syndrome, want_posterior)
    except Exception:
        logger.exception("shot failed in worker", extra={"shot": index, "pid": os.getpid()})
        raise


class _Progress:
    def __init__(self, total: int):
        self.total, self.done, self.start = total, 0, time.perf_counter()
        self._next_log = self.start + 30.0
        self._step = max(1, total // 10)

    def advance(self) -> None:
        self.done += 1
        now = time.perf_counter()
        if self.done % self._step == 0 or now >= self._next_log or self.done == self.total:
            elapsed = now - self.start
            logger.info(
                "decoding progress",
                extra={
                    "done": self.done,
                    "total": self.total,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / self.done * (self.total - self.done),
                },
            )
            self._next_log = now + 30.0


def decode_all(
    problem: Problem, syndromes: np.ndarray, posterior_shots: int, workers: int, scratch: Path
) -> list[ShotResult]:
    """Decode every syndrome, one fresh decoder per shot, returning results in syndrome order."""
    total = syndromes.shape[0]
    results: list[ShotResult | None] = [None] * total
    progress = _Progress(total)
    if workers == 1:
        workdir = scratch / "worker-0"
        workdir.mkdir()
        with contextlib.chdir(workdir):
            decoder = ShotDecoder(problem, workdir)
            for s in range(total):
                try:
                    results[s] = decoder.decode(s, syndromes[s], s < posterior_shots)
                except Exception:
                    logger.exception("shot failed", extra={"shot": s})
                    raise
                progress.advance()
        return results  # type: ignore[return-value]

    level = logging.getLogger().getEffectiveLevel()
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(problem, str(scratch), level),
    )
    try:
        futures = [executor.submit(_run_in_worker, s, syndromes[s], s < posterior_shots) for s in range(total)]
        for future in as_completed(futures):
            result = future.result()
            results[result.index] = result
            progress.advance()
    except BaseException:
        logger.exception("parallel decoding aborted; cancelling remaining shots", extra={"done": progress.done, "total": total})
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------
# Inputs


def manifest_path(path: Path) -> str:
    """An input path as a manifest records it: relative to the working directory when it lies
    inside it, otherwise only its final component, so manifests hold no machine-specific paths."""
    resolved, cwd = path.resolve(), Path.cwd().resolve()
    return resolved.relative_to(cwd).as_posix() if resolved.is_relative_to(cwd) else resolved.name


@dataclass(frozen=True)
class Artifact:
    path: Path
    indptr: np.ndarray  # int64 [m + 1]
    indices: np.ndarray  # int64 [nnz]
    priors: np.ndarray  # float64 [n]
    num_detectors: int
    num_columns: int
    sha256: dict[str, str]
    circuit_sha256: str | None


def load_artifact(path: Path) -> Artifact:
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    for key in ("num_detectors", "num_columns", "sha256"):
        if key not in manifest:
            raise ValueError(f"artifact manifest {manifest_file} has no '{key}'")
    sha = {}
    for name in _ARTIFACT_FILES:
        if name not in manifest["sha256"]:
            raise ValueError(f"artifact manifest {manifest_file} has no sha256 for {name}")
        actual = _sha256(path / name)
        if actual != manifest["sha256"][name]:
            raise ValueError(f"{path / name} does not match its manifest checksum ({actual} != {manifest['sha256'][name]})")
        sha[name] = actual

    indptr = np.load(path / "H_indptr.npy", allow_pickle=False)
    indices = np.load(path / "H_indices.npy", allow_pickle=False)
    priors = np.load(path / "priors.npy", allow_pickle=False)
    m, n = int(manifest["num_detectors"]), int(manifest["num_columns"])
    if indptr.dtype != np.uint32 or indices.dtype != np.uint32 or priors.dtype != np.float64:
        raise ValueError(f"artifact dtypes {indptr.dtype}, {indices.dtype}, {priors.dtype}; expected uint32, uint32, float64")
    if indptr.shape != (m + 1,) or indices.ndim != 1 or priors.shape != (n,):
        raise ValueError(f"artifact shapes {indptr.shape}, {indices.shape}, {priors.shape} do not fit m={m}, n={n}")
    indptr, indices = indptr.astype(np.int64), indices.astype(np.int64)
    if indptr[0] != 0 or indptr[-1] != indices.size or np.any(np.diff(indptr) < 0):
        raise ValueError("H_indptr is not a valid CSR offset array")
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"H_indices has entries outside [0, {n})")
    row_of = np.repeat(np.arange(m), np.diff(indptr))
    if np.any(np.diff(indices)[row_of[1:] == row_of[:-1]] <= 0):
        raise ValueError("H_indices is not strictly ascending within every row")
    if not np.all(np.isfinite(priors) & (priors > 0.0) & (priors < 1.0)):
        raise ValueError("artifact priors must lie strictly between 0 and 1 (rtd-export prunes p = 0 and p = 1)")
    logger.info(
        "artifact loaded",
        extra={"artifact": str(path), "num_detectors": m, "num_columns": n, "nnz_H": int(indices.size)},
    )
    return Artifact(
        path=path,
        indptr=indptr,
        indices=indices,
        priors=priors,
        num_detectors=m,
        num_columns=n,
        sha256=sha,
        circuit_sha256=manifest.get("source_circuit", {}).get("sha256"),
    )


def load_shots(shots: Path, first: int, count: int, artifact: Artifact) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_file = shots / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"shots manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    expected_sha = manifest.get("sha256", {}).get("detectors.npy")
    if expected_sha is None:
        raise ValueError(f"shots manifest {manifest_file} has no sha256 for detectors.npy")
    detectors_sha = _sha256(shots / "detectors.npy")
    if detectors_sha != expected_sha:
        raise ValueError(f"{shots / 'detectors.npy'} does not match its manifest checksum")

    circuit_sha = manifest.get("sha256", {}).get("circuit.stim")
    if circuit_sha is None or artifact.circuit_sha256 is None:
        logger.warning(
            "cannot confirm the shots were sampled from the artifact's circuit",
            extra={"shots_circuit_sha256": circuit_sha, "artifact_circuit_sha256": artifact.circuit_sha256},
        )
    elif circuit_sha != artifact.circuit_sha256:
        raise ValueError(
            f"shots in {shots} come from circuit {circuit_sha}, but the artifact was exported from {artifact.circuit_sha256}"
        )

    all_detectors = np.load(shots / "detectors.npy", mmap_mode="r", allow_pickle=False)
    if all_detectors.dtype != np.uint8 or all_detectors.ndim != 2 or all_detectors.shape[1] != artifact.num_detectors:
        raise ValueError(
            f"detectors.npy is {all_detectors.dtype} {all_detectors.shape}; expected uint8 [shots, {artifact.num_detectors}]"
        )
    if first + count > all_detectors.shape[0]:
        raise ValueError(f"--first {first} --count {count} exceeds the {all_detectors.shape[0]} shots in {shots}")
    syndromes = np.ascontiguousarray(all_detectors[first : first + count], dtype=np.uint8)
    if np.any(syndromes > 1):
        raise ValueError("detectors.npy has entries other than 0 and 1")
    source = {
        "type": "shots",
        "path": manifest_path(shots),
        "first": first,
        "count": count,
        "detectors_sha256": expected_sha,
    }
    logger.info("shots loaded", extra={"shots": str(shots), "first": first, "count": count})
    return syndromes, source


def single_column_syndromes(artifact: Artifact, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Zero syndrome, then H[:, j] for `count` distinct columns j drawn from default_rng(seed)."""
    n, m = artifact.num_columns, artifact.num_detectors
    columns = np.random.default_rng(seed).choice(n, size=count, replace=False).astype(np.int64)
    h_csc = sp.csr_matrix(
        (np.ones(artifact.indices.size, dtype=np.uint8), artifact.indices, artifact.indptr), shape=(m, n)
    ).tocsc()
    syndromes = np.zeros((count + 1, m), dtype=np.uint8)
    for i, j in enumerate(columns.tolist(), start=1):
        syndromes[i, h_csc.indices[h_csc.indptr[j] : h_csc.indptr[j + 1]]] = 1
    source = {"type": "single_columns", "count": count, "seed": seed, "columns": columns.tolist()}
    logger.info("single-column syndromes built", extra={"columns": count, "seed": seed})
    return syndromes, columns, source


def gamma_table(rows: int, seed: int, interval: tuple[float, float], n: int) -> np.ndarray:
    lo, hi = interval
    return np.ascontiguousarray(np.random.default_rng(seed).uniform(lo, hi, size=(rows, n)), dtype=np.float64)


def log_prior_ratios(priors: np.ndarray) -> list[float]:
    """ln((1 - p) / p) per column with the C library's log, as relay_bp computes it."""
    return [math.log((1.0 - p) / p) for p in priors.tolist()]


def decoding_weight(decoding: np.ndarray, lam: list[float]) -> float:
    """Sum of the finite lam[j] over decoding[j] = 1, added one at a time in ascending j.

    Same order and filter as relay_bp's decoding quality, so the double result is reproducible.
    """
    total = 0.0
    for j in np.flatnonzero(decoding).tolist():
        value = lam[j]
        if math.isfinite(value):
            total += value
    return total


# ---------------------------------------------------------------------------------------------
# Run


def _check_outputs(arrays: dict[str, np.ndarray], expected: dict[str, tuple[np.dtype, tuple[int, ...]]]) -> None:
    for name, (dtype, shape) in expected.items():
        array = arrays[name]
        if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
            raise RuntimeError(f"{name}: dtype {array.dtype} shape {array.shape}, expected {np.dtype(dtype)} {shape}")


def _check_syndromes(h_int: sp.csr_matrix, syndromes: np.ndarray, decoding: np.ndarray, success: np.ndarray) -> None:
    for s in range(syndromes.shape[0]):
        matches = np.array_equal((h_int @ decoding[s].astype(np.int64)) % 2, syndromes[s].astype(np.int64))
        if matches != bool(success[s]):
            raise RuntimeError(f"shot {s}: success={bool(success[s])} but H e == syndrome is {matches}")


def _algorithm_kwargs(args: argparse.Namespace, gammas: np.ndarray | None) -> dict[str, Any]:
    common = {
        "alpha": args.alpha,
        "alpha_iteration_scaling_factor": args.alpha_scaling,
        "gamma0": args.gamma0,
        "data_scale_value": None,
        "max_data_value": None,
    }
    if args.decoder == "min_sum":
        return {"max_iter": args.max_iter, **common, "int_bits": None, "frac_bits": None}
    return {
        **common,
        "pre_iter": args.pre_iter,
        "num_sets": args.num_sets,
        "set_max_iter": args.set_max_iter,
        "gamma_dist_interval": tuple(args.gamma_interval),
        "explicit_gammas": gammas,
        "stop_nconv": args.stop_nconv,
        "stopping_criterion": args.stopping,
        "logging": True,
        "seed": 0,
    }


def _json_config(kwargs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {"check_matrix": "artifact H as scipy.sparse.csr_matrix, uint8", "error_priors": "artifact priors.npy"}
    for key, value in kwargs.items():
        if key == "explicit_gammas":
            config[key] = "gammas.npy"
        elif isinstance(value, tuple):
            config[key] = list(value)
        else:
            config[key] = value
    if args.decoder == "relay":
        config["gamma_table"] = {
            "rows": args.gamma_rows,
            "seed": args.gamma_seed,
            "interval": list(args.gamma_interval),
            "generator": "numpy.random.default_rng(seed).uniform(interval[0], interval[1], size=(rows, num_columns))",
            "leg_row": "relay leg r >= 1 uses row r mod rows",
        }
    return config


def run(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it")

    artifact = load_artifact(args.artifact)
    n, m = artifact.num_columns, artifact.num_detectors
    columns = None
    if args.shots is not None:
        syndromes, source = load_shots(args.shots, args.first, args.count, artifact)
    else:
        if args.single_columns > n:
            raise ValueError(f"--single-columns {args.single_columns} exceeds the {n} columns of H")
        syndromes, columns, source = single_column_syndromes(artifact, args.single_columns, args.seed)
    num_shots = syndromes.shape[0]
    if args.posterior_shots > num_shots:
        raise ValueError(f"--posterior-shots {args.posterior_shots} exceeds the {num_shots} syndromes decoded")

    gammas = None
    if args.decoder == "relay":
        gammas = gamma_table(args.gamma_rows, args.gamma_seed, tuple(args.gamma_interval), n)
    kwargs = _algorithm_kwargs(args, gammas)
    class_name = _CLASS_NAMES[(args.decoder, args.float)]
    problem = Problem(class_name, artifact.indptr, artifact.indices, (m, n), artifact.priors, kwargs)
    workers = min(args.workers, num_shots)

    with tempfile.TemporaryDirectory(prefix="rtd-golden-") as scratch_dir:
        scratch = Path(scratch_dir).resolve()
        # One construction up front surfaces constructor errors and native warnings once, before
        # any worker starts.
        probe_dir = scratch / "probe"
        probe_dir.mkdir()
        with contextlib.chdir(probe_dir):
            _, stderr_text = ShotDecoder(problem, probe_dir).construct()
        if stderr_text:
            logger.warning("relay_bp printed to stderr while constructing the decoder", extra={"stderr": stderr_text.strip()})
        logger.info(
            "decoding started",
            extra={"relay_bp_class": class_name, "syndromes": num_shots, "workers": workers, "source": source["type"]},
        )
        t0 = time.perf_counter()
        results = decode_all(problem, syndromes, args.posterior_shots, workers, scratch)
        decode_wall = time.perf_counter() - t0

    decoding = np.ascontiguousarray(np.stack([r.decoding for r in results]), dtype=np.uint8)
    success = np.array([r.success for r in results], dtype=np.uint8)
    iterations = np.array([r.iterations for r in results], dtype=np.int64)
    lam = log_prior_ratios(artifact.priors)
    weight = np.array(
        [decoding_weight(decoding[s], lam) if success[s] else math.inf for s in range(num_shots)], dtype=np.float64
    )

    arrays: dict[str, np.ndarray] = {
        "detectors.npy": syndromes,
        "decoding.npy": decoding,
        "success.npy": success,
        "iterations.npy": iterations,
        "weight.npy": weight,
    }
    expected = {
        "detectors.npy": (np.uint8, (num_shots, m)),
        "decoding.npy": (np.uint8, (num_shots, n)),
        "success.npy": (np.uint8, (num_shots,)),
        "iterations.npy": (np.int64, (num_shots,)),
        "weight.npy": (np.float64, (num_shots,)),
    }
    if args.posterior_shots:
        arrays["posterior.npy"] = np.ascontiguousarray(np.stack([results[s].posterior for s in range(args.posterior_shots)]))
        expected["posterior.npy"] = (np.float64, (args.posterior_shots, n))
        if not np.all(np.isfinite(arrays["posterior.npy"])):
            logger.warning("posterior ratios contain non-finite values", extra={"count": int(np.sum(~np.isfinite(arrays["posterior.npy"])))})
    if columns is not None:
        arrays["columns.npy"] = columns
        expected["columns.npy"] = (np.int64, (columns.size,))
    legs_summary: dict[str, Any] = {}
    if gammas is not None:
        legs_ptr = np.zeros(num_shots + 1, dtype=np.int64)
        legs_ptr[1:] = np.cumsum([r.legs.iterations.size for r in results])
        arrays["gammas.npy"] = gammas
        arrays["legs_ptr.npy"] = legs_ptr
        arrays["leg_iterations.npy"] = np.concatenate([r.legs.iterations for r in results])
        arrays["leg_converged.npy"] = np.concatenate([r.legs.converged for r in results])
        arrays["leg_unique_best.npy"] = np.concatenate([r.legs.unique_best for r in results])
        total_legs = int(legs_ptr[-1])
        expected["gammas.npy"] = (np.float64, (args.gamma_rows, n))
        expected["legs_ptr.npy"] = (np.int64, (num_shots + 1,))
        for name, dtype in (("leg_iterations.npy", np.int64), ("leg_converged.npy", np.uint8), ("leg_unique_best.npy", np.uint8)):
            expected[name] = (dtype, (total_legs,))
        per_shot = np.add.reduceat(arrays["leg_iterations.npy"], legs_ptr[:-1]) if num_shots else np.zeros(0)
        if np.any(np.diff(legs_ptr) < 1) or not np.array_equal(per_shot, iterations):
            raise RuntimeError("leg records do not partition the shots or do not sum to the reported iterations")
        legs_summary = {
            "legs": total_legs,
            "mean_legs": total_legs / num_shots,
            "converged_in_leg0": int(arrays["leg_converged.npy"][legs_ptr[:-1]].sum()),
            "converged_legs": int(arrays["leg_converged.npy"].sum()),
            "unique_best_legs": int(arrays["leg_unique_best.npy"].sum()),
        }

    _check_outputs(arrays, expected)
    h_int = sp.csr_matrix((np.ones(artifact.indices.size, dtype=np.int64), artifact.indices, artifact.indptr), shape=(m, n))
    _check_syndromes(h_int, syndromes, decoding, success)
    if not np.array_equal(np.isfinite(weight), success.astype(bool)):
        raise RuntimeError("weight is not finite exactly on the converged shots")
    logger.info("self-checks passed", extra={"syndromes": num_shots})

    out.mkdir(parents=True, exist_ok=True)
    # Without a manifest a half-written directory cannot pass for a complete one.
    (out / "manifest.json").unlink(missing_ok=True)
    for name in OUTPUT_FILES:
        if name not in arrays and name != "manifest.json" and (out / name).exists():
            (out / name).unlink()
            logger.warning("removed output of an earlier run", extra={"file": str(out / name)})
    for name, array in arrays.items():
        np.save(out / name, array)
        reloaded = np.load(out / name, allow_pickle=False)
        if reloaded.dtype != array.dtype or not np.array_equal(reloaded, array):
            raise RuntimeError(f"{out / name} does not read back as written")

    converged = int(success.sum())
    construct_sum = sum(r.construct_seconds for r in results)
    decode_sum = sum(r.decode_seconds for r in results)
    manifest = {
        "format_version": GOLDEN_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "decoder": args.decoder,
        "float": args.float,
        "relay_bp_class": class_name,
        "fresh_decoder_per_shot": True,
        "config": _json_config(kwargs, args),
        "artifact": {
            "path": manifest_path(artifact.path),
            "num_detectors": m,
            "num_columns": n,
            "sha256": artifact.sha256,
        },
        "syndromes": source,
        "num_shots": num_shots,
        "posterior_shots": args.posterior_shots,
        "versions": {
            "relay_bp": _relay_bp_source(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "stim": stim.__version__,
            "rtd": version("rtd"),
        },
        "timing": {
            "workers": workers,
            "total_seconds": time.perf_counter() - t_start,
            "decode_wall_seconds": decode_wall,
            "construct_seconds_sum": construct_sum,
            "decode_seconds_sum": decode_sum,
            "seconds_per_iteration": decode_sum / int(iterations.sum()) if iterations.sum() else None,
        },
        "summary": {
            "shots": num_shots,
            "converged": converged,
            "converged_fraction": converged / num_shots,
            "mean_iterations": float(iterations.mean()),
            "max_iterations": int(iterations.max()),
            **legs_summary,
        },
        "sha256": {name: _sha256(out / name) for name in sorted(arrays)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("golden outputs written", extra={"out": str(out), **manifest["summary"], **manifest["timing"]})
    return manifest


# ---------------------------------------------------------------------------------------------
# Command line


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return value


def _float_or_none(text: str) -> float | None:
    return None if text.lower() == "none" else _finite_float(text)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True, help="directory written by rtd-export")
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shots", type=Path, default=None, help="directory written by rtd-sample")
    source.add_argument("--single-columns", type=int, default=None, metavar="K", help="decode 0 and K single columns of H")
    unset = argparse.SUPPRESS
    parser.add_argument("--first", type=int, default=unset, help="first shot to decode (with --shots)")
    parser.add_argument("--count", type=int, default=unset, help="number of shots to decode (with --shots)")
    parser.add_argument("--seed", type=int, default=unset, help="column-sampling seed (with --single-columns)")
    parser.add_argument("--decoder", required=True, choices=DECODERS)
    parser.add_argument("--float", required=True, choices=FLOAT_TYPES)
    parser.add_argument("--alpha", type=_float_or_none, default=unset, help="normalisation factor, 0 = adaptive, or 'none'")
    parser.add_argument("--alpha-scaling", type=_finite_float, default=unset, help="alpha_iteration_scaling_factor")
    parser.add_argument("--gamma0", type=_float_or_none, default=unset, help="initial memory strength, or 'none'")
    parser.add_argument("--max-iter", type=int, default=unset, help="min_sum: iteration limit")
    parser.add_argument("--pre-iter", type=int, default=unset, help="relay: iteration limit of leg 0")
    parser.add_argument("--num-sets", type=int, default=unset, help="relay: legs after leg 0")
    parser.add_argument("--set-max-iter", type=int, default=unset, help="relay: iteration limit of legs >= 1")
    parser.add_argument("--stopping", choices=STOPPING_CRITERIA, default=unset, help="relay: stopping criterion")
    parser.add_argument("--stop-nconv", type=int, default=unset, help="relay: converged legs that end 'nconv'")
    parser.add_argument("--gamma-rows", type=int, default=unset, metavar="T", help="relay: rows of the gamma table")
    parser.add_argument("--gamma-seed", type=int, default=unset, help="relay: gamma-table seed")
    parser.add_argument(
        "--gamma-interval", type=_finite_float, nargs=2, default=unset, metavar=("LO", "HI"), help="relay: gamma range"
    )
    parser.add_argument("--posterior-shots", type=int, default=0, metavar="K", help="save posteriors of the first K shots")
    parser.add_argument("--workers", type=int, default=1, help="decoding processes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Require every algorithm flag of the chosen decoder, reject the others, and range-check them."""
    given = vars(args)

    def require(flags: dict[str, str], why: str) -> None:
        missing = [flag for dest, flag in flags.items() if dest not in given]
        if missing:
            raise ValueError(f"{', '.join(missing)} required {why}")

    def reject(flags: dict[str, str], why: str) -> None:
        extra = [flag for dest, flag in flags.items() if dest in given]
        if extra:
            raise ValueError(f"{', '.join(extra)} only valid {why}")

    require(_COMMON_FLAGS, f"for --decoder {args.decoder}")
    if args.decoder == "min_sum":
        require(_MIN_SUM_FLAGS, "for --decoder min_sum")
        reject(_RELAY_FLAGS, "with --decoder relay")
    else:
        require(_RELAY_FLAGS, "for --decoder relay")
        reject(_MIN_SUM_FLAGS, "with --decoder min_sum")
    if args.shots is not None:
        require(_SHOTS_FLAGS, "with --shots")
        reject(_SINGLE_COLUMN_FLAGS, "with --single-columns")
        if args.first < 0 or args.count < 1:
            raise ValueError(f"--first must be >= 0 and --count >= 1 (got {args.first}, {args.count})")
    else:
        require(_SINGLE_COLUMN_FLAGS, "with --single-columns")
        reject(_SHOTS_FLAGS, "with --shots")
        if args.single_columns < 1:
            raise ValueError(f"--single-columns must be >= 1 (got {args.single_columns})")

    if args.decoder == "min_sum" and args.max_iter < 1:
        raise ValueError(f"--max-iter must be >= 1 (got {args.max_iter})")
    if args.decoder == "relay":
        for dest, low in (("pre_iter", 1), ("num_sets", 0), ("set_max_iter", 1), ("stop_nconv", 1), ("gamma_rows", 1)):
            if given[dest] < low:
                raise ValueError(f"{_RELAY_FLAGS[dest]} must be >= {low} (got {given[dest]})")
        lo, hi = args.gamma_interval
        if not lo < hi:
            raise ValueError(f"--gamma-interval needs LO < HI (got {lo}, {hi}); relay_bp panics otherwise")
        if args.gamma0 is None:
            logger.warning("--gamma0 none: relay_bp ignores memory strengths, so every relay leg is plain BP")
    if args.alpha is not None and args.alpha < 0:
        logger.warning("relay_bp replaces a negative alpha with 1", extra={"alpha": args.alpha})
    if args.alpha == 0 and args.alpha_scaling <= 0:
        logger.warning("adaptive alpha with a non-positive --alpha-scaling is 1 in every iteration", extra={"alpha_scaling": args.alpha_scaling})
    if args.posterior_shots < 0:
        raise ValueError(f"--posterior-shots must be >= 0 (got {args.posterior_shots})")
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1 (got {args.workers})")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("run started", extra={"cli_args": cli_args})
    try:
        validate_args(args)
        run(args)
    except ImportError:
        logger.exception("relay_bp is not installed; run `uv sync --group reference`", extra={"cli_args": cli_args})
        return 1
    except Exception:
        logger.exception("run failed", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
