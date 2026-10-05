"""Inner decoders for the windowed reference decoder (rtd.window_ref).

Every inner decoder solves one window: given the window's check matrix H (scipy CSR, local rows
by local columns), its priors p and the window syndrome s, it returns a correction e with
H e = s when it finds one. The result is reported as the ascending list of local columns with
e_j = 1 (the support), the BP iterations and legs spent, whether e satisfies the syndrome, and the
weight W(e) = sum of ln((1 - p_j) / p_j) over the support (+inf when no solution was found).

Weights follow rtd.golden's rule: math.log per column, added one at a time in ascending local
column index, skipping non-finite terms, so double results are reproducible bit for bit.

Available decoders:
    relay       IBM's relay_bp (Relay-BP), one freshly constructed decoder per window decode, with
                an explicit memory-strength table per window shape
    pymatching  minimum-weight perfect matching (graph-like windows only: every column touches
                one or two rows)
    bplsd       BP+LSD from the ldpc package
    brute       exact minimum weight by enumerating the whole solution space (tiny windows only)
"""

from __future__ import annotations

import contextlib
import logging
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from rtd import gf2

logger = logging.getLogger("rtd.window_ref")

INNER_KINDS = ("relay", "pymatching", "bplsd", "brute")


@dataclass(frozen=True)
class InnerContext:
    """What the stream loop tells an inner decoder about the window it is solving."""

    shot: int
    window: int
    attempt: int
    shape: int
    gamma_stream: int  # shot + (window << 32) + (attempt << 56), modulo 2^64
    converge: np.ndarray  # uint8 per local row: 1 = H e = s must hold on this row
    iteration_cap: int | None


@dataclass(frozen=True)
class InnerResult:
    support: np.ndarray  # int64, ascending local column indices with e_j = 1
    iterations: int
    legs: int
    converged: bool
    weight: float  # W(e) if converged, else +inf
    cap_hit: bool = False
    num_optimal: int = 1  # brute force only: how many solutions share the minimum weight


class InnerDecoder(Protocol):
    name: str

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------------------------
# Shared helpers


def log_ratios(priors: np.ndarray) -> list[float]:
    """ln((1 - p) / p) per column with the C library's log, as relay_bp and rtd.golden compute it."""
    return [math.log((1.0 - p) / p) for p in priors.tolist()]


def support_weight(support: np.ndarray, lam: list[float]) -> float:
    """Sum of the finite lam[j] over the support, added one at a time in ascending j."""
    total = 0.0
    for j in np.sort(support).tolist():
        value = lam[j]
        if math.isfinite(value):
            total += value
    return total


def syndrome_of(h: sp.csr_matrix, support: np.ndarray) -> np.ndarray:
    """H e mod 2 for the e whose support is given, as uint8 per row."""
    e = np.zeros(h.shape[1], dtype=np.int64)
    e[support] = 1
    return (np.asarray(h @ e).ravel() % 2).astype(np.uint8)


def satisfies(h: sp.csr_matrix, support: np.ndarray, syndrome: np.ndarray, rows: np.ndarray | None = None) -> bool:
    """True when H e = s holds on every row (or on the rows where `rows` is 1)."""
    diff = syndrome_of(h, support) ^ syndrome.astype(np.uint8)
    if rows is not None:
        diff = diff[rows.astype(bool)]
    return not bool(diff.any())


class _ShapeCache:
    """Per-shape objects built from a window matrix, rebuilt if a different matrix arrives under
    the same shape index (a decoder reused across plans)."""

    def __init__(self) -> None:
        self._entries: dict[int, tuple[sp.csr_matrix, np.ndarray, Any]] = {}

    def get(self, shape: int, h: sp.csr_matrix, priors: np.ndarray) -> Any | None:
        entry = self._entries.get(shape)
        if entry is not None and entry[0] is h and entry[1] is priors:
            return entry[2]
        return None

    def put(self, shape: int, h: sp.csr_matrix, priors: np.ndarray, value: Any) -> Any:
        self._entries[shape] = (h, priors, value)
        return value


def _require_all_rows(ctx: InnerContext, decoder: str) -> None:
    if not bool(np.all(ctx.converge)):
        raise ValueError(
            f"{decoder} checks H e = s on every row; convergence rows C' < W need a decoder with a row mask"
        )


def _require_no_cap(ctx: InnerContext, decoder: str) -> None:
    if ctx.iteration_cap is not None:
        raise ValueError(f"{decoder} has no iteration budget; iteration_cap must be null")


# ---------------------------------------------------------------------------------------------
# Relay-BP (relay_bp)


@dataclass(frozen=True)
class RelayParams:
    """relay_bp constructor arguments, named as in rtd-golden's flags."""

    float_type: str  # "f32" | "f64"
    alpha: float | None  # None = scale 1; 0 = relay_bp's iteration-dependent rule
    alpha_scaling: float
    gamma0: float | None
    pre_iter: int
    num_sets: int
    set_max_iter: int
    stopping: str  # "nconv" | "all" | "pre_iter"
    stop_nconv: int
    gamma_interval: tuple[float, float]

    def kwargs(self, gammas: np.ndarray) -> dict[str, Any]:
        """The keyword arguments rtd.golden passes to relay_bp's RelayDecoder."""
        return {
            "alpha": self.alpha,
            "alpha_iteration_scaling_factor": self.alpha_scaling,
            "gamma0": self.gamma0,
            "data_scale_value": None,
            "max_data_value": None,
            "pre_iter": self.pre_iter,
            "num_sets": self.num_sets,
            "set_max_iter": self.set_max_iter,
            "gamma_dist_interval": tuple(self.gamma_interval),
            "explicit_gammas": gammas,
            "stop_nconv": self.stop_nconv,
            "stopping_criterion": self.stopping,
            "logging": True,
            "seed": 0,
        }

    def check_params(self) -> dict[str, Any]:
        """The subset rtd.golden.check_relay_legs reads."""
        return {
            "pre_iter": self.pre_iter,
            "num_sets": self.num_sets,
            "set_max_iter": self.set_max_iter,
            "stopping_criterion": self.stopping,
            "stop_nconv": self.stop_nconv,
        }


@dataclass(frozen=True)
class SeededShapeGammas:
    """Memory strengths of window shape i: default_rng([seed, i]).uniform(low, high, (rows, n_i)).

    Relay leg r >= 1 of any window of shape i uses row r mod rows (relay_bp's rule)."""

    rows: int
    seed: int
    low: float
    high: float

    def __call__(self, shape: int, n: int) -> np.ndarray:
        table = np.random.default_rng([self.seed, shape]).uniform(self.low, self.high, (self.rows, n))
        return np.ascontiguousarray(table, dtype=np.float64)


@dataclass(frozen=True)
class FixedGammas:
    """One table for every shape (the whole-shot identity check uses the golden's own table)."""

    table: np.ndarray = field(repr=False)

    def __call__(self, shape: int, n: int) -> np.ndarray:
        if self.table.ndim != 2 or self.table.shape[1] != n:
            raise ValueError(f"fixed gamma table has shape {self.table.shape}, window shape {shape} has {n} columns")
        return np.ascontiguousarray(self.table, dtype=np.float64)


class RelayInner:
    """relay_bp's Relay-BP, one newly constructed decoder per window decode.

    relay_bp keeps per-leg records that are not reset between decodes and writes them to
    relay_logging.out in the current directory, so each decode gets a fresh decoder and runs
    inside a private working directory (rtd.golden.ShotDecoder does both)."""

    name = "relay"

    def __init__(self, params: RelayParams, gammas: SeededShapeGammas | FixedGammas):
        from rtd import golden  # imports relay_bp lazily

        self._golden = golden
        self._params = params
        self._gammas = gammas
        self._class_name = golden._CLASS_NAMES[("relay", params.float_type)]
        self._workdir = Path(tempfile.mkdtemp(prefix="rtd-window-relay-")).resolve()
        self._cache = _ShapeCache()
        self.gamma_tables: dict[int, np.ndarray] = {}
        logger.debug("relay inner decoder ready", extra={"relay_class": self._class_name, "workdir": str(self._workdir)})

    def _shot_decoder(self, h: sp.csr_matrix, priors: np.ndarray, shape: int) -> Any:
        cached = self._cache.get(shape, h, priors)
        if cached is not None:
            return cached
        n = h.shape[1]
        table = self._gammas(shape, n)
        self.gamma_tables[shape] = table
        problem = self._golden.Problem(
            class_name=self._class_name,
            indptr=h.indptr.astype(np.int64),
            indices=h.indices.astype(np.int64),
            shape=(int(h.shape[0]), int(n)),
            priors=np.ascontiguousarray(priors, dtype=np.float64),
            kwargs=self._params.kwargs(table),
        )
        decoder = self._golden.ShotDecoder(problem, self._workdir)
        logger.debug("relay decoder prepared for shape", extra={"shape_index": shape, "rows": h.shape[0], "columns": n})
        return self._cache.put(shape, h, priors, decoder)

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        _require_all_rows(ctx, "relay_bp")
        _require_no_cap(ctx, "relay_bp")
        decoder = self._shot_decoder(h, priors, ctx.shape)
        try:
            with contextlib.chdir(self._workdir):
                shot = decoder.decode(ctx.shot, np.ascontiguousarray(syndrome, dtype=np.uint8), False)
            self._golden.check_relay_legs(shot.legs, shot.success, shot.iterations, self._params.check_params())
        except Exception:
            logger.exception(
                "relay_bp window decode failed; aborting the shot",
                extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt, "shape_index": ctx.shape},
            )
            raise
        support = np.flatnonzero(shot.decoding).astype(np.int64)
        if shot.success and not satisfies(h, support, syndrome):
            raise RuntimeError(f"shot {ctx.shot} window {ctx.window}: relay_bp reports success but H e != s")
        weight = support_weight(support, log_ratios(priors)) if shot.success else math.inf
        return InnerResult(
            support=support,
            iterations=int(shot.iterations),
            legs=int(shot.legs.iterations.size),
            converged=bool(shot.success),
            weight=weight,
        )

    def close(self) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Minimum-weight perfect matching (PyMatching)


class MatchingInner:
    """PyMatching on a graph-like window: every column touches one row (an edge to the boundary)
    or two rows. Parallel edges keep the smaller weight, which does not change the minimum."""

    name = "pymatching"

    def __init__(self) -> None:
        self._cache = _ShapeCache()

    def _matching(self, h: sp.csr_matrix, priors: np.ndarray, shape: int) -> Any:
        cached = self._cache.get(shape, h, priors)
        if cached is not None:
            return cached
        import pymatching

        degrees = np.diff(h.tocsc().indptr)
        if degrees.size and (degrees.min() < 1 or degrees.max() > 2):
            raise ValueError(f"PyMatching needs every column to touch 1 or 2 rows; shape {shape} has degrees {degrees.min()}..{degrees.max()}")
        n = h.shape[1]
        matching = pymatching.Matching.from_check_matrix(
            sp.csc_matrix(h, dtype=np.uint8),
            weights=np.array(log_ratios(priors), dtype=np.float64),
            faults_matrix=sp.identity(n, dtype=np.uint8, format="csc"),
            merge_strategy="smallest-weight",
        )
        logger.debug("matching graph built for shape", extra={"shape_index": shape, "rows": h.shape[0], "columns": n})
        return self._cache.put(shape, h, priors, matching)

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        _require_all_rows(ctx, "PyMatching")
        matching = self._matching(h, priors, ctx.shape)
        try:
            e = np.asarray(matching.decode(np.ascontiguousarray(syndrome, dtype=np.uint8)), dtype=np.uint8)
        except Exception:
            logger.exception(
                "PyMatching window decode failed; the window reports no solution",
                extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt, "shape_index": ctx.shape},
            )
            return InnerResult(np.zeros(0, dtype=np.int64), 0, 1, False, math.inf)
        support = np.flatnonzero(e).astype(np.int64)
        ok = satisfies(h, support, syndrome)
        if not ok:
            logger.warning(
                "PyMatching correction does not satisfy the window syndrome",
                extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt},
            )
        weight = support_weight(support, log_ratios(priors)) if ok else math.inf
        return InnerResult(support, 0, 1, ok, weight)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------
# BP+LSD (ldpc)


@dataclass(frozen=True)
class BpLsdParams:
    """ldpc.BpLsdDecoder settings. The defaults are Lee, English, Bartlett's: min-sum BP with
    30 iterations, then LSD of order 0, run on every decode as their decoder does."""

    max_iter: int = 30
    bp_method: str = "minimum_sum"
    ms_scaling_factor: float = 1.0
    schedule: str = "parallel"
    lsd_method: str = "LSD_0"
    lsd_order: int = 0
    always_run_lsd: bool = True


class BpLsdInner:
    name = "bplsd"

    def __init__(self, params: BpLsdParams | None = None):
        self._params = params or BpLsdParams()
        self._cache = _ShapeCache()

    def _decoder(self, h: sp.csr_matrix, priors: np.ndarray, shape: int) -> Any:
        cached = self._cache.get(shape, h, priors)
        if cached is not None:
            return cached
        from ldpc import BpLsdDecoder

        p = self._params
        decoder = BpLsdDecoder(
            sp.csr_matrix(h, dtype=np.uint8),
            error_channel=[float(x) for x in priors],
            max_iter=p.max_iter,
            bp_method=p.bp_method,
            ms_scaling_factor=p.ms_scaling_factor,
            schedule=p.schedule,
            lsd_method=p.lsd_method,
            lsd_order=p.lsd_order,
            always_run_lsd=p.always_run_lsd,
        )
        logger.debug("BP+LSD decoder built for shape", extra={"shape_index": shape, "rows": h.shape[0], "columns": h.shape[1]})
        return self._cache.put(shape, h, priors, decoder)

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        _require_all_rows(ctx, "BP+LSD")
        _require_no_cap(ctx, "BP+LSD")
        decoder = self._decoder(h, priors, ctx.shape)
        try:
            e = np.asarray(decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8)), dtype=np.uint8)
            iterations = int(decoder.iter)
        except Exception:
            logger.exception(
                "BP+LSD window decode failed; the window reports no solution",
                extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt, "shape_index": ctx.shape},
            )
            return InnerResult(np.zeros(0, dtype=np.int64), 0, 1, False, math.inf)
        support = np.flatnonzero(e).astype(np.int64)
        ok = satisfies(h, support, syndrome)
        weight = support_weight(support, log_ratios(priors)) if ok else math.inf
        return InnerResult(support, iterations, 1, ok, weight)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------
# Exact minimum weight by enumeration


def minimum_weight_solutions(h_dense: np.ndarray, syndrome: np.ndarray, lam: list[float], max_free: int) -> tuple[np.ndarray, float, int] | None:
    """All solutions of H e = s are e0 + span(null space of H). Enumerate them, return the one
    of least weight (ties: the lexicographically smallest support), its weight and the number of
    solutions sharing that weight. None if H e = s has no solution.

    Solutions are bit masks in uint64, so at most 64 columns; 2^(n - rank) solutions are listed,
    at most 2^max_free."""
    rows, n = h_dense.shape
    if n > 64:
        raise ValueError(f"brute force handles at most 64 columns, window has {n}")
    augmented = np.hstack([h_dense.astype(np.uint8) & 1, (syndrome.astype(np.uint8) & 1).reshape(-1, 1)])
    rref, pivots = gf2.row_reduce(augmented)
    if n in pivots:
        return None
    x0 = np.zeros(n, dtype=np.uint8)
    for r, pc in enumerate(pivots):
        x0[pc] = rref[r, n]
    basis = gf2.nullspace(h_dense) if n else np.zeros((0, 0), dtype=np.uint8)
    if basis.shape[0] > max_free:
        raise ValueError(f"window has 2^{basis.shape[0]} solutions, more than the brute-force limit 2^{max_free}")
    bit = np.left_shift(np.uint64(1), np.arange(n, dtype=np.uint64))

    def mask(v: np.ndarray) -> np.uint64:
        return np.bitwise_or.reduce(bit[v.astype(bool)], initial=np.uint64(0))

    solutions = np.array([mask(x0)], dtype=np.uint64)
    for b in basis:
        solutions = np.concatenate([solutions, solutions ^ mask(b)])
    weights = np.zeros(solutions.size, dtype=np.float64)
    for j in range(n):  # ascending j, adding 0.0 where e_j = 0, which leaves every partial sum unchanged
        if math.isfinite(lam[j]):
            weights += np.where((solutions >> np.uint64(j)) & np.uint64(1), lam[j], 0.0)
    best = float(weights.min())
    optimal = solutions[weights == best]
    supports = sorted(tuple(j for j in range(n) if (int(s) >> j) & 1) for s in optimal)
    return np.array(supports[0], dtype=np.int64), best, int(optimal.size)


class BruteForceInner:
    """Exact minimum-weight decoder for tiny windows; also reports ties."""

    name = "brute"

    def __init__(self, max_free: int = 22):
        self._max_free = max_free

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        _require_all_rows(ctx, "brute force")
        lam = log_ratios(priors)
        found = minimum_weight_solutions(h.toarray().astype(np.uint8), syndrome, lam, self._max_free)
        if found is None:
            logger.warning("window syndrome has no solution", extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt})
            return InnerResult(np.zeros(0, dtype=np.int64), 0, 1, False, math.inf)
        support, _, count = found
        # Recomputed with the shared rule so every inner decoder reports weights identically.
        return InnerResult(support, 0, 1, True, support_weight(support, lam), num_optimal=count)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------
# Picklable description, so worker processes can build their own decoder


@dataclass(frozen=True)
class InnerSpec:
    kind: str
    relay: RelayParams | None = None
    gammas: SeededShapeGammas | FixedGammas | None = None
    bplsd: BpLsdParams | None = None
    brute_max_free: int = 22

    def build(self) -> InnerDecoder:
        t0 = time.perf_counter()
        if self.kind == "relay":
            if self.relay is None or self.gammas is None:
                raise ValueError("the relay inner decoder needs relay parameters and a gamma table source")
            decoder: InnerDecoder = RelayInner(self.relay, self.gammas)
        elif self.kind == "pymatching":
            decoder = MatchingInner()
        elif self.kind == "bplsd":
            decoder = BpLsdInner(self.bplsd)
        elif self.kind == "brute":
            decoder = BruteForceInner(self.brute_max_free)
        else:
            raise ValueError(f"unknown inner decoder {self.kind!r}; expected one of {INNER_KINDS}")
        logger.debug("inner decoder built", extra={"inner": self.kind, "seconds": time.perf_counter() - t0})
        return decoder
