"""Split an XYZ decoding problem into its X-type and Z-type halves (XZ decoding).

A CSS code's X-type detectors see only the Z component of a fault and the Z-type detectors only
its X component, and the X-basis logical observables are flipped only by Z components (and vice
versa). So H and A separate into two independent problems once every fault is restricted to the
rows of one detector type:

    half x: rows = X-type detectors (det_type 0), observables = X-basis observables (type 0)
    half z: rows = Z-type detectors (det_type 1), observables = Z-basis observables (type 1)

Within a half, a fault whose restricted column is empty is invisible and dropped, and faults
whose restricted columns coincide are one event for this half: they are merged into a single
column that occurs when an odd number of its members occur,

    p = p * (1 - q) + q * (1 - p)        folded over the members in ascending XYZ index,

and that column keeps the smallest XYZ index as its representative. Columns are ordered by
representative, rows keep their XYZ order (so each half is still grouped by round).

The observable pairing is not assumed: every fault that flips a paired observable must have a
row in the half (otherwise the half could not see a logical error), and all members of a merged
column must flip the same paired observables (otherwise merging would lose information). Either
violation is an error.

Half artifact directory (an ordinary artifact plus the maps back to the XYZ problem):
    H_indptr.npy, H_indices.npy   uint32  CSR of the half's H
    A_indptr.npy, A_indices.npy   uint32  CSR of the half's A (paired observables only)
    priors.npy                    float64 merged prior of each column
    col_to_dem.npy                        the representative's entry of the XYZ col_to_dem
    col_members_ptr.npy, col_members.npy  uint32  CSR: XYZ columns merged into each column
    det_check.npy, det_round.npy, det_type.npy   the XYZ metadata of the kept rows
    det_global.npy                uint32  XYZ row of each row
    obs_global.npy                uint32  XYZ observable of each observable
    syndrome_bias.npy, observables_bias.npy      sliced, only if the XYZ artifact has them
    manifest.json                 counts, degree statistics, the XYZ circuit's checksum, "xz" block

Shots split per half the same way: detectors[:, det_global], observables[:, obs_global].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from rtd import log

logger = logging.getLogger("rtd.xz")

HALVES: dict[str, int] = {"x": 0, "z": 1}  # half -> the det_type and observable_type it keeps
ARTIFACT_FORMAT_VERSION = 1
_REQUIRED_FILES = ("H_indptr.npy", "H_indices.npy", "A_indptr.npy", "A_indices.npy", "priors.npy", "det_type.npy")


class SplitError(ValueError):
    """The XYZ problem does not separate into the requested half."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_file_atomic(path: Path, data: bytes) -> None:
    """Replaces `path` with `data` so that readers see the old file or the complete new one.

    The bytes go to a temporary file in the same directory, which is flushed to disk and then
    renamed over `path`; the directory is synced so the rename survives a power loss. A full disk
    or a crash therefore never leaves `path` empty or truncated.
    """
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        logger.error(
            "atomic write failed; the previous file (if any) is unchanged",
            exc_info=True,
            extra={"path": str(path), "bytes": len(data), "recovery": "temporary file removed"},
        )
        tmp.unlink(missing_ok=True)
        raise
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _degree_stats(degrees: np.ndarray) -> dict[str, float]:
    if degrees.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "zero": 0}
    return {
        "min": int(degrees.min()),
        "max": int(degrees.max()),
        "mean": float(degrees.mean()),
        "zero": int(np.sum(degrees == 0)),
    }


def _csr(indptr: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    data = np.ones(indices.size, dtype=np.uint8)
    return sp.csr_matrix((data, indices.astype(np.int64), indptr.astype(np.int64)), shape=shape)


def _to_uint32_csr(m: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    csr = sp.csr_matrix(m, dtype=np.uint8)
    csr.sum_duplicates()
    csr.sort_indices()
    if csr.nnz >= 2**32 or max(csr.shape) >= 2**32:
        raise SplitError(f"matrix too large for uint32 indices: shape {csr.shape}, nnz {csr.nnz}")
    return csr.indptr.astype(np.uint32), csr.indices.astype(np.uint32)


def merged_prior(probabilities: list[float]) -> float:
    """Probability that an odd number of independent events occur, folded left to right.

    Python floats are IEEE doubles and each operation rounds once (no fused multiply-add), which is
    the exact evaluation the C++ window builder uses for the same merge.
    """
    p = probabilities[0]
    for q in probabilities[1:]:
        p = p * (1.0 - q) + q * (1.0 - p)
    return p


@dataclass(frozen=True)
class HalfSplit:
    """One half of an XYZ problem. Column g of the half is XYZ columns members[ptr[g]:ptr[g+1]]."""

    half: str
    rows: np.ndarray  # uint32 [m_T], XYZ row of each row
    observables: np.ndarray  # uint32 [k_T], XYZ observable of each observable
    h_indptr: np.ndarray  # uint32 [m_T + 1]
    h_indices: np.ndarray  # uint32
    a_indptr: np.ndarray  # uint32 [k_T + 1]
    a_indices: np.ndarray  # uint32
    priors: np.ndarray  # float64 [n_T]
    members_ptr: np.ndarray  # uint32 [n_T + 1]
    members: np.ndarray  # uint32
    parent_columns: int
    dropped_columns: int  # XYZ columns with no row of this type

    @property
    def num_rows(self) -> int:
        return int(self.rows.size)

    @property
    def num_columns(self) -> int:
        return int(self.priors.size)

    @property
    def representatives(self) -> np.ndarray:
        return self.members[self.members_ptr[:-1].astype(np.int64)]

    @property
    def group_sizes(self) -> np.ndarray:
        return np.diff(self.members_ptr.astype(np.int64))

    @property
    def merged_groups(self) -> int:
        """Columns of the half that merge two or more XYZ columns."""
        return int(np.sum(self.group_sizes > 1))

    @property
    def merged_columns(self) -> int:
        """XYZ columns that belong to a merged group."""
        sizes = self.group_sizes
        return int(sizes[sizes > 1].sum())


def infer_observable_type(h: sp.csr_matrix, a: sp.csr_matrix, det_type: np.ndarray) -> np.ndarray:
    """Pairs each observable with the detector type that sees every fault flipping it.

    Used when no observable_type.npy is available. An observable paired with neither type, or with
    both (every fault flipping it has rows of both types), cannot be assigned and is an error.
    """
    a_csr = sp.csr_matrix(a)
    k = a.shape[0]
    has_type = {
        t: np.diff(sp.csc_matrix(h[np.flatnonzero(det_type == t), :]).indptr) > 0 for t in HALVES.values()
    }
    result = np.full(k, 255, dtype=np.uint8)
    for o in range(k):
        flipping = a_csr.indices[a_csr.indptr[o] : a_csr.indptr[o + 1]]
        if flipping.size == 0:
            raise SplitError(f"observable {o} is flipped by no fault; its basis cannot be inferred")
        fits = [t for t in HALVES.values() if bool(np.all(has_type[t][flipping]))]
        if len(fits) != 1:
            raise SplitError(
                f"observable {o} fits detector types {fits}; the pairing cannot be inferred, "
                "pass the shots directory (observable_type.npy)"
            )
        result[o] = fits[0]
    return result


def split_problem(
    h: sp.csr_matrix,
    a: sp.csr_matrix,
    priors: np.ndarray,
    det_type: np.ndarray,
    observable_type: np.ndarray,
    half: str,
) -> HalfSplit:
    """Restricts H to one detector type, drops empty columns, merges identical ones and keeps the
    paired observables, verifying that the pairing holds (module docstring)."""
    if half not in HALVES:
        raise SplitError(f"unknown half {half!r}; expected one of {sorted(HALVES)}")
    t = HALVES[half]
    m, n = h.shape
    k = a.shape[0]
    if a.shape[1] != n or priors.shape != (n,) or det_type.shape != (m,) or observable_type.shape != (k,):
        raise SplitError(
            f"inconsistent shapes: H {h.shape}, A {a.shape}, priors {priors.shape}, "
            f"det_type {det_type.shape}, observable_type {observable_type.shape}"
        )

    rows = np.flatnonzero(det_type == t)
    paired = np.flatnonzero(observable_type == t)
    restricted = sp.csc_matrix(h[rows, :])
    restricted.sort_indices()
    r_ptr, r_idx = restricted.indptr, restricted.indices
    degree = np.diff(r_ptr)
    kept = degree > 0

    a_paired = sp.csc_matrix(a[paired, :])
    a_paired.sort_indices()
    a_ptr, a_idx = a_paired.indptr, a_paired.indices
    flips_paired = np.diff(a_ptr) > 0
    invisible = np.flatnonzero(flips_paired & ~kept)
    if invisible.size:
        raise SplitError(
            f"half {half}: {invisible.size} faults flip a paired observable but have no detector of "
            f"type {t} (first XYZ columns {invisible[:10].tolist()}); the problem does not separate"
        )

    # Group kept columns by their restricted support; dict order = ascending representative.
    groups: dict[bytes, list[int]] = {}
    for j in np.flatnonzero(kept).tolist():
        groups.setdefault(r_idx[r_ptr[j] : r_ptr[j + 1]].tobytes(), []).append(j)

    n_t = len(groups)
    members_ptr = np.zeros(n_t + 1, dtype=np.int64)
    members = np.empty(int(kept.sum()), dtype=np.int64)
    merged = np.empty(n_t, dtype=np.float64)
    prior_list = priors.tolist()
    pos = 0
    for g, cols in enumerate(groups.values()):
        members[pos : pos + len(cols)] = cols
        pos += len(cols)
        members_ptr[g + 1] = pos
        if len(cols) == 1:
            merged[g] = prior_list[cols[0]]
            continue
        merged[g] = merged_prior([prior_list[j] for j in cols])
        rep_obs = a_idx[a_ptr[cols[0]] : a_ptr[cols[0] + 1]]
        for j in cols[1:]:
            if not np.array_equal(a_idx[a_ptr[j] : a_ptr[j + 1]], rep_obs):
                raise SplitError(
                    f"half {half}: XYZ columns {cols[0]} and {j} have the same type-{t} detectors but "
                    "flip different paired observables; merging them would lose a logical error"
                )

    reps = members[members_ptr[:-1]]
    h_half = restricted[:, reps]
    a_half = a_paired[:, reps]
    h_indptr, h_indices = _to_uint32_csr(h_half)
    a_indptr, a_indices = _to_uint32_csr(a_half)
    return HalfSplit(
        half=half,
        rows=rows.astype(np.uint32),
        observables=paired.astype(np.uint32),
        h_indptr=h_indptr,
        h_indices=h_indices,
        a_indptr=a_indptr,
        a_indices=a_indices,
        priors=merged,
        members_ptr=members_ptr.astype(np.uint32),
        members=members.astype(np.uint32),
        parent_columns=n,
        dropped_columns=int(n - kept.sum()),
    )


@dataclass(frozen=True)
class XyzArtifact:
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    h: sp.csr_matrix
    a: sp.csr_matrix
    priors: np.ndarray
    col_to_dem: np.ndarray | None
    det_check: np.ndarray | None
    det_round: np.ndarray | None
    det_type: np.ndarray
    syndrome_bias: np.ndarray | None
    observables_bias: np.ndarray | None


def load_xyz_artifact(path: Path) -> XyzArtifact:
    """Loads an artifact written by rtd-export, verifying every file against its manifest."""
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    if "xz" in manifest:
        raise SplitError(f"{path} is already one half of an XZ split ({manifest['xz'].get('half')})")
    recorded = manifest.get("sha256", {})
    for name in _REQUIRED_FILES:
        if name not in recorded:
            raise SplitError(f"artifact manifest {manifest_file} has no sha256 for {name}")
    t0 = time.perf_counter()
    for name, expected in recorded.items():
        actual = sha256_file(path / name)
        if actual != expected:
            raise SplitError(f"{path / name} does not match its manifest checksum ({actual} != {expected})")
    logger.debug("artifact checksums verified", extra={"artifact": str(path), "seconds": time.perf_counter() - t0})

    def load(name: str) -> np.ndarray | None:
        file = path / name
        return np.load(file, allow_pickle=False) if file.is_file() else None

    m, n, k = int(manifest["num_detectors"]), int(manifest["num_columns"]), int(manifest["num_observables"])
    h = _csr(np.load(path / "H_indptr.npy"), np.load(path / "H_indices.npy"), (m, n))
    a = _csr(np.load(path / "A_indptr.npy"), np.load(path / "A_indices.npy"), (k, n))
    priors = np.load(path / "priors.npy")
    det_type = np.load(path / "det_type.npy")
    if priors.dtype != np.float64 or priors.shape != (n,):
        raise SplitError(f"priors.npy is {priors.dtype} {priors.shape}; expected float64 ({n},)")
    if det_type.shape != (m,):
        raise SplitError(f"det_type.npy has shape {det_type.shape}; expected ({m},)")
    unknown = np.setdiff1d(np.unique(det_type), list(HALVES.values()))
    if unknown.size:
        raise SplitError(f"det_type.npy holds values {unknown.tolist()} besides 0 (X) and 1 (Z)")
    return XyzArtifact(
        path=path,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_file),
        h=h,
        a=a,
        priors=priors,
        col_to_dem=load("col_to_dem.npy"),
        det_check=load("det_check.npy"),
        det_round=load("det_round.npy"),
        det_type=det_type,
        syndrome_bias=load("syndrome_bias.npy"),
        observables_bias=load("observables_bias.npy"),
    )


def _write_dir_atomically(out: Path, overwrite: bool, write: Any) -> None:
    """Writes into a sibling temporary directory, then renames it to `out`."""
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.parent / f".{out.name}.partial-{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        write(partial)
        if out.exists():
            shutil.rmtree(out)
        partial.rename(out)
    except BaseException:
        logger.exception("writing failed; removing the partial directory", extra={"out": str(out), "partial": str(partial)})
        shutil.rmtree(partial, ignore_errors=True)
        raise


def write_half(parent: XyzArtifact, split: HalfSplit, out: Path, observable_type_source: str, overwrite: bool = False) -> dict[str, Any]:
    """Writes one half as an artifact directory loadable by rtd_decode and rtd.golden."""
    files: dict[str, np.ndarray] = {
        "H_indptr.npy": split.h_indptr,
        "H_indices.npy": split.h_indices,
        "A_indptr.npy": split.a_indptr,
        "A_indices.npy": split.a_indices,
        "priors.npy": split.priors,
        "col_members_ptr.npy": split.members_ptr,
        "col_members.npy": split.members,
        "det_global.npy": split.rows,
        "obs_global.npy": split.observables,
        "det_type.npy": parent.det_type[split.rows],
    }
    reps = split.representatives.astype(np.int64)
    rows = split.rows.astype(np.int64)
    if parent.col_to_dem is not None:
        files["col_to_dem.npy"] = parent.col_to_dem[reps]
    if parent.det_check is not None:
        files["det_check.npy"] = parent.det_check[rows]
    if parent.det_round is not None:
        files["det_round.npy"] = parent.det_round[rows]
    if parent.syndrome_bias is not None:
        files["syndrome_bias.npy"] = parent.syndrome_bias[rows]
    if parent.observables_bias is not None:
        files["observables_bias.npy"] = parent.observables_bias[split.observables.astype(np.int64)]

    row_degree = np.diff(split.h_indptr.astype(np.int64))
    col_degree = np.bincount(split.h_indices.astype(np.int64), minlength=split.num_columns)
    if row_degree.size and row_degree.min() == 0:
        logger.warning(
            "half has detectors no fault flips",
            extra={"half": split.half, "empty_rows": int(np.sum(row_degree == 0))},
        )
    if split.observables.size == 0:
        logger.warning("half has no paired observable; it can detect but never fail", extra={"half": split.half})

    manifest: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "source_circuit": parent.manifest.get("source_circuit"),
        "num_detectors": split.num_rows,
        "num_columns": split.num_columns,
        "num_observables": int(split.observables.size),
        "nnz_H": int(split.h_indices.size),
        "nnz_A": int(split.a_indices.size),
        "row_degree": _degree_stats(row_degree),
        "column_degree": _degree_stats(col_degree),
        "prior_range": [float(split.priors.min()), float(split.priors.max())] if split.num_columns else None,
        "pruning": parent.manifest.get("pruning"),
        "from_dem": parent.manifest.get("from_dem"),
        "versions": {**parent.manifest.get("versions", {}), "xz_split": {"rtd": version("rtd"), "numpy": np.__version__, "scipy": version("scipy")}},
        "xz": {
            "half": split.half,
            "detector_type": HALVES[split.half],
            "observable_type": HALVES[split.half],
            "observable_type_source": observable_type_source,
            "parent": str(parent.path),
            "parent_manifest_sha256": parent.manifest_sha256,
            "parent_columns": split.parent_columns,
            "dropped_columns": split.dropped_columns,
            "merged_groups": split.merged_groups,
            "merged_columns": split.merged_columns,
            "merge_rule": "p = p * (1.0 - q) + q * (1.0 - p), ascending XYZ index",
        },
    }

    def write(directory: Path) -> None:
        for name, array in files.items():
            np.save(directory / name, array)
        manifest["sha256"] = {name: sha256_file(directory / name) for name in sorted(files)}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _write_dir_atomically(out, overwrite, write)
    logger.info(
        "half written",
        extra={
            "half": split.half,
            "out": str(out),
            "rows": split.num_rows,
            "columns": split.num_columns,
            "edges": manifest["nnz_H"],
            "observables": manifest["num_observables"],
            "merged_groups": split.merged_groups,
            "merged_columns": split.merged_columns,
            "dropped_columns": split.dropped_columns,
        },
    )
    return manifest


def resolve_observable_type(parent: XyzArtifact, observable_type: np.ndarray | None, shots: Path | None) -> tuple[np.ndarray, str]:
    k = parent.a.shape[0]
    if observable_type is not None:
        source = "given"
    elif shots is not None:
        file = shots / "observable_type.npy"
        if not file.is_file():
            raise FileNotFoundError(f"{file} not found; omit the shots directory to infer the pairing")
        observable_type = np.load(file, allow_pickle=False)
        source = f"shots:{shots}"
    else:
        observable_type = infer_observable_type(parent.h, parent.a, parent.det_type)
        logger.info("observable pairing inferred from the matrices", extra={"observable_type": observable_type.tolist()})
        return observable_type, "inferred"
    observable_type = np.asarray(observable_type, dtype=np.uint8)
    if observable_type.shape != (k,):
        raise SplitError(f"observable_type has shape {observable_type.shape}; the artifact has {k} observables")
    return observable_type, source


def split_artifact(
    artifact: Path,
    out: Path,
    observable_type: np.ndarray | None = None,
    shots: Path | None = None,
    overwrite: bool = False,
    halves: tuple[str, ...] = ("x", "z"),
) -> dict[str, dict[str, Any]]:
    """Writes `out/x` and `out/z` (or the requested halves) from an XYZ artifact directory."""
    t0 = time.perf_counter()
    parent = load_xyz_artifact(artifact)
    logger.info(
        "XYZ artifact loaded",
        extra={
            "artifact": str(artifact),
            "rows": parent.h.shape[0],
            "columns": parent.h.shape[1],
            "edges": int(parent.h.nnz),
            "observables": parent.a.shape[0],
            "seconds": time.perf_counter() - t0,
        },
    )
    obs_type, source = resolve_observable_type(parent, observable_type, shots)
    manifests = {}
    for half in halves:
        t0 = time.perf_counter()
        split = split_problem(parent.h, parent.a, parent.priors, parent.det_type, obs_type, half)
        logger.info("half split", extra={"half": half, "seconds": time.perf_counter() - t0})
        manifests[half] = write_half(parent, split, out / half, source, overwrite)
    return manifests


# ---------------------------------------------------------------------------------------------
# Shots


@dataclass(frozen=True)
class HalfIndex:
    """Which XYZ detectors and observables one half keeps."""

    half: str
    det_global: np.ndarray  # int64 [m_T]
    obs_global: np.ndarray  # int64 [k_T]
    circuit_sha256: str | None
    artifact: Path | None = None

    @classmethod
    def from_artifact(cls, half_dir: Path) -> HalfIndex:
        manifest = json.loads((half_dir / "manifest.json").read_text())
        if "xz" not in manifest:
            raise SplitError(f"{half_dir} is not one half of an XZ split (its manifest has no 'xz' block)")
        for name in ("det_global.npy", "obs_global.npy"):
            expected = manifest.get("sha256", {}).get(name)
            if expected is None or sha256_file(half_dir / name) != expected:
                raise SplitError(f"{half_dir / name} is missing or does not match its manifest checksum")
        return cls(
            half=manifest["xz"]["half"],
            det_global=np.load(half_dir / "det_global.npy").astype(np.int64),
            obs_global=np.load(half_dir / "obs_global.npy").astype(np.int64),
            circuit_sha256=(manifest.get("source_circuit") or {}).get("sha256"),
            artifact=half_dir,
        )

    @classmethod
    def from_split(cls, split: HalfSplit, circuit_sha256: str | None = None) -> HalfIndex:
        return cls(
            half=split.half,
            det_global=split.rows.astype(np.int64),
            obs_global=split.observables.astype(np.int64),
            circuit_sha256=circuit_sha256,
        )


def split_shots(detectors: np.ndarray, observables: np.ndarray, half: HalfIndex) -> tuple[np.ndarray, np.ndarray]:
    """The half's syndromes and observable flips: columns det_global and obs_global of the XYZ ones."""
    if detectors.ndim != 2 or observables.ndim != 2 or detectors.shape[0] != observables.shape[0]:
        raise SplitError(f"shots have shapes {detectors.shape} and {observables.shape}; expected [S, m] and [S, k]")
    if half.det_global.size and half.det_global.max() >= detectors.shape[1]:
        raise SplitError(f"half {half.half} keeps XYZ row {int(half.det_global.max())}; the shots have {detectors.shape[1]}")
    if half.obs_global.size and half.obs_global.max() >= observables.shape[1]:
        raise SplitError(f"half {half.half} keeps observable {int(half.obs_global.max())}; the shots have {observables.shape[1]}")
    return (
        np.ascontiguousarray(detectors[:, half.det_global], dtype=np.uint8),
        np.ascontiguousarray(observables[:, half.obs_global], dtype=np.uint8),
    )


def write_shots_dir(
    out: Path,
    detectors: np.ndarray,
    observables: np.ndarray,
    manifest: dict[str, Any],
    circuit_sha256: str,
    metadata: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Writes a shots directory that rtd_decode's load_shots accepts.

    load_shots links shots to an artifact through manifest["sha256"]["circuit.stim"]: the checksum
    of the circuit both were produced from, recorded here even though the circuit file itself is
    not copied into the directory.
    """
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "detectors.npy", np.ascontiguousarray(detectors, dtype=np.uint8))
    np.save(out / "observables.npy", np.ascontiguousarray(observables, dtype=np.uint8))
    for name, array in (metadata or {}).items():
        np.save(out / name, array)
    full = dict(manifest)
    full["shots"] = int(detectors.shape[0])
    full["sha256"] = {
        "circuit.stim": circuit_sha256,
        "detectors.npy": sha256_file(out / "detectors.npy"),
        "observables.npy": sha256_file(out / "observables.npy"),
    }
    # Written last and atomically: a complete manifest marks a complete directory.
    write_file_atomic(out / "manifest.json", (json.dumps(full, indent=2) + "\n").encode())
    return full


def split_shots_dir(
    shots: Path,
    split: Path,
    out: Path,
    first: int = 0,
    count: int | None = None,
    overwrite: bool = False,
    halves: tuple[str, ...] = ("x", "z"),
) -> dict[str, dict[str, Any]]:
    """Writes `out/<half>` shots directories from an XYZ shots directory and an XZ split."""
    manifest_file = shots / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"shots manifest not found: {manifest_file}")
    parent_manifest = json.loads(manifest_file.read_text())
    circuit_sha = parent_manifest.get("sha256", {}).get("circuit.stim")
    if circuit_sha is None:
        raise SplitError(f"{manifest_file} records no circuit checksum; the halves could not be linked to it")
    t0 = time.perf_counter()
    for name in ("detectors.npy", "observables.npy"):
        expected = parent_manifest.get("sha256", {}).get(name)
        if expected is None or sha256_file(shots / name) != expected:
            raise SplitError(f"{shots / name} is missing or does not match its manifest checksum")
    logger.debug("shots checksums verified", extra={"shots": str(shots), "seconds": time.perf_counter() - t0})

    all_dets = np.load(shots / "detectors.npy", mmap_mode="r", allow_pickle=False)
    all_obs = np.load(shots / "observables.npy", mmap_mode="r", allow_pickle=False)
    total = all_dets.shape[0]
    count = total - first if count is None else count
    if first < 0 or count < 0 or first + count > total:
        raise ValueError(f"--first {first} --count {count} does not fit the {total} shots in {shots}")
    dets = np.asarray(all_dets[first : first + count])
    obs = np.asarray(all_obs[first : first + count])

    results = {}
    for half in halves:
        index = HalfIndex.from_artifact(split / half)
        if index.circuit_sha256 != circuit_sha:
            raise SplitError(
                f"split half {split / half} comes from circuit {index.circuit_sha256}, the shots from {circuit_sha}"
            )
        d, o = split_shots(dets, obs, index)
        manifest = {k: v for k, v in parent_manifest.items() if k not in ("sha256", "shots", "created", "sampling")}
        manifest["created"] = datetime.now(timezone.utc).isoformat()
        manifest["counts"] = {**parent_manifest.get("counts", {}), "detectors": int(d.shape[1]), "observables": int(o.shape[1])}
        manifest["xz"] = {
            "half": half,
            "parent_shots": str(shots),
            "parent_manifest_sha256": sha256_file(manifest_file),
            "first": first,
            "count": count,
            "artifact": str(split / half),
        }
        metadata = {
            name: np.load(split / half / name)
            for name in ("det_check.npy", "det_round.npy", "det_type.npy")
            if (split / half / name).is_file()
        }
        metadata["observable_type.npy"] = np.full(o.shape[1], HALVES[half], dtype=np.uint8)

        def write(
            directory: Path, half: str = half, d: np.ndarray = d, o: np.ndarray = o, manifest: dict = manifest, metadata: dict = metadata
        ) -> None:
            results[half] = write_shots_dir(directory, d, o, manifest, circuit_sha, metadata)

        _write_dir_atomically(out / half, overwrite, write)
        logger.info("half shots written", extra={"half": half, "out": str(out / half), "shots": count, "detectors": int(d.shape[1])})
    return results


# ---------------------------------------------------------------------------------------------
# Command-line tools


def _parse_split_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, required=True, help="XYZ artifact written by rtd-export")
    parser.add_argument(
        "--shots",
        type=Path,
        default=None,
        help="shots directory whose observable_type.npy gives the observable bases (default: infer from the matrices)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <artifact>_xz)")
    parser.add_argument("--halves", default="x,z", help="comma-separated halves to write (default x,z)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_split_shots_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an XYZ shots directory into the halves of an XZ split.")
    parser.add_argument("--shots", type=Path, required=True, help="XYZ shots directory written by rtd-sample")
    parser.add_argument("--split", type=Path, required=True, help="XZ split directory written by rtd-xz-split")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <shots>_xz)")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--halves", default="x,z", help="comma-separated halves to write (default x,z)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _halves(text: str) -> tuple[str, ...]:
    halves = tuple(h.strip() for h in text.split(",") if h.strip())
    unknown = [h for h in halves if h not in HALVES]
    if not halves or unknown:
        raise ValueError(f"--halves must list some of {sorted(HALVES)}, got {text!r}")
    return halves


def main(argv: list[str] | None = None) -> int:
    args = _parse_split_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("run started", extra={"cli_args": cli_args})
    t0 = time.perf_counter()
    try:
        out = args.out if args.out is not None else args.artifact.with_name(args.artifact.name + "_xz")
        split_artifact(args.artifact, out, shots=args.shots, overwrite=args.overwrite, halves=_halves(args.halves))
    except Exception:
        logger.exception("run failed; nothing was renamed into place", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed", extra={"seconds": time.perf_counter() - t0})
    return 0


def main_shots(argv: list[str] | None = None) -> int:
    args = _parse_split_shots_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("run started", extra={"cli_args": cli_args})
    t0 = time.perf_counter()
    try:
        out = args.out if args.out is not None else args.shots.with_name(args.shots.name + "_xz")
        split_shots_dir(args.shots, args.split, out, args.first, args.count, args.overwrite, _halves(args.halves))
    except Exception:
        logger.exception("run failed; nothing was renamed into place", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed", extra={"seconds": time.perf_counter() - t0})
    return 0


if __name__ == "__main__":
    sys.exit(main())
