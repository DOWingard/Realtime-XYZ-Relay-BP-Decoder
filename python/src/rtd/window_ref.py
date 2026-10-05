"""Sliding-window decoding reference: window plans and the stream decoding loop, written for
clarity over speed, with a pluggable inner decoder (rtd.window_ref_inner).

The problem. A memory experiment has rounds r = 1 .. Rt (Rt = R + 1, the last one the noiseless
readout). Its m detectors are grouped by round, M per round, rows sorted by round. Fault column j
of H flips detectors in rounds s(j) and s(j) + 1 only, where s(j) is the earliest round it
touches. Column j of A lists the logical observables it flips.

Windows. Window k starts at round t_k = 1 + k C and spans W rounds; it decodes the columns with
s(j) in [t_k, t_k + W) against the residual syndrome of those rounds and commits the columns with
s(j) in [t_k, t_k + C). The columns with s(j) = t_k + W - 1 lose their rows in round t_k + W,
which the window cannot see; those that become identical are merged into one column with
p = p1 (1 - p2) + p2 (1 - p1), the probability that an odd number of them occurs. The first
window whose span reaches Rt is final: it covers rounds [t_k, Rt] and commits everything left.
Committing a column flips its rows in the residual (the carry into the next window) and XORs its
observables into the logical frame.

The merge is optional for the exact boundary (build_plan(..., merge_truncated=False)): every
column of the window then stays its own local column with its own prior, in ascending global
order. That is the window problem of decoders that slice H without merging (Lee, English and
Bartlett's sliding BP+LSD), and the only way to reproduce their windows when H itself holds
identical columns with different priors, which no truncation merge could keep apart.

Boundaries. "exact" builds every window from the true matrix. "uniform" uses the bulk window
(the one at k = 1) at every position, zero-padding rounds past Rt, and maps its committed
columns to global columns by shifting in time; columns with no global counterpart are virtual.

Non-convergence policies: commit_anyway (commit the inner decoder's output regardless), flag (the
same, and mark the shot), defer (retry the same start with the window widened by C rounds, up to
max_deferrals times).

Command line (python -m rtd.window_ref ...):
    dump-plan      write a plan (plan.json and per-shape / per-placement arrays)
    compare-plans  compare two dumped plans (identical JSON up to key order, identical arrays)
    decode         decode shots with any inner decoder and write per-shot and per-window arrays
    golden         decode with relay_bp and explicit memory-strength tables and write a golden
                   directory for the C++ windowed decoder
    toy-artifact   write the 3-bit repetition-code circuit and export its artifact
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from rtd import log
from rtd.window_ref_inner import (
    INNER_KINDS,
    BpLsdParams,
    InnerContext,
    InnerDecoder,
    InnerResult,
    InnerSpec,
    RelayParams,
    SeededShapeGammas,
    log_ratios,
    support_weight,
)

logger = logging.getLogger("rtd.window_ref")

BOUNDARIES = ("exact", "uniform")
POLICIES = ("commit_anyway", "defer", "flag")
VIRTUAL = 0xFFFFFFFF
_MASK64 = (1 << 64) - 1


class PlanError(ValueError):
    """A problem or window specification the window layer cannot handle."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------------------------
# Specification


@dataclass(frozen=True)
class WindowSpec:
    width: int  # W, rounds per window
    commit: int  # C, rounds committed per window
    converge_rounds: int  # C', rounds on which H e = s must hold
    boundary: str
    on_failure: str
    max_deferrals: int
    iteration_cap: int | None = None

    def validate(self) -> None:
        if not 1 <= self.commit < self.width:
            raise PlanError("commit_out_of_range", f"need 1 <= commit < width, got commit {self.commit}, width {self.width}")
        if not self.commit <= self.converge_rounds <= self.width:
            raise PlanError("converge_out_of_range", f"need commit <= converge_rounds <= width, got {self.converge_rounds}")
        if self.boundary not in BOUNDARIES:
            raise PlanError("invalid_spec", f"boundary {self.boundary!r} not in {BOUNDARIES}")
        if self.on_failure not in POLICIES:
            raise PlanError("invalid_spec", f"on_failure {self.on_failure!r} not in {POLICIES}")
        if self.max_deferrals < 0:
            raise PlanError("invalid_spec", f"max_deferrals must be >= 0 (got {self.max_deferrals})")
        if self.on_failure != "defer" and self.max_deferrals != 0:
            raise PlanError("deferrals_without_defer", f"max_deferrals {self.max_deferrals} needs on_failure defer")
        if self.iteration_cap is not None and self.iteration_cap < 1:
            raise PlanError("zero_iteration_cap", f"iteration_cap must be >= 1 or null (got {self.iteration_cap})")

    def plan_json(self) -> dict[str, Any]:
        """The "spec" object of plan.json: the values that determine the plan."""
        return {
            "width": self.width,
            "commit": self.commit,
            "converge_rounds": self.converge_rounds,
            "boundary": self.boundary,
            "on_failure": self.on_failure,
            "max_deferrals": self.max_deferrals,
        }

    def window_json(self) -> dict[str, Any]:
        """The "window" object of an rtd_decode version-2 spec."""
        return {"mode": "sliding", **self.plan_json(), "iteration_cap": self.iteration_cap}


# ---------------------------------------------------------------------------------------------
# Problem and time structure


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _csr(indptr: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    h = sp.csr_matrix((np.ones(indices.size, dtype=np.uint8), indices.astype(np.int64), indptr.astype(np.int64)), shape=shape)
    h.sort_indices()
    return h


@dataclass(frozen=True)
class GlobalProblem:
    """The whole-shot decoding problem: H (m x n), A (k x n), priors and detector rounds."""

    h: sp.csr_matrix
    a: sp.csc_matrix
    priors: np.ndarray
    det_round: np.ndarray
    path: Path | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str | None = None

    @property
    def m(self) -> int:
        return int(self.h.shape[0])

    @property
    def n(self) -> int:
        return int(self.h.shape[1])

    @property
    def k(self) -> int:
        return int(self.a.shape[0])

    @cached_property
    def h_csc(self) -> sp.csc_matrix:
        """H by columns, row indices ascending within each column."""
        hc = self.h.tocsc()
        hc.sort_indices()
        return hc

    @classmethod
    def from_arrays(cls, h: sp.spmatrix | np.ndarray, a: sp.spmatrix | np.ndarray, priors: np.ndarray, det_round: np.ndarray) -> GlobalProblem:
        hc = sp.csr_matrix(h, dtype=np.uint8)
        hc.sort_indices()
        ac = sp.csc_matrix(a, dtype=np.uint8)
        ac.sort_indices()
        return cls(hc, ac, np.ascontiguousarray(priors, dtype=np.float64), np.asarray(det_round, dtype=np.int64))

    def column_class(self) -> np.ndarray:
        """Bit o of entry j is 1 when column j of A flips observable o."""
        if self.k > 64:
            raise PlanError("too_many_observables", f"{self.k} observables; the frame holds at most 64")
        cls_ = np.zeros(self.n, dtype=np.uint64)
        a_csr = sp.csr_matrix(self.a)
        for o in range(self.k):
            cols = a_csr.indices[a_csr.indptr[o] : a_csr.indptr[o + 1]]
            cls_[cols] ^= np.uint64(1 << o)
        return cls_


def load_problem(path: Path, verify: bool = True) -> GlobalProblem:
    """Load an artifact directory written by rtd-export (or any directory with the same files)."""
    t0 = time.perf_counter()
    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"artifact manifest not found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    if verify:
        for name, expected in manifest.get("sha256", {}).items():
            actual = _sha256_file(path / name)
            if actual != expected:
                raise ValueError(f"{path / name} does not match its manifest checksum ({actual} != {expected})")
    for name in ("syndrome_bias.npy", "observables_bias.npy"):
        if (path / name).exists():
            raise ValueError(f"{path / name} exists: artifacts with decided (p = 1) faults are not supported by the window reference")
    m, n = int(manifest["num_detectors"]), int(manifest["num_columns"])
    h = _csr(np.load(path / "H_indptr.npy"), np.load(path / "H_indices.npy"), (m, n))
    a_csr = np.load(path / "A_indptr.npy"), np.load(path / "A_indices.npy")
    k = int(a_csr[0].size - 1)
    a = _csr(a_csr[0], a_csr[1], (k, n)).tocsc()
    a.sort_indices()
    priors = np.load(path / "priors.npy").astype(np.float64)
    det_round = np.load(path / "det_round.npy").astype(np.int64)
    if priors.shape != (n,) or det_round.shape != (m,):
        raise ValueError(f"artifact {path}: priors {priors.shape} / det_round {det_round.shape} do not fit m={m}, n={n}")
    problem = GlobalProblem(h, a, priors, det_round, path, manifest, _sha256_file(manifest_file))
    logger.info(
        "artifact loaded",
        extra={"artifact": str(path), "rows": m, "columns": n, "observables": k, "nnz_H": int(h.nnz), "seconds": time.perf_counter() - t0},
    )
    return problem


@dataclass(frozen=True)
class TimeStructure:
    rounds_total: int  # Rt
    per_round: int  # M
    earliest: np.ndarray  # s(j), int64 [n]


def time_structure(problem: GlobalProblem) -> TimeStructure:
    """Rows must be grouped by round, rounds 1 .. Rt with M rows each; every column must touch
    rounds s(j) and possibly s(j) + 1, nothing else."""
    rounds = problem.det_round
    if rounds.size == 0:
        raise PlanError("no_rows", "the problem has no detectors")
    if rounds[0] != 1:
        raise PlanError("bad_round_numbering", f"the first detector must be in round 1 (got {int(rounds[0])})")
    if np.any(np.diff(rounds) < 0):
        bad = int(np.flatnonzero(np.diff(rounds) < 0)[0]) + 1
        raise PlanError("rows_not_grouped_by_round", f"detector {bad} has round {int(rounds[bad])} after round {int(rounds[bad - 1])}")
    per_round = int(np.sum(rounds == 1))
    rt = int(rounds[-1])
    counts = np.bincount(rounds, minlength=rt + 1)[1:]
    if np.any(counts != per_round):
        r = int(np.flatnonzero(counts != per_round)[0]) + 1
        raise PlanError("unequal_round_sizes", f"round {r} holds {int(counts[r - 1])} detectors, round 1 holds {per_round}")
    hc = problem.h_csc
    degrees = np.diff(hc.indptr)
    if degrees.size and degrees.min() == 0:
        raise PlanError("empty_column", f"column {int(np.flatnonzero(degrees == 0)[0])} touches no detector")
    row_round = rounds[hc.indices]
    starts = hc.indptr[:-1]
    earliest = np.minimum.reduceat(row_round, starts) if problem.n else np.zeros(0, dtype=np.int64)
    latest = np.maximum.reduceat(row_round, starts) if problem.n else np.zeros(0, dtype=np.int64)
    if np.any(latest > earliest + 1):
        j = int(np.flatnonzero(latest > earliest + 1)[0])
        raise PlanError("column_spans_rounds", f"column {j} touches rounds {int(earliest[j])}..{int(latest[j])}; at most two consecutive allowed")
    return TimeStructure(rt, per_round, earliest.astype(np.int64))


# ---------------------------------------------------------------------------------------------
# Plan


@dataclass(frozen=True)
class Shape:
    """One window's local decoding problem, shared by every placement with identical content."""

    index: int
    h: sp.csr_matrix  # uint8, local rows x local columns, sorted indices
    priors: np.ndarray  # float64 per local column
    commit: np.ndarray  # uint8 per local column
    converge: np.ndarray  # uint8 per local row
    cls: np.ndarray  # uint64 per local column: observables flipped, 0 for uncommitted columns
    merged_columns: int  # local columns that stand for two or more global columns
    h_csc: sp.csc_matrix = field(repr=False)
    lam: list[float] = field(repr=False)

    @property
    def rows(self) -> int:
        return int(self.h.shape[0])

    @property
    def columns(self) -> int:
        return int(self.h.shape[1])

    @property
    def edges(self) -> int:
        return int(self.h.nnz)

    def indptr(self) -> np.ndarray:
        return self.h.indptr.astype(np.uint32)

    def indices(self) -> np.ndarray:
        return self.h.indices.astype(np.uint32)

    def key(self) -> tuple[Any, ...]:
        return (
            self.rows,
            self.columns,
            self.indptr().tobytes(),
            self.indices().tobytes(),
            self.priors.tobytes(),
            self.commit.tobytes(),
            self.cls.tobytes(),
            self.converge.tobytes(),
        )


@dataclass(frozen=True)
class Placement:
    window: int  # k
    attempt: int  # a
    shape: int
    first_round: int  # t_k
    rounds: int
    commit_rounds: int
    final: bool
    columns: np.ndarray  # uint32 per local column: representative global column, or the mapped one / VIRTUAL
    members_ptr: np.ndarray | None  # uint32 CSR of the global members of each local column (exact only)
    members: np.ndarray | None

    def json(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "attempt": self.attempt,
            "shape": self.shape,
            "first_round": self.first_round,
            "rounds": self.rounds,
            "commit_rounds": self.commit_rounds,
            "final": self.final,
        }


@dataclass
class WindowPlan:
    spec: WindowSpec
    rounds_total: int
    per_round: int
    num_positions: int
    num_columns: int
    shapes: list[Shape]
    placements: list[Placement]
    merge_truncated: bool = True  # False: truncated identical columns stay separate (exact only)
    _index: dict[tuple[int, int], Placement] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {(p.window, p.attempt): p for p in self.placements}

    def placement(self, window: int, attempt: int) -> Placement | None:
        return self._index.get((window, attempt))

    def json(self) -> dict[str, Any]:
        return {
            "spec": self.spec.plan_json(),
            "rounds_total": self.rounds_total,
            "detectors_per_round": self.per_round,
            "num_positions": self.num_positions,
            "shapes": [
                {"index": s.index, "rows": s.rows, "columns": s.columns, "edges": s.edges, "merged_columns": s.merged_columns}
                for s in self.shapes
            ],
            "placements": [p.json() for p in self.placements],
        }

    def statistics(self) -> dict[str, Any]:
        committed_virtual = 0
        for p in self.placements:
            shape = self.shapes[p.shape]
            committed_virtual += int(np.sum((p.columns == VIRTUAL) & (shape.commit == 1)))
        return {
            "shapes": len(self.shapes),
            "placements": len(self.placements),
            "positions": self.num_positions,
            "max_rows": max(s.rows for s in self.shapes),
            "max_columns": max(s.columns for s in self.shapes),
            "max_edges": max(s.edges for s in self.shapes),
            "merged_columns": [s.merged_columns for s in self.shapes],
            "committed_virtual_columns": committed_virtual,
        }


@dataclass
class _Local:
    """A window's local problem before shape deduplication."""

    shape: Shape
    reps: np.ndarray  # int64 representative global column per local column
    members: list[np.ndarray]


def _local_problem(
    problem: GlobalProblem, ts: TimeStructure, cls_: np.ndarray, spec: WindowSpec, t: int, attempt: int, final: bool, merge_truncated: bool = True
) -> _Local:
    """The exact local problem of the window starting at round t, attempt `attempt`; without
    merge_truncated every column of the window is its own local column."""
    M, rt, C = ts.per_round, ts.rounds_total, spec.commit
    s = ts.earliest
    if final:
        last = rt
        in_window = s >= t
        in_commit = in_window
        converge_last = rt
    else:
        last = t + spec.width + attempt * C - 1
        in_window = (s >= t) & (s <= last)
        in_commit = (s >= t) & (s < t + C)
        converge_last = t + spec.converge_rounds + attempt * C - 1
    row_lo, row_hi = (t - 1) * M, last * M
    hc = problem.h_csc

    # Columns in ascending global order; when merging, those in the last round (non-final) are
    # grouped by their support restricted to the window, the first member being the smallest
    # global index.
    entries: list[tuple[int, list[int], tuple[int, ...]]] = []
    groups: dict[tuple[int, ...], int] = {}
    for j in np.flatnonzero(in_window).tolist():
        rows = hc.indices[hc.indptr[j] : hc.indptr[j + 1]]
        local_rows = tuple(int(r) - row_lo for r in rows if row_lo <= r < row_hi)
        if merge_truncated and not final and s[j] == last:
            at = groups.get(local_rows)
            if at is not None:
                entries[at][1].append(j)
                continue
            groups[local_rows] = len(entries)
        entries.append((j, [j], local_rows))
    entries.sort(key=lambda e: e[0])

    n_local = len(entries)
    priors = np.empty(n_local, dtype=np.float64)
    commit = np.zeros(n_local, dtype=np.uint8)
    local_cls = np.zeros(n_local, dtype=np.uint64)
    row_idx: list[int] = []
    col_idx: list[int] = []
    merged = 0
    for col, (rep, members, local_rows) in enumerate(entries):
        p = float(problem.priors[members[0]])
        for q in problem.priors[members[1:]].tolist():
            p = p * (1.0 - q) + q * (1.0 - p)
        priors[col] = p
        if len(members) > 1:
            merged += 1
        committed = bool(np.all(in_commit[members]))
        if len(members) > 1 and committed:
            raise AssertionError(f"merged column {rep} lies in the commit set of the window at round {t}")
        if committed:
            commit[col] = 1
            local_cls[col] = cls_[rep]
        row_idx.extend(local_rows)
        col_idx.extend([col] * len(local_rows))
    n_rows = row_hi - row_lo
    h = sp.csr_matrix((np.ones(len(row_idx), dtype=np.uint8), (np.array(row_idx, dtype=np.int64), np.array(col_idx, dtype=np.int64))), shape=(n_rows, n_local))
    h.sum_duplicates()
    h.sort_indices()
    row_rounds = t + np.arange(n_rows) // M
    converge = (row_rounds <= converge_last).astype(np.uint8)
    h_csc = h.tocsc()
    h_csc.sort_indices()
    shape = Shape(-1, h, priors, commit, converge, local_cls, merged, h_csc, log_ratios(priors))
    return _Local(shape, np.array([e[0] for e in entries], dtype=np.int64), [np.array(e[1], dtype=np.int64) for e in entries])


def _members_csr(members: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(members) + 1, dtype=np.uint32)
    ptr[1:] = np.cumsum([m.size for m in members])
    flat = np.concatenate(members).astype(np.uint32) if members else np.zeros(0, dtype=np.uint32)
    return ptr, flat


class _ShapeTable:
    def __init__(self) -> None:
        self.shapes: list[Shape] = []
        self._by_key: dict[tuple[Any, ...], int] = {}

    def index_of(self, shape: Shape) -> int:
        key = shape.key()
        if key not in self._by_key:
            self._by_key[key] = len(self.shapes)
            self.shapes.append(dataclasses.replace(shape, index=len(self.shapes)))
        return self._by_key[key]


def build_plan(problem: GlobalProblem, spec: WindowSpec, merge_truncated: bool = True) -> WindowPlan:
    """The window plan of `problem` under `spec`. merge_truncated=False (exact boundary only)
    keeps every column of a window as its own local column: no truncation merge."""
    t0 = time.perf_counter()
    spec.validate()
    if not merge_truncated and spec.boundary != "exact":
        raise PlanError("invalid_spec", f"merge_truncated=False applies to the exact boundary only (got {spec.boundary!r})")
    ts = time_structure(problem)
    bad = ~((problem.priors >= 0.0) & (problem.priors < 1.0))
    if bad.any():
        j = int(np.flatnonzero(bad)[0])
        raise PlanError("invalid_prior", f"column {j} has prior {problem.priors[j]!r}, outside [0, 1)")
    cls_ = problem.column_class()
    plan = _build_uniform(problem, ts, cls_, spec) if spec.boundary == "uniform" else _build_exact(problem, ts, cls_, spec, merge_truncated)
    logger.info(
        "window plan built",
        extra={"spec": spec.window_json(), "merge_truncated": merge_truncated, **plan.statistics(), "seconds": time.perf_counter() - t0},
    )
    return plan


def _build_exact(problem: GlobalProblem, ts: TimeStructure, cls_: np.ndarray, spec: WindowSpec, merge_truncated: bool = True) -> WindowPlan:
    rt, C, W = ts.rounds_total, spec.commit, spec.width
    table = _ShapeTable()
    placements: list[Placement] = []

    def add(k: int, t: int, attempt: int, final: bool) -> None:
        local = _local_problem(problem, ts, cls_, spec, t, attempt, final, merge_truncated)
        rounds = rt - t + 1 if final else W + attempt * C
        ptr, flat = _members_csr(local.members)
        placements.append(
            Placement(k, attempt, table.index_of(local.shape), t, rounds, rounds if final else C, final, local.reps.astype(np.uint32), ptr, flat)
        )

    k = 0
    while True:
        t = 1 + k * C
        final = t + W - 1 >= rt
        add(k, t, 0, final)
        if spec.on_failure == "defer" and not final:
            for a in range(1, spec.max_deferrals + 1):
                final_a = t + W + a * C - 1 >= rt
                add(k, t, a, final_a)
                if final_a:
                    break
        if final:
            break
        k += 1
    return WindowPlan(spec, rt, ts.per_round, k + 1, problem.n, table.shapes, placements, merge_truncated)


def _build_uniform(problem: GlobalProblem, ts: TimeStructure, cls_: np.ndarray, spec: WindowSpec) -> WindowPlan:
    rt, M, C, W = ts.rounds_total, ts.per_round, spec.commit, spec.width
    t1 = 1 + C
    if W >= rt:
        raise PlanError("no_bulk_window", f"width {W} >= Rt = {rt}: window 0 is final, so there is no bulk window 1")
    for a in range(spec.max_deferrals + 1):
        if t1 + W + a * C > rt - 1:
            raise PlanError(
                "no_bulk_window",
                f"the bulk window at round {t1} with width {W + a * C} (attempt {a}) reaches round {t1 + W + a * C - 1}; "
                f"it must end by round {rt - 2} so its cut round stays below the readout round {rt}",
            )
    locals_ = [_local_problem(problem, ts, cls_, spec, t1, a, False) for a in range(spec.max_deferrals + 1)]
    shapes = [dataclasses.replace(lp.shape, index=a) for a, lp in enumerate(locals_)]

    # Full global support of every column, to find the column a shifted bulk column corresponds to.
    hc = problem.h_csc
    by_support: dict[bytes, list[int]] = {}
    for j in range(problem.n):
        by_support.setdefault(hc.indices[hc.indptr[j] : hc.indptr[j + 1]].astype(np.int64).tobytes(), []).append(j)

    positions = -(-rt // C)
    placements: list[Placement] = []
    for k in range(positions):
        t = 1 + k * C
        shift = (t - t1) * M
        for a, lp in enumerate(locals_):
            mapped = np.full(lp.reps.size, VIRTUAL, dtype=np.uint32)
            for col, g in enumerate(lp.reps.tolist()):
                rows = hc.indices[hc.indptr[g] : hc.indptr[g + 1]].astype(np.int64) + shift
                if rows.min() < 0 or rows.max() >= problem.m:
                    continue
                matches = by_support.get(rows.tobytes(), [])
                if len(matches) > 1:
                    raise PlanError("ambiguous_mapping", f"bulk column {g} shifted to round {t} matches global columns {matches}")
                if matches:
                    j = matches[0]
                    if ts.earliest[j] != ts.earliest[g] + (t - t1):
                        raise AssertionError(f"mapped column {j} has s = {ts.earliest[j]}, expected {ts.earliest[g] + t - t1}")
                    mapped[col] = j
            placements.append(Placement(k, a, a, t, W + a * C, C, False, mapped, None, None))
    return WindowPlan(spec, rt, M, positions, problem.n, shapes, placements)


# ---------------------------------------------------------------------------------------------
# Plan dump, load and comparison


def dump_plan(plan: WindowPlan, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s in plan.shapes:
        np.save(out / f"shape_{s.index}_H_indptr.npy", s.indptr())
        np.save(out / f"shape_{s.index}_H_indices.npy", s.indices())
        np.save(out / f"shape_{s.index}_priors.npy", s.priors.astype(np.float64))
        np.save(out / f"shape_{s.index}_commit.npy", s.commit.astype(np.uint8))
        np.save(out / f"shape_{s.index}_converge.npy", s.converge.astype(np.uint8))
        np.save(out / f"shape_{s.index}_class.npy", s.cls.astype(np.uint64))
    for p in plan.placements:
        stem = f"placement_{p.window}_{p.attempt}"
        np.save(out / f"{stem}_columns.npy", p.columns.astype(np.uint32))
        if p.members_ptr is not None and p.members is not None:
            np.save(out / f"{stem}_members_ptr.npy", p.members_ptr.astype(np.uint32))
            np.save(out / f"{stem}_members.npy", p.members.astype(np.uint32))
    # Written last so that a directory with plan.json is complete.
    (out / "plan.json").write_text(json.dumps(plan.json(), indent=2) + "\n")
    logger.info("plan written", extra={"out": str(out), "shapes": len(plan.shapes), "placements": len(plan.placements)})


def compare_plan_dirs(a: Path, b: Path) -> list[str]:
    """Differences between two dumped plans: JSON compared as parsed objects, arrays by dtype,
    shape and bytes (so float64 priors compare bitwise). Empty list = identical."""
    problems: list[str] = []
    ja, jb = json.loads((a / "plan.json").read_text()), json.loads((b / "plan.json").read_text())
    if ja != jb:
        for key in sorted(set(ja) | set(jb)):
            if ja.get(key) != jb.get(key):
                problems.append(f"plan.json key {key!r} differs")
    names_a = {p.name for p in a.glob("*.npy")}
    names_b = {p.name for p in b.glob("*.npy")}
    for name in sorted(names_a ^ names_b):
        problems.append(f"{name} only in {a if name in names_a else b}")
    for name in sorted(names_a & names_b):
        xa, xb = np.load(a / name), np.load(b / name)
        if xa.dtype != xb.dtype or xa.shape != xb.shape or xa.tobytes() != xb.tobytes():
            problems.append(f"{name} differs ({xa.dtype}{list(xa.shape)} vs {xb.dtype}{list(xb.shape)})")
    return problems


# ---------------------------------------------------------------------------------------------
# Stream decoding


@dataclass
class WindowRecord:
    iterations: int = 0
    legs: int = 0
    attempts: int = 0
    converged: bool = True
    cap_hit: bool = False
    weight: float = 0.0
    committed_weight: float = 0.0
    unexplained: int = 0
    flagged: bool = False
    virtual: int = 0
    decode_ns: int = 0
    committed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint32))
    residual_first_round: np.ndarray | None = None  # the residual's first round before decoding
    solution: np.ndarray | None = None  # representatives (exact) of the returned support
    num_optimal: int = 1
    frame_after: int = 0


@dataclass
class ShotOutcome:
    windows: list[WindowRecord]
    frame: int
    residual: np.ndarray  # uint8 [m], the syndrome left after every commit

    @property
    def success(self) -> bool:
        return all(w.converged for w in self.windows)

    @property
    def flagged(self) -> bool:
        return any(w.flagged for w in self.windows)

    @property
    def iterations(self) -> int:
        return sum(w.iterations for w in self.windows)

    @property
    def legs(self) -> int:
        return sum(w.legs for w in self.windows)

    @property
    def weight(self) -> float:
        total = 0.0
        for w in self.windows:
            total += w.committed_weight
        return total


def gamma_stream(shot: int, window: int, attempt: int) -> int:
    return (shot + (window << 32) + (attempt << 56)) & _MASK64


def decode_shot(plan: WindowPlan, syndrome: np.ndarray, shot: int, inner: InnerDecoder) -> ShotOutcome:
    """Decode one shot window by window, exactly as the stream decoder does."""
    spec = plan.spec
    M, real_rows = plan.per_round, plan.rounds_total * plan.per_round
    resid = np.array(syndrome, dtype=np.uint8, copy=True)
    if resid.shape != (real_rows,):
        raise ValueError(f"shot {shot}: syndrome has shape {resid.shape}, expected ({real_rows},)")
    frame = 0
    records: list[WindowRecord] = []
    complete = False
    for k in range(plan.num_positions):
        if complete:
            # A final deferral attempt already decided every remaining column.
            records.append(WindowRecord(frame_after=frame))
            continue
        attempt, iterations, legs, ns = 0, 0, 0, 0
        first_round_resid: np.ndarray | None = None
        flagged = False
        while True:
            p = plan.placement(k, attempt)
            assert p is not None
            shape = plan.shapes[p.shape]
            lo = (p.first_round - 1) * M
            real = max(0, min(shape.rows, real_rows - lo))
            s_loc = np.zeros(shape.rows, dtype=np.uint8)
            s_loc[:real] = resid[lo : lo + real]
            if first_round_resid is None:
                first_round_resid = s_loc[:M].copy()
            ctx = InnerContext(shot, k, attempt, p.shape, gamma_stream(shot, k, attempt), shape.converge, spec.iteration_cap)
            t0 = time.perf_counter_ns()
            res = inner.decode(shape.h, shape.priors, s_loc, ctx)
            ns += time.perf_counter_ns() - t0
            iterations += res.iterations
            legs += res.legs
            if res.converged:
                break
            if spec.on_failure == "defer" and attempt < spec.max_deferrals and plan.placement(k, attempt + 1) is not None:
                logger.debug("window deferred", extra={"shot": shot, "window": k, "attempt": attempt})
                attempt += 1
                continue
            flagged = spec.on_failure != "commit_anyway"
            logger.debug("window not converged", extra={"shot": shot, "window": k, "attempt": attempt, "policy": spec.on_failure})
            break
        record = _commit(plan, p, shape, res, resid, lo, real)
        frame ^= record.flip
        record.window.iterations, record.window.legs, record.window.attempts = iterations, legs, attempt + 1
        record.window.flagged = flagged
        record.window.decode_ns = ns
        record.window.residual_first_round = first_round_resid
        record.window.frame_after = frame
        records.append(record.window)
        if p.final:
            complete = True
    return ShotOutcome(records, frame, resid)


@dataclass
class _Committed:
    window: WindowRecord
    flip: int


def _commit(plan: WindowPlan, p: Placement, shape: Shape, res: InnerResult, resid: np.ndarray, lo: int, real: int) -> _Committed:
    support = np.sort(np.asarray(res.support, dtype=np.int64))
    committed_local = support[shape.commit[support] == 1] if support.size else support
    flip = 0
    for col in committed_local.tolist():
        rows = shape.h_csc.indices[shape.h_csc.indptr[col] : shape.h_csc.indptr[col + 1]]
        rows = rows[rows < real]
        resid[lo + rows] ^= 1
        flip ^= int(shape.cls[col])
    global_cols = p.columns[committed_local].astype(np.uint32) if committed_local.size else np.zeros(0, dtype=np.uint32)
    virtual = int(np.sum(global_cols == VIRTUAL))
    global_cols = np.sort(global_cols[global_cols != VIRTUAL])
    committed_weight = support_weight(committed_local, shape.lam)
    M = plan.per_round
    hi = lo + real if p.final else min(lo + p.commit_rounds * M, lo + real)
    unexplained = int(resid[lo:hi].sum())
    window = WindowRecord(
        converged=res.converged,
        cap_hit=res.cap_hit,
        weight=res.weight if res.converged else math.inf,
        committed_weight=committed_weight,
        unexplained=unexplained,
        virtual=virtual,
        committed=global_cols,
        solution=p.columns[support].astype(np.uint32) if support.size else np.zeros(0, dtype=np.uint32),
        num_optimal=res.num_optimal,
    )
    return _Committed(window, flip)


def frame_bits(frame: int, k: int) -> np.ndarray:
    return np.array([(frame >> o) & 1 for o in range(k)], dtype=np.uint8)


# ---------------------------------------------------------------------------------------------
# Lee, English, Bartlett's detector rule (the same sets, stated through detectors)


def detector_rule_sets(problem: GlobalProblem, width: int, commit: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Window k's columns are the not-yet-committed faults touching any detector of rounds
    [t_k, t_k + W) (clipped at Rt); it commits those that touch a detector of rounds
    [t_k, t_k + C), or all of them in the final window. Returns (columns, commits) per window."""
    ts = time_structure(problem)
    rt, M = ts.rounds_total, ts.per_round
    committed = np.zeros(problem.n, dtype=bool)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    k = 0
    while True:
        t = 1 + k * commit
        final = t + width - 1 >= rt
        last = rt if final else t + width - 1
        rows = problem.h[(t - 1) * M : last * M]
        touching = np.zeros(problem.n, dtype=bool)
        touching[rows.indices] = True
        columns = touching & ~committed
        if final:
            commits = columns
        else:
            commit_rows = problem.h[(t - 1) * M : (t - 1 + commit) * M]
            touches_commit = np.zeros(problem.n, dtype=bool)
            touches_commit[commit_rows.indices] = True
            commits = columns & touches_commit
        committed |= commits
        out.append((np.flatnonzero(columns), np.flatnonzero(commits)))
        if final:
            return out
        k += 1


# ---------------------------------------------------------------------------------------------
# Batch decoding (optionally over worker processes)

_WORKER: tuple[WindowPlan, InnerDecoder] | None = None


def _init_worker(plan: WindowPlan, inner_spec: InnerSpec, level: int) -> None:
    global _WORKER
    log.configure(level)
    try:
        os.chdir(tempfile.mkdtemp(prefix=f"rtd-window-worker-{os.getpid()}-"))
        _WORKER = (plan, inner_spec.build())
    except Exception:
        logger.exception("window worker initialisation failed", extra={"pid": os.getpid(), "inner": inner_spec.kind})
        raise


def _decode_chunk(first: int, syndromes: np.ndarray) -> list[ShotOutcome]:
    if _WORKER is None:
        raise RuntimeError("worker process was not initialised")
    plan, inner = _WORKER
    out = []
    for i in range(syndromes.shape[0]):
        try:
            out.append(decode_shot(plan, syndromes[i], first + i, inner))
        except Exception:
            logger.exception("shot failed in window worker", extra={"shot": first + i, "pid": os.getpid()})
            raise
    return out


def decode_shots(plan: WindowPlan, syndromes: np.ndarray, inner_spec: InnerSpec, first: int = 0, workers: int = 1) -> list[ShotOutcome]:
    """Decode every syndrome (row s has shot index first + s), in order."""
    t0 = time.perf_counter()
    total = syndromes.shape[0]
    logger.info("window decoding started", extra={"shots": total, "inner": inner_spec.kind, "workers": workers})
    if workers <= 1 or total <= 1:
        inner = inner_spec.build()
        try:
            outcomes = []
            for i in range(total):
                try:
                    outcomes.append(decode_shot(plan, syndromes[i], first + i, inner))
                except Exception:
                    logger.exception("shot failed", extra={"shot": first + i})
                    raise
                if (i + 1) % max(1, total // 10) == 0:
                    logger.info("window decoding progress", extra={"done": i + 1, "total": total, "elapsed_seconds": time.perf_counter() - t0})
        finally:
            inner.close()
    else:
        bounds = np.linspace(0, total, min(workers, total) * 4 + 1).astype(int)
        level = logging.getLogger().getEffectiveLevel()
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(plan, inner_spec, level)) as pool:
            try:
                futures = [pool.submit(_decode_chunk, first + int(lo), syndromes[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True) if hi > lo]
                outcomes = [o for f in futures for o in f.result()]
            except BaseException:
                logger.exception("parallel window decoding aborted; cancelling remaining chunks", extra={"shots": total})
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    logger.info("window decoding finished", extra={"shots": total, "seconds": time.perf_counter() - t0})
    return outcomes


def outcome_arrays(outcomes: Sequence[ShotOutcome], plan: WindowPlan, k: int, observables: np.ndarray | None) -> dict[str, np.ndarray]:
    """Per-shot and per-window arrays under rtd_decode's names and dtypes."""
    S, K = len(outcomes), plan.num_positions
    arrays: dict[str, np.ndarray] = {}

    def per_window(name: str, dtype: Any, get: Any) -> None:
        arrays[name] = np.array([[get(w) for w in o.windows] for o in outcomes], dtype=dtype).reshape(S, K)

    per_window("win_iterations", np.uint32, lambda w: w.iterations)
    per_window("win_legs", np.uint32, lambda w: w.legs)
    per_window("win_attempts", np.uint8, lambda w: w.attempts)
    per_window("win_converged", np.uint8, lambda w: w.converged)
    per_window("win_cap_hit", np.uint8, lambda w: w.cap_hit)
    per_window("win_weight", np.float64, lambda w: w.weight)
    per_window("win_committed_weight", np.float64, lambda w: w.committed_weight)
    per_window("win_unexplained", np.uint32, lambda w: w.unexplained)
    per_window("win_flagged", np.uint8, lambda w: w.flagged)
    per_window("win_virtual", np.uint32, lambda w: w.virtual)
    per_window("win_decode_ns", np.uint64, lambda w: w.decode_ns)
    commits = [w.committed for o in outcomes for w in o.windows]
    ptr = np.zeros(S * K + 1, dtype=np.uint64)
    ptr[1:] = np.cumsum([c.size for c in commits])
    arrays["commit_ptr"] = ptr
    arrays["commit_faults"] = np.concatenate(commits).astype(np.uint32) if commits else np.zeros(0, dtype=np.uint32)
    predicted = np.stack([frame_bits(o.frame, k) for o in outcomes]) if S else np.zeros((0, k), dtype=np.uint8)
    arrays["predicted_observables"] = predicted.astype(np.uint8).reshape(S, k)
    if observables is not None:
        arrays["logical_failure"] = np.any(predicted != observables, axis=1).astype(np.uint8) if S else np.zeros(0, dtype=np.uint8)
    arrays["flagged"] = np.array([o.flagged for o in outcomes], dtype=np.uint8)
    arrays["success"] = np.array([o.success for o in outcomes], dtype=np.uint8)
    arrays["iterations"] = np.array([o.iterations for o in outcomes], dtype=np.uint32)
    arrays["legs"] = np.array([o.legs for o in outcomes], dtype=np.uint32)
    arrays["best_leg"] = np.full(S, -1, dtype=np.int32)
    arrays["weight"] = np.array([o.weight for o in outcomes], dtype=np.float64)
    arrays["decode_ns"] = np.array([sum(w.decode_ns for w in o.windows) for o in outcomes], dtype=np.uint64)
    return arrays


def check_outcomes(problem: GlobalProblem, plan: WindowPlan, syndromes: np.ndarray, outcomes: Sequence[ShotOutcome]) -> dict[str, int]:
    """Internal consistency of decoded shots; raises on any violation.

    - residual left over = the committed rounds' unexplained counts summed over windows;
    - exact boundary: each global column committed at most once per shot, only inside its
      window's commit set, and H c + residual = sigma, A c = frame;
    - every window converged => residual = 0 (so H c = sigma for the exact boundary)."""
    exact = plan.spec.boundary == "exact"
    hc = problem.h_csc
    cls_ = problem.column_class()
    all_converged = 0
    for s, o in enumerate(outcomes):
        if sum(w.unexplained for w in o.windows) != int(o.residual.sum()):
            counts = [w.unexplained for w in o.windows]
            raise RuntimeError(f"shot {s}: unexplained counts {counts} do not add up to the residual weight {int(o.residual.sum())}")
        if o.success:
            all_converged += 1
            if o.residual.any():
                raise RuntimeError(f"shot {s}: every window converged but {int(o.residual.sum())} detectors remain unexplained")
        if not exact:
            continue
        committed = np.concatenate([w.committed for w in o.windows]).astype(np.int64) if o.windows else np.zeros(0, dtype=np.int64)
        if np.unique(committed).size != committed.size:
            raise RuntimeError(f"shot {s}: a column was committed twice")
        for k, w in enumerate(o.windows):
            if w.attempts == 0:
                continue
            p = plan.placement(k, w.attempts - 1)
            assert p is not None
            allowed = set(p.columns[plan.shapes[p.shape].commit == 1].tolist())
            if not set(w.committed.tolist()) <= allowed:
                raise RuntimeError(f"shot {s} window {k}: committed columns outside its commit set")
        c = np.zeros(problem.n, dtype=np.int64)
        c[committed] = 1
        hcx = (np.asarray(hc @ c).ravel() % 2).astype(np.uint8)
        if not np.array_equal(hcx ^ o.residual, syndromes[s].astype(np.uint8)):
            raise RuntimeError(f"shot {s}: H c + residual != sigma")
        frame = 0
        for j in committed.tolist():
            frame ^= int(cls_[j])
        if frame != o.frame:
            raise RuntimeError(f"shot {s}: frame {o.frame} != A c = {frame}")
    return {"shots": len(outcomes), "all_windows_converged": all_converged}


# ---------------------------------------------------------------------------------------------
# Shots


def load_shots_dir(shots: Path, first: int, count: int, problem: GlobalProblem) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """detectors[first:first+count] and observables from an rtd-sample directory, checked against
    the manifest's checksums and the artifact's source circuit."""
    manifest = json.loads((shots / "manifest.json").read_text())
    sha = manifest.get("sha256", {})
    for name in ("detectors.npy", "observables.npy"):
        if name not in sha:
            raise ValueError(f"shots manifest {shots / 'manifest.json'} has no sha256 for {name}")
        if _sha256_file(shots / name) != sha[name]:
            raise ValueError(f"{shots / name} does not match its manifest checksum")
    artifact_circuit = problem.manifest.get("source_circuit", {}).get("sha256")
    if sha.get("circuit.stim") is None or artifact_circuit is None:
        logger.warning("cannot confirm the shots come from the artifact's circuit", extra={"shots": str(shots)})
    elif sha["circuit.stim"] != artifact_circuit:
        raise ValueError(f"shots in {shots} come from circuit {sha['circuit.stim']}, the artifact from {artifact_circuit}")
    detectors = np.load(shots / "detectors.npy", mmap_mode="r")
    observables = np.load(shots / "observables.npy", mmap_mode="r")
    if detectors.shape[1] != problem.m or observables.shape[1] != problem.k:
        raise ValueError(f"shots are {detectors.shape} / {observables.shape}, artifact has m={problem.m}, k={problem.k}")
    if first < 0 or count < 1 or first + count > detectors.shape[0]:
        raise ValueError(f"first {first} count {count} outside the {detectors.shape[0]} shots of {shots}")
    d = np.ascontiguousarray(detectors[first : first + count], dtype=np.uint8)
    o = np.ascontiguousarray(observables[first : first + count], dtype=np.uint8)
    source = {
        "path": _manifest_path(shots),
        "first": first,
        "count": count,
        "detectors_sha256": sha["detectors.npy"],
        "observables_sha256": sha["observables.npy"],
    }
    logger.info("shots loaded", extra={"shots": str(shots), "first": first, "count": count})
    return d, o, source


def _manifest_path(path: Path) -> str:
    resolved, cwd = path.resolve(), Path.cwd().resolve()
    return resolved.relative_to(cwd).as_posix() if resolved.is_relative_to(cwd) else resolved.name


# ---------------------------------------------------------------------------------------------
# Command line


def _add_window_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--commit", type=int, required=True)
    p.add_argument("--converge", type=int, required=True, help="C', rounds on which H e = s must hold")
    p.add_argument("--boundary", choices=BOUNDARIES, required=True)
    p.add_argument("--on-failure", choices=POLICIES, required=True)
    p.add_argument("--max-deferrals", type=int, required=True)
    p.add_argument("--iteration-cap", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")


_RELAY_FLAGS = {
    "float": "--float",
    "alpha": "--alpha",
    "alpha_scaling": "--alpha-scaling",
    "gamma0": "--gamma0",
    "pre_iter": "--pre-iter",
    "num_sets": "--num-sets",
    "set_max_iter": "--set-max-iter",
    "stopping": "--stopping",
    "stop_nconv": "--stop-nconv",
    "gamma_rows": "--gamma-rows",
    "gamma_seed": "--gamma-seed",
    "gamma_interval": "--gamma-interval",
}


def _add_relay_flags(p: argparse.ArgumentParser) -> None:
    """relay_bp's settings. None has a default (relay_bp's Python defaults differ from the
    Relay-BP paper); relay_from_args reports every missing one."""
    from rtd.golden import STOPPING_CRITERIA, _finite_float, _float_or_none

    unset = argparse.SUPPRESS
    p.add_argument("--float", choices=("f32", "f64"), default=unset)
    p.add_argument("--alpha", type=_float_or_none, default=unset, help="check-message scale, 0 = adaptive, 'none' = 1")
    p.add_argument("--alpha-scaling", type=_finite_float, default=unset)
    p.add_argument("--gamma0", type=_float_or_none, default=unset, help="leg-0 memory strength, or 'none'")
    p.add_argument("--pre-iter", type=int, default=unset)
    p.add_argument("--num-sets", type=int, default=unset)
    p.add_argument("--set-max-iter", type=int, default=unset)
    p.add_argument("--stopping", choices=STOPPING_CRITERIA, default=unset)
    p.add_argument("--stop-nconv", type=int, default=unset)
    p.add_argument("--gamma-rows", type=int, default=unset, metavar="T")
    p.add_argument("--gamma-seed", type=int, default=unset)
    p.add_argument("--gamma-interval", type=_finite_float, nargs=2, default=unset, metavar=("LO", "HI"))


def spec_from_args(args: argparse.Namespace) -> WindowSpec:
    spec = WindowSpec(args.width, args.commit, args.converge, args.boundary, args.on_failure, args.max_deferrals, args.iteration_cap)
    spec.validate()
    return spec


def relay_from_args(args: argparse.Namespace) -> tuple[RelayParams, SeededShapeGammas]:
    given = vars(args)
    missing = [flag for dest, flag in _RELAY_FLAGS.items() if dest not in given]
    if missing:
        raise ValueError(f"the relay inner decoder needs {', '.join(missing)}")
    for dest, low in (("pre_iter", 1), ("num_sets", 0), ("set_max_iter", 1), ("stop_nconv", 1), ("gamma_rows", 1)):
        if given[dest] < low:
            raise ValueError(f"{_RELAY_FLAGS[dest]} must be >= {low} (got {given[dest]})")
    lo, hi = args.gamma_interval
    if not lo < hi:
        raise ValueError(f"--gamma-interval needs LO < HI (got {lo}, {hi})")
    params = RelayParams(
        float_type=args.float,
        alpha=args.alpha,
        alpha_scaling=args.alpha_scaling,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stopping=args.stopping,
        stop_nconv=args.stop_nconv,
        gamma_interval=(lo, hi),
    )
    return params, SeededShapeGammas(args.gamma_rows, args.gamma_seed, lo, hi)


def _prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)


def _cmd_dump_plan(args: argparse.Namespace) -> None:
    _prepare_out(args.out, args.overwrite)
    problem = load_problem(args.artifact)
    dump_plan(build_plan(problem, spec_from_args(args), merge_truncated=not args.no_merge_truncated), args.out)


def _cmd_compare(args: argparse.Namespace) -> int:
    problems = compare_plan_dirs(args.a, args.b)
    for p in problems:
        logger.error("plans differ", extra={"difference": p})
    logger.info("plan comparison finished", extra={"a": str(args.a), "b": str(args.b), "differences": len(problems)})
    return 1 if problems else 0


def _cmd_decode(args: argparse.Namespace) -> None:
    from rtd.window_ref_golden import write_arrays

    _prepare_out(args.out, args.overwrite)
    problem = load_problem(args.artifact)
    spec = spec_from_args(args)
    plan = build_plan(problem, spec, merge_truncated=not args.no_merge_truncated)
    syndromes, observables, source = load_shots_dir(args.shots, args.first, args.count, problem)
    if args.inner == "relay":
        params, gammas = relay_from_args(args)
        inner_spec = InnerSpec("relay", relay=params, gammas=gammas)
    elif args.inner == "bplsd":
        inner_spec = InnerSpec("bplsd", bplsd=BpLsdParams(max_iter=args.bplsd_max_iter, lsd_order=args.lsd_order))
    else:
        inner_spec = InnerSpec(args.inner)
    outcomes = decode_shots(plan, syndromes, inner_spec, args.first, args.workers)
    check = check_outcomes(problem, plan, syndromes, outcomes)
    arrays = outcome_arrays(outcomes, plan, problem.k, observables)
    if not args.save_commits:
        arrays.pop("commit_ptr")
        arrays.pop("commit_faults")
    write_arrays(args.out, arrays)
    summary = {
        "shots": len(outcomes),
        "logical_failures": int(arrays["logical_failure"].sum()),
        "flagged": int(arrays["flagged"].sum()),
        "windows_not_converged": int((arrays["win_converged"] == 0).sum()),
        **check,
    }
    run = {
        "window": spec.window_json(),
        "merge_truncated": plan.merge_truncated,
        "inner": args.inner,
        "artifact": _manifest_path(args.artifact),
        "shots": source,
        "plan": plan.statistics(),
        "summary": summary,
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    logger.info("decode outputs written", extra={"out": str(args.out), **summary})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m rtd.window_ref", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    no_merge_help = "exact boundary: keep truncated identical columns as separate local columns (no merge)"
    dump = sub.add_parser("dump-plan", help="write a window plan")
    _add_window_flags(dump)
    dump.add_argument("--no-merge-truncated", action="store_true", help=no_merge_help)

    cmp_ = sub.add_parser("compare-plans", help="compare two dumped plans")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)

    dec = sub.add_parser("decode", help="decode shots window by window")
    _add_window_flags(dec)
    dec.add_argument("--shots", type=Path, required=True)
    dec.add_argument("--first", type=int, default=0)
    dec.add_argument("--count", type=int, required=True)
    dec.add_argument("--inner", choices=INNER_KINDS, required=True)
    dec.add_argument("--workers", type=int, default=1)
    dec.add_argument("--save-commits", action="store_true")
    dec.add_argument("--bplsd-max-iter", type=int, default=30)
    dec.add_argument("--lsd-order", type=int, default=0)
    dec.add_argument("--no-merge-truncated", action="store_true", help=no_merge_help)
    _add_relay_flags(dec)

    gold = sub.add_parser("golden", help="write a windowed golden (relay_bp inner decoder)")
    _add_window_flags(gold)
    gold.add_argument("--shots", type=Path, required=True)
    gold.add_argument("--first", type=int, default=0)
    gold.add_argument("--count", type=int, required=True)
    gold.add_argument("--workers", type=int, default=1)
    _add_relay_flags(gold)

    toy = sub.add_parser("toy-artifact", help="write the 3-bit repetition-code circuit and its artifact")
    toy.add_argument("--out", type=Path, required=True)
    toy.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("run started", extra={"cli_args": cli_args})
    try:
        if args.command == "dump-plan":
            _cmd_dump_plan(args)
        elif args.command == "compare-plans":
            return _cmd_compare(args)
        elif args.command == "decode":
            _cmd_decode(args)
        elif args.command == "golden":
            from rtd.window_ref_golden import run_golden

            run_golden(args)
        elif args.command == "toy-artifact":
            from rtd.window_ref_toy import write_toy_artifact

            write_toy_artifact(args.out, args.overwrite)
    except PlanError:
        logger.exception("window plan rejected", extra={"cli_args": cli_args})
        return 2
    except ImportError:
        logger.exception("an optional decoder package is missing; install the reference group", extra={"cli_args": cli_args})
        return 1
    except Exception:
        logger.exception("run failed", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
