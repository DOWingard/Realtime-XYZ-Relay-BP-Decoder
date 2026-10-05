"""Netlist of one window decoding graph, in a neutral format (netlist.json + netlist.npz) for a
hardware implementation of the fixed-point windowed decoder.

What a window decoder is built from (Maurer et al., Algorithm 2): the window check matrix H~
(W M rows: W rounds of M detectors; N~ fault columns), the quantised priors lambda_int of its
columns, the commit mask m_com (columns decided by this window), the convergence mask (rows on
which H~ e = s must hold), the carry map (which rows of the next window's first round a committed
column flips: Algorithm 2's u = (H~ e_com)[C M : (C + 1) M]) and the frame matrix A~ (the logical
observables a committed column flips).

The bulk window of a bivariate-bicycle code is also invariant under the group Z_L x Z_M' acting on
the check index c = M' a + b of every detector, (a, b) -> (a + s mod L, b + t mod M'), with round and
check type unchanged (the gross code: L = 12, M' = 6, 72 shifts). So H~ is a lift of a base graph:
every row and every column belongs to an orbit of L M' nodes, and H~ is the base matrix with each
nonzero replaced by a shift. The netlist records each node's orbit and shift (when the action is
free and exact, else no lift), which is what a partly serial design that processes f of the L M'
copies per clock needs.

Arrays of netlist.npz (n local columns, rows local rows):
    row_round int32 [rows]            round of each row relative to the window's first round (0 based)
    row_type uint8, row_check int32    detector type (0 = X, 1 = Z) and check index c
    converge uint8 [rows]              convergence mask
    H_indptr, H_indices uint32         H~ in CSR (rows in round order, columns ascending)
    Hc_indptr, Hc_indices uint32       the same matrix in CSC
    col_round int32 [n]                s(l) relative to the window's first round
    prior float64 [n], lambda_int int8 [n]   fault probability and its quantised log-likelihood
    commit uint8 [n]                   commit mask
    frame uint64 [n]                   A~ by column: bit o = observable o (0 for uncommitted columns)
    global_column uint32 [n]           global column of each local column at the placement the netlist was taken
                                       from (0xFFFFFFFF = virtual)
    carry_indptr uint32 [n + 1], carry_rows uint32   rows of the next window's first round each
                                       committed column flips (empty for uncommitted ones)
    lift arrays, when the lift exists: row_orbit, col_orbit int32; row_shift, col_shift int16 [.,2];
    base_row, base_col int32, base_shift int16 [E_base, 2]: the base graph as (row orbit, column
    orbit, shift of the row relative to the column); edge (i, l) exists iff for some base edge,
    orbit(i) = base_row, orbit(l) = base_col and shift(i) - shift(l) = base_shift (mod L, M')
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from rtd import log
from rtd.fixed_ref import FORMATS, FixedFormat
from rtd.window_ref import GlobalProblem, Shape, WindowPlan, WindowSpec, _manifest_path, _prepare_out, _sha256_file, build_plan, load_problem

logger = logging.getLogger("rtd.fixed_netlist")

NETLIST_FORMAT_VERSION = 1
GROUP = (12, 6)  # Z12 x Z6 of the bivariate-bicycle code; check c = 6 a + b


@dataclass
class Lift:
    row_orbit: np.ndarray
    row_shift: np.ndarray  # int16 [rows, 2]
    col_orbit: np.ndarray
    col_shift: np.ndarray  # int16 [n, 2]
    base_row: np.ndarray
    base_col: np.ndarray
    base_shift: np.ndarray  # int16 [E_base, 2]

    @property
    def base_rows(self) -> int:
        return int(self.row_orbit.max()) + 1 if self.row_orbit.size else 0

    @property
    def base_columns(self) -> int:
        return int(self.col_orbit.max()) + 1 if self.col_orbit.size else 0


def _row_action(row_round: np.ndarray, row_type: np.ndarray, row_check: np.ndarray, s: int, t: int, group: tuple[int, int]) -> np.ndarray | None:
    """The local row each row goes to under the shift (s, t), or None if one leaves the window."""
    na, nb = group
    a, b = row_check // nb, row_check % nb
    target = ((a + s) % na) * nb + (b + t) % nb
    index = {(int(r), int(ty), int(c)): i for i, (r, ty, c) in enumerate(zip(row_round, row_type, row_check, strict=True))}
    out = np.empty(row_round.size, dtype=np.int64)
    for i in range(row_round.size):
        j = index.get((int(row_round[i]), int(row_type[i]), int(target[i])))
        if j is None:
            return None
        out[i] = j
    return out


PRIOR_RTOL = 1e-12  # a merged column combines its members' probabilities in an orbit-dependent order


def lift_structure(
    shape: Shape, row_round: np.ndarray, row_type: np.ndarray, row_check: np.ndarray, lambda_int: np.ndarray, group: tuple[int, int] = GROUP
) -> Lift | None:
    """Orbits and shifts of rows and columns of a window under Z_L x Z_M' (group = (L, M'); checks
    c = M' a + b), or None when the window graph is not invariant or the action is not free.
    Invariant means: H~, the commit and convergence masks and the quantised priors exactly, the
    float priors to a relative PRIOR_RTOL (a merged column's probability combines its members',
    with rounding that depends on their order)."""
    return graph_lift(shape.h, shape.priors, lambda_int, shape.commit, shape.converge, row_round, row_type, row_check, group)


def graph_lift(
    h: sp.csr_matrix,
    priors: np.ndarray,
    lambda_int: np.ndarray,
    column_label: np.ndarray,
    row_label: np.ndarray,
    row_round: np.ndarray,
    row_type: np.ndarray,
    row_check: np.ndarray,
    group: tuple[int, int] = GROUP,
) -> Lift | None:
    """The lift of a Tanner graph whose columns carry (prior, lambda_int, column_label) and rows
    carry row_label, all of which the group must preserve."""
    rows_n, cols_n = h.shape
    na, nb = group
    if row_check.size and row_check.max() >= na * nb:
        return None
    hc = sp.csc_matrix(h)
    hc.sort_indices()
    supports = {tuple(hc.indices[hc.indptr[j] : hc.indptr[j + 1]].tolist()): j for j in range(cols_n)}
    if len(supports) != cols_n:
        logger.warning("graph has identical columns; no lift", extra={"columns": cols_n})
        return None
    generators = {}
    for name, (s, t) in (("a", (1, 0)), ("b", (0, 1))):
        rows = _row_action(row_round, row_type, row_check, s, t, group)
        if rows is None or not np.array_equal(row_label[rows], row_label):
            return None
        cols = np.empty(cols_n, dtype=np.int64)
        for j in range(cols_n):
            image = tuple(sorted(rows[hc.indices[hc.indptr[j] : hc.indptr[j + 1]]].tolist()))
            k = supports.get(image)
            if k is None or lambda_int[k] != lambda_int[j] or column_label[k] != column_label[j]:
                return None
            if abs(priors[k] - priors[j]) > PRIOR_RTOL * priors[j]:
                return None
            cols[j] = k
        generators[name] = (rows, cols)
    ra, ca = generators["a"]
    rb, cb = generators["b"]

    def orbits(perm_a: np.ndarray, perm_b: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray] | None:
        orbit = np.full(size, -1, dtype=np.int32)
        shift = np.zeros((size, 2), dtype=np.int16)
        count = 0
        for start in range(size):
            if orbit[start] >= 0:
                continue
            x = start
            for s in range(na):
                y = x
                for t in range(nb):
                    if orbit[y] >= 0:
                        return None  # a shift fixes a node: the action is not free
                    orbit[y], shift[y] = count, (s, t)
                    y = perm_b[y]
                x = perm_a[x]
            if x != start:
                return None
            count += 1
        return orbit, shift

    r = orbits(ra, rb, rows_n)
    c = orbits(ca, cb, cols_n)
    if r is None or c is None:
        return None
    row_orbit, row_shift = r
    col_orbit, col_shift = c
    coo = sp.coo_matrix(h)
    rep = col_shift[coo.col].astype(np.int64)
    rel = (row_shift[coo.row].astype(np.int64) - rep) % np.array([na, nb])
    keys = np.unique(np.stack([row_orbit[coo.row], col_orbit[coo.col], rel[:, 0], rel[:, 1]], axis=1), axis=0)
    if keys.shape[0] * na * nb != coo.nnz:
        return None
    return Lift(row_orbit, row_shift, col_orbit, col_shift, keys[:, 0].astype(np.int32), keys[:, 1].astype(np.int32), keys[:, 2:].astype(np.int16))


def _carry(shape: Shape, commit_rounds: int, per_round: int) -> tuple[np.ndarray, np.ndarray]:
    """Per column, the rows of the next window's first round (global round t_k + C) it flips if
    committed, as local rows [0, M) of that window."""
    hc = shape.h.tocsc()
    hc.sort_indices()
    lo, hi = commit_rounds * per_round, (commit_rounds + 1) * per_round
    ptr = np.zeros(shape.columns + 1, dtype=np.uint32)
    rows: list[np.ndarray] = []
    for j in range(shape.columns):
        r = hc.indices[hc.indptr[j] : hc.indptr[j + 1]]
        r = (r[(r >= lo) & (r < hi)] - lo) if shape.commit[j] else r[:0]
        rows.append(r.astype(np.uint32))
        ptr[j + 1] = ptr[j] + r.size
    return ptr, (np.concatenate(rows) if rows else np.zeros(0, dtype=np.uint32))


def write_netlist(problem: GlobalProblem, artifact: Path, spec: WindowSpec, fmt: FixedFormat, out: Path, window: int = 1, group: tuple[int, int] = GROUP) -> dict[str, Any]:
    """The netlist of placement (window, 0) of the plan: for the uniform boundary every placement
    uses this one bulk shape."""
    t0 = time.perf_counter()
    plan: WindowPlan = build_plan(problem, spec)
    placement = plan.placement(window, 0)
    if placement is None:
        raise ValueError(f"the plan has no window {window}")
    if placement.final:
        raise ValueError(f"window {window} is final; a netlist describes a window that commits C rounds")
    shape = plan.shapes[placement.shape]
    M = plan.per_round
    first_row = (placement.first_round - 1) * M
    global_rows = np.arange(first_row, first_row + shape.rows)
    if global_rows[-1] >= problem.m:
        raise ValueError("the placement reaches past the readout round; choose an earlier window")
    det_round = np.load(artifact / "det_round.npy").astype(np.int64)
    row_round = (det_round[global_rows] - placement.first_round).astype(np.int32)
    row_type = np.load(artifact / "det_type.npy").astype(np.uint8)[global_rows]
    row_check = np.load(artifact / "det_check.npy").astype(np.int32)[global_rows]
    hc = shape.h.tocsc()
    hc.sort_indices()
    if shape.columns and np.diff(hc.indptr).min() == 0:
        raise ValueError("the window has a column without rows; its round is undefined")
    first_local_row = np.array([hc.indices[hc.indptr[j]] for j in range(shape.columns)], dtype=np.int64)
    lam = np.array([math.log((1.0 - p) / p) if p > 0 else math.inf for p in shape.priors.tolist()])
    lambda_int = fmt.quantise_llr(lam).astype(np.int8)
    carry_ptr, carry_rows = _carry(shape, spec.commit, M)
    arrays: dict[str, np.ndarray] = {
        "row_round": row_round,
        "row_type": row_type,
        "row_check": row_check,
        "converge": shape.converge.astype(np.uint8),
        "H_indptr": shape.indptr(),
        "H_indices": shape.indices(),
        "Hc_indptr": hc.indptr.astype(np.uint32),
        "Hc_indices": hc.indices.astype(np.uint32),
        "col_round": row_round[first_local_row].astype(np.int32),
        "prior": shape.priors.astype(np.float64),
        "lambda_int": lambda_int,
        "commit": shape.commit.astype(np.uint8),
        "frame": shape.cls.astype(np.uint64),
        "global_column": placement.columns.astype(np.uint32),
        "carry_indptr": carry_ptr,
        "carry_rows": carry_rows,
    }
    t_lift = time.perf_counter()
    lift = lift_structure(shape, row_round, row_type, row_check, lambda_int, group)
    lift_seconds = time.perf_counter() - t_lift
    if lift is not None:
        arrays.update(
            row_orbit=lift.row_orbit, row_shift=lift.row_shift, col_orbit=lift.col_orbit, col_shift=lift.col_shift,
            base_row=lift.base_row, base_col=lift.base_col, base_shift=lift.base_shift,
        )  # fmt: skip
        logger.info("window graph is a lift of a base graph", extra={"base_rows": lift.base_rows, "base_columns": lift.base_columns, "base_edges": int(lift.base_row.size), "seconds": lift_seconds})
    else:
        logger.warning("window graph is not invariant under the group; netlist written without lift", extra={"group": list(group), "seconds": lift_seconds})
    _prepare_out(out, overwrite=True)
    np.savez_compressed(out / "netlist.npz", **arrays)
    degrees = np.diff(hc.indptr)
    row_degrees = np.diff(shape.h.indptr)
    description = {
        "format_version": NETLIST_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": _manifest_path(artifact), "manifest_sha256": problem.manifest_sha256},
        "window": spec.window_json(),
        "placement": placement.json(),
        "shape": shape.index,
        "detectors_per_round": M,
        "rows": shape.rows,
        "columns": shape.columns,
        "edges": shape.edges,
        "committed_columns": int(shape.commit.sum()),
        "merged_columns": shape.merged_columns,
        "observables": problem.k,
        "column_degree": {"min": int(degrees.min()), "max": int(degrees.max()), "mean": float(degrees.mean())},
        "row_degree": {"min": int(row_degrees.min()), "max": int(row_degrees.max()), "mean": float(row_degrees.mean())},
        "carry_edges": int(carry_rows.size),
        "arithmetic": fmt.name,
        "lambda_int": "sat(round(S ln((1 - p)/p))), round half away from zero, sat = clip to +-(2^N - 1)",
        "lift": None
        if lift is None
        else {
            "group": list(group),
            "action": f"Z{group[0]} x Z{group[1]} on the check index c = {group[1]} a + b: (a, b) -> (a + s mod {group[0]}, b + t mod {group[1]})",
            "base_rows": lift.base_rows,
            "base_columns": lift.base_columns,
            "base_edges": int(lift.base_row.size),
            "copies": group[0] * group[1],
            "prior_rtol": PRIOR_RTOL,
        },
        "arrays": __doc__.split("Arrays of netlist.npz")[1].strip(),
        "sha256_npz": _sha256_file(out / "netlist.npz"),
    }
    (out / "netlist.json").write_text(json.dumps(description, indent=2) + "\n")
    logger.info("netlist written", extra={"out": str(out), "rows": shape.rows, "columns": shape.columns, "edges": shape.edges, "seconds": time.perf_counter() - t0})
    return description


@dataclass
class Netlist:
    description: dict[str, Any]
    h: sp.csr_matrix
    arrays: dict[str, np.ndarray]

    @property
    def carry(self) -> sp.csc_matrix:
        """M x n: entry (i, l) = 1 when committing l flips row i of the next window."""
        a = self.arrays
        data = np.ones(a["carry_rows"].size, dtype=np.uint8)
        return sp.csc_matrix((data, a["carry_rows"].astype(np.int64), a["carry_indptr"].astype(np.int64)), shape=(self.description["detectors_per_round"], self.h.shape[1]))

    def frame_matrix(self) -> np.ndarray:
        """k x n uint8: A~ restricted to committed columns."""
        k = self.description["observables"]
        return ((self.arrays["frame"][None, :] >> np.arange(k, dtype=np.uint64)[:, None]) & np.uint64(1)).astype(np.uint8)


def read_netlist(path: Path) -> Netlist:
    """Reads and checks a netlist: checksum, sizes, CSR against CSC, carry map against H~, and
    (when present) the lift against H~."""
    description = json.loads((path / "netlist.json").read_text())
    if _sha256_file(path / "netlist.npz") != description["sha256_npz"]:
        raise ValueError(f"{path / 'netlist.npz'} does not match netlist.json")
    with np.load(path / "netlist.npz", allow_pickle=False) as data:
        a = {name: data[name] for name in data.files}
    m, n = description["rows"], description["columns"]
    h = sp.csr_matrix((np.ones(a["H_indices"].size, dtype=np.uint8), a["H_indices"].astype(np.int64), a["H_indptr"].astype(np.int64)), shape=(m, n))
    hc = sp.csc_matrix((np.ones(a["Hc_indices"].size, dtype=np.uint8), a["Hc_indices"].astype(np.int64), a["Hc_indptr"].astype(np.int64)), shape=(m, n))
    if h.nnz != description["edges"] or (h != hc).nnz:
        raise ValueError("the CSR and CSC forms of H~ differ")
    for name, size in (("prior", n), ("lambda_int", n), ("commit", n), ("frame", n), ("col_round", n), ("converge", m), ("row_round", m)):
        if a[name].shape != (size,):
            raise ValueError(f"{name} has shape {a[name].shape}, expected ({size},)")
    netlist = Netlist(description, h, a)
    M, C = description["detectors_per_round"], description["window"]["commit"]
    expected = (hc[C * M : (C + 1) * M] @ sp.diags(a["commit"].astype(np.uint8), dtype=np.uint8)).tocsc()
    if (netlist.carry != expected.astype(bool).astype(np.uint8)).nnz:
        raise ValueError("the carry map is not the committed columns' rows in round C of the window")
    if "base_row" in a:
        na, nb = description["lift"]["group"]
        coo = h.tocoo()
        rel = (a["row_shift"][coo.row].astype(np.int64) - a["col_shift"][coo.col].astype(np.int64)) % np.array([na, nb])
        edges = {tuple(e) for e in np.stack([a["row_orbit"][coo.row], a["col_orbit"][coo.col], rel[:, 0], rel[:, 1]], axis=1).tolist()}
        base = {tuple(e) for e in np.stack([a["base_row"], a["base_col"], a["base_shift"][:, 0], a["base_shift"][:, 1]], axis=1).tolist()}
        if edges != base or len(base) * na * nb != h.nnz:
            raise ValueError("the base graph does not reproduce H~")
    return netlist


def dem_lift(artifact: Path, group: tuple[int, int] = GROUP) -> dict[str, Any]:
    """The lift of a whole artifact's Tanner graph (H, priors), with the orbit counts per round:
    check orbits by detector round and type, fault orbits by the earliest round s(j) they touch."""
    t0 = time.perf_counter()
    problem = load_problem(artifact)
    det_round = np.load(artifact / "det_round.npy").astype(np.int64)
    det_type = np.load(artifact / "det_type.npy").astype(np.uint8)
    det_check = np.load(artifact / "det_check.npy").astype(np.int32)
    zeros_n = np.zeros(problem.n, dtype=np.uint8)
    lift = graph_lift(problem.h, problem.priors, zeros_n, zeros_n, np.zeros(problem.m, dtype=np.uint8), det_round, det_type, det_check, group)
    if lift is None:
        logger.warning("artifact graph is not a lift", extra={"artifact": str(artifact), "group": list(group)})
        return {"artifact": _manifest_path(artifact), "group": list(group), "lift": False}
    hc = problem.h_csc
    earliest = det_round[hc.indices[hc.indptr[:-1]]]
    orbit_round = np.zeros(lift.base_columns, dtype=np.int64)
    orbit_round[lift.col_orbit] = earliest
    row_orbit_round = np.zeros(lift.base_rows, dtype=np.int64)
    row_orbit_round[lift.row_orbit] = det_round
    row_orbit_type = np.zeros(lift.base_rows, dtype=np.int64)
    row_orbit_type[lift.row_orbit] = det_type
    rounds = sorted(set(det_round.tolist()))
    checks = {int(r): {int(t): int(np.sum((row_orbit_round == r) & (row_orbit_type == t))) for t in sorted(set(det_type.tolist()))} for r in rounds}
    faults = {int(r): int(np.sum(orbit_round == r)) for r in rounds}
    edges_by_round = {int(r): int(np.sum(orbit_round[lift.base_col] == r)) for r in rounds}
    result = {
        "artifact": _manifest_path(artifact),
        "group": list(group),
        "lift": True,
        "rows": problem.m,
        "columns": problem.n,
        "edges": int(problem.h.nnz),
        "base_rows": lift.base_rows,
        "base_columns": lift.base_columns,
        "base_edges": int(lift.base_row.size),
        "check_orbits_by_round_and_type": checks,
        "fault_orbits_by_earliest_round": faults,
        "base_edges_by_earliest_round": edges_by_round,
        "prior_rtol": PRIOR_RTOL,
        "seconds": time.perf_counter() - t0,
    }
    logger.info("artifact lift verified", extra={k: result[k] for k in ("artifact", "base_rows", "base_columns", "base_edges", "seconds")})
    return result


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rtd.fixed_netlist", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("window", help="write the netlist of one window")
    w.add_argument("--artifact", type=Path, required=True)
    w.add_argument("--width", type=int, required=True)
    w.add_argument("--commit", type=int, required=True)
    w.add_argument("--converge", type=int, required=True)
    w.add_argument("--boundary", choices=("exact", "uniform"), default="uniform")
    w.add_argument("--window", type=int, default=1, help="placement (window, 0) to describe; the uniform boundary uses one shape everywhere")
    w.add_argument("--format", choices=FORMATS, required=True)
    w.add_argument("--group", type=int, nargs=2, default=list(GROUP), metavar=("L", "M"), help="cyclic group orders of the bivariate-bicycle lift")
    w.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("lift", help="verify the lift of a whole artifact and print its orbit counts as JSON")
    d.add_argument("--artifact", type=Path, required=True)
    d.add_argument("--group", type=int, nargs=2, default=list(GROUP), metavar=("L", "M"))
    d.add_argument("--out", type=Path, help="also write the JSON here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        if args.command == "lift":
            result = dem_lift(args.artifact, tuple(args.group))
            text = json.dumps(result, indent=2)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text + "\n")
            print(text)
            return 0 if result["lift"] else 1
        spec = WindowSpec(args.width, args.commit, args.converge, args.boundary, "commit_anyway", 0, None)
        spec.validate()
        problem = load_problem(args.artifact)
        write_netlist(problem, args.artifact, spec, FixedFormat.parse(args.format), args.out, args.window, tuple(args.group))
        read_netlist(args.out)
    except (ValueError, OSError) as e:
        logger.exception("netlist command failed", extra={"command": args.command, "artifact": str(args.artifact), "error_type": type(e).__name__})
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
