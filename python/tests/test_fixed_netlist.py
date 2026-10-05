"""Window netlists: the lift finder on a constructed lifted graph, the netlist writer and reader on
bb18 and gross windows, and the Z12 x Z6 lift of the gross artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rtd.fixed_netlist import dem_lift, graph_lift, main as netlist_main, read_netlist, write_netlist
from rtd.fixed_ref import FixedFormat
from rtd.window_ref import WindowSpec, build_plan, load_problem

REPO = Path(__file__).resolve().parents[2]
BB18_ARTIFACT = REPO / "test" / "fixtures" / "bb18_choi_r9" / "artifact"
GROSS = REPO / "data" / "artifacts" / "gross_choi_p0.003"
GROSS_X = REPO / "data" / "artifacts" / "gross_choi_p0.003_xz" / "x"
GROSS_X_R24 = REPO / "data" / "artifacts" / "gross_choi_p0.003_r24_compat_xz" / "x"
INT4 = FixedFormat.parse("int4.2.8")


def _lifted_graph(group: tuple[int, int], rounds: int, base_columns: list[list[tuple[int, int, int]]]) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """H of the lift: base column o with entries (round, s, t) has, at copy (a, b), rows
    (round, check 6-style index ((s + a) mod L) M' + (t + b) mod M')."""
    na, nb = group
    copies = na * nb
    rows, cols = [], []
    col = 0
    for entries in base_columns:
        for a in range(na):
            for b in range(nb):
                for r, s, t in entries:
                    rows.append(r * copies + ((s + a) % na) * nb + (t + b) % nb)
                    cols.append(col)
                col += 1
    h = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(rounds * copies, col))
    h.sort_indices()
    row_round = np.repeat(np.arange(rounds), copies).astype(np.int64)
    row_check = np.tile(np.arange(copies), rounds).astype(np.int32)
    return h, row_round, row_check


def _toy() -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = [[(0, 0, 0), (0, 1, 0), (1, 0, 1)], [(0, 0, 0), (1, 2, 1)], [(1, 0, 0), (1, 1, 1), (0, 2, 0)], [(0, 1, 1)]]
    h, row_round, row_check = _lifted_graph((3, 2), 2, base)
    priors = np.repeat(np.array([0.01, 0.02, 0.01, 0.003]), 6)
    return h, row_round, row_check, priors, INT4.quantise_llr(np.log((1 - priors) / priors)).astype(np.int8)


def test_the_lift_of_a_constructed_lifted_graph_is_found():
    h, row_round, row_check, priors, lam = _toy()
    zeros_n, zeros_m = np.zeros(h.shape[1], dtype=np.uint8), np.zeros(h.shape[0], dtype=np.uint8)
    lift = graph_lift(h, priors, lam, zeros_n, zeros_m, row_round, zeros_m, row_check, (3, 2))
    assert lift is not None
    assert (lift.base_rows, lift.base_columns, lift.base_row.size) == (2, 4, 9)
    # Every node's orbit and shift reproduce H: (i, l) is an edge iff its (orbits, relative shift) is a base edge.
    coo = h.tocoo()
    rel = (lift.row_shift[coo.row].astype(int) - lift.col_shift[coo.col].astype(int)) % np.array([3, 2])
    got = {tuple(e) for e in np.stack([lift.row_orbit[coo.row], lift.col_orbit[coo.col], rel[:, 0], rel[:, 1]], axis=1).tolist()}
    assert got == {tuple(e) for e in np.stack([lift.base_row, lift.base_col, lift.base_shift[:, 0], lift.base_shift[:, 1]], axis=1).tolist()}
    assert np.bincount(lift.col_orbit).tolist() == [6] * 4


def test_a_graph_that_breaks_the_symmetry_has_no_lift():
    h, row_round, row_check, priors, lam = _toy()
    zeros_n, zeros_m = np.zeros(h.shape[1], dtype=np.uint8), np.zeros(h.shape[0], dtype=np.uint8)
    args = (row_round, zeros_m, row_check, (3, 2))
    lifted = graph_lift(h, priors, lam, zeros_n, zeros_m, *args)
    assert lifted is not None
    coo = h.tocoo()
    dropped = h.tolil()
    dropped[coo.row[0], coo.col[0]] = 0
    assert graph_lift(dropped.tocsr(), priors, lam, zeros_n, zeros_m, *args) is None
    tilted = priors.copy()
    tilted[1] *= 1.0 + 1e-9
    assert graph_lift(h, tilted, lam, zeros_n, zeros_m, *args) is None
    tilted[1] = priors[1] * (1.0 + 1e-15)  # below the rounding tolerance of merged priors
    assert graph_lift(h, tilted, lam, zeros_n, zeros_m, *args) is not None
    label = zeros_n.copy()
    label[2] = 1
    assert graph_lift(h, priors, lam, label, zeros_m, *args) is None
    row_label = zeros_m.copy()
    row_label[0] = 1
    assert graph_lift(h, priors, lam, zeros_n, row_label, *args) is None
    assert graph_lift(h, priors, lam, zeros_n, zeros_m, row_round, zeros_m, row_check, (2, 3)) is None


def test_bb18_window_netlist_round_trip(tmp_path):
    problem = load_problem(BB18_ARTIFACT)
    spec = WindowSpec(4, 2, 2, "uniform", "commit_anyway", 0, None)
    description = write_netlist(problem, BB18_ARTIFACT, spec, INT4, tmp_path / "nl")
    netlist = read_netlist(tmp_path / "nl")
    plan = build_plan(problem, spec)
    placement = plan.placement(1, 0)
    shape = plan.shapes[placement.shape]
    assert (netlist.h != shape.h).nnz == 0
    a = netlist.arrays
    lam = np.array([math.log((1 - p) / p) for p in shape.priors])
    assert np.array_equal(a["lambda_int"], INT4.quantise_llr(lam))
    assert np.array_equal(a["commit"], shape.commit) and np.array_equal(a["converge"], shape.converge)
    assert np.array_equal(a["global_column"], placement.columns)
    assert description["rows"] == 4 * plan.per_round and description["window"]["commit"] == 2
    # The carry map: rows of round C of the window each committed column flips.
    carry = netlist.carry.toarray()
    dense = shape.h.toarray()
    M = plan.per_round
    assert np.array_equal(carry, dense[2 * M : 3 * M] * shape.commit[None, :])
    frame = netlist.frame_matrix()
    assert frame.shape == (problem.k, shape.columns)
    assert np.array_equal(frame.T @ (1 << np.arange(problem.k, dtype=np.uint64)), shape.cls)
    assert not frame[:, shape.commit == 0].any()
    # The rounds of rows and columns are relative to the window.
    assert a["row_round"].min() == 0 and a["row_round"].max() == 3
    assert np.all(a["col_round"] >= 0) and np.all(a["col_round"] <= 3)


def test_the_reader_rejects_a_changed_netlist(tmp_path):
    problem = load_problem(BB18_ARTIFACT)
    write_netlist(problem, BB18_ARTIFACT, WindowSpec(4, 2, 2, "uniform", "commit_anyway", 0, None), INT4, tmp_path / "nl")
    description = json.loads((tmp_path / "nl" / "netlist.json").read_text())
    description["sha256_npz"] = "0" * 64
    (tmp_path / "nl" / "netlist.json").write_text(json.dumps(description))
    with pytest.raises(ValueError, match="does not match"):
        read_netlist(tmp_path / "nl")


def test_the_netlist_cli_reports_bad_windows(tmp_path):
    base = ["window", "--artifact", str(BB18_ARTIFACT), "--width", "4", "--commit", "2", "--converge", "2", "--format", "int5.2.8"]
    assert netlist_main([*base, "--out", str(tmp_path / "ok")]) == 0
    assert read_netlist(tmp_path / "ok").description["arithmetic"] == "int5.2.8"
    assert netlist_main([*base, "--window", "40", "--out", str(tmp_path / "far")]) == 2


needs_gross = pytest.mark.skipif(not (GROSS / "manifest.json").exists(), reason="gross artifacts are not present")


@needs_gross
def test_the_gross_artifact_is_a_z12_x_z6_lift():
    result = dem_lift(GROSS)
    assert (result["rows"], result["columns"]) == (1872, 71280)
    assert (result["base_rows"], result["base_columns"], result["base_edges"]) == (26, 990, 5832)
    assert all(types == {0: 1, 1: 1} for types in result["check_orbits_by_round_and_type"].values())
    bulk = [result["fault_orbits_by_earliest_round"][r] for r in range(2, 13)]
    assert bulk == [82] * 11


@needs_gross
def test_the_gross_x_half_and_its_bulk_window_are_lifts(tmp_path):
    x = dem_lift(GROSS_X)
    assert (x["base_rows"], x["base_columns"], x["base_edges"]) == (13, 122, 426)
    problem = load_problem(GROSS_X_R24)
    description = write_netlist(problem, GROSS_X_R24, WindowSpec(12, 8, 8, "uniform", "commit_anyway", 0, None), INT4, tmp_path / "nl")
    assert description["lift"]["base_rows"] == 12 and description["lift"]["copies"] == 72
    assert description["lift"]["base_edges"] * 72 == description["edges"]
    netlist = read_netlist(tmp_path / "nl")  # checks the base graph against H~
    assert netlist.description["lift"]["base_columns"] * 72 == netlist.description["columns"]
