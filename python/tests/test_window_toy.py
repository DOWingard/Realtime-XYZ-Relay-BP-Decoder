"""The hand-derived toy fixture (test/fixtures/toy_rep3): windowed decoding of a 3-bit repetition
code, checked window by window against values worked out on paper (DERIVATION.md)."""

import json
from pathlib import Path

import numpy as np
import pytest

from rtd.export import parse_dem
from rtd.window_ref import GlobalProblem, WindowSpec, build_plan, decode_shot, load_problem
from rtd.window_ref_inner import BruteForceInner, MatchingInner
from rtd.window_ref_toy import toy_circuit

TOY = Path(__file__).resolve().parents[2] / "test" / "fixtures" / "toy_rep3"
EXPECTED = json.loads((TOY / "expected.json").read_text())
CASES = EXPECTED["cases"]


@pytest.fixture(scope="module")
def toy() -> GlobalProblem:
    return load_problem(TOY)


def _syndrome(problem: GlobalProblem, faults: list[int]) -> np.ndarray:
    e = np.zeros(problem.n, dtype=np.int64)
    e[faults] = 1
    return (np.asarray(problem.h @ e).ravel() % 2).astype(np.uint8)


def test_artifact_is_the_circuit_model(toy):
    """The committed circuit is the generator's, and the artifact is its error model, column for column."""
    assert (TOY / "circuit.stim").read_text() == str(toy_circuit()) + "\n"
    h, a, priors = parse_dem(toy_circuit().detector_error_model(decompose_errors=False))
    assert (toy.h != h.tocsr()).nnz == 0
    assert (toy.a != a).nnz == 0
    assert np.array_equal(toy.priors, priors)


def test_loadable_as_an_ordinary_artifact():
    from rtd.golden import load_artifact

    artifact = load_artifact(TOY)
    assert (artifact.num_detectors, artifact.num_columns) == (14, 30)


def test_column_table(toy):
    """Column 5 (r - 1) + i is, for i = 0..4: q1 flip, c0 misread, q0 flip, q2 flip, c1 misread."""
    hc = toy.h.tocsc()
    cls_ = toy.column_class()
    for r in range(1, 7):
        b, row = 5 * (r - 1), 2 * (r - 1)
        expected = {b: [row, row + 1], b + 1: [row, row + 2], b + 2: [row], b + 3: [row + 1], b + 4: [row + 1, row + 3]}
        for j, rows in expected.items():
            assert hc.indices[hc.indptr[j] : hc.indptr[j + 1]].tolist() == rows
            assert int(cls_[j]) == (1 if j == b + 2 else 0)
            assert toy.priors[j] == (0.02 if j in (b + 1, b + 4) else 0.01)


@pytest.mark.parametrize(("width", "commit", "starts", "final_start"), [(3, 1, [1, 2, 3, 4, 5], 5), (4, 2, [1, 3, 5], 5)])
def test_window_layout(toy, width, commit, starts, final_start):
    plan = build_plan(toy, WindowSpec(width, commit, width, "exact", "commit_anyway", 0))
    assert [p.first_round for p in plan.placements] == starts
    assert [p.final for p in plan.placements] == [t == final_start for t in starts]
    for p in plan.placements:
        shape = plan.shapes[p.shape]
        if p.final:
            assert shape.merged_columns == 0 and bool(np.all(shape.commit == 1))
            continue
        last = p.first_round + width - 1
        b = 5 * (last - 1)
        # The two merged columns of the last round, each named by its smaller member.
        members = [p.members[p.members_ptr[c] : p.members_ptr[c + 1]].tolist() for c in range(p.columns.size)]
        merged = {int(p.columns[c]): mem for c, mem in enumerate(members) if len(mem) > 1}
        assert merged == {b + 1: [b + 1, b + 2], b + 3: [b + 3, b + 4]}
        local = {int(g): c for c, g in enumerate(p.columns)}
        assert shape.priors[local[b + 1]] == 0.02 * (1.0 - 0.01) + 0.01 * (1.0 - 0.02)
        assert shape.priors[local[b + 3]] == 0.01 * (1.0 - 0.02) + 0.02 * (1.0 - 0.01)
        committed = sorted(int(p.columns[c]) for c in np.flatnonzero(shape.commit))
        assert committed == list(range(5 * (p.first_round - 1), 5 * (p.first_round - 1 + commit)))


def _run_case(toy, case, inner):
    plan = build_plan(toy, WindowSpec(case["width"], case["commit"], case["width"], "exact", "commit_anyway", 0))
    syndrome = _syndrome(toy, case["injected_faults"])
    return plan, syndrome, decode_shot(plan, syndrome, 0, inner)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_hand_derivation_brute_force(toy, case):
    plan, syndrome, outcome = _run_case(toy, case, BruteForceInner())
    assert syndrome.tolist() == case["syndrome"]
    assert int(toy.column_class()[case["injected_faults"]].astype(np.int64).sum() % 2) == case["true_observables"]
    assert len(outcome.windows) == len(case["windows"])
    hc = toy.h.tocsc()
    committed_so_far = np.zeros(toy.n, dtype=np.int64)
    M = plan.per_round
    for record, want in zip(outcome.windows, case["windows"], strict=True):
        k = want["window"]
        placement = plan.placement(k, 0)
        assert placement.first_round == want["first_round"]
        lo = (placement.first_round - 1) * M
        carry = (np.asarray(hc @ committed_so_far).ravel() % 2)[lo : lo + M]
        assert carry.tolist() == want["carry"], f"window {k}: carry"
        assert record.residual_first_round.tolist() == want["residual_first_round_before_decode"], f"window {k}: residual"
        assert (syndrome[lo : lo + M] ^ carry).tolist() == want["residual_first_round_before_decode"]
        assert sorted(record.solution.tolist()) == want["solution"], f"window {k}: solution"
        assert record.num_optimal == 1, f"window {k}: minimum-weight solution not unique"
        assert record.committed.tolist() == want["committed"], f"window {k}: commits"
        assert record.frame_after == want["frame_after"], f"window {k}: frame"
        assert record.converged and record.unexplained == 0
        committed_so_far[record.committed.astype(np.int64)] = 1
    assert outcome.frame == case["final_frame"]
    assert not outcome.residual.any()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_matching_inner_agrees(toy, case):
    """Every toy column touches at most two detectors, so matching is an exact minimum-weight
    decoder here and must reproduce the hand-derived commits and frames."""
    pytest.importorskip("pymatching")
    _, _, outcome = _run_case(toy, case, MatchingInner())
    assert [w.committed.tolist() for w in outcome.windows] == [w["committed"] for w in case["windows"]]
    assert [w.frame_after for w in outcome.windows] == [w["frame_after"] for w in case["windows"]]
    assert outcome.frame == case["final_frame"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_bplsd_inner_runs(toy, case):
    pytest.importorskip("ldpc")
    from rtd.window_ref_inner import BpLsdInner

    _, _, outcome = _run_case(toy, case, BpLsdInner())
    assert outcome.success and not outcome.residual.any()
    assert outcome.frame == case["final_frame"]
