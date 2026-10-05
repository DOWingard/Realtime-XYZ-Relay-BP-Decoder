"""The windowing reference: time structure, plans, the detector rule, the commit partition, plan
dumps, the stream loop's policies, the whole-shot identity against relay_bp's goldens, and the
committed windowed goldens."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rtd.window_ref import (
    VIRTUAL,
    GlobalProblem,
    PlanError,
    WindowSpec,
    build_plan,
    check_outcomes,
    compare_plan_dirs,
    decode_shot,
    decode_shots,
    detector_rule_sets,
    dump_plan,
    load_problem,
    main,
    time_structure,
)
from rtd.window_ref_inner import BruteForceInner, FixedGammas, InnerContext, InnerResult, InnerSpec, RelayParams

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "test" / "fixtures"
BB18 = FIXTURES / "bb18_choi"
BB18_R9 = FIXTURES / "bb18_choi_r9"
TOY = FIXTURES / "toy_rep3"


@pytest.fixture(scope="module")
def bb18() -> GlobalProblem:
    return load_problem(BB18 / "artifact")


@pytest.fixture(scope="module")
def bb18_r9() -> GlobalProblem:
    return load_problem(BB18_R9 / "artifact")


@pytest.fixture(scope="module")
def toy() -> GlobalProblem:
    return load_problem(TOY)


def random_problem(seed: int, rounds: int = 7, per_round: int = 4, n: int = 60, k: int = 3) -> GlobalProblem:
    """Random columns, each touching a non-empty set of rows in round s and any rows of s + 1,
    with duplicates allowed in the last round so that merging happens."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n):
        s = int(rng.integers(1, rounds + 1))
        lo = (s - 1) * per_round
        rows = set((lo + np.flatnonzero(rng.random(per_round) < 0.4)).tolist()) or {lo + int(rng.integers(per_round))}
        if s < rounds:
            rows |= set((lo + per_round + np.flatnonzero(rng.random(per_round) < 0.4)).tolist())
        cols.append(sorted(rows))
    m = rounds * per_round
    h = np.zeros((m, n), dtype=np.uint8)
    for j, rows in enumerate(cols):
        h[rows, j] = 1
    a = (rng.random((k, n)) < 0.3).astype(np.uint8)
    det_round = np.repeat(np.arange(1, rounds + 1), per_round)
    return GlobalProblem.from_arrays(h, a, rng.uniform(0.001, 0.2, n), det_round)


def _plan_args(width, commit, converge, boundary, policy, deferrals, out) -> list[str]:
    args = ["dump-plan", "--artifact", str(BB18_R9 / "artifact"), "--width", str(width), "--commit", str(commit)]
    args += ["--converge", str(converge), "--boundary", boundary, "--on-failure", policy, "--max-deferrals", str(deferrals)]
    return args + ["--out", str(out)]


def _params_from_config(float_type: str, c: dict, interval: tuple) -> RelayParams:
    """RelayParams from relay_bp constructor arguments as rtd-golden records them."""
    return RelayParams(
        float_type, c["alpha"], c["alpha_iteration_scaling_factor"], c["gamma0"], c["pre_iter"], c["num_sets"],
        c["set_max_iter"], c["stopping_criterion"], c["stop_nconv"], interval,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Time structure and specification


def _with(problem: GlobalProblem, h=None, det_round=None) -> GlobalProblem:
    return GlobalProblem.from_arrays(problem.h if h is None else h, problem.a, problem.priors, problem.det_round if det_round is None else det_round)


def test_time_structure_of_the_fixtures(bb18, bb18_r9, toy):
    assert (time_structure(bb18).rounds_total, time_structure(bb18).per_round) == (4, 18)
    assert (time_structure(bb18_r9).rounds_total, time_structure(bb18_r9).per_round) == (10, 18)
    ts = time_structure(toy)
    assert (ts.rounds_total, ts.per_round) == (7, 2)
    assert ts.earliest.tolist() == np.repeat(np.arange(1, 7), 5).tolist()


def test_time_structure_rejections(toy):
    rounds = toy.det_round.copy()
    rounds[3] = 3  # round 2 loses a row, round 3 gains one
    with pytest.raises(PlanError, match="unequal_round_sizes"):
        time_structure(_with(toy, det_round=rounds))
    with pytest.raises(PlanError, match="bad_round_numbering"):
        time_structure(_with(toy, det_round=toy.det_round + 1))
    rounds = toy.det_round.copy()
    rounds[[3, 4]] = rounds[[4, 3]]
    with pytest.raises(PlanError, match="rows_not_grouped_by_round"):
        time_structure(_with(toy, det_round=rounds))
    h = toy.h.toarray()
    h[12, 0] = 1  # column 0 (round 1) now also touches round 7
    with pytest.raises(PlanError, match="column_spans_rounds"):
        time_structure(_with(toy, h=h))
    h = toy.h.toarray()
    h[:, 5] = 0
    with pytest.raises(PlanError, match="empty_column"):
        time_structure(_with(toy, h=h))
    priors = toy.priors.copy()
    priors[3] = 1.0
    with pytest.raises(PlanError, match="invalid_prior"):
        build_plan(GlobalProblem.from_arrays(toy.h, toy.a, priors, toy.det_round), WindowSpec(3, 1, 3, "exact", "flag", 0))


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        (WindowSpec(3, 3, 3, "exact", "commit_anyway", 0), "commit_out_of_range"),
        (WindowSpec(3, 0, 3, "exact", "commit_anyway", 0), "commit_out_of_range"),
        (WindowSpec(4, 2, 1, "exact", "commit_anyway", 0), "converge_out_of_range"),
        (WindowSpec(4, 2, 5, "exact", "commit_anyway", 0), "converge_out_of_range"),
        (WindowSpec(4, 2, 4, "exact", "flag", 1), "deferrals_without_defer"),
        (WindowSpec(4, 2, 4, "exact", "defer", -1), "invalid_spec"),
        (WindowSpec(4, 2, 4, "sideways", "flag", 0), "invalid_spec"),
        (WindowSpec(4, 2, 4, "exact", "retry", 0), "invalid_spec"),
        (WindowSpec(4, 2, 4, "exact", "flag", 0, 0), "zero_iteration_cap"),
    ],
)
def test_spec_validation(spec, code):
    with pytest.raises(PlanError, match=code):
        spec.validate()


# ---------------------------------------------------------------------------------------------
# Plan properties


def _members(p, c):
    return p.members[p.members_ptr[c] : p.members_ptr[c + 1]].astype(np.int64)


def _check_exact_plan(problem: GlobalProblem, W: int, C: int) -> None:
    plan = build_plan(problem, WindowSpec(W, C, W, "exact", "commit_anyway", 0))
    ts = time_structure(problem)
    M = ts.per_round
    hc = problem.h.tocsc()
    rule = detector_rule_sets(problem, W, C)
    zero = [p for p in plan.placements if p.attempt == 0]
    assert len(zero) == len(rule) == plan.num_positions
    seen = np.zeros(problem.n, dtype=np.int64)
    commits_by_window = []
    for p, (rule_cols, rule_commits) in zip(zero, rule, strict=True):
        shape = plan.shapes[p.shape]
        members = [_members(p, c) for c in range(p.columns.size)]
        cols = np.sort(np.concatenate(members))
        commits = np.sort(np.concatenate([members[c] for c in np.flatnonzero(shape.commit)]))
        # Lee et al.'s detector rule builds the same column and commit sets.
        assert np.array_equal(cols, rule_cols)
        assert np.array_equal(commits, rule_commits)
        seen[commits] += 1
        commits_by_window.append(set(commits.tolist()))
        lo = (p.first_round - 1) * M
        for c, mem in enumerate(members):
            if mem.size > 1:
                assert shape.commit[c] == 0, "merged column in a commit set"
                assert set(ts.earliest[mem].tolist()) == {p.first_round + W - 1}
            # Each local column's rows are its representative's rows inside the window.
            rows = hc.indices[hc.indptr[mem[0]] : hc.indptr[mem[0] + 1]]
            inside = rows[(rows >= lo) & (rows < lo + shape.rows)] - lo
            assert shape.h_csc.indices[shape.h_csc.indptr[c] : shape.h_csc.indptr[c + 1]].tolist() == inside.tolist()
        # Local columns ascend by representative, local rows are the window's rows in order.
        assert np.all(np.diff(p.columns.astype(np.int64)) > 0)
        assert shape.rows == (problem.m - lo if p.final else W * M)
    # The commit sets partition the columns.
    assert np.all(seen == 1)
    # Every row of a committed round is touched only by that window's commits or the previous one's.
    for k, p in enumerate(zero):
        lo = (p.first_round - 1) * M
        hi = problem.m if p.final else lo + C * M
        allowed = commits_by_window[k] | (commits_by_window[k - 1] if k else set())
        assert set(problem.h[lo:hi].indices.tolist()) <= allowed


@pytest.mark.parametrize(("W", "C"), [(2, 1), (3, 1), (3, 2), (4, 1), (4, 3), (5, 2)])
def test_exact_plan_invariants_bb18(bb18, bb18_r9, W, C):
    _check_exact_plan(bb18_r9, W, C)
    if W < 4:
        _check_exact_plan(bb18, W, C)


@pytest.mark.parametrize("seed", range(12))
def test_exact_plan_invariants_random(seed):
    problem = random_problem(seed)
    for W, C in [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3), (7, 2), (9, 4)]:
        _check_exact_plan(problem, W, C)


def test_merging_happens_on_random_problems():
    merged = sum(s.merged_columns for seed in range(12) for s in build_plan(random_problem(seed), WindowSpec(3, 1, 3, "exact", "commit_anyway", 0)).shapes)
    assert merged > 0


def test_bulk_shapes_repeat(bb18_r9):
    """The circuit is the same every bulk round, so the bulk windows share one shape."""
    plan = build_plan(bb18_r9, WindowSpec(4, 2, 4, "exact", "commit_anyway", 0))
    assert [p.shape for p in plan.placements] == [0, 1, 1, 2]


def test_deferral_placements(toy):
    plan = build_plan(toy, WindowSpec(3, 1, 3, "exact", "defer", 2))
    got = [(p.window, p.attempt, p.first_round, p.rounds, p.commit_rounds, p.final) for p in plan.placements]
    assert got == [
        (0, 0, 1, 3, 1, False), (0, 1, 1, 4, 1, False), (0, 2, 1, 5, 1, False),
        (1, 0, 2, 3, 1, False), (1, 1, 2, 4, 1, False), (1, 2, 2, 5, 1, False),
        (2, 0, 3, 3, 1, False), (2, 1, 3, 4, 1, False), (2, 2, 3, 5, 5, True),
        (3, 0, 4, 3, 1, False), (3, 1, 4, 4, 4, True),
        (4, 0, 5, 3, 3, True),
    ]  # fmt: skip
    # Convergence rows grow with the attempt: C' + a C rounds.
    assert plan.shapes[plan.placement(0, 1).shape].converge.tolist() == [1] * 8
    narrow = build_plan(toy, WindowSpec(3, 1, 1, "exact", "defer", 1))
    assert narrow.shapes[narrow.placement(0, 0).shape].converge.tolist() == [1, 1, 0, 0, 0, 0]
    assert narrow.shapes[narrow.placement(0, 1).shape].converge.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]


def test_identity_plan_is_the_global_problem(bb18):
    plan = build_plan(bb18, WindowSpec(9, 2, 9, "exact", "commit_anyway", 0))
    assert len(plan.placements) == 1 and plan.placements[0].final
    shape = plan.shapes[0]
    assert (shape.h != bb18.h).nnz == 0
    assert shape.priors.tobytes() == bb18.priors.tobytes()
    assert np.array_equal(plan.placements[0].columns, np.arange(bb18.n))
    assert np.array_equal(shape.cls, bb18.column_class())


# ---------------------------------------------------------------------------------------------
# Uniform boundary


def test_uniform_plan(bb18_r9):
    exact = build_plan(bb18_r9, WindowSpec(4, 2, 4, "exact", "commit_anyway", 0))
    uniform = build_plan(bb18_r9, WindowSpec(4, 2, 4, "uniform", "commit_anyway", 0))
    assert len(uniform.shapes) == 1 and uniform.num_positions == 5
    assert uniform.shapes[0].key() == exact.shapes[exact.placement(1, 0).shape].key()
    assert np.array_equal(uniform.placement(1, 0).columns, exact.placement(1, 0).columns)
    assert all(not p.final and p.rounds == 4 and p.commit_rounds == 2 for p in uniform.placements)
    # Position 2 (rounds 5-8) is bulk: the mapping is a pure shift by two rounds, column for column.
    ts = time_structure(bb18_r9)
    commit = uniform.shapes[0].commit == 1
    base = uniform.placement(1, 0).columns.astype(np.int64)
    mapped = uniform.placement(2, 0).columns
    assert not np.any(mapped == VIRTUAL)
    assert np.array_equal(ts.earliest[mapped.astype(np.int64)], ts.earliest[base] + 2)
    # Position 3 (rounds 7-10) holds the readout round: its buffer columns have no shifted
    # counterpart, its committed ones (rounds 7, 8) all do.
    virtual = uniform.placement(3, 0).columns == VIRTUAL
    assert virtual.any() and not (virtual & commit).any()
    # Position 0 reaches before round 1 and position 4 past the readout: committed virtual columns.
    assert (uniform.placement(0, 0).columns[commit] == VIRTUAL).any()
    assert (uniform.placement(4, 0).columns[commit] == VIRTUAL).any()
    assert uniform.statistics()["committed_virtual_columns"] == int(sum(((p.columns == VIRTUAL) & commit).sum() for p in uniform.placements))


@pytest.mark.parametrize(("W", "D", "ok"), [(4, 0, True), (4, 1, True), (4, 2, False), (5, 0, True), (5, 1, False), (9, 0, False)])
def test_uniform_needs_a_bulk_window(bb18_r9, W, D, ok):
    spec = WindowSpec(W, 2, W, "uniform", "defer" if D else "flag", D)
    if ok:
        assert len(build_plan(bb18_r9, spec).shapes) == D + 1
    else:
        with pytest.raises(PlanError, match="no_bulk_window"):
            build_plan(bb18_r9, spec)


# ---------------------------------------------------------------------------------------------
# Plan dump


@pytest.mark.parametrize(
    "spec",
    [WindowSpec(4, 2, 3, "exact", "defer", 2), WindowSpec(4, 2, 4, "uniform", "defer", 1), WindowSpec(3, 1, 3, "exact", "flag", 0)],
    ids=["exact_defer", "uniform_defer", "exact_flag"],
)
def test_dump_plan_round_trip(bb18_r9, tmp_path, spec):
    plan = build_plan(bb18_r9, spec)
    dump_plan(plan, tmp_path / "py")
    doc = json.loads((tmp_path / "py" / "plan.json").read_text())
    assert doc == json.loads(json.dumps(plan.json()))
    assert doc["spec"] == {
        "width": spec.width,
        "commit": spec.commit,
        "converge_rounds": spec.converge_rounds,
        "boundary": spec.boundary,
        "on_failure": spec.on_failure,
        "max_deferrals": spec.max_deferrals,
    }
    assert doc["rounds_total"] == 10 and doc["detectors_per_round"] == 18
    for s in plan.shapes:
        d = tmp_path / "py"
        indptr, indices = np.load(d / f"shape_{s.index}_H_indptr.npy"), np.load(d / f"shape_{s.index}_H_indices.npy")
        assert indptr.dtype == np.uint32 and indices.dtype == np.uint32
        back = sp.csr_matrix((np.ones(indices.size, dtype=np.uint8), indices, indptr), shape=(s.rows, s.columns))
        assert (back != s.h).nnz == 0
        assert np.load(d / f"shape_{s.index}_priors.npy").tobytes() == s.priors.tobytes()
        assert np.load(d / f"shape_{s.index}_class.npy").dtype == np.uint64
    for p in plan.placements:
        cols = np.load(tmp_path / "py" / f"placement_{p.window}_{p.attempt}_columns.npy")
        assert cols.dtype == np.uint32 and np.array_equal(cols, p.columns)
        has_members = (tmp_path / "py" / f"placement_{p.window}_{p.attempt}_members.npy").exists()
        assert has_members == (spec.boundary == "exact")
    # The command line writes the same files, and a changed prior bit is detected.
    args = _plan_args(spec.width, spec.commit, spec.converge_rounds, spec.boundary, spec.on_failure, spec.max_deferrals, tmp_path / "cli")
    assert main(args) == 0
    assert compare_plan_dirs(tmp_path / "py", tmp_path / "cli") == []
    shutil.copytree(tmp_path / "cli", tmp_path / "bad")
    priors = np.load(tmp_path / "bad" / "shape_0_priors.npy")
    priors[0] = np.nextafter(priors[0], 1.0)
    np.save(tmp_path / "bad" / "shape_0_priors.npy", priors)
    assert compare_plan_dirs(tmp_path / "py", tmp_path / "bad") == [f"shape_0_priors.npy differs (float64[{priors.size}] vs float64[{priors.size}])"]


def _cpp_plan_tool() -> Path | None:
    for preset in ("release", "debug"):
        for candidate in sorted((REPO / "build").glob(f"*{preset}*/src/rtd_window_plan")):
            if candidate.is_file():
                return candidate
    return None


@pytest.mark.parametrize(
    ("artifact", "W", "C", "CV", "boundary", "policy", "D"),
    [
        (TOY, 3, 1, 2, "exact", "defer", 2),
        (BB18 / "artifact", 3, 2, 3, "exact", "defer", 1),
        (BB18_R9 / "artifact", 5, 2, 5, "exact", "defer", 2),
        (BB18_R9 / "artifact", 4, 2, 3, "uniform", "defer", 1),
    ],
)
def test_plan_matches_cpp_tool(tmp_path, artifact, W, C, CV, boundary, policy, D):
    """The C++ plan dump tool and this reference write identical plans."""
    import subprocess

    tool = _cpp_plan_tool()
    if tool is None:
        pytest.skip("rtd_window_plan has not been built")
    flags = ["--artifact", str(artifact), "--width", str(W), "--commit", str(C), "--converge", str(CV)]
    flags += ["--boundary", boundary, "--on-failure", policy, "--max-deferrals", str(D)]
    subprocess.run([str(tool), *flags, "--out", str(tmp_path / "cpp")], check=True, capture_output=True)
    assert main(["dump-plan", *flags, "--out", str(tmp_path / "py")]) == 0
    assert compare_plan_dirs(tmp_path / "cpp", tmp_path / "py") == []


def test_dump_plan_rejects_impossible_uniform(tmp_path):
    assert main(_plan_args(5, 2, 5, "uniform", "defer", 1, tmp_path / "x")) == 2


# ---------------------------------------------------------------------------------------------
# Stream loop and policies


class ScriptedInner:
    """The brute-force decoder, except that chosen (window, attempt) pairs report no solution
    and return an empty correction."""

    name = "scripted"

    def __init__(self, fail: set[tuple[int, int]]):
        self._fail = fail
        self._brute = BruteForceInner()
        self.calls: list[tuple[int, int, int]] = []

    def decode(self, h, priors, syndrome, ctx: InnerContext) -> InnerResult:
        self.calls.append((ctx.window, ctx.attempt, ctx.gamma_stream))
        if (ctx.window, ctx.attempt) in self._fail:
            return InnerResult(np.zeros(0, dtype=np.int64), 7, 2, False, math.inf)
        return self._brute.decode(h, priors, syndrome, ctx)

    def close(self) -> None:
        pass


def _toy_syndrome(toy, faults):
    e = np.zeros(toy.n, dtype=np.int64)
    e[faults] = 1
    return (np.asarray(toy.h @ e).ravel() % 2).astype(np.uint8)


@pytest.mark.parametrize("policy", ["commit_anyway", "flag"])
def test_failed_window_commits_its_output(toy, policy):
    plan = build_plan(toy, WindowSpec(3, 1, 3, "exact", policy, 0))
    syndrome = _toy_syndrome(toy, [6, 15, 27])
    inner = ScriptedInner({(1, 0)})
    out = decode_shot(plan, syndrome, 5, inner)
    w = out.windows[1]
    assert not w.converged and math.isinf(w.weight) and w.attempts == 1
    assert (w.iterations, w.legs) == (7, 2)
    assert w.flagged == (policy == "flag") and out.flagged == (policy == "flag")
    # Nothing was committed, so round 2's detector stays unexplained, and later windows go on.
    assert w.committed.size == 0 and w.unexplained == 1
    assert not out.success and out.residual.sum() == 1
    assert [c[0] for c in inner.calls] == [0, 1, 2, 3, 4]
    assert inner.calls[1][2] == 5 + (1 << 32)
    check_outcomes(toy, plan, syndrome[None, :], [out])


def test_defer_widens_then_succeeds(toy):
    plan = build_plan(toy, WindowSpec(3, 1, 3, "exact", "defer", 2))
    syndrome = _toy_syndrome(toy, [6, 15, 27])
    inner = ScriptedInner({(1, 0)})
    out = decode_shot(plan, syndrome, 0, inner)
    w = out.windows[1]
    assert w.attempts == 2 and w.converged and not w.flagged
    assert (w.iterations, w.legs) == (7, 3)
    assert w.committed.tolist() == [6]
    assert inner.calls[2][2] == (1 << 32) + (1 << 56)
    assert out.success and out.frame == 1


def test_defer_exhausted_flags(toy):
    plan = build_plan(toy, WindowSpec(3, 1, 3, "exact", "defer", 2))
    out = decode_shot(plan, _toy_syndrome(toy, [6, 15, 27]), 0, ScriptedInner({(1, 0), (1, 1), (1, 2)}))
    assert out.windows[1].attempts == 3 and out.windows[1].flagged and out.flagged


def test_final_deferral_completes_the_stream(toy):
    """Window 3's second attempt reaches the readout, is final and commits everything left, so
    window 4 is not decoded and records an empty window."""
    plan = build_plan(toy, WindowSpec(3, 1, 3, "exact", "defer", 2))
    syndrome = _toy_syndrome(toy, [6, 15, 27])
    inner = ScriptedInner({(3, 0)})
    out = decode_shot(plan, syndrome, 0, inner)
    assert [c[:2] for c in inner.calls] == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
    w3, w4 = out.windows[3], out.windows[4]
    assert w3.attempts == 2 and w3.converged and w3.committed.tolist() == [15, 27]
    assert (w4.attempts, w4.iterations, w4.legs, w4.converged, w4.weight, w4.committed.size, w4.unexplained) == (0, 0, 0, True, 0.0, 0, 0)
    assert out.success and out.frame == 1 and not out.residual.any()
    check_outcomes(toy, plan, syndrome[None, :], [out])


def test_uniform_stream_on_the_toy(toy):
    """The toy's bulk window is the same as its first, so a uniform decode of these faults
    commits the same columns, except that the last position's rounds past the readout are
    virtual and its readout round is modelled as a bulk round."""
    plan = build_plan(toy, WindowSpec(3, 1, 3, "uniform", "commit_anyway", 0))
    assert plan.num_positions == 7
    syndrome = _toy_syndrome(toy, [6, 15])
    out = decode_shot(plan, syndrome, 0, BruteForceInner())
    assert [w.committed.tolist() for w in out.windows] == [[], [6], [], [15], [], [], []]
    assert out.success and not out.residual.any()
    check_outcomes(toy, plan, syndrome[None, :], [out])


# ---------------------------------------------------------------------------------------------
# relay_bp: identity and goldens


def _relay_params(manifest: dict) -> RelayParams:
    c = manifest["config"]
    return _params_from_config(manifest["float"], c, tuple(c["gamma_dist_interval"]))


@pytest.mark.parametrize("golden", ["relay_f32", "relay_f64"])
@pytest.mark.parametrize(("W", "C"), [(4, 1), (10, 3)])
def test_identity_with_whole_shot_golden(bb18, golden, W, C):
    """With W >= Rt there is one final window, the whole problem in the same order, so the
    windowed decode is relay_bp's whole-shot decode: same e, iterations, legs and weight."""
    pytest.importorskip("relay_bp", reason="needs `uv sync --group reference`")
    g = BB18 / golden
    manifest = json.loads((g / "manifest.json").read_text())
    plan = build_plan(bb18, WindowSpec(W, C, W, "exact", "commit_anyway", 0))
    detectors = np.load(g / "detectors.npy")
    inner = InnerSpec("relay", relay=_relay_params(manifest), gammas=FixedGammas(np.load(g / "gammas.npy")))
    outcomes = decode_shots(plan, detectors, inner)
    decoding, iterations = np.load(g / "decoding.npy"), np.load(g / "iterations.npy")
    weight, legs_ptr = np.load(g / "weight.npy"), np.load(g / "legs_ptr.npy")
    for s, o in enumerate(outcomes):
        (w,) = o.windows
        e = np.zeros(bb18.n, dtype=np.uint8)
        e[w.committed.astype(np.int64)] = 1
        assert np.array_equal(e, decoding[s]), f"shot {s}: e"
        assert w.iterations == iterations[s], f"shot {s}: iterations"
        assert w.legs == legs_ptr[s + 1] - legs_ptr[s], f"shot {s}: legs"
        assert np.float64(w.weight).tobytes() == weight[s].tobytes(), f"shot {s}: weight"
        assert o.weight == (w.weight if w.converged else o.weight)


GOLDENS = sorted(p for fx in (BB18, BB18_R9) for p in fx.glob("window_*") if (p / "manifest.json").exists())


@pytest.mark.parametrize("golden", GOLDENS, ids=[f"{p.parent.name}/{p.name}" for p in GOLDENS])
def test_window_golden_consistent(golden):
    from rtd.window_ref_golden import golden_problem, load_golden

    manifest, arrays = load_golden(golden)
    problem = golden_problem(manifest, REPO)
    spec = WindowSpec(**{k: v for k, v in manifest["window"].items() if k != "mode"})
    plan = build_plan(problem, spec)
    assert manifest["num_shapes"] == len(plan.shapes) and manifest["num_positions"] == plan.num_positions
    assert manifest["plan"] == json.loads(json.dumps(plan.json()))
    assert golden.name == f"window_{spec.width}_{spec.commit}_{spec.boundary}_{spec.on_failure}"
    S, K, k = manifest["count"], plan.num_positions, problem.k
    for name, dtype, shape in [
        ("win_iterations", np.uint32, (S, K)), ("win_legs", np.uint32, (S, K)), ("win_attempts", np.uint8, (S, K)),
        ("win_converged", np.uint8, (S, K)), ("win_weight", np.float64, (S, K)), ("win_committed_weight", np.float64, (S, K)),
        ("win_unexplained", np.uint32, (S, K)), ("win_flagged", np.uint8, (S, K)), ("win_virtual", np.uint32, (S, K)),
        ("commit_ptr", np.uint64, (S * K + 1,)), ("predicted_observables", np.uint8, (S, k)), ("logical_failure", np.uint8, (S,)),
        ("flagged", np.uint8, (S,)), ("detectors", np.uint8, (S, problem.m)), ("observables", np.uint8, (S, k)),
    ]:  # fmt: skip
        assert arrays[name].dtype == dtype and arrays[name].shape == shape, name
    for i, shape in enumerate(plan.shapes):
        table = np.load(golden / "gammas" / f"shape_{i}.npy")
        assert table.dtype == np.float64 and table.shape == (manifest["gamma"]["rows"], shape.columns)
        g = manifest["gamma"]
        assert table.tobytes() == np.random.default_rng([g["seed"], i]).uniform(g["low"], g["high"], table.shape).tobytes()
    spec_json = json.loads((golden / "spec.json").read_text())
    assert spec_json["version"] == 2 and spec_json["window"] == manifest["window"]
    assert spec_json["gamma_source"] == {"type": "explicit_shapes", "directory": "gammas"}

    attempts, conv = arrays["win_attempts"], arrays["win_converged"].astype(bool)
    decoded = attempts > 0
    assert np.array_equal(np.isinf(arrays["win_weight"]), decoded & ~conv)
    assert np.all(arrays["flagged"] == arrays["win_flagged"].any(axis=1))
    if spec.on_failure == "commit_anyway":
        assert not arrays["flagged"].any()
    else:
        assert np.array_equal(arrays["win_flagged"].astype(bool), decoded & ~conv)
    assert np.array_equal(arrays["logical_failure"], (arrays["predicted_observables"] != arrays["observables"]).any(axis=1).astype(np.uint8))
    ptr, faults = arrays["commit_ptr"].astype(np.int64), arrays["commit_faults"].astype(np.int64)
    assert ptr[0] == 0 and ptr[-1] == faults.size and np.all(np.diff(ptr) >= 0)
    assert np.all((faults >= 0) & (faults < problem.n))
    cls_ = problem.column_class()
    hc = problem.h.tocsc()
    for s in range(S):
        c = faults[ptr[s * K] : ptr[(s + 1) * K]]
        assert np.unique(c).size == c.size, f"shot {s}: a column committed twice"
        if spec.boundary != "exact":
            continue
        frame = 0
        for j in c.tolist():
            frame ^= int(cls_[j])
        assert [(frame >> o) & 1 for o in range(k)] == arrays["predicted_observables"][s].tolist()
        e = np.zeros(problem.n, dtype=np.int64)
        e[c] = 1
        residual = (np.asarray(hc @ e).ravel() % 2).astype(np.uint8) ^ arrays["detectors"][s]
        assert int(residual.sum()) == int(arrays["win_unexplained"][s].sum())
        if conv[s][decoded[s]].all():
            assert not residual.any(), f"shot {s}: every window converged but H c != sigma"
    if golden.parent == BB18:
        # The R = 3 goldens decode exactly the syndromes of the whole-shot goldens.
        assert np.array_equal(arrays["detectors"], np.load(BB18 / "relay_f32" / "detectors.npy"))


@pytest.mark.parametrize("golden", GOLDENS, ids=[f"{p.parent.name}/{p.name}" for p in GOLDENS])
def test_window_golden_reproduces(golden):
    """Re-decoding a few shots gives the golden's arrays bit for bit: the first two, and up to two
    of each kind with a non-converged, deferred or skipped window."""
    pytest.importorskip("relay_bp", reason="needs `uv sync --group reference`")
    from rtd.window_ref import outcome_arrays
    from rtd.window_ref_golden import GOLDEN_ARRAYS, golden_problem, load_golden
    from rtd.window_ref_inner import SeededShapeGammas

    manifest, arrays = load_golden(golden)
    problem = golden_problem(manifest, REPO)
    spec = WindowSpec(**{k: v for k, v in manifest["window"].items() if k != "mode"})
    plan = build_plan(problem, spec)
    d, g, cfg = manifest["decoder"], manifest["gamma"], manifest["config"]
    params = _params_from_config(d["float"], cfg, (g["low"], g["high"]))
    attempts, converged = arrays["win_attempts"], arrays["win_converged"]
    picks = {0, 1}
    for kind in ((converged == 0).any(axis=1), (attempts > 1).any(axis=1), (attempts == 0).any(axis=1)):
        picks |= set(np.flatnonzero(kind)[:2].tolist())
    shots = sorted(picks)
    inner = InnerSpec("relay", relay=params, gammas=SeededShapeGammas(g["rows"], g["seed"], g["low"], g["high"]))
    for s in shots:
        (outcome,) = decode_shots(plan, arrays["detectors"][s : s + 1], inner, manifest["first"] + s)
        fresh = outcome_arrays([outcome], plan, problem.k, arrays["observables"][s : s + 1])
        K = plan.num_positions
        ptr = arrays["commit_ptr"].astype(np.int64)
        for name in GOLDEN_ARRAYS:
            if name == "commit_ptr":
                want = arrays[name][s * K : (s + 1) * K + 1] - arrays[name][s * K]
            elif name == "commit_faults":
                want = arrays[name][ptr[s * K] : ptr[(s + 1) * K]]
            else:
                want = arrays[name][s : s + 1]
            assert fresh[name].dtype == want.dtype and fresh[name].tobytes() == want.tobytes(), f"shot {s}: {name}"


def test_decode_cli_on_the_toy(tmp_path, toy):
    """decode with the brute-force inner decoder on shots sampled from the toy circuit."""
    import stim

    from rtd.window_ref_toy import toy_circuit

    circuit = toy_circuit()
    dets, obs = circuit.compile_detector_sampler(seed=5).sample(40, separate_observables=True)
    shots = tmp_path / "shots"
    shots.mkdir()
    np.save(shots / "detectors.npy", dets.astype(np.uint8))
    np.save(shots / "observables.npy", obs.astype(np.uint8))
    sha = {name: hashlib.sha256((shots / name).read_bytes()).hexdigest() for name in ("detectors.npy", "observables.npy")}
    sha["circuit.stim"] = toy.manifest["source_circuit"]["sha256"]
    (shots / "manifest.json").write_text(json.dumps({"sha256": sha, "rounds": 6}))
    out = tmp_path / "out"
    args = ["decode", "--artifact", str(TOY), "--shots", str(shots), "--count", "40", "--width", "4", "--commit", "2", "--converge", "4"]
    args += ["--boundary", "exact", "--on-failure", "flag", "--max-deferrals", "0", "--inner", "brute", "--save-commits", "--out", str(out)]
    assert main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["summary"]["shots"] == 40 and run["summary"]["windows_not_converged"] == 0
    assert np.load(out / "win_iterations.npy").shape == (40, 3)
    assert np.load(out / "commit_ptr.npy").shape == (121,)
    assert stim.__version__


# ---------------------------------------------------------------------------------------------
# Exact boundary without the truncation merge


def with_duplicates(seed: int, rounds: int = 7, per_round: int = 4, n: int = 60) -> GlobalProblem:
    """random_problem plus an exact copy of every fifth column with a prior of its own, columns
    shuffled: identical (H, A) columns with different priors, as a DEM folded over a repeated
    block keeps them."""
    base = random_problem(seed, rounds=rounds, per_round=per_round, n=n)
    rng = np.random.default_rng(seed + 1000)
    copies = np.arange(0, base.n, 5)
    h = sp.hstack([base.h_csc, base.h_csc[:, copies]], format="csc")
    a = sp.hstack([base.a, base.a[:, copies]], format="csc")
    priors = np.concatenate([base.priors, rng.uniform(0.001, 0.2, copies.size)])
    order = rng.permutation(h.shape[1])
    return GlobalProblem.from_arrays(h[:, order], a[:, order], priors[order], base.det_round)


@pytest.mark.parametrize("seed", range(6))
def test_no_merge_plan_is_every_column_of_the_window(seed):
    """Every local column is one global column of J_k, in ascending order, with its own prior;
    the commit sets are the merged plan's and Lee et al.'s detector rule's."""
    problem = with_duplicates(seed)
    M = time_structure(problem).per_round
    cls_ = problem.column_class()
    merges = 0
    for W, C in [(2, 1), (3, 1), (4, 2), (5, 3)]:
        spec = WindowSpec(W, C, W, "exact", "commit_anyway", 0)
        plan = build_plan(problem, spec, merge_truncated=False)
        merged = build_plan(problem, spec)
        assert not plan.merge_truncated and merged.merge_truncated
        assert [s.key() for s in merged.shapes] == [s.key() for s in build_plan(problem, spec, merge_truncated=True).shapes]
        assert all(s.merged_columns == 0 for s in plan.shapes)
        merges += sum(s.merged_columns for s in merged.shapes)
        rule = detector_rule_sets(problem, W, C)
        for p, q, (cols, commits) in zip(plan.placements, merged.placements, rule, strict=True):
            shape, mshape = plan.shapes[p.shape], merged.shapes[q.shape]
            assert np.array_equal(p.members_ptr, np.arange(p.columns.size + 1, dtype=np.uint32))
            assert np.array_equal(p.members, p.columns)
            assert np.array_equal(p.columns.astype(np.int64), cols)
            assert np.array_equal(p.columns[shape.commit == 1].astype(np.int64), commits)
            assert np.array_equal(q.columns[mshape.commit == 1], p.columns[shape.commit == 1])
            assert shape.priors.tobytes() == problem.priors[cols].tobytes()
            assert np.array_equal(shape.cls, np.where(shape.commit == 1, cls_[cols], 0).astype(np.uint64))
            lo = (p.first_round - 1) * M
            assert (shape.h != problem.h[lo : lo + shape.rows][:, cols]).nnz == 0
            assert shape.columns >= mshape.columns
    # The comparison means something only if the default plan merged columns somewhere.
    assert merges > 0


def test_no_merge_keeps_duplicate_columns_apart():
    """A duplicated pair truncated at a window's last round is one column with the folded prior
    in the merged plan and two columns with their own priors without the merge."""
    problem = with_duplicates(3)
    ts = time_structure(problem)
    spec = WindowSpec(3, 1, 3, "exact", "commit_anyway", 0)
    plan, merged = build_plan(problem, spec, merge_truncated=False), build_plan(problem, spec)
    hc = problem.h_csc
    supports = [hc.indices[hc.indptr[j] : hc.indptr[j + 1]].tobytes() for j in range(problem.n)]
    found = 0
    for p, q in zip(plan.placements, merged.placements, strict=True):
        if p.final:
            continue
        last = p.first_round + spec.width - 1
        for c in range(q.columns.size):
            mem = _members(q, c)
            if mem.size == 2 and supports[mem[0]] == supports[mem[1]] and ts.earliest[mem[0]] == last:
                found += 1
                a, b = problem.priors[mem].tolist()
                assert merged.shapes[q.shape].priors[c] == a * (1.0 - b) + b * (1.0 - a)
                at = np.searchsorted(p.columns, mem.astype(np.uint32))
                assert np.array_equal(p.columns[at], mem.astype(np.uint32))
                assert plan.shapes[p.shape].priors[at].tolist() == [a, b]
    assert found > 0


def test_no_merge_needs_the_exact_boundary(bb18_r9):
    with pytest.raises(PlanError) as err:
        build_plan(bb18_r9, WindowSpec(4, 2, 4, "uniform", "commit_anyway", 0), merge_truncated=False)
    assert err.value.code == "invalid_spec"


def _lee_sliding(problem: GlobalProblem, syndrome: np.ndarray, width: int, commit: int, inner) -> np.ndarray:
    """Lee, English and Bartlett's sliding-window loop restated from their code: window w holds
    the detectors of times [wC, wC + W - 1] and the faults touching them that are not committed
    yet (ascending, global priors, never merged); it commits the faults touching times
    [wC, wC + C - 1], the final window (its last time reaches the readout) everything; committed
    faults update the syndrome on every row. Returns the committed correction's support."""
    times = problem.det_round - 1
    last_time = int(times.max())
    h = problem.h
    committed = np.zeros(problem.n, dtype=bool)
    pred = np.zeros(problem.n, dtype=bool)
    syn = syndrome.astype(np.uint8).copy()
    w = 0
    while True:
        start, end = w * commit, w * commit + width - 1
        final = end >= last_time
        rows = np.flatnonzero((times >= start) & (times <= end))
        faults = (h[rows].getnnz(axis=0) > 0) & ~committed
        cols = np.flatnonzero(faults)
        ctx = InnerContext(0, w, 0, -1, 0, np.ones(rows.size, dtype=np.uint8), None)
        res = inner.decode(sp.csr_matrix(h[rows][:, cols]), problem.priors[cols], syn[rows], ctx)
        e = np.zeros(problem.n, dtype=bool)
        e[cols[res.support]] = True
        if final:
            commit_mask = faults
        else:
            commit_rows = np.flatnonzero((times >= start) & (times < start + commit))
            commit_mask = (h[commit_rows].getnnz(axis=0) > 0) & ~committed
        to_commit = e & commit_mask
        committed |= commit_mask
        syn ^= (np.asarray(h @ to_commit.astype(np.int64)).ravel() % 2).astype(np.uint8)
        pred ^= to_commit
        if final:
            return np.flatnonzero(pred)
        w += 1


@pytest.mark.parametrize("seed", range(4))
def test_no_merge_decode_is_lee_sliding_window(seed):
    """With the same exact-minimum-weight inner decoder, the no-merge stream decode commits the
    same correction as Lee et al.'s loop on every shot, duplicates included."""
    problem = with_duplicates(seed, rounds=6, n=30)
    rng = np.random.default_rng(seed + 7)
    inner = BruteForceInner()
    for W, C in [(2, 1), (3, 1), (4, 2)]:
        plan = build_plan(problem, WindowSpec(W, C, W, "exact", "commit_anyway", 0), merge_truncated=False)
        syndromes, outcomes = [], []
        for shot in range(12):
            e = rng.random(problem.n) < problem.priors
            s = (np.asarray(problem.h @ e.astype(np.int64)).ravel() % 2).astype(np.uint8)
            out = decode_shot(plan, s, shot, inner)
            ours = np.sort(np.concatenate([w.committed for w in out.windows]).astype(np.int64))
            assert np.array_equal(ours, _lee_sliding(problem, s, W, C, inner)), (W, C, shot)
            syndromes.append(s)
            outcomes.append(out)
        check_outcomes(problem, plan, np.array(syndromes), outcomes)


def test_no_merge_cli(tmp_path, toy):
    """dump-plan and decode take --no-merge-truncated; the toy's (3, 1) plan merges a truncated
    pair by default and keeps it apart with the flag."""
    import stim

    from rtd.window_ref_toy import toy_circuit

    base = ["--artifact", str(TOY), "--width", "3", "--commit", "1", "--converge", "3", "--boundary", "exact"]
    base += ["--on-failure", "commit_anyway", "--max-deferrals", "0"]
    assert main(["dump-plan", *base, "--out", str(tmp_path / "merged")]) == 0
    assert main(["dump-plan", *base, "--no-merge-truncated", "--out", str(tmp_path / "plain")]) == 0
    merged = json.loads((tmp_path / "merged" / "plan.json").read_text())
    plain = json.loads((tmp_path / "plain" / "plan.json").read_text())
    assert sum(s["merged_columns"] for s in merged["shapes"]) > 0
    assert all(s["merged_columns"] == 0 for s in plain["shapes"])
    assert plain["spec"] == merged["spec"] and plain["placements"] == merged["placements"]
    assert compare_plan_dirs(tmp_path / "merged", tmp_path / "plain")
    assert not compare_plan_dirs(tmp_path / "plain", _dump(build_plan(toy, WindowSpec(3, 1, 3, "exact", "commit_anyway", 0), merge_truncated=False), tmp_path / "api"))
    dets, obs = toy_circuit().compile_detector_sampler(seed=11).sample(20, separate_observables=True)
    shots = tmp_path / "shots"
    shots.mkdir()
    np.save(shots / "detectors.npy", dets.astype(np.uint8))
    np.save(shots / "observables.npy", obs.astype(np.uint8))
    sha = {name: hashlib.sha256((shots / name).read_bytes()).hexdigest() for name in ("detectors.npy", "observables.npy")}
    sha["circuit.stim"] = toy.manifest["source_circuit"]["sha256"]
    (shots / "manifest.json").write_text(json.dumps({"sha256": sha, "rounds": 6}))
    out = tmp_path / "out"
    args = ["decode", *base, "--shots", str(shots), "--count", "20", "--inner", "brute", "--no-merge-truncated", "--out", str(out)]
    assert main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["merge_truncated"] is False and run["summary"]["shots"] == 20
    assert stim.__version__


def _dump(plan, out: Path) -> Path:
    dump_plan(plan, out)
    return out
