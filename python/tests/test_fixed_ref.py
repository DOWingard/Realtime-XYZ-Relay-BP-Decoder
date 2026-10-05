"""The fixed-point emulator: format arithmetic (same values as the C++ policy tests), the
vectorised relay decoder against a loop-by-loop naive decoder, windowing, goldens and test
vectors."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rtd.fixed_golden import load_golden, load_vectors, main as golden_main
from rtd.fixed_ref import AlphaRule, FixedFormat, FixedInnerSpec, FixedRelay, RelayParams, UniformGammas, round_half_away, table_rows
from rtd.window_ref import WindowSpec, build_plan, check_outcomes, decode_shot, decode_shots, load_problem, load_shots_dir, outcome_arrays
from rtd.window_ref_inner import SeededShapeGammas

REPO = Path(__file__).resolve().parents[2]
BB18_R9 = REPO / "test" / "fixtures" / "bb18_choi_r9"
INT4 = FixedFormat.parse("int4.2.8")
INT6 = FixedFormat.parse("int6.2.8")


# ---- the format ------------------------------------------------------------------------------


def test_format_names_and_limits():
    assert INT4.name == "int4.2.8" and INT4.max_magnitude == 15 and INT4.memory_shift == 3
    assert INT6.max_magnitude == 63
    assert FixedFormat.parse("int7.16.32").name == "int7.16.32"
    for bad in ("int8.2.8", "int4.2.6", "float", "int4.0.8", "int4.2.64"):
        with pytest.raises(ValueError):
            FixedFormat.parse(bad)


def test_rounding_is_half_away_from_zero_without_double_rounding():
    assert round_half_away(np.nextafter(0.5, 0.0)) == 0.0
    assert round_half_away(0.5) == 1.0 and round_half_away(-0.5) == -1.0
    assert round_half_away(2.4999) == 2.0 and round_half_away(-2.5) == -3.0


def test_quantised_priors_match_the_cpp_policy():
    llr = np.array([2.25, 2.2, -2.25, 0.0, 7.24, 7.25, 40.0, -40.0, math.inf, math.log((1 - 0.003) / 0.003)])
    assert INT4.quantise_llr(llr).tolist() == [5, 4, -5, 0, 14, 15, 15, -15, 15, 12]
    assert INT6.quantise_llr(np.array([40.0]))[0] == 63


def test_memory_strengths_become_beta_times_m():
    gammas = np.array([0.125, -0.24, 0.66, 0.0, 1.0, -1.0, -3.0, 2.0, 0.0625])
    assert INT4.quantise_gamma(gammas).tolist() == [7, 10, 3, 8, 0, 16, 16, 0, 8]


def test_shift_and_add_multiplier_reproduces_the_worked_example():
    assert INT4.product(np.array([15, 8, 4, 2, 1, -15]), np.array(7)).tolist() == [11, 7, 3, 1, 0, -11]
    x = np.arange(-15, 16)
    assert np.array_equal(INT4.product(x, np.array(8)), x)
    assert np.array_equal(INT4.product(x, np.array(16)), 2 * x)
    assert not INT4.product(x, np.array(0)).any()
    for beta in range(17):
        exact = np.abs(x) * beta // 8
        assert np.all(np.abs(INT4.product(x, np.array(beta))) <= exact)


def test_bias_matches_the_cpp_policy():
    cases = [(12, -9, 8, 12), (12, -9, 0, -9), (12, -40, 0, -15), (12, -6, 7, 8), (12, 3, 16, 15), (3, 12, 16, -6)]
    for lam, marginal, beta, expected in cases:
        assert int(INT4.bias(np.array(lam), np.array(marginal), np.array(beta))) == expected
    for beta in range(17):
        assert int(INT4.bias(np.array(12), np.array(12), np.array(beta))) == 12


def test_alpha_rules():
    adaptive = AlphaRule("adaptive")
    assert [adaptive.shift(t) for t in range(3)] == [1, 2, 3]
    assert adaptive.shift(29) == 30 and adaptive.shift(30) is None
    assert AlphaRule("constant", value=1.0).shift(0) is None
    assert AlphaRule("constant", value=0.75).shift(5) == 2
    assert AlphaRule("constant", value=0.5).shift(0) == 1
    for ok in (AlphaRule("constant", value=1.0), AlphaRule("constant", value=0.875), adaptive):
        ok.check_fixed()
    for bad in (AlphaRule("constant", value=0.8), AlphaRule("constant", value=0.25), AlphaRule("adaptive", scaling=2.0)):
        with pytest.raises(ValueError):
            bad.check_fixed()


def test_uniform_gammas_are_reproducible_and_in_range():
    g = UniformGammas(5, -0.24, 0.66)
    a = g.row(3, 1, 500)
    assert np.array_equal(a, g.row(3, 1, 500))
    assert not np.array_equal(a, g.row(3, 2, 500)) and not np.array_equal(a, g.row(4, 1, 500))
    assert a.min() >= -0.24 and a.max() < 0.66
    assert abs(a.mean() - 0.21) < 0.05


# ---- the relay decoder against a naive one -------------------------------------------------


def naive_decode(fmt: FixedFormat, h: np.ndarray, priors: np.ndarray, syndrome: np.ndarray, params: RelayParams, gammas, cap: int | None = None):
    """Relay-BP in intN.S.M edge by edge, with Python integers; shares nothing with FixedRelay
    except the quantisation of the double inputs."""
    m, n = h.shape
    rows_of = [[i for i in range(m) if h[i, j]] for j in range(n)]
    cols_of = [[j for j in range(n) if h[i, j]] for i in range(m)]
    top = fmt.max_magnitude
    lam = [int(v) for v in fmt.quantise_llr(np.array([math.log((1 - p) / p) for p in priors]))]

    def sat(x: int) -> int:
        return max(-top, min(top, x))

    def times(x: int, beta: int) -> int:
        total = sum((beta << b) >> fmt.memory_shift for b in range(fmt.bits) if (abs(x) >> b) & 1)
        return -total if x < 0 else total

    marginal = list(lam)

    def leg(max_iter: int, beta: list[int], memory: bool):
        nonlocal marginal
        nu = {(i, j): lam[j] for i in range(m) for j in cols_of[i]}
        hard = [0] * n
        for t in range(max_iter):
            k = params.alpha.shift(t)
            mu = {}
            for i in range(m):
                for j in cols_of[i]:
                    others = [nu[(i, q)] for q in cols_of[i] if q != j]
                    least = min([abs(x) for x in others] + [top])
                    negative = bool(syndrome[i]) ^ (sum(x < 0 for x in others) % 2 == 1)
                    scaled = least if k is None else least - (least >> k)
                    mu[(i, j)] = -scaled if negative else scaled
            for j in range(n):
                bias = sat(times(lam[j], beta[j]) + sat(marginal[j]) - times(sat(marginal[j]), beta[j])) if memory else lam[j]
                total = bias + sum(mu[(i, j)] for i in rows_of[j])
                for i in rows_of[j]:
                    nu[(i, j)] = sat(total - mu[(i, j)])
                marginal[j] = total
                hard[j] = int(total <= 0)
            if all((sum(hard[j] for j in cols_of[i]) % 2) == syndrome[i] for i in range(m)):
                return t + 1, True, list(hard)
        return max_iter, False, list(hard)

    def weight(hard):
        return sum(lam[j] for j in range(n) if hard[j]) / fmt.scale

    memory = params.gamma0 is not None
    beta0 = [int(fmt.quantise_gamma(params.gamma0 if memory else 0.0))] * n
    budget0 = params.pre_iter if cap is None else min(params.pre_iter, cap)
    it, conv, hard = leg(budget0, beta0, memory)
    total, legs, best, best_w, nconv, best_hard = it, 1, (0 if conv else None), (weight(hard) if conv else math.inf), int(conv), hard
    cap_hit = False
    done = conv and (params.stopping == "after_leg0" or (params.stopping == "after_n_converged" and nconv >= params.stop_count))
    if not conv and budget0 < params.pre_iter:
        cap_hit = True
    elif not done:
        for r in range(1, params.num_sets + 1):
            budget = params.set_max_iter
            if cap is not None:
                if cap - total == 0:
                    cap_hit = True
                    break
                budget = min(budget, cap - total)
            beta = [int(b) for b in fmt.quantise_gamma(gammas(0, r))]
            it, conv, hard = leg(budget, beta, True)
            total += it
            legs += 1
            if not conv:
                if budget < params.set_max_iter:
                    cap_hit = True
                    break
                continue
            nconv += 1
            if weight(hard) < best_w:
                best_w, best, best_hard = weight(hard), r, hard
            if params.stopping == "after_n_converged" and nconv >= params.stop_count:
                break
    return {"iterations": total, "legs": legs, "best_leg": best, "weight": best_w, "support": [j for j in range(n) if best_hard[j]], "cap_hit": cap_hit}


@pytest.mark.parametrize("fmt_name", ["int4.2.8", "int6.2.8"])
@pytest.mark.parametrize("alpha", [AlphaRule("adaptive"), AlphaRule("constant", value=1.0), AlphaRule("constant", value=0.75)])
def test_vectorised_decoder_matches_the_naive_decoder(fmt_name, alpha):
    fmt = FixedFormat.parse(fmt_name)
    rng = np.random.default_rng(7)
    checked = converged = 0
    for trial in range(6):
        m, n = 8, 16
        h = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            h[rng.choice(m, size=rng.integers(1, 4), replace=False), j] = 1
        priors = rng.uniform(0.001, 0.15, n)
        table = rng.uniform(-0.24, 0.66, (3, n))
        params = RelayParams(0.125, 8, 5, 6, ["after_n_converged", "all_legs", "after_leg0"][trial % 3], 2, alpha)
        relay = FixedRelay(fmt, sp.csr_matrix(h), priors, params)
        for shot in range(5):
            error = (rng.uniform(size=n) < priors).astype(np.uint8)
            syndrome = (h.astype(np.int64) @ error % 2).astype(np.uint8)
            for cap in (None, 11):
                got = relay.decode(syndrome, table_rows(table), 0, None, cap)
                want = naive_decode(fmt, h, priors, syndrome, params, table_rows(table), cap)
                assert got.iterations == want["iterations"] and got.legs == want["legs"]
                assert got.best_leg == want["best_leg"] and got.cap_hit == want["cap_hit"]
                assert got.weight == want["weight"] or (math.isinf(got.weight) and math.isinf(want["weight"]))
                assert got.support.tolist() == want["support"]
                checked += 1
                converged += got.success
    assert checked == 60 and 0 < converged < checked


def test_convergence_mask_and_cap_zero():
    h = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    params = RelayParams(0.125, 10, 5, 0, "after_leg0", 1, AlphaRule("constant", value=1.0))
    relay = FixedRelay(INT4, h, np.array([0.01, 0.01, 0.01]), params)
    res = relay.decode(np.array([1, 0], dtype=np.uint8), None)
    assert res.success and res.support.tolist() == [0] and res.iterations == 1
    assert res.weight == int(INT4.quantise_llr(np.array([math.log(99)]))[0]) / 2
    none = relay.decode(np.array([1, 0], dtype=np.uint8), None, cap=0)
    assert none.cap_hit and not none.success and none.legs == 0 and none.support.size == 0
    masked = relay.decode(np.array([1, 1], dtype=np.uint8), None, converge=np.array([0, 1], dtype=np.uint8))
    assert masked.success


# ---- windows ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bb18():
    problem = load_problem(BB18_R9 / "artifact")
    syndromes, observables, _ = load_shots_dir(BB18_R9 / "shots", 0, 12, problem)
    return problem, syndromes, observables


PARAMS = RelayParams(0.125, 30, 20, 8, "after_n_converged", 3, AlphaRule("adaptive"))


def test_window_as_wide_as_the_shot_is_the_whole_shot_decode(bb18):
    problem, syndromes, _ = bb18
    rt = 10
    plan = build_plan(problem, WindowSpec(rt, 1, rt, "exact", "commit_anyway", 0, None))
    assert len(plan.shapes) == 1 and plan.num_positions == 1
    gammas = SeededShapeGammas(3, 17, -0.24, 0.66)
    inner = FixedInnerSpec(INT4, PARAMS, gammas).build()
    whole = FixedRelay(INT4, problem.h, problem.priors, PARAMS)
    table = gammas(0, problem.n)
    for s in range(4):
        outcome = decode_shot(plan, syndromes[s], s, inner)
        res = whole.decode(syndromes[s], table_rows(table))
        assert outcome.windows[0].iterations == res.iterations and outcome.windows[0].converged == res.success
        assert outcome.windows[0].committed.tolist() == res.support.tolist()
        assert outcome.windows[0].weight == res.weight


@pytest.mark.parametrize("spec", [WindowSpec(4, 2, 4, "exact", "defer", 2, None), WindowSpec(4, 2, 3, "uniform", "flag", 0, 150), WindowSpec(3, 1, 2, "exact", "commit_anyway", 0, 60)])
def test_windowed_decodes_are_consistent(bb18, spec):
    problem, syndromes, observables = bb18
    plan = build_plan(problem, spec)
    outcomes = decode_shots(plan, syndromes, FixedInnerSpec(INT4, PARAMS, SeededShapeGammas(3, 17, -0.24, 0.66)))
    check_outcomes(problem, plan, syndromes, outcomes)
    arrays = outcome_arrays(outcomes, plan, problem.k, observables)
    if spec.iteration_cap is not None:
        assert arrays["win_iterations"].max() <= spec.iteration_cap * (1 + spec.max_deferrals)
    assert arrays["win_attempts"].max() >= 1


# ---- goldens and test vectors ----------------------------------------------------------------


def test_golden_and_vectors_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    out = tmp_path / "golden"
    argv = [
        "golden", "--artifact", str(BB18_R9 / "artifact"), "--shots", str(BB18_R9 / "shots"), "--count", "6",
        "--width", "4", "--commit", "2", "--converge", "4", "--boundary", "uniform", "--on-failure", "defer", "--max-deferrals", "1",
        "--iteration-cap", "200", "--format", "int4.2.8", "--alpha", "adaptive", "--gamma0", "0.125", "--pre-iter", "30",
        "--set-max-iter", "20", "--num-sets", "8", "--stopping", "after_n_converged", "--stop-count", "3", "--gamma-rows", "3",
        "--gamma-seed", "17", "--gamma-interval", "-0.24", "0.66", "--out", str(out),
    ]  # fmt: skip
    assert golden_main(argv) == 0
    manifest, arrays = load_golden(out)
    assert manifest["decoder"]["arithmetic"] == "int4.2.8"
    spec = json.loads((out / "spec.json").read_text())
    assert spec["arithmetic"] == "int4.2.8" and "policy" not in spec
    assert arrays["win_iterations"].shape == (6, manifest["num_positions"])
    assert golden_main(["vectors", str(out), "--repo-root", str(REPO)]) == 0
    description, vectors = load_vectors(out)
    assert description["arithmetic"] == "int4.2.8" and description["shots"] == 6
    # The carry of window k equals the detectors of its first round XOR the rows every earlier
    # committed local column flips there.
    M, K = description["detectors_per_round"], description["num_positions"]
    problem = load_problem(BB18_R9 / "artifact")
    plan = build_plan(problem, WindowSpec(4, 2, 4, "uniform", "defer", 1, 200))
    for s in range(6):
        resid = vectors["detectors"][s].astype(np.uint8).copy()
        for k in range(K):
            a = int(vectors["win_attempts"][s, k])
            p = plan.placement(k, a - 1)
            lo = (p.first_round - 1) * M
            assert np.array_equal(vectors["win_carry"][s, k], resid[lo : lo + M])
            shape = plan.shapes[p.shape]
            i = s * K + k
            local = vectors["commit_local"][vectors["commit_local_ptr"][i] : vectors["commit_local_ptr"][i + 1]]
            support = vectors["solution_local"][vectors["solution_ptr"][i] : vectors["solution_ptr"][i + 1]]
            assert set(local.tolist()) == {int(j) for j in support if shape.commit[j]}
            for col in local.tolist():
                rows = shape.h_csc.indices[shape.h_csc.indptr[col] : shape.h_csc.indptr[col + 1]]
                rows = rows[lo + rows < resid.size]
                resid[lo + rows] ^= 1
        assert int(vectors["win_frame"][s, K - 1]) == sum(int(b) << o for o, b in enumerate(vectors["predicted_observables"][s]))
    assert vectors["shape0_beta"].dtype == np.uint8 and vectors["shape0_lambda"].dtype == np.int8
    assert vectors["shape0_beta"].max() <= 16 and np.abs(vectors["shape0_lambda"]).max() <= 15
