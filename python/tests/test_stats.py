import math

import numpy as np
import pytest
from scipy import optimize
from scipy.stats import binomtest

from rtd import stats


# ---------------------------------------------------------------------------------------------
# Rates


def test_wilson_matches_the_harness_bit_for_bit():
    # 8 failures in 100,000 shots, as written by rtd_decode's run.json for the Relay-5 run.
    estimate = stats.wilson(8, 100_000)
    assert estimate.value == 8e-05
    assert estimate.lo == 4.053852835999305e-05
    assert estimate.hi == 0.00015786843812503332


@pytest.mark.parametrize(("k", "n"), [(0, 10), (1, 10), (5, 10), (10, 10), (13, 1000), (8, 100_000)])
def test_wilson_matches_scipy(k, n):
    reference = binomtest(k, n).proportion_ci(method="wilson")
    estimate = stats.wilson(k, n)
    assert estimate.lo == pytest.approx(reference.low, rel=1e-12, abs=1e-15)
    assert estimate.hi == pytest.approx(reference.high, rel=1e-12, abs=1e-15)


def test_wilson_extremes_are_exact_and_symmetric():
    assert stats.wilson(0, 50).lo == 0.0
    assert stats.wilson(50, 50).hi == 1.0
    low, high = stats.wilson(7, 40), stats.wilson(33, 40)
    assert low.lo == pytest.approx(1.0 - high.hi, abs=1e-15)
    assert low.hi == pytest.approx(1.0 - high.lo, abs=1e-15)


@pytest.mark.parametrize(("k", "n"), [(0, 0), (5, 4), (-1, 10), (1.5, 10)])
def test_wilson_rejects_invalid_counts(k, n):
    with pytest.raises(ValueError):
        stats.wilson(k, n)


def test_block_per_cycle_known_value():
    # README: P_block = 8e-5 over 12 rounds is 6.7e-6 per cycle (Mueller et al.: (7 +- 1)e-6).
    assert stats.block_per_cycle(8e-5, 12) == pytest.approx(6.6669e-6, rel=1e-4)
    assert stats.block_per_cycle(0.0, 12) == 0.0
    assert stats.block_per_cycle(1.0, 12) == 1.0
    per_cycle = stats.wilson_block_per_cycle(8, 100_000, 12)
    assert per_cycle.lo == pytest.approx(3.4e-6, rel=0.01)
    assert per_cycle.hi == pytest.approx(1.3e-5, rel=0.02)


def test_block_per_cycle_is_vectorised_and_inverts():
    P = np.array([1e-6, 1e-3, 0.3])
    rate = stats.block_per_cycle(P, 24)
    assert isinstance(rate, np.ndarray)
    np.testing.assert_allclose(-np.expm1(24 * np.log1p(-rate)), P, rtol=1e-12)


def test_per_qubit_per_cycle():
    rate = stats.per_qubit_per_cycle(8e-5, 12, 12)
    assert rate == pytest.approx(8e-5 / (12 * 12), rel=1e-3)  # small-rate limit P / (k R)
    # Exact inversion: each qubit's per-cycle flip q composes to (1 - 2q)^R over R cycles.
    q = stats.per_qubit_per_cycle(0.2, 5, 3)
    fidelity = (1.0 - 2.0 * q) ** 5
    assert 1.0 - ((1.0 + fidelity) / 2.0) ** 3 == pytest.approx(0.2, rel=1e-12)


@pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
def test_rates_reject_non_probabilities(bad):
    with pytest.raises(ValueError):
        stats.block_per_cycle(bad, 12)
    with pytest.raises(ValueError):
        stats.per_qubit_per_cycle(bad, 12, 12)


def test_rates_reject_zero_rounds_and_unreachable_per_qubit_rate():
    with pytest.raises(ValueError):
        stats.block_per_cycle(0.1, 0)
    with pytest.raises(ValueError):
        stats.per_qubit_per_cycle(1.0 - 2.0 ** -3, 12, 3)


# ---------------------------------------------------------------------------------------------
# Paired comparison


def test_paired_counts():
    ref = np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8)
    test = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    counts = stats.paired_counts(ref, test)
    assert counts == (2, 1, 1, 2)
    assert counts.shots == 6


def test_paired_counts_rejects_mismatch_and_non_binary():
    with pytest.raises(ValueError):
        stats.paired_counts([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        stats.paired_counts([0, 2], [0, 1])


def test_paired_ratio_log_normal_known_value():
    # V = 50/30; Var(ln V) = 1/50 + 1/30 - 2*10/1500 = 0.04, so the interval is V e^(+-1.96*0.2).
    ratio = stats.paired_ratio(10, 20, 40)
    assert ratio.method == "log-normal"
    assert ratio.value == pytest.approx(5.0 / 3.0)
    assert ratio.log_se == pytest.approx(0.2)
    assert ratio.lo == pytest.approx(5.0 / 3.0 * math.exp(-stats.Z95 * 0.2))
    assert ratio.hi == pytest.approx(5.0 / 3.0 * math.exp(stats.Z95 * 0.2))


def test_paired_ratio_shared_failures_cancel():
    # Adding shared failures narrows the interval on ln V only through the denominators.
    wide, narrow = stats.paired_ratio(0, 20, 40), stats.paired_ratio(1000, 20, 40)
    assert narrow.hi / narrow.lo < wide.hi / wide.lo


def test_paired_ratio_degenerate_patterns():
    undefined = stats.paired_ratio(0, 0, 0)
    assert math.isnan(undefined.value) and undefined.lo == 0.0 and undefined.hi == math.inf
    assert undefined.method == "undefined"

    # Test arm never fails: V = 0, exact Clopper-Pearson upper bound on pi with 0 of D successes.
    zero = stats.paired_ratio(0, 5, 0)
    pi_hi = binomtest(0, 5).proportion_ci(method="exact").high
    assert zero.value == 0.0 and zero.lo == 0.0
    assert zero.hi == pytest.approx(pi_hi / (1.0 - pi_hi), rel=1e-9)
    assert zero.method == "exact-conditional"

    infinite = stats.paired_ratio(0, 0, 5)
    pi_lo = binomtest(5, 5).proportion_ci(method="exact").low
    assert infinite.value == math.inf and infinite.hi == math.inf
    assert infinite.lo == pytest.approx(pi_lo / (1.0 - pi_lo), rel=1e-9)

    shared = stats.paired_ratio(10, 0, 0)
    d_up = -math.log(0.025)
    assert shared.value == 1.0 and shared.method == "poisson-zero-discordant"
    assert shared.lo == pytest.approx(10.0 / (10.0 + d_up))
    assert shared.hi == pytest.approx((10.0 + d_up) / 10.0)


def test_paired_bootstrap_agrees_with_the_delta_method():
    rng = np.random.default_rng(11)
    shots = 400_000
    ref = rng.random(shots) < 4e-4
    test = ref & (rng.random(shots) < 0.8) | (rng.random(shots) < 2e-4)
    counts = stats.paired_counts(ref, test)
    analytic = stats.paired_ratio(*counts[:3])
    boot = stats.paired_bootstrap_ratio(ref, test, 4000, seed=5)
    assert boot.value == pytest.approx(analytic.value)
    assert boot.lo == pytest.approx(analytic.lo, rel=0.05)
    assert boot.hi == pytest.approx(analytic.hi, rel=0.05)


def test_paired_bootstrap_is_seeded():
    rng = np.random.default_rng(2)
    ref, test = rng.random(5000) < 0.02, rng.random(5000) < 0.03
    first = stats.paired_bootstrap_ratio(ref, test, 500, seed=9)
    assert stats.paired_bootstrap_ratio(ref, test, 500, seed=9) == first
    assert stats.paired_bootstrap_ratio(ref, test, 500, seed=10) != first


def test_paired_bootstrap_without_failures():
    boot = stats.paired_bootstrap_ratio(np.zeros(10, bool), np.zeros(10, bool), 50, seed=1)
    assert math.isnan(boot.value) and boot.lo == 0.0 and boot.hi == math.inf


# ---------------------------------------------------------------------------------------------
# Bootstrap of means and quantiles


def test_bootstrap_mean_normal_data():
    rng = np.random.default_rng(3)
    x = rng.normal(10.0, 2.0, size=20_000)
    estimate = stats.bootstrap_mean(x, 1000, seed=4)
    assert estimate.value == pytest.approx(np.mean(x))
    half = stats.Z95 * np.std(x) / math.sqrt(x.size)
    assert estimate.hi - estimate.lo == pytest.approx(2 * half, rel=0.1)
    assert estimate.lo < np.mean(x) < estimate.hi


def test_bootstrap_mean_integer_path_agrees_with_index_path():
    # Few distinct values take the multinomial path; the same data as distinct floats take the
    # index path. Both resample shots with replacement, so their intervals agree statistically.
    rng = np.random.default_rng(6)
    x = rng.geometric(0.01, size=30_000)
    jittered = x + rng.uniform(0, 1e-9, size=x.size)
    counts_path = stats.bootstrap_mean(x, 1000, seed=1)
    index_path = stats.bootstrap_mean(jittered, 1000, seed=1)
    assert counts_path.hi - counts_path.lo == pytest.approx(index_path.hi - index_path.lo, rel=0.15)
    assert stats.bootstrap_mean(x, 300, seed=1) == stats.bootstrap_mean(x, 300, seed=1)


def test_bootstrap_rejects_non_finite():
    with pytest.raises(ValueError):
        stats.bootstrap_mean([1.0, math.inf], 10, seed=1)
    with pytest.raises(ValueError):
        stats.bootstrap_mean([], 10, seed=1)


def test_nearest_rank_definition():
    x = np.arange(1, 101)
    assert stats.nearest_rank(x, 0.5) == 50
    assert stats.nearest_rank(x, 0.99) == 99
    assert stats.nearest_rank(x, 0.07) == 7  # float 0.07 * 100 = 7.000000000000001
    assert stats.nearest_rank(x, 1.0) == 100
    assert stats.nearest_rank(x, 0.0) == 1
    rng = np.random.default_rng(0)
    y = rng.integers(0, 1000, size=777)
    for q in (0.1, 0.5, 0.9, 0.99, 0.999):
        assert stats.nearest_rank(y, q) == np.quantile(y, q, method="inverted_cdf")


def test_bootstrap_quantiles_bracket_the_estimate():
    rng = np.random.default_rng(8)
    x = rng.geometric(0.004, size=50_000)
    levels = [0.5, 0.9, 0.99, 0.999]
    estimates = stats.bootstrap_quantiles(x, levels, 500, seed=2)
    for q, estimate in zip(levels, estimates):
        assert estimate.value == stats.nearest_rank(x, q)
        assert estimate.lo <= estimate.value <= estimate.hi
    assert stats.bootstrap_quantile(x, 0.9, 500, seed=2) == estimates[1]
    # The same levels through the index path (distinct floats) give nearby bounds.
    floats = stats.bootstrap_quantiles(x + rng.uniform(0, 1e-6, x.size), [0.5], 200, seed=2)[0]
    assert abs(floats.lo - estimates[0].lo) <= 0.02 * estimates[0].value


# ---------------------------------------------------------------------------------------------
# Anchors


def test_anchor_pass_readme_values():
    # Our 6.7e-6 (3.4e-6 - 1.3e-5) against Mueller's (7 +- 1)e-6 passes ...
    assert stats.anchor_pass(3.4e-6, 1.3e-5, (6e-6, 8e-6), 1.0)
    # ... our mean iterations 335.6 +- 1.2 against 330.8 +- 0.5 does not.
    assert not stats.anchor_pass(334.4, 336.8, (330.3, 331.3), 1.0)


def test_anchor_pass_factor_band():
    assert stats.anchor_pass(1.2e-5, 2e-5, 1e-5, 1.25)  # band [8e-6, 1.25e-5]
    assert not stats.anchor_pass(1.3e-5, 2e-5, 1e-5, 1.25)
    assert stats.anchor_pass(1.25e-5, 2e-5, 1e-5, 1.25)  # touching counts as overlap
    assert stats.anchor_pass(1e-6, 2.5e-4, (3e-4, 5e-4), 1.3)  # range with a reading factor
    assert stats.anchor_pass(1e-6, 2.5e-4, np.float64(2e-4), 1.3)


@pytest.mark.parametrize(("lo", "hi", "value", "factor"),
                         [(1.0, 2.0, 1.0, 0.9), (2.0, 1.0, 1.0, 1.3), (1.0, 2.0, -1.0, 1.3),
                          (1.0, 2.0, 1.0, math.inf), (math.nan, 2.0, 1.0, 1.3)])
def test_anchor_pass_rejects_invalid(lo, hi, value, factor):
    with pytest.raises(ValueError):
        stats.anchor_pass(lo, hi, value, factor)


# ---------------------------------------------------------------------------------------------
# Confidence signals


def test_risk_coverage_hand_example():
    result = stats.risk_coverage([5, 1, 3, 3, 0], [1, 0, 1, 0, 0], [0.0, 0.2, 0.4, 0.6, 1.0])
    np.testing.assert_array_equal(result.discarded, [0, 1, 2, 3, 5])
    np.testing.assert_array_equal(result.kept, [5, 4, 3, 2, 0])
    np.testing.assert_array_equal(result.failures, [2, 1, 0, 0, 0])
    np.testing.assert_allclose(result.rate[:4], [0.4, 0.25, 0.0, 0.0])
    assert math.isnan(result.rate[4]) and result.lo[4] == 0.0 and result.hi[4] == 1.0
    expected = stats.wilson(1, 4)
    assert (result.lo[1], result.hi[1]) == pytest.approx((expected.lo, expected.hi))
    np.testing.assert_allclose(result.discarded_fraction, [0.0, 0.2, 0.4, 0.6, 1.0])
    assert result.as_dict()["kept"] == [5, 4, 3, 2, 0]


def test_risk_coverage_ties_discard_the_lower_index_first():
    assert stats.risk_coverage([1.0, 1.0], [1, 0], [0.5]).rate[0] == 0.0
    assert stats.risk_coverage([1.0, 1.0], [0, 1], [0.5]).rate[0] == 1.0
    # Integer scores tie the same way.
    assert stats.risk_coverage(np.array([7, 7, 7], dtype=np.uint32), [1, 1, 0], [2 / 3]).rate[0] == 0.0


def test_risk_coverage_rounds_the_exact_fraction():
    # 0.29 * 100 is 28.999999999999996 in floating point; the exact decimal gives 29 shots.
    result = stats.risk_coverage(np.arange(100.0), np.zeros(100), [0.29, 0.005, 0.015])
    np.testing.assert_array_equal(result.discarded, [29, 1, 2])


def test_risk_coverage_rejects_nan_scores_and_bad_fractions():
    with pytest.raises(ValueError):
        stats.risk_coverage([1.0, math.nan], [0, 1], [0.1])
    with pytest.raises(ValueError):
        stats.risk_coverage([1.0, 2.0], [0, 1], [1.5])
    with pytest.raises(ValueError):
        stats.risk_coverage([1.0, 2.0], [0, 1, 1], [0.1])


def test_fit_logistic_slope_single_gap_closed_form():
    # All informative shots at D = 1 with 20 of 100 failing: 1/(1 + e^kappa) = 0.2, kappa = ln 4.
    # Shots at D = 0 carry no information about kappa.
    delta = np.r_[np.ones(100), np.zeros(10)]
    fail = np.r_[np.ones(20), np.zeros(80), np.ones(5), np.zeros(5)]
    fit = stats.fit_logistic_slope(delta, fail)
    assert fit.kappa == pytest.approx(math.log(4.0), rel=1e-10)
    assert fit.se == pytest.approx(1.0 / math.sqrt(100 * 0.2 * 0.8), rel=1e-9)

    # The likelihood interval maps to the likelihood interval of the proportion at D = 1.
    def drop(kappa):
        p = 1.0 / (1.0 + math.exp(kappa))
        return 20 * math.log(p) + 80 * math.log(1.0 - p)

    peak = drop(math.log(4.0))
    lo = optimize.brentq(lambda k: drop(k) - peak + stats.Z95**2 / 2, 0.0, math.log(4.0))
    hi = optimize.brentq(lambda k: drop(k) - peak + stats.Z95**2 / 2, math.log(4.0), 10.0)
    assert fit.lo == pytest.approx(lo, rel=1e-8)
    assert fit.hi == pytest.approx(hi, rel=1e-8)
    assert (fit.shots, fit.failures) == (110, 25)


def test_fit_logistic_slope_recovers_a_simulated_slope():
    rng = np.random.default_rng(21)
    delta = rng.exponential(3.0, size=100_000)
    fail = rng.random(delta.size) < 1.0 / (1.0 + np.exp(0.9 * delta))
    fit = stats.fit_logistic_slope(delta, fail)
    assert fit.lo < 0.9 < fit.hi
    assert fit.hi - fit.lo < 0.05


def test_fit_logistic_slope_negative_slope():
    rng = np.random.default_rng(4)
    delta = rng.normal(0, 2, size=20_000)
    fail = rng.random(delta.size) < 1.0 / (1.0 + np.exp(-0.5 * delta))
    fit = stats.fit_logistic_slope(delta, fail)
    assert fit.lo < -0.5 < fit.hi < 0


def test_fit_logistic_slope_separation_and_degenerate_data():
    separated = stats.fit_logistic_slope(np.r_[np.zeros(5), np.ones(50)],
                                         np.r_[np.ones(5), np.zeros(50)])
    assert separated.kappa == math.inf and separated.hi == math.inf
    # l(kappa) = -5 ln 2 - 50 softplus(-kappa) reaches its supremum minus z^2/2 at the lower bound.
    assert 50 * math.log1p(math.exp(-separated.lo)) == pytest.approx(stats.Z95**2 / 2, rel=1e-8)

    flat = stats.fit_logistic_slope(np.zeros(10), np.r_[np.ones(3), np.zeros(7)])
    assert math.isnan(flat.kappa) and flat.lo == -math.inf and flat.hi == math.inf

    reversed_ = stats.fit_logistic_slope(np.r_[np.ones(4), np.ones(6)], np.r_[np.ones(4), np.ones(6)])
    assert reversed_.kappa == -math.inf and reversed_.lo == -math.inf and math.isfinite(reversed_.hi)


def test_detector_density():
    detectors = np.array([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_allclose(stats.detector_density(detectors), [0.5, 0.0])
    with pytest.raises(ValueError):
        stats.detector_density(np.zeros(4))


def test_norm_fraction():
    ptr = [0, 2, 2, 5]
    weights = [3.0, 4.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(stats.norm_fraction(ptr, weights, 10.0, 2.0),
                               [0.5, 0.0, math.sqrt(3.0) / 10.0])
    np.testing.assert_allclose(stats.norm_fraction(ptr, weights, 10.0, 1.0), [0.7, 0.0, 0.3])
    np.testing.assert_allclose(stats.norm_fraction(ptr, weights, [10.0, 5.0, 2.0], math.inf),
                               [0.4, 0.0, 0.5])
    with pytest.raises(ValueError):
        stats.norm_fraction([0, 2], [1.0, 2.0], 0.0, 2.0)
    with pytest.raises(ValueError):
        stats.norm_fraction([0, 3], [1.0, 2.0], 1.0, 2.0)


def test_min_round_time():
    # 145 mean iterations at 24 ns, committing 8 rounds per window: 435 ns per round.
    assert stats.min_round_time(145.0, 24e-9, 8) == pytest.approx(4.35e-7)
    with pytest.raises(ValueError):
        stats.min_round_time(145.0, 24e-9, 0)
