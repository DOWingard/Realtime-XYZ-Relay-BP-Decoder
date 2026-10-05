"""Interval estimates, paired comparisons and confidence-signal statistics for decoding campaigns.

Every function is pure: numpy arrays or numbers in, floats (or small named tuples of floats) out.
Degenerate inputs are handled by an explicit, documented rule or rejected with ValueError; no
function divides by zero silently. Intervals are two-sided at the normal quantile z (default 95%).

Conventions
    P_block             probability that any of the k logical observables of a shot is wrong.
    block rate/cycle    p_L = 1 - (1 - P_block)^(1/R) over R noisy rounds (Bravyi et al. 2024,
                        Mueller et al. 2025).
    per qubit/cycle     p_L = [1 - (2 (1 - P_block)^(1/k) - 1)^(1/R)] / 2 (Gu et al. 2026); it
                        additionally assumes that the k logical qubits fail independently.
    nearest rank        the q-quantile of n values is the value of rank ceil(q n), clamped to
                        [1, n]; q is taken as the exact decimal it was written as, so 0.07 of 100
                        values is rank 7, not 8 as float rounding of 0.07 * 100 would give.

Return types
    Estimate(value, lo, hi)                     wilson, wilson_block_per_cycle, bootstrap_*
    PairedCounts(n11, n10, n01, n00)            paired_counts
    PairedRatio(value, lo, hi, log_se, method)  paired_ratio
    RiskCoverage (arrays per requested fraction) risk_coverage
    LogisticSlope(kappa, lo, hi, se, ...)       fit_logistic_slope
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special

logger = logging.getLogger("rtd.stats")

# Two-sided 95% normal quantile, the value the C++ harness uses.
Z95 = 1.959963984540054

# Bootstrap resamples are drawn in chunks so that one chunk holds at most this many values.
_CHUNK_ELEMENTS = 4_000_000


class Estimate(NamedTuple):
    """A point estimate with a two-sided interval."""

    value: float
    lo: float
    hi: float


class PairedCounts(NamedTuple):
    """Outcome counts of two decoders on the same shots (1 = failure)."""

    n11: int  # both fail
    n10: int  # only the reference fails
    n01: int  # only the test decoder fails
    n00: int  # neither fails

    @property
    def shots(self) -> int:
        return self.n11 + self.n10 + self.n01 + self.n00


class PairedRatio(NamedTuple):
    """V = (test failure rate) / (reference failure rate) on the same shots.

    method is "log-normal" for the delta-method interval on ln V, or names the rule used for a
    degenerate count pattern (see paired_ratio). log_se is NaN unless method is "log-normal".
    """

    value: float
    lo: float
    hi: float
    log_se: float
    method: str


class LogisticSlope(NamedTuple):
    """Maximum-likelihood kappa of P(fail | D) = 1 / (1 + exp(kappa D)) with a likelihood interval."""

    kappa: float
    lo: float
    hi: float
    se: float  # 1 / sqrt(Fisher information) at kappa; NaN when kappa is infinite or undefined
    log_likelihood: float  # at kappa (its supremum when kappa is infinite)
    shots: int
    failures: int


@dataclass(frozen=True, slots=True)
class RiskCoverage:
    """Kept-shot failure rate against the requested discarded fraction, one entry per fraction."""

    fraction: NDArray[np.float64]  # requested discarded fraction f
    discarded: NDArray[np.int64]  # shots discarded, nearest integer to f * shots
    kept: NDArray[np.int64]
    failures: NDArray[np.int64]  # failures among the kept shots
    rate: NDArray[np.float64]  # failures / kept; NaN when nothing is kept
    lo: NDArray[np.float64]  # Wilson interval of rate
    hi: NDArray[np.float64]
    shots: int

    @property
    def discarded_fraction(self) -> NDArray[np.float64]:
        """The fraction actually discarded, discarded / shots."""
        return self.discarded / self.shots

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready lists."""
        return {
            "shots": self.shots,
            "fraction": self.fraction.tolist(),
            "discarded": self.discarded.tolist(),
            "kept": self.kept.tolist(),
            "failures": self.failures.tolist(),
            "rate": self.rate.tolist(),
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
        }


# ---------------------------------------------------------------------------------------------
# Validation helpers


def _invalid(message: str, **context: Any) -> ValueError:
    logger.error(message, extra={"validation": context})
    return ValueError(message)


def _alpha(z: float) -> float:
    """Two-sided tail probability of the standard normal quantile z."""
    if not (math.isfinite(z) and z > 0.0):
        raise _invalid(f"z must be a positive finite number, got {z!r}", z=z)
    return math.erfc(z / math.sqrt(2.0))


def _count(value: Any, name: str) -> int:
    as_float = float(value)
    if not (math.isfinite(as_float) and as_float >= 0.0 and as_float == math.floor(as_float)):
        raise _invalid(f"{name} must be a non-negative integer, got {value!r}", **{name: value})
    return int(as_float)


def _binary(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise _invalid(f"{name} must be one-dimensional, got shape {array.shape}", name=name)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise _invalid(f"{name} must be boolean or numeric 0/1, got dtype {array.dtype}", name=name)
    if np.any((array != 0) & (array != 1)):
        raise _invalid(f"{name} must contain only 0 and 1", name=name)
    return array.astype(np.bool_)


def _finite_1d(values: ArrayLike, name: str) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise _invalid(f"{name} must be a non-empty one-dimensional array, got shape {array.shape}",
                       name=name)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise _invalid(f"{name} must be real numeric, got dtype {array.dtype}", name=name)
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise _invalid(f"{name} contains NaN or infinite values; filter them before calling",
                       name=name, nonfinite=int(np.count_nonzero(~np.isfinite(array))))
    return array


def _exact_decimal(q: float) -> Fraction:
    """The decimal number q was written as (0.07 -> 7/100), not its binary approximation."""
    return Fraction(repr(float(q)))


def _nearest_rank_index(q: float, n: int) -> int:
    """0-based index of the nearest-rank q-quantile among n sorted values."""
    if not (0.0 <= q <= 1.0):
        raise _invalid(f"quantile level must lie in [0, 1], got {q!r}", q=q)
    rank = math.ceil(_exact_decimal(q) * n)
    return min(max(rank, 1), n) - 1


# ---------------------------------------------------------------------------------------------
# Rates


def _wilson_bounds(k: NDArray[np.float64], n: NDArray[np.float64], z: float
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Same operation order as the C++ harness, so both produce identical doubles.
    phat = k / n
    z2 = z * z
    denominator = 1.0 + (z2 / n)
    centre = (phat + (z2 / (2.0 * n))) / denominator
    half = z * np.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))) / denominator
    # The bounds are exactly 0 and 1 at the extremes; the formula leaves rounding residue there.
    lo = np.where(k == 0, 0.0, np.maximum(0.0, centre - half))
    hi = np.where(k == n, 1.0, np.minimum(1.0, centre + half))
    return lo, hi


def wilson(k: int, n: int, z: float = Z95) -> Estimate:
    """Wilson score interval of the binomial proportion k / n (no continuity correction).

    Raises ValueError for n = 0 (no trials, no estimate) and for k > n.
    """
    kk, nn = _count(k, "k"), _count(n, "n")
    if nn == 0:
        raise _invalid("wilson: n = 0 trials, the proportion is undefined", k=kk, n=nn)
    if kk > nn:
        raise _invalid(f"wilson: k = {kk} exceeds n = {nn}", k=kk, n=nn)
    _alpha(z)  # validates z
    lo, hi = _wilson_bounds(np.float64(kk), np.float64(nn), z)
    return Estimate(kk / nn, float(lo), float(hi))


def _probability_array(P: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(P, dtype=np.float64)
    if not np.all((array >= 0.0) & (array <= 1.0)):  # also rejects NaN
        raise _invalid(f"{name} must lie in [0, 1]", **{name: P})
    return array


def _scalar_or_array(array: NDArray[np.float64]) -> float | NDArray[np.float64]:
    return float(array) if array.ndim == 0 else array


def block_per_cycle(P: ArrayLike, R: int) -> float | NDArray[np.float64]:
    """Block rate per cycle p_L = 1 - (1 - P)^(1/R), computed as -expm1(log1p(-P) / R).

    Monotone increasing in P, so it maps the bounds of an interval on P to bounds on p_L.
    Accepts a scalar (returns float) or an array (returns an array).
    """
    rounds = _count(R, "R")
    if rounds == 0:
        raise _invalid("block_per_cycle: R must be at least 1", R=R)
    probability = _probability_array(P, "P")
    with np.errstate(divide="ignore"):  # P = 1 gives log1p(-1) = -inf and the exact rate 1
        return _scalar_or_array(-np.expm1(np.log1p(-probability) / rounds))


def per_qubit_per_cycle(P: ArrayLike, R: int, k: int) -> float | NDArray[np.float64]:
    """Per logical qubit per cycle: [1 - (2 (1 - P)^(1/k) - 1)^(1/R)] / 2.

    2 (1 - P)^(1/k) - 1 is the per-qubit fidelity 1 - 2q over all R cycles when each of the k
    qubits fails independently like a binary symmetric channel. There is no solution for
    P >= 1 - 2^(-k); such inputs raise ValueError.
    """
    rounds, qubits = _count(R, "R"), _count(k, "k")
    if rounds == 0 or qubits == 0:
        raise _invalid("per_qubit_per_cycle: R and k must be at least 1", R=R, k=k)
    probability = _probability_array(P, "P")
    with np.errstate(divide="ignore"):  # P = 1 is rejected just below, through -inf
        fidelity_minus_one = 2.0 * np.expm1(np.log1p(-probability) / qubits)
    if np.any(fidelity_minus_one <= -1.0):
        raise _invalid(f"per_qubit_per_cycle: P >= 1 - 2^-{qubits} has no per-qubit rate", P=P, k=k)
    return _scalar_or_array(-np.expm1(np.log1p(fidelity_minus_one) / rounds) / 2.0)


def wilson_block_per_cycle(k: int, n: int, R: int, z: float = Z95) -> Estimate:
    """Block rate per cycle of k failures in n shots of R rounds, with the mapped Wilson interval."""
    block = wilson(k, n, z)
    return Estimate(*(float(block_per_cycle(v, R)) for v in block))


# ---------------------------------------------------------------------------------------------
# Paired comparison of two decoders on the same shots


def paired_counts(fail_ref: ArrayLike, fail_test: ArrayLike) -> PairedCounts:
    """Count (both, reference only, test only, neither) failures over the same shots."""
    ref, test = _binary(fail_ref, "fail_ref"), _binary(fail_test, "fail_test")
    if ref.shape != test.shape:
        raise _invalid(f"paired_counts: {ref.size} reference shots against {test.size} test shots",
                       ref=ref.size, test=test.size)
    n11 = int(np.count_nonzero(ref & test))
    n10 = int(np.count_nonzero(ref & ~test))
    n01 = int(np.count_nonzero(~ref & test))
    return PairedCounts(n11, n10, n01, int(ref.size) - n11 - n10 - n01)


def paired_ratio(n11: int, n10: int, n01: int, z: float = Z95) -> PairedRatio:
    """V = (n11 + n01) / (n11 + n10) with an interval, for rare failures on paired shots.

    Normal case (both arms have failures and at least one pair is discordant): the delta method on
    ln V under the multinomial model of the counts gives
        Var(ln V) = 1/(n11 + n01) + 1/(n11 + n10) - 2 n11 / ((n11 + n01)(n11 + n10))
                  = (n10 + n01) / ((n11 + n01)(n11 + n10)),
    and the interval is V exp(+-z sqrt(Var)). The shared failures n11 cancel from the numerator,
    which is why pairing needs far fewer shots than two independent runs.

    Degenerate patterns, where that formula gives 0/0, an infinite ratio or a zero-width interval:
    - no failures in either arm: V undefined -> (NaN, 0, inf), method "undefined".
    - n11 = 0 and one arm has no failures: then V = n01 / n10, and conditional on the
      D = n10 + n01 discordant shots n01 ~ Binomial(D, pi) with V = pi / (1 - pi). The exact
      Clopper-Pearson bound for 0 or D successes gives
        test never fails:      V = 0,   hi = pi_hi / (1 - pi_hi),  pi_hi = 1 - (alpha/2)^(1/D)
        reference never fails: V = inf, lo = pi_lo / (1 - pi_lo),  pi_lo = (alpha/2)^(1/D)
      method "exact-conditional".
    - failures only in both arms together (n10 = n01 = 0 < n11): V = 1, but no discordant shot
      was seen. The expected number of discordant shots is bounded by the exact Poisson limit for
      zero events, D_up = -ln(alpha/2), and the extreme allocations of D_up discordant shots give
      [n11 / (n11 + D_up), (n11 + D_up) / n11]; method "poisson-zero-discordant".
    alpha = erfc(z / sqrt(2)) is the two-sided level matching z.
    """
    a11, a10, a01 = _count(n11, "n11"), _count(n10, "n10"), _count(n01, "n01")
    alpha = _alpha(z)
    test_failures, ref_failures, discordant = a11 + a01, a11 + a10, a10 + a01
    if test_failures == 0 and ref_failures == 0:
        logger.warning("paired_ratio: no failures in either arm, the ratio is undefined",
                       extra={"n11": a11, "n10": a10, "n01": a01})
        return PairedRatio(math.nan, 0.0, math.inf, math.nan, "undefined")
    if discordant == 0:
        d_up = -math.log(alpha / 2.0)
        logger.warning("paired_ratio: no discordant shots; interval from the zero-event Poisson limit",
                       extra={"n11": a11, "discordant_upper": d_up})
        return PairedRatio(1.0, a11 / (a11 + d_up), (a11 + d_up) / a11, math.nan,
                           "poisson-zero-discordant")
    if test_failures == 0:
        pi_hi = 1.0 - (alpha / 2.0) ** (1.0 / discordant)
        logger.warning("paired_ratio: the test arm has no failures; exact conditional bound",
                       extra={"n10": a10})
        return PairedRatio(0.0, 0.0, pi_hi / (1.0 - pi_hi), math.nan, "exact-conditional")
    if ref_failures == 0:
        pi_lo = (alpha / 2.0) ** (1.0 / discordant)
        logger.warning("paired_ratio: the reference arm has no failures; exact conditional bound",
                       extra={"n01": a01})
        return PairedRatio(math.inf, pi_lo / (1.0 - pi_lo), math.inf, math.nan, "exact-conditional")
    value = test_failures / ref_failures
    log_se = math.sqrt(1.0 / test_failures + 1.0 / ref_failures
                       - 2.0 * a11 / (test_failures * ref_failures))
    return PairedRatio(value, value * math.exp(-z * log_se), value * math.exp(z * log_se), log_se,
                       "log-normal")


def _ratio_point(counts: PairedCounts) -> float:
    numerator, denominator = counts.n11 + counts.n01, counts.n11 + counts.n10
    if denominator == 0:
        return math.nan if numerator == 0 else math.inf
    return numerator / denominator


def paired_bootstrap_ratio(fail_ref: ArrayLike, fail_test: ArrayLike, B: int, seed: int,
                           z: float = Z95) -> Estimate:
    """Percentile bootstrap of V = (test failures) / (reference failures) over shots.

    Resampling shots with replacement is the same as drawing the four outcome counts from a
    multinomial with the observed frequencies, which is what is drawn (B draws of 4 numbers
    instead of B resamples of every shot). value = V of the observed counts. Resamples with no
    failure in either arm have no ratio; they are left out and counted in a warning. A resample
    with reference failures 0 and test failures > 0 has V = inf. The bounds are the order
    statistics at alpha/2 and 1 - alpha/2 of the remaining ratios (no interpolation).
    """
    counts = paired_counts(fail_ref, fail_test)
    resamples = _count(B, "B")
    if resamples == 0:
        raise _invalid("paired_bootstrap_ratio: B must be at least 1", B=B)
    if counts.shots == 0:
        raise _invalid("paired_bootstrap_ratio: no shots", shots=0)
    alpha = _alpha(z)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    probabilities = np.array(counts, dtype=np.float64) / counts.shots
    draws = rng.multinomial(counts.shots, probabilities, size=resamples)
    numerator = draws[:, 0] + draws[:, 2]
    denominator = draws[:, 0] + draws[:, 1]
    defined = (numerator > 0) | (denominator > 0)
    with np.errstate(divide="ignore"):
        ratios = numerator[defined] / denominator[defined]
    if ratios.size < resamples:
        logger.warning("paired_bootstrap_ratio: resamples without failures have no ratio and are "
                       "left out", extra={"left_out": int(resamples - ratios.size), "B": resamples})
    value = _ratio_point(counts)
    if ratios.size == 0:
        return Estimate(value, 0.0, math.inf)
    lo, hi = np.quantile(ratios, [alpha / 2.0, 1.0 - alpha / 2.0], method="inverted_cdf")
    logger.debug("paired bootstrap done", extra={"B": resamples, "shots": counts.shots,
                                                 "seconds": time.perf_counter() - started})
    return Estimate(value, float(lo), float(hi))


# ---------------------------------------------------------------------------------------------
# Bootstrap of means and quantiles (heavy-tailed iteration counts)


def _bootstrap(x: ArrayLike, B: int, seed: int, name: str,
               per_resample: Callable[[NDArray[np.int64] | None, NDArray[Any] | None,
                                       NDArray[Any], int], NDArray[np.float64]]
               ) -> NDArray[np.float64]:
    """Run B resamples of x and return per_resample's statistics, one row per resample.

    Resampling shots with replacement equals drawing the multiplicities of the distinct values
    from a multinomial. When x has few distinct values (iteration counts) the multinomial is far
    cheaper, so it is used when there are at most n/8 of them; otherwise indices are resampled.
    per_resample(multiplicities, samples, distinct_values, n) receives either a [chunk, U] matrix of
    multiplicities of the ascending distinct values, or a [chunk, n] matrix of resampled values.
    """
    data = _finite_1d(x, name)
    resamples = _count(B, "B")
    if resamples == 0:
        raise _invalid(f"{name}: B must be at least 1", B=B)
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = data.size
    values, multiplicity = np.unique(data, return_counts=True)
    use_multinomial = values.size * 8 <= n
    width = values.size if use_multinomial else n
    chunk = max(1, _CHUNK_ELEMENTS // width)
    rows = []
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        if use_multinomial:
            counts = rng.multinomial(n, multiplicity / n, size=size)
            rows.append(per_resample(counts, None, values, n))
        else:
            samples = data[rng.integers(0, n, size=(size, n))]
            rows.append(per_resample(None, samples, values, n))
    result = np.concatenate(rows, axis=0)
    logger.debug("bootstrap done", extra={"statistic": name, "B": resamples, "n": n,
                                          "distinct": int(values.size),
                                          "method": "multinomial" if use_multinomial else "index",
                                          "seconds": time.perf_counter() - started})
    return result


def bootstrap_mean(x: ArrayLike, B: int, seed: int, z: float = Z95) -> Estimate:
    """Mean of x with a percentile bootstrap interval (order statistics, no interpolation).

    x must be finite (e.g. drop the +inf weights of non-converged shots before calling).
    """
    alpha = _alpha(z)
    data = _finite_1d(x, "bootstrap_mean")

    def mean(counts: NDArray[np.int64] | None, samples: NDArray[Any] | None,
             values: NDArray[Any], n: int) -> NDArray[np.float64]:
        if counts is not None:
            return (counts @ values.astype(np.float64)) / n
        assert samples is not None
        return samples.mean(axis=1, dtype=np.float64)

    means = _bootstrap(data, B, seed, "bootstrap_mean", mean)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0], method="inverted_cdf")
    return Estimate(float(np.mean(data, dtype=np.float64)), float(lo), float(hi))


def nearest_rank(x: ArrayLike, q: float) -> float:
    """Nearest-rank q-quantile of x (the smallest value with at least a fraction q at or below it)."""
    data = _finite_1d(x, "nearest_rank")
    index = _nearest_rank_index(q, data.size)
    return float(np.partition(data, index)[index])


def bootstrap_quantiles(x: ArrayLike, qs: Sequence[float], B: int, seed: int,
                        z: float = Z95) -> list[Estimate]:
    """Nearest-rank quantiles of x at every level in qs, each with a percentile bootstrap interval.

    All levels share the same resamples. Bounds are order statistics (no interpolation).
    """
    alpha = _alpha(z)
    data = _finite_1d(x, "bootstrap_quantiles")
    levels = [float(q) for q in qs]
    if not levels:
        raise _invalid("bootstrap_quantiles: no quantile levels given")
    n = data.size
    indices = np.array([_nearest_rank_index(q, n) for q in levels], dtype=np.int64)

    def quantiles(counts: NDArray[np.int64] | None, samples: NDArray[Any] | None,
                  values: NDArray[Any], size: int) -> NDArray[np.float64]:
        if counts is not None:
            cumulative = np.cumsum(counts, axis=1)
            # The value of rank r is the first distinct value whose cumulative count reaches r.
            positions = np.stack([np.count_nonzero(cumulative <= index, axis=1) for index in indices],
                                 axis=1)
            return values[positions].astype(np.float64)
        assert samples is not None
        return np.partition(samples, np.unique(indices), axis=1)[:, indices].astype(np.float64)

    table = _bootstrap(data, B, seed, "bootstrap_quantiles", quantiles)
    sorted_data = np.sort(data)
    estimates = []
    for column, index in enumerate(indices):
        lo, hi = np.quantile(table[:, column], [alpha / 2.0, 1.0 - alpha / 2.0],
                             method="inverted_cdf")
        estimates.append(Estimate(float(sorted_data[index]), float(lo), float(hi)))
    return estimates


def bootstrap_quantile(x: ArrayLike, q: float, B: int, seed: int, z: float = Z95) -> Estimate:
    """Nearest-rank q-quantile of x with a percentile bootstrap interval."""
    return bootstrap_quantiles(x, [q], B, seed, z)[0]


# ---------------------------------------------------------------------------------------------
# Anchors: our interval against a published value


def intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """True when the closed intervals [a_lo, a_hi] and [b_lo, b_hi] share a point."""
    for lo, hi in ((a_lo, a_hi), (b_lo, b_hi)):
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise _invalid(f"interval [{lo}, {hi}] is not ordered", lo=lo, hi=hi)
    return a_lo <= b_hi and b_lo <= a_hi


def anchor_pass(lo: float, hi: float, value: float | tuple[float, float], factor: float) -> bool:
    """Acceptance rule for a published anchor.

    Our estimate has the interval [lo, hi]; the published value v comes with a multiplicative
    error factor f (its own error bar for a value printed in a paper, the stated reading error
    for a value read off a figure). The anchor passes when [lo, hi] overlaps [v / f, v f].
    A published range "a-b" is passed as value = (a, b) and gives the band [a / f, b f].
    """
    if not (math.isfinite(factor) and factor >= 1.0):
        raise _invalid(f"anchor factor must be finite and >= 1, got {factor!r}", factor=factor)
    if np.ndim(value) == 0:
        v_lo = v_hi = float(value)  # type: ignore[arg-type]
    else:
        v_lo, v_hi = (float(v) for v in value)  # type: ignore[union-attr]
    if not (v_lo > 0.0 and v_hi >= v_lo and math.isfinite(v_hi)):
        raise _invalid(f"published value must be positive and finite, got {value!r}", value=value)
    return intervals_overlap(lo, hi, v_lo / factor, v_hi * factor)


# ---------------------------------------------------------------------------------------------
# Confidence signals


def risk_coverage(score: ArrayLike, fail: ArrayLike, fractions: Sequence[float],
                  z: float = Z95) -> RiskCoverage:
    """Failure rate of the kept shots after discarding the least trustworthy fraction f.

    A larger score means less trustworthy. For each f the round(f * n) shots with the largest
    scores are discarded (halves rounded up, f taken as the exact decimal written). Ties in the
    score are broken by shot index: among equal scores the lower index is discarded first, so the
    result is deterministic. The kept-shot rate carries a Wilson interval; when nothing is kept
    (f = 1) the rate is NaN with the interval [0, 1]. NaN scores are rejected: a signal that is
    undefined for a shot must be mapped to a value by the caller.
    """
    _alpha(z)  # validates z
    failures = _binary(fail, "fail")
    scores = np.asarray(score)
    if scores.shape != failures.shape:
        raise _invalid(f"risk_coverage: {scores.size} scores against {failures.size} outcomes",
                       scores=scores.size, outcomes=failures.size)
    scores = _finite_1d(scores, "score")
    requested = np.asarray(fractions, dtype=np.float64)
    if requested.ndim != 1 or requested.size == 0 or not np.all((requested >= 0) & (requested <= 1)):
        raise _invalid("risk_coverage: fractions must be a non-empty list of values in [0, 1]",
                       fractions=list(np.atleast_1d(requested)))
    n = int(scores.size)
    # lexsort orders by the last key first: score descending (by reversing an ascending sort)
    # with index ascending inside a tie (the reversed ascending order of -index).
    order = np.lexsort((-np.arange(n), scores))[::-1]
    discarded_failures = np.concatenate(([0], np.cumsum(failures[order], dtype=np.int64)))
    total_failures = int(discarded_failures[-1])
    discarded = np.array([math.floor(_exact_decimal(f) * n + Fraction(1, 2)) for f in requested],
                         dtype=np.int64)
    kept = n - discarded
    kept_failures = total_failures - discarded_failures[discarded]
    rate = np.full(requested.size, np.nan)
    lo = np.zeros(requested.size)
    hi = np.ones(requested.size)
    some = kept > 0
    if not np.all(some):
        logger.warning("risk_coverage: a fraction discards every shot; its rate is undefined",
                       extra={"fractions": requested[~some].tolist()})
    rate[some] = kept_failures[some] / kept[some]
    lo[some], hi[some] = _wilson_bounds(kept_failures[some].astype(np.float64),
                                        kept[some].astype(np.float64), z)
    return RiskCoverage(requested, discarded, kept, kept_failures, rate, lo, hi, n)


def fit_logistic_slope(delta: ArrayLike, fail: ArrayLike, z: float = Z95) -> LogisticSlope:
    """Maximum-likelihood slope kappa of P(fail | D) = 1 / (1 + exp(kappa D)).

    The model has no intercept: a gap D = 0 means the two best classes are equally likely, so the
    decoder fails half the time. kappa = 1 is the ideal for D a true log-likelihood ratio (natural
    logarithm). The log-likelihood
        l(kappa) = -sum_fail softplus(kappa D) - sum_success softplus(-kappa D)
    is concave, so the score has one root. The interval is the profile-likelihood interval
    {kappa : 2 (l_max - l(kappa)) <= z^2}, which stays valid with few failures.

    Degenerate data, handled explicitly:
    - every D = 0: kappa is not identifiable -> (NaN, -inf, inf).
    - no failure with D > 0 and no success with D < 0 (e.g. no failures at all when D >= 0): the
      likelihood increases without bound in kappa, kappa = +inf, and only the lower bound is
      finite (the upper is +inf). Symmetrically kappa = -inf.
    """
    _alpha(z)  # validates z
    gaps = _finite_1d(delta, "delta").astype(np.float64)
    outcome = _binary(fail, "fail")
    if gaps.shape != outcome.shape:
        raise _invalid(f"fit_logistic_slope: {gaps.size} gaps against {outcome.size} outcomes",
                       gaps=gaps.size, outcomes=outcome.size)
    started = time.perf_counter()
    shots, failures = int(gaps.size), int(np.count_nonzero(outcome))
    values, inverse = np.unique(gaps, return_inverse=True)
    n_at = np.bincount(inverse).astype(np.float64)
    k_at = np.bincount(inverse, weights=outcome.astype(np.float64))
    s_at = n_at - k_at
    zero = values == 0.0
    ties = float(n_at[zero].sum())  # shots at D = 0 contribute ln(1/2) each for every kappa
    d, k, s = values[~zero], k_at[~zero], s_at[~zero]

    def log_likelihood(kappa: float) -> float:
        t = kappa * d
        return float(-(k @ np.logaddexp(0.0, t)) - (s @ np.logaddexp(0.0, -t))) - ties * math.log(2.0)

    def score(kappa: float) -> float:
        return float(d @ (s - (k + s) * special.expit(kappa * d)))

    def information(kappa: float) -> float:
        p = special.expit(kappa * d)
        return float(((k + s) * d * d) @ (p * (1.0 - p)))

    if d.size == 0:
        logger.warning("fit_logistic_slope: every gap is 0, kappa is not identifiable",
                       extra={"shots": shots})
        return LogisticSlope(math.nan, -math.inf, math.inf, math.nan,
                             -ties * math.log(2.0), shots, failures)

    positive, negative = d > 0, d < 0
    # Limits of the score as kappa -> +inf and -inf; the maximum is finite only if they bracket 0.
    score_up = float(-(d[positive] @ k[positive]) + d[negative] @ s[negative])
    score_down = float(d[positive] @ s[positive] - d[negative] @ k[negative])
    scale = 1.0 / float(np.median(np.abs(d)))
    target_drop = z * z / 2.0

    def crossing(f: Callable[[float], float], start: float, step: float) -> float:
        """Root of f beyond start in the direction of step; f(start) >= 0 and f -> -inf there."""
        a, b = start, start + step
        for _ in range(200):
            if f(b) < 0.0:
                return float(optimize.brentq(f, min(a, b), max(a, b), xtol=1e-15, rtol=1e-14,
                                             maxiter=500))
            a, b = b, b + 2.0 * (b - a)
        raise RuntimeError("fit_logistic_slope: no likelihood crossing found")

    if score_up >= 0.0 or score_down <= 0.0:
        kappa = math.inf if score_up >= 0.0 else -math.inf
        # The supremum: terms with a vanishing tail tend to 0, only the ties remain.
        supremum = -ties * math.log(2.0)
        level = supremum - target_drop

        def excess(x: float) -> float:
            return log_likelihood(x) - level

        # The likelihood is monotone towards the infinite kappa; walk from 0 to the crossing.
        towards = 1.0 if kappa > 0 else -1.0
        if excess(0.0) >= 0.0:
            bound = crossing(excess, 0.0, -towards * scale)
        else:
            bound = crossing(lambda x: -excess(x), 0.0, towards * scale)
        lo, hi = (bound, math.inf) if kappa > 0 else (-math.inf, bound)
        logger.warning("fit_logistic_slope: the data separate, the slope is infinite",
                       extra={"kappa": kappa, "bound": bound, "shots": shots, "failures": failures})
        return LogisticSlope(kappa, lo, hi, math.nan, supremum, shots, failures)

    # Bracket the root of the decreasing score, then solve.
    a, b = 0.0, 0.0
    step = scale
    if score(0.0) > 0.0:
        b = step
        while score(b) > 0.0:
            a, b, step = b, b + 2.0 * step, 2.0 * step
    else:
        a = -step
        while score(a) < 0.0:
            b, a, step = a, a - 2.0 * step, 2.0 * step
    kappa = float(optimize.brentq(score, a, b, xtol=1e-15, rtol=1e-14, maxiter=500))
    peak = log_likelihood(kappa)
    level = peak - target_drop

    def excess_at(x: float) -> float:
        return log_likelihood(x) - level

    lo = crossing(excess_at, kappa, -scale)
    hi = crossing(excess_at, kappa, scale)
    info = information(kappa)
    se = 1.0 / math.sqrt(info) if info > 0.0 else math.nan
    logger.debug("logistic slope fitted", extra={"kappa": kappa, "lo": lo, "hi": hi,
                                                 "shots": shots, "failures": failures,
                                                 "seconds": time.perf_counter() - started})
    return LogisticSlope(kappa, lo, hi, se, peak, shots, failures)


def detector_density(detectors: ArrayLike) -> NDArray[np.float64]:
    """Fraction of detectors equal to 1 in each shot ([shots, m] array in, [shots] out).

    A confidence baseline known before decoding: larger means less trustworthy.
    """
    array = np.asarray(detectors)
    if array.ndim != 2 or array.shape[1] == 0:
        raise _invalid(f"detector_density: expected a [shots, m] array with m > 0, got {array.shape}")
    return np.count_nonzero(array, axis=1) / array.shape[1]


def norm_fraction(ptr: ArrayLike, weights: ArrayLike, total: ArrayLike, alpha: float
                  ) -> NDArray[np.float64]:
    """Cluster norm fraction Q^(alpha) = (sum_i w_i^alpha)^(1/alpha) / total, one value per shot.

    Shot s has the clusters weights[ptr[s]:ptr[s+1]] (w_i = sum of lambda_j over cluster i, or the
    cluster size for the size variant); total is the sum of lambda_j over the whole fault set (a
    scalar or one value per shot). alpha = inf keeps only the largest cluster. A shot with no
    clusters has Q = 0. total must be positive. Larger means less trustworthy.
    """
    offsets = np.asarray(ptr, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if offsets.ndim != 1 or offsets.size < 1 or offsets[0] != 0 or offsets[-1] != w.size \
            or np.any(np.diff(offsets) < 0):
        raise _invalid("norm_fraction: ptr must be non-decreasing from 0 to len(weights)")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise _invalid("norm_fraction: cluster weights must be finite and non-negative")
    if not (alpha > 0):
        raise _invalid(f"norm_fraction: alpha must be positive, got {alpha!r}", alpha=alpha)
    shots = offsets.size - 1
    denominators = np.broadcast_to(np.asarray(total, dtype=np.float64), (shots,))
    if not np.all(denominators > 0):
        raise _invalid("norm_fraction: total weight must be positive")
    shot_of = np.repeat(np.arange(shots), np.diff(offsets))
    if math.isinf(alpha):
        norms = np.zeros(shots)
        np.maximum.at(norms, shot_of, w)
    else:
        norms = np.bincount(shot_of, weights=w ** alpha, minlength=shots) ** (1.0 / alpha)
    return norms / denominators


def min_round_time(mean_iterations: float, iteration_time: float, commit_rounds: int) -> float:
    """Smallest round time at which a window stream keeps up on average: E[i] tau_it / C.

    The queue of windows stays bounded only if the mean decode time of a window, E[i] tau_it, is
    below the time C tau_rd in which the next C rounds arrive. Monotone in E[i], so it maps the
    bounds of an interval on the mean.
    """
    rounds = _count(commit_rounds, "commit_rounds")
    if rounds == 0 or not (mean_iterations >= 0 and iteration_time > 0):
        raise _invalid("min_round_time: need mean_iterations >= 0, iteration_time > 0, C >= 1",
                       mean_iterations=mean_iterations, iteration_time=iteration_time, C=commit_rounds)
    return mean_iterations * iteration_time / rounds
