"""Integer emulator of the fixed-point Relay-BP decoder in the format intN.S.M, whole-shot or
windowed: the golden model the C++ fixed-point backend must match bit for bit.

The format (Maurer et al., arXiv:2510.21600, section 4.2 and App. C). Belief propagation in the
log-likelihood domain is unchanged when every log-likelihood is multiplied by the same constant,
so priors lambda, messages mu (check to variable) and nu (variable to check) and marginals M are
all multiplied by S and rounded to integers. Messages have N magnitude bits and an explicit sign:
they live in [-(2^N - 1), 2^N - 1]. A memory strength gamma is stored as beta = 1 - gamma,
scaled by M = 2^m: beta_int = round(beta M). The memory product gamma x is computed as
x - beta(x)x, where

    beta(x)x = sign(x) * sum over the set bits b of |x| of floor(beta_int * 2^b / M)

truncates every partial product before adding them (15 (x) 7 at M = 8 is 7 + 3 + 1 + 0 = 11).

One iteration (the flooding schedule of the C++ kernels):
    check i:     parity = s_i XOR (number of negative inputs); every edge gets the other edges'
                 smallest magnitude, scaled by alpha = 1 - 2^-k as x - (x >> k), with the sign
                 parity XOR its own sign
    variable j:  bias = sat(beta(x)lambda + M~ - beta(x)M~), M~ = sat(M)   (with memory)
                 sigma = bias + sum of incoming mu (exact); nu_e = sat(sigma - mu_e);
                 M = sigma; e_j = [sigma <= 0]
The first check pass of every leg reads the priors as its inputs (messages restart from the
priors); marginals carry over from leg to leg. The relay controller is Relay-BP's (Mueller et al.,
arXiv:2506.01779, Algorithm 1) with the budget rule of rtd_core: before each leg the remaining
iterations are cap - used; none left stops the decode.

Choices where the paper is silent (the C++ policy makes the same ones):
    rounding            half away from zero (for S lambda and for beta M)
    saturation          bias and stored marginal clip to +-(2^N - 1); the hard decision uses the
                        unsaturated sigma; sigma = 0 counts as an error
    accumulators        exact (wide enough never to overflow)
    alpha schedule      adaptive: k = t + 1 with t the iteration index within the leg (restarts
                        every leg); constant alpha must be 1 or 1 - 2^-k
    memory strengths    beta clipped to [0, 2]
    solution weight     sum of lambda_int over the support, divided by S once

Every quantity is an integer except the solution weight (an exact integer over S) and the
quantisation of the double-precision inputs (priors, memory strengths), which both programs do
with the same IEEE operations.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from rtd.window_ref_inner import InnerContext, InnerResult, _ShapeCache

logger = logging.getLogger("rtd.fixed_ref")

_MASK64 = (1 << 64) - 1
IDENTITY_SHIFT = 31  # a scaling shift this large leaves every magnitude unchanged (alpha = 1)
FORMATS = ("int4.2.8", "int5.2.8", "int6.2.8")  # the formats compiled into rtd_core


def round_half_away(v: np.ndarray | float) -> np.ndarray:
    """Round half away from zero without double rounding: |v| - floor(|v|) is exact."""
    x = np.asarray(v, dtype=np.float64)
    a = np.abs(x)
    r = np.floor(a)
    with np.errstate(invalid="ignore"):  # inf - inf: an infinite input stays infinite
        r = r + (a - r >= 0.5)
    return np.where(x < 0.0, -r, r)


@dataclass(frozen=True)
class FixedFormat:
    """intN.S.M: N magnitude bits, log-likelihood scale S, memory-strength scale M (a power of 2)."""

    bits: int
    scale: int
    memory: int

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 7:
            raise ValueError(f"magnitude bits must be in [1, 7], got {self.bits}")
        if not 1 <= self.scale <= 64:
            raise ValueError(f"the log-likelihood scale must be in [1, 64], got {self.scale}")
        if self.memory < 1 or self.memory > 32 or self.memory & (self.memory - 1):
            raise ValueError(f"the memory scale must be a power of two up to 32, got {self.memory}")

    @classmethod
    def parse(cls, name: str) -> FixedFormat:
        match = re.fullmatch(r"int(\d+)\.(\d+)\.(\d+)", name)
        if match is None:
            raise ValueError(f"{name!r} is not a format of the form intN.S.M")
        return cls(int(match[1]), int(match[2]), int(match[3]))

    @property
    def name(self) -> str:
        return f"int{self.bits}.{self.scale}.{self.memory}"

    @property
    def max_magnitude(self) -> int:
        return (1 << self.bits) - 1

    @property
    def memory_shift(self) -> int:
        return self.memory.bit_length() - 1

    def saturate(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, -self.max_magnitude, self.max_magnitude)

    def quantise_llr(self, llr: np.ndarray) -> np.ndarray:
        """lambda_int = sat(round(S lambda)); +inf becomes the largest magnitude."""
        llr = np.asarray(llr, dtype=np.float64)
        scaled = round_half_away(llr * float(self.scale))
        out = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=float(self.max_magnitude), neginf=-float(self.max_magnitude)), -self.max_magnitude, self.max_magnitude)
        return out.astype(np.int64)

    def quantise_gamma(self, gamma: np.ndarray | float) -> np.ndarray:
        """beta_int = round((1 - gamma) M), clipped to [0, 2M]."""
        g = np.asarray(gamma, dtype=np.float64)
        beta = round_half_away((1.0 - g) * float(self.memory))
        beta = np.where(np.isnan(g), float(self.memory), beta)
        return np.clip(beta, 0, 2 * self.memory).astype(np.int64)

    def product(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """beta(x)x: each set bit b of |x| adds floor(beta 2^b / M); the sign of x is kept."""
        x = np.asarray(x, dtype=np.int64)
        beta = np.asarray(beta, dtype=np.int64)
        magnitude = np.abs(x)
        total = np.zeros(np.broadcast_shapes(x.shape, beta.shape), dtype=np.int64)
        for b in range(self.bits):
            total += np.where((magnitude >> b) & 1, (beta << b) >> self.memory_shift, 0)
        return np.where(x < 0, -total, total)

    def bias(self, lam: np.ndarray, marginal: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """sat(beta(x)lambda + M~ - beta(x)M~) with M~ = sat(M)."""
        stored = self.saturate(marginal)
        return self.saturate(self.product(lam, beta) + stored - self.product(stored, beta))


@dataclass(frozen=True)
class AlphaRule:
    """Min-sum scaling: constant alpha, or adaptive alpha(t) = 1 - 2^-((t + 1) / scaling)."""

    rule: str  # "constant" | "adaptive"
    value: float = 1.0  # constant
    scaling: float = 1.0  # adaptive

    def check_fixed(self) -> None:
        """A fixed-point format scales by shift and subtract: every alpha must be 1 or 1 - 2^-k."""
        if self.rule == "adaptive":
            if self.scaling != 1.0:
                raise ValueError(f"adaptive alpha needs scaling 1 in a fixed-point format, got {self.scaling}")
            return
        if self.rule != "constant":
            raise ValueError(f"unknown alpha rule {self.rule!r}")
        if self.value == 1.0:
            return
        rest = 1.0 - self.value
        if 0.0 < self.value < 1.0 and rest <= 0.5:
            mantissa, _ = math.frexp(rest)
            if mantissa == 0.5:
                return
        raise ValueError(f"constant alpha {self.value} is not 1 or 1 - 2^-k for a whole k >= 1")

    def shift(self, t: int) -> int | None:
        """k of alpha = 1 - 2^-k at iteration t of a leg, or None for alpha = 1."""
        if self.rule == "adaptive":
            k = t + 1
        elif self.value == 1.0:
            return None
        else:
            k = -math.frexp(1.0 - self.value)[1] + 1
        return None if k >= IDENTITY_SHIFT else k

    def json(self) -> dict[str, Any]:
        return {"rule": "constant", "value": self.value} if self.rule == "constant" else {"rule": "adaptive", "scaling": self.scaling}


@dataclass(frozen=True)
class RelayParams:
    """The relay schedule: leg 0 of pre_iter iterations with memory gamma0 (None: no memory term),
    then num_sets relay legs of set_max_iter iterations each."""

    gamma0: float | None
    pre_iter: int
    set_max_iter: int
    num_sets: int
    stopping: str  # "after_n_converged" | "after_leg0" | "all_legs"
    stop_count: int
    alpha: AlphaRule

    def validate(self) -> None:
        if self.pre_iter < 1:
            raise ValueError("pre_iter must be >= 1")
        if self.num_sets > 0 and self.set_max_iter < 1:
            raise ValueError("set_max_iter must be >= 1 when num_sets > 0")
        if self.num_sets > 0 and self.gamma0 is None:
            raise ValueError("relay legs need a memory term (gamma0)")
        if self.stopping not in ("after_n_converged", "after_leg0", "all_legs"):
            raise ValueError(f"unknown stopping rule {self.stopping!r}")
        if self.stopping == "after_n_converged" and self.stop_count < 1:
            raise ValueError("after_n_converged needs a count >= 1")
        self.alpha.check_fixed()

    def stopping_json(self) -> dict[str, Any]:
        if self.stopping == "after_n_converged":
            return {"rule": "after_n_converged", "count": self.stop_count}
        return {"rule": self.stopping}


# ---------------------------------------------------------------------------------------------
# Memory strengths of the relay legs


def _splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step: (new state, output)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


@dataclass(frozen=True)
class UniformGammas:
    """rtd_core's UniformGammaGenerator: (seed, stream, leg) hashed by SplitMix64 into a
    xoshiro256+ state; gamma = low + (high - low) u with u = (x >> 11) 2^-53, kept below high."""

    seed: int
    low: float
    high: float

    def row(self, stream: int, leg: int, n: int) -> np.ndarray:
        _, key = _splitmix64(self.seed & _MASK64)
        key ^= stream & _MASK64
        _, key = _splitmix64(key)
        key ^= leg & _MASK64
        _, key = _splitmix64(key)
        s = []
        state = key
        for _ in range(4):
            state, word = _splitmix64(state)
            s.append(word)
        s0, s1, s2, s3 = s
        span = self.high - self.low
        below_high = math.nextafter(self.high, self.low)
        out = np.empty(n, dtype=np.float64)
        for j in range(n):
            result = (s0 + s3) & _MASK64
            t = (s1 << 17) & _MASK64
            s2 ^= s0
            s3 ^= s1
            s1 ^= s2
            s0 ^= s3
            s2 ^= t
            s3 = ((s3 << 45) | (s3 >> 19)) & _MASK64
            u = float(result >> 11) * 2.0**-53
            out[j] = min(self.low + span * u, below_high)
        return out


GammaRows = Callable[[int, int], np.ndarray]  # (stream, leg) -> gamma per column
# (shape index, columns) -> the shape's [T, n] table, e.g. SeededShapeGammas or FixedGammas.
ShapeTables = Callable[[int, int], np.ndarray]


def table_rows(table: np.ndarray) -> GammaRows:
    """Relay leg r >= 1 uses row r mod T of an explicit [T, n] table, for every stream."""
    rows = table.shape[0]
    return lambda stream, leg: table[leg % rows]


# ---------------------------------------------------------------------------------------------
# The decoder


@dataclass
class LegOutcome:
    iterations: int
    converged: bool
    weight: float  # +inf unless converged


@dataclass
class FixedResult:
    success: bool
    iterations: int
    legs: int
    best_leg: int | None
    weight: float
    cap_hit: bool
    support: np.ndarray  # int64, ascending
    leg_records: list[LegOutcome] = field(default_factory=list)
    marginals: np.ndarray | None = None  # sigma of the last iteration run (integers)


class FixedGraph:
    """Edge arrays of one parity-check matrix, in row-major order."""

    def __init__(self, h: sp.csr_matrix):
        h = sp.csr_matrix(h)
        h.sort_indices()
        self.m, self.n = int(h.shape[0]), int(h.shape[1])
        indptr = h.indptr.astype(np.int64)
        degree = np.diff(indptr)
        self.e_col = h.indices.astype(np.int64)
        self.e_row = np.repeat(np.arange(self.m, dtype=np.int64), degree)
        self.row_starts = indptr[:-1][degree > 0]
        self.nonempty = np.flatnonzero(degree > 0)
        if np.any(np.bincount(self.e_col, minlength=self.n) == 0):
            logger.warning("a column has no rows; its marginal is its bias alone", extra={"rows": self.m, "columns": self.n})

    def row_min(self, values: np.ndarray, empty: int) -> np.ndarray:
        out = np.full(self.m, empty, dtype=np.int64)
        if self.row_starts.size:
            out[self.nonempty] = np.minimum.reduceat(values, self.row_starts)
        return out

    def row_count(self, flags: np.ndarray) -> np.ndarray:
        return np.bincount(self.e_row, weights=flags, minlength=self.m).astype(np.int64)


class FixedRelay:
    """Relay-BP in intN.S.M on one decoding problem (H, priors)."""

    def __init__(self, fmt: FixedFormat, h: sp.csr_matrix, priors: np.ndarray, params: RelayParams):
        params.validate()
        self.fmt = fmt
        self.params = params
        self.graph = FixedGraph(h)
        priors = np.asarray(priors, dtype=np.float64)
        if priors.shape != (self.graph.n,):
            raise ValueError(f"{priors.size} priors for {self.graph.n} columns")
        # One division and one log per column with the C library's log, as rtd_core computes them.
        llr = np.array([math.log((1.0 - p) / p) if p > 0.0 else math.inf for p in priors.tolist()], dtype=np.float64)
        self.lam = fmt.quantise_llr(llr)
        self._beta = np.zeros(self.graph.n, dtype=np.int64)
        self._marginal = self.lam.copy()

    # -- one leg ------------------------------------------------------------------------------

    def _check_pass(self, nu: np.ndarray, syndrome: np.ndarray, shift: int | None) -> np.ndarray:
        g, top = self.graph, self.fmt.max_magnitude
        negative = nu < 0
        magnitude = np.abs(nu)
        parity = (g.row_count(negative) & 1).astype(bool) ^ syndrome
        min1 = g.row_min(magnitude, top)
        is_min = magnitude == min1[g.e_row]
        repeated = g.row_count(is_min) >= 2
        other = g.row_min(np.where(is_min, top, magnitude), top)
        min2 = np.where(repeated, min1, other)
        if shift is not None:
            min1 = min1 - (min1 >> shift)
            min2 = min2 - (min2 >> shift)
        out = np.where(is_min, min2[g.e_row], min1[g.e_row])
        return np.where(parity[g.e_row] != negative, -out, out)

    def _run_leg(self, syndrome: np.ndarray, converge: np.ndarray | None, max_iter: int, memory: bool) -> tuple[LegOutcome, np.ndarray]:
        g, fmt = self.graph, self.fmt
        nu = self.lam[g.e_col].copy()
        hard = np.zeros(g.n, dtype=bool)
        for t in range(max_iter):
            eta = self._check_pass(nu, syndrome, self.params.alpha.shift(t))
            incoming = np.bincount(g.e_col, weights=eta, minlength=g.n).astype(np.int64)
            bias = fmt.bias(self.lam, self._marginal, self._beta) if memory else self.lam
            sigma = bias + incoming
            nu = fmt.saturate(sigma[g.e_col] - eta)
            self._marginal = sigma
            hard = sigma <= 0
            residual = (g.row_count(hard[g.e_col]) & 1).astype(bool) ^ syndrome
            if converge is not None:
                residual &= converge
            if not residual.any():
                support = np.flatnonzero(hard)
                weight = float(int(self.lam[support].sum())) / float(fmt.scale)
                return LegOutcome(t + 1, True, weight), hard
        return LegOutcome(max_iter, False, math.inf), hard

    # -- the relay ------------------------------------------------------------------------------

    def decode(self, syndrome: np.ndarray, gammas: GammaRows | None, stream: int = 0, converge: np.ndarray | None = None, cap: int | None = None) -> FixedResult:
        p, fmt = self.params, self.fmt
        s = np.asarray(syndrome).astype(bool)
        if s.shape != (self.graph.m,):
            raise ValueError(f"syndrome has shape {s.shape}, the problem has {self.graph.m} rows")
        mask = None if converge is None or bool(np.all(converge)) else np.asarray(converge).astype(bool)
        memory = p.gamma0 is not None
        self._marginal = self.lam.copy()
        if memory:
            self._beta = np.full(self.graph.n, int(fmt.quantise_gamma(p.gamma0)), dtype=np.int64)
        if cap is not None and cap == 0:
            return FixedResult(False, 0, 0, None, math.inf, True, np.zeros(0, dtype=np.int64), [], self._marginal.copy())
        leg0_budget = min(p.pre_iter, cap) if cap is not None else p.pre_iter
        first, hard = self._run_leg(s, mask, leg0_budget, memory)
        records = [first]
        best_hard = hard.copy()
        iterations, legs, converged = first.iterations, 1, 0
        best_leg: int | None = None
        best_weight = math.inf
        if first.converged:
            converged, best_leg, best_weight = 1, 0, first.weight
        cap_hit = False
        stop_after_leg0 = first.converged and (p.stopping == "after_leg0" or (p.stopping == "after_n_converged" and converged >= p.stop_count))
        if not first.converged and leg0_budget < p.pre_iter:
            cap_hit = True
        elif not stop_after_leg0:
            for leg in range(1, p.num_sets + 1):
                budget = p.set_max_iter
                if cap is not None:
                    remaining = cap - iterations
                    if remaining == 0:
                        cap_hit = True
                        break
                    budget = min(budget, remaining)
                if gammas is None:
                    raise ValueError("relay legs need memory strengths")
                self._beta = fmt.quantise_gamma(gammas(stream, leg))
                outcome, hard = self._run_leg(s, mask, budget, True)
                records.append(outcome)
                legs += 1
                iterations += outcome.iterations
                if not outcome.converged:
                    if budget < p.set_max_iter:
                        cap_hit = True
                        break
                    continue
                converged += 1
                if outcome.weight < best_weight:
                    best_weight, best_leg = outcome.weight, leg
                    best_hard = hard.copy()
                if p.stopping == "after_n_converged" and converged >= p.stop_count:
                    break
        return FixedResult(
            success=best_leg is not None,
            iterations=iterations,
            legs=legs,
            best_leg=best_leg,
            weight=best_weight,
            cap_hit=cap_hit,
            support=np.flatnonzero(best_hard).astype(np.int64),
            leg_records=records,
            marginals=self._marginal.copy(),
        )


# ---------------------------------------------------------------------------------------------
# Inner decoder of the windowing reference (rtd.window_ref.decode_shot)


class FixedInner:
    """The integer relay decoder as the inner decoder of every window. Memory strengths of the
    relay legs come from an explicit table per window shape (leg r uses row r mod T) or from
    rtd_core's uniform generator keyed by the window's stream."""

    name = "fixed"

    def __init__(self, fmt: FixedFormat, params: RelayParams, gammas: ShapeTables | UniformGammas):
        params.validate()
        self._fmt = fmt
        self._params = params
        self._gammas = gammas
        self._cache = _ShapeCache()
        self.gamma_tables: dict[int, np.ndarray] = {}

    def _decoder(self, h: sp.csr_matrix, priors: np.ndarray, shape: int) -> tuple[FixedRelay, GammaRows]:
        cached = self._cache.get(shape, h, priors)
        if cached is not None:
            return cached
        relay = FixedRelay(self._fmt, h, priors, self._params)
        if isinstance(self._gammas, UniformGammas):
            generator = self._gammas
            n = h.shape[1]
            rows: GammaRows = lambda stream, leg: generator.row(stream, leg, n)  # noqa: E731
        else:
            table = self._gammas(shape, h.shape[1])
            self.gamma_tables[shape] = table
            rows = table_rows(table)
        logger.debug("fixed-point decoder prepared for shape", extra={"shape_index": shape, "rows": h.shape[0], "columns": h.shape[1], "format": self._fmt.name})
        return self._cache.put(shape, h, priors, (relay, rows))

    def decode(self, h: sp.csr_matrix, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        relay, rows = self._decoder(h, priors, ctx.shape)
        try:
            res = relay.decode(syndrome, rows, ctx.gamma_stream, ctx.converge, ctx.iteration_cap)
        except Exception:
            logger.exception("fixed-point window decode failed; aborting the shot", extra={"shot": ctx.shot, "window": ctx.window, "attempt": ctx.attempt, "shape_index": ctx.shape})
            raise
        return InnerResult(support=res.support, iterations=res.iterations, legs=res.legs, converged=res.success, weight=res.weight, cap_hit=res.cap_hit)

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class FixedInnerSpec:
    """Picklable description from which window_ref's worker processes build a FixedInner."""

    fmt: FixedFormat
    params: RelayParams
    gammas: ShapeTables | UniformGammas
    kind: str = "fixed"

    def build(self) -> FixedInner:
        t0 = time.perf_counter()
        inner = FixedInner(self.fmt, self.params, self.gammas)
        logger.debug("fixed-point inner decoder built", extra={"format": self.fmt.name, "seconds": time.perf_counter() - t0})
        return inner
