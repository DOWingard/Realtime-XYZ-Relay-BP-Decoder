#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>

#include "rtd/core/arith.hpp"
#include "rtd/core/config.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// Integer min-sum arithmetic in the format intN.S.M of IBM's FPGA Relay-BP decoder (Maurer et al.,
// arXiv:2510.21600, §4.2, §4.2.1, App. C).
//
// BP in the log-likelihood domain is invariant under a common scale factor, so every
// log-likelihood quantity (priors λ, check-to-variable messages μ, variable-to-check messages ν,
// marginals M) is multiplied by S and rounded to an integer. N is the number of magnitude bits;
// messages carry an explicit sign, so they range over [−(2^N − 1), 2^N − 1]. A memory strength γ
// is stored as β = 1 − γ ∈ [0, 2] scaled by M = 2^m, and γ·x is computed as x − β⊗x, where β⊗x
// is the shift-and-add product
//
//     β⊗x = sign(x) · Σ_{b : bit b of |x| is set} ⌊β_int · 2^b / M⌋,   β_int = round(β·M),
//
// which truncates every partial product before the sum (App. C, Table 2: 15 ⊗ 7 at M = 8 is
// 7 + 3 + 1 + 0 = 11, where 15·7/8 = 13.125).
//
// One variable update with memory, as in the relay adder of Fig. 3c:
//     bias Λ = sat(β⊗λ + M̃ − β⊗M̃),  M̃ = sat(M)     = (1 − γ)λ + γM up to truncation
//     σ = Λ + Σ_k μ_k                                  exact, in a wide accumulator
//     ν_k = sat(σ − μ_k),  M ← σ,  ê = [σ ≤ 0]
// and one check update is min-sum with the two smallest magnitudes scaled by α = 1 − 2^(−k),
// computed as x − (x >> k).
//
// The paper does not publish every detail. The choices made here, with their reasons:
//  1. Rounding. "Standard rounding" (§4.2) is taken as round half away from zero, for priors
//     (S·λ) and for memory strengths (β·M).
//  2. Saturation limits. The bias and the stored marginal both clip to the message range
//     ±(2^N − 1). The shift-and-add multiplier then serves both of its inputs (λ and M) with one
//     width, and the bias replaces, at the same width, the N-bit prior that a plain BP variable
//     node adds. The hard decision uses the unsaturated σ (Fig. 3b takes ê from σ_sum), and
//     clipping never changes a sign, so storing σ and clipping it when it is next read as M̃ is
//     the same as clipping it when it is stored.
//  3. Accumulator width. σ and every partial sum are exact: an adder tree of
//     N + 1 + ⌈log2(d + 1)⌉ bits for a column of degree d cannot overflow, and 32 bits hold that
//     for any degree this decoder meets. Integer addition is associative, so the exact σ − μ_k
//     does not depend on the order of the additions.
//  4. The zero marginal. HD(x) = (1 − sgn x)/2 leaves x = 0 undefined; σ = 0 counts as an error,
//     the rule of relay_bp and of the floating-point policies.
//  5. Scaling schedule. α·x = x − (x >> k): the shifted term is truncated, so α·x rounds up. With
//     AdaptiveAlpha{1}, k = t + 1 with t the iteration index within the leg, so α runs
//     1/2, 3/4, 7/8, … and restarts at every leg, as the messages restart from the priors at every
//     leg (Algorithm 1 counts t per leg). Constant α = 1 is the identity.
//  6. Memory strengths clip to β ∈ [0, 2] (γ ∈ [−1, 1]), the range §4.2.1 states.
//  7. Priors. λ = +∞ (a fault with p = 0) quantises to the largest magnitude like any large
//     prior: the gateware stores only λ_int and cannot tell them apart, so the floating-point
//     rule "a saturated prior ignores memory" has no counterpart. A negative λ (p > 1/2, never
//     produced by a detector error model) stays signed; the gateware's priors are unsigned.
//  8. Solution weight. W(ê) = Σ_{ê_j = 1} λ_int,j, summed exactly and reported as W/S in
//     log-likelihood units. The gateware compares the solutions of its legs using the only priors
//     it holds, the quantised ones.
//  9. Signed zero. Messages are stored as two's complement integers, in which zero has one sign.
//     The gateware's sign-magnitude −0 can only leave a check node (a zero minimum with odd
//     parity), where the adder of the receiving variable node reads it as 0; the messages that
//     enter a check node come from two's complement sums or from the unsigned priors, and are
//     never −0.
template <unsigned N, unsigned S, unsigned M>
struct FixedArith {
    static_assert(N >= 1 && N <= 7, "messages are stored as 8-bit integers with an explicit sign");
    static_assert(S >= 1 && S <= 64, "the log-likelihood scale must be a small positive integer");
    static_assert(std::has_single_bit(M) && M <= 32,
                  "the memory scale must be a power of two, with 2M representable in a message");

    using msg_t = std::int8_t;
    using acc_t = std::int32_t;

    static constexpr unsigned magnitude_bits = N;
    static constexpr unsigned llr_scale = S;
    static constexpr unsigned memory_scale = M;
    static constexpr unsigned memory_shift = static_cast<unsigned>(std::countr_zero(M));
    static constexpr int max_magnitude = (1 << N) - 1;
    static constexpr int max_beta = 2 * static_cast<int>(M);
    // A scaling shift this large leaves every magnitude unchanged: α = 1.
    static constexpr int identity_shift = 31;
    static constexpr bool unit_alpha_is_identity = true;
    // Messages are two's complement bytes with is_negative(x) = x < 0, magnitude(x) = |x| and
    // with_sign(m, s) = s ? −m : m, which lets the check pass run on byte vectors.
    static constexpr bool byte_messages = true;

private:
    static constexpr std::size_t name_length = [] {
        std::size_t length = 5; // "int" + two dots
        for (unsigned v : {N, S, M}) {
            do {
                ++length;
                v /= 10;
            } while (v != 0);
        }
        return length;
    }();

    static constexpr std::array<char, name_length> name_chars = [] {
        std::array<char, name_length> out{};
        std::size_t at = 0;
        for (const char c : std::string_view("int")) {
            out[at++] = c;
        }
        const auto put = [&](unsigned v) {
            std::array<char, 10> digits{};
            std::size_t count = 0;
            do {
                digits[count++] = static_cast<char>('0' + (v % 10));
                v /= 10;
            } while (v != 0);
            while (count > 0) {
                out[at++] = digits[--count];
            }
        };
        put(N);
        out[at++] = '.';
        put(S);
        out[at++] = '.';
        put(M);
        return out;
    }();

    // β⊗x for every β ∈ [0, 2M] and x ∈ [0, 2^N − 1]; the gateware's shift-and-add network,
    // tabulated.
    static constexpr std::size_t num_magnitudes = std::size_t{1} << N;
    static constexpr std::size_t num_betas = (2 * std::size_t{M}) + 1;
    static constexpr std::array<std::array<std::int16_t, num_magnitudes>, num_betas> products =
        [] {
            std::array<std::array<std::int16_t, num_magnitudes>, num_betas> table{};
            for (int beta = 0; beta <= max_beta; ++beta) {
                for (int x = 0; x <= max_magnitude; ++x) {
                    int sum = 0;
                    for (unsigned b = 0; b < N; ++b) {
                        if (((x >> b) & 1) != 0) {
                            sum += (beta << b) >> memory_shift;
                        }
                    }
                    table[static_cast<std::size_t>(beta)][static_cast<std::size_t>(x)] =
                        static_cast<std::int16_t>(sum);
                }
            }
            return table;
        }();

public:
    static constexpr std::string_view name{name_chars.data(), name_chars.size()};

    // Round half away from zero, written so that it never rounds twice: |v| − ⌊|v|⌋ is exact.
    [[nodiscard]] static double round_half_away(double v) noexcept {
        const double a = std::fabs(v);
        double r = std::floor(a);
        if (a - r >= 0.5) {
            r += 1.0;
        }
        return v < 0.0 ? -r : r;
    }

    [[nodiscard]] static constexpr int saturate(std::int64_t x) noexcept {
        return static_cast<int>(std::clamp<std::int64_t>(x, -max_magnitude, max_magnitude));
    }

    [[nodiscard]] static constexpr msg_t max_message() noexcept {
        return static_cast<msg_t>(max_magnitude);
    }

    // λ_int = sat(round(S·λ)); +∞ maps to the largest magnitude (choice 7).
    [[nodiscard]] static msg_t from_llr(double llr) noexcept {
        if (llr == std::numeric_limits<double>::infinity()) {
            return max_message();
        }
        if (std::isnan(llr)) {
            return msg_t{0};
        }
        const double scaled = round_half_away(llr * static_cast<double>(S));
        const double clipped =
            std::clamp(scaled, -static_cast<double>(max_magnitude), static_cast<double>(max_magnitude));
        return static_cast<msg_t>(clipped);
    }

    // β_int = round((1 − γ)·M), clipped to [0, 2M] (choice 6).
    [[nodiscard]] static msg_t from_gamma(double gamma) noexcept {
        if (std::isnan(gamma)) {
            return static_cast<msg_t>(M);
        }
        const double beta = 1.0 - gamma;
        const double scaled = round_half_away(beta * static_cast<double>(M));
        return static_cast<msg_t>(std::clamp(scaled, 0.0, static_cast<double>(max_beta)));
    }

    // The shift k of α = 1 − 2^(−k), rounded to the nearest integer in −log2(1 − α) and clipped to
    // [1, identity_shift]; α ≥ 1 is the identity. Only α of exactly this form is represented
    // exactly; fixed_alpha_problem() tells a caller which rules those are.
    [[nodiscard]] static msg_t from_alpha(double alpha) noexcept {
        if (!(alpha < 1.0)) {
            return static_cast<msg_t>(identity_shift);
        }
        const double exponent = -std::log2(1.0 - alpha);
        const double rounded = std::clamp(round_half_away(exponent), 1.0,
                                          static_cast<double>(identity_shift));
        return static_cast<msg_t>(rounded);
    }

    [[nodiscard]] static bool is_negative(msg_t x) noexcept { return x < 0; }
    [[nodiscard]] static msg_t magnitude(msg_t x) noexcept {
        return static_cast<msg_t>(x < 0 ? -x : x);
    }
    [[nodiscard]] static msg_t with_sign(msg_t magnitude, bool negative) noexcept {
        return negative ? static_cast<msg_t>(-magnitude) : magnitude;
    }

    // α·|μ| = |μ| − (|μ| >> k); |μ| ≥ 0 and 1 ≤ k ≤ 31, so both convert to unsigned bytes
    // unchanged.
    [[nodiscard]] static msg_t scale(msg_t magnitude, msg_t shift) noexcept {
        const auto x = static_cast<std::uint8_t>(magnitude);
        return static_cast<msg_t>(x - (x >> static_cast<std::uint8_t>(shift)));
    }

    [[nodiscard]] static acc_t widen(msg_t x) noexcept { return x; }
    [[nodiscard]] static acc_t accumulate(acc_t sum, msg_t x) noexcept { return sum + x; }
    [[nodiscard]] static acc_t combine(acc_t a, acc_t b) noexcept { return a + b; }
    [[nodiscard]] static msg_t to_msg(acc_t sum) noexcept {
        return static_cast<msg_t>(saturate(sum));
    }

    // β⊗x with the sign of x; |x| must not exceed 2^N − 1 and β must lie in [0, 2M].
    [[nodiscard]] static int product(int x, int beta) noexcept {
        const auto& row = products[static_cast<std::size_t>(beta)];
        return x < 0 ? -row[static_cast<std::size_t>(-x)] : row[static_cast<std::size_t>(x)];
    }

    // Λ = sat(β⊗λ + M̃ − β⊗M̃) with M̃ = sat(M).
    [[nodiscard]] static acc_t mix(msg_t lambda, acc_t marginal, msg_t beta) noexcept {
        const int stored = saturate(marginal);
        const int bias = product(lambda, beta) + stored - product(stored, beta);
        return saturate(bias);
    }

    [[nodiscard]] static bool is_error(acc_t marginal) noexcept { return marginal <= 0; }
    [[nodiscard]] static constexpr acc_t zero() noexcept { return 0; }
    // Log-likelihood units: the integer divided by S.
    [[nodiscard]] static double to_double(acc_t x) noexcept {
        return static_cast<double>(x) / static_cast<double>(S);
    }

    // W(ê) = Σ λ_int over the support, exact, divided by S once (choice 8).
    [[nodiscard]] static double solution_weight(std::span<const double> llr,
                                                std::span<const index_t> support) noexcept {
        std::int64_t sum = 0;
        for (const index_t j : support) {
            sum += from_llr(llr[j]);
        }
        return static_cast<double>(sum) / static_cast<double>(S);
    }
};

using Int4_2_8 = FixedArith<4, 2, 8>;
using Int5_2_8 = FixedArith<5, 2, 8>;
using Int6_2_8 = FixedArith<6, 2, 8>;

static_assert(MessageArithmetic<Int4_2_8>);
static_assert(MessageArithmetic<Int5_2_8>);
static_assert(MessageArithmetic<Int6_2_8>);

template <class A>
inline constexpr bool is_fixed_arith = false;
template <unsigned N, unsigned S, unsigned M>
inline constexpr bool is_fixed_arith<FixedArith<N, S, M>> = true;

// Why `rule` cannot be run exactly by a fixed-point format, or nothing when it can: every α it
// produces must be 1 or 1 − 2^(−k) for a whole k ≥ 1. That holds for ConstantAlpha{1},
// ConstantAlpha{1 − 2^(−k)} and AdaptiveAlpha{1}, whose α(t) = 1 − 2^(−(t + 1)).
[[nodiscard]] inline std::optional<std::string> fixed_alpha_problem(const AlphaRule& rule) {
    if (const auto* constant = std::get_if<ConstantAlpha>(&rule)) {
        const double alpha = constant->value;
        if (alpha == 1.0) {
            return std::nullopt;
        }
        const double rest = 1.0 - alpha;
        if (alpha > 0.0 && alpha < 1.0 && rest <= 0.5) {
            int exponent = 0;
            const double mantissa = std::frexp(rest, &exponent);
            if (mantissa == 0.5) {
                return std::nullopt;
            }
        }
        return std::format("constant alpha {} is not 1 or 1 - 2^-k for a whole k >= 1; a "
                           "fixed-point format scales by shift and subtract",
                           alpha);
    }
    const auto& adaptive = std::get<AdaptiveAlpha>(rule);
    if (adaptive.scaling == 1.0) {
        return std::nullopt;
    }
    return std::format("adaptive alpha with scaling {} gives alpha(t) = 1 - 2^-((t+1)/{}), which "
                       "is not 1 - 2^-k for whole k; a fixed-point format needs scaling 1",
                       adaptive.scaling, adaptive.scaling);
}

} // namespace rtd
