#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <string_view>

namespace rtd {

// An arithmetic policy fixes the number format of the decoder: how priors, memory strengths and
// messages are represented and which operations combine them. The kernels are written once against
// this interface, so every format runs the identical schedule, tie rule and summation order and a
// difference in accuracy isolates the arithmetic.
//
// msg_t holds an edge message, a quantised prior λ_j and a memory strength γ_j.
// acc_t holds a marginal M_j and every running sum; it may be wider than msg_t.
template <class A>
concept MessageArithmetic =
    std::regular<typename A::msg_t> && std::regular<typename A::acc_t> &&
    std::totally_ordered<typename A::msg_t> &&
    requires(double d, A::msg_t x, A::acc_t s, bool negative) {
        { A::name } -> std::convertible_to<std::string_view>;
        // Quantise a prior log-odds; +∞ (a fault that never happens) maps to max_message().
        { A::from_llr(d) } -> std::same_as<typename A::msg_t>;
        { A::from_gamma(d) } -> std::same_as<typename A::msg_t>;
        { A::from_alpha(d) } -> std::same_as<typename A::msg_t>;
        // Sign rule used for the check-node parity and for every outgoing sign.
        { A::is_negative(x) } -> std::same_as<bool>;
        { A::magnitude(x) } -> std::same_as<typename A::msg_t>;
        { A::with_sign(x, negative) } -> std::same_as<typename A::msg_t>;
        // Min-sum normalisation α·|μ|.
        { A::scale(x, x) } -> std::same_as<typename A::msg_t>;
        { A::widen(x) } -> std::same_as<typename A::acc_t>;
        { A::accumulate(s, x) } -> std::same_as<typename A::acc_t>;
        { A::combine(s, s) } -> std::same_as<typename A::acc_t>;
        { A::to_msg(s) } -> std::same_as<typename A::msg_t>;
        // Local evidence with memory: Λ = (1 − γ)·λ + γ·M.
        { A::mix(x, s, x) } -> std::same_as<typename A::acc_t>;
        // Hard decision: the fault is declared present when M ≤ 0.
        { A::is_error(s) } -> std::same_as<bool>;
        // What a check with a single neighbour sends before scaling.
        { A::max_message() } -> std::same_as<typename A::msg_t>;
        { A::zero() } -> std::same_as<typename A::acc_t>;
        { A::to_double(s) } -> std::same_as<double>;
        // True when scale(x, from_alpha(1.0)) == x for every x, so α = 1 can skip the multiply.
        { A::unit_alpha_is_identity } -> std::convertible_to<bool>;
    };

// IEEE binary floating point. Every operation is one correctly rounded IEEE operation, in the
// order written; the core is compiled with -ffp-contract=off so a·b + c is never fused.
template <std::floating_point T>
struct FloatArith {
    using msg_t = T;
    using acc_t = T;

    static constexpr std::string_view name = sizeof(T) == 4 ? "f32" : "f64";
    static constexpr bool unit_alpha_is_identity = true;

    [[nodiscard]] static constexpr msg_t max_message() noexcept {
        return std::numeric_limits<T>::max();
    }

    [[nodiscard]] static msg_t from_llr(double llr) noexcept {
        return llr == std::numeric_limits<double>::infinity() ? max_message()
                                                              : static_cast<T>(llr);
    }
    [[nodiscard]] static msg_t from_gamma(double gamma) noexcept { return static_cast<T>(gamma); }
    [[nodiscard]] static msg_t from_alpha(double alpha) noexcept { return static_cast<T>(alpha); }

    // The sign bit, so −0.0 counts as negative.
    [[nodiscard]] static bool is_negative(msg_t x) noexcept { return std::signbit(x); }
    [[nodiscard]] static msg_t magnitude(msg_t x) noexcept { return std::fabs(x); }
    [[nodiscard]] static msg_t with_sign(msg_t magnitude, bool negative) noexcept {
        return negative ? -magnitude : magnitude;
    }
    [[nodiscard]] static msg_t scale(msg_t magnitude, msg_t alpha) noexcept {
        return alpha * magnitude;
    }

    [[nodiscard]] static acc_t widen(msg_t x) noexcept { return x; }
    [[nodiscard]] static acc_t accumulate(acc_t sum, msg_t x) noexcept { return sum + x; }
    [[nodiscard]] static acc_t combine(acc_t a, acc_t b) noexcept { return a + b; }
    [[nodiscard]] static msg_t to_msg(acc_t sum) noexcept { return sum; }

    // A saturated prior (a fault with p = 0) keeps its prior: memory never pulls it towards the
    // marginal. Otherwise (1 − γ) is rounded to T first, then two products and one sum.
    [[nodiscard]] static acc_t mix(msg_t lambda, acc_t marginal, msg_t gamma) noexcept {
        if (lambda == max_message()) {
            return lambda;
        }
        const T keep = T{1} - gamma;
        const T prior_part = lambda * keep;
        const T memory_part = marginal * gamma;
        return prior_part + memory_part;
    }

    [[nodiscard]] static bool is_error(acc_t marginal) noexcept { return marginal <= T{0}; }
    [[nodiscard]] static constexpr acc_t zero() noexcept { return T{0}; }
    [[nodiscard]] static double to_double(acc_t x) noexcept { return static_cast<double>(x); }
};

using F32 = FloatArith<float>;
using F64 = FloatArith<double>;

static_assert(MessageArithmetic<F32>);
static_assert(MessageArithmetic<F64>);

} // namespace rtd
