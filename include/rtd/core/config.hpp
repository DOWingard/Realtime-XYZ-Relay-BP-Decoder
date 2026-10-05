#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace rtd {

// Min-sum normalisation α: every check-to-variable magnitude is multiplied by α.
//
// ConstantAlpha{1} is plain (unnormalised) min-sum. AdaptiveAlpha{s} uses
// α(t) = 1 − 2^(−(t+1)/s), where t is the iteration index within the current leg, starting at 0:
// small α damps early iterations and α → 1 as the leg proceeds.
struct ConstantAlpha {
    double value = 1.0;
};
struct AdaptiveAlpha {
    double scaling = 1.0;
};
using AlphaRule = std::variant<ConstantAlpha, AdaptiveAlpha>;

// α for iteration t of a leg, in double (each backend quantises it).
[[nodiscard]] double alpha_at(const AlphaRule& rule, std::uint32_t iteration) noexcept;

// True when α is exactly 1 on every iteration, so the multiply can be compiled out.
[[nodiscard]] bool is_unit_alpha(const AlphaRule& rule) noexcept;

struct MinSumConfig {
    AlphaRule alpha = ConstantAlpha{1.0};
    // Memory strength of the first leg. Unset means no memory term at all: the local evidence of
    // every variable is its prior λ_j, and any γ handed to the backend is ignored.
    std::optional<double> gamma0;
};

// When the relay stops. Every rule also stops after the last configured leg.
//
// AfterLeg0: stop after leg 0 if it converged; otherwise run every relay leg with no early stop.
// AfterNConverged{S}: stop as soon as S legs (leg 0 included) have converged.
// AllLegs: never stop early.
struct AfterLeg0 {};
struct AfterNConverged {
    std::uint32_t count = 1;
};
struct AllLegs {};
using StoppingRule = std::variant<AfterLeg0, AfterNConverged, AllLegs>;

// The leg schedule. Leg 0 runs up to pre_iter iterations with memory strength γ₀; relay legs
// r = 1..num_sets run up to set_max_iter iterations each with per-variable strengths from a
// GammaSource. Plain min-sum is {num_sets = 0, AfterLeg0} with gamma0 unset; Mem-BP is the same
// with gamma0 set. There are no default budgets: pre_iter = 0 fails validation.
struct RelayConfig {
    std::uint32_t pre_iter = 0;
    std::uint32_t set_max_iter = 0;
    std::uint32_t num_sets = 0;
    StoppingRule stopping = AfterLeg0{};
};

struct ConfigError {
    enum class Code : std::uint8_t {
        invalid_alpha,
        invalid_gamma0,
        relay_without_memory,
        zero_pre_iter,
        zero_set_max_iter,
        zero_stop_count,
        missing_gamma_source,
        gamma_width_mismatch,
    };
    Code code;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(ConfigError::Code code) noexcept;

// Rejects configurations that are meaningless or silently degenerate: a non-positive or
// non-finite α, a non-finite γ₀, relay legs without a memory term (every leg would then repeat
// plain min-sum from the same priors), zero iteration budgets and a zero convergence count.
[[nodiscard]] std::expected<void, ConfigError> validate(const MinSumConfig& min_sum,
                                                        const RelayConfig& relay);

// Number of legs the configuration can run: 1 + num_sets.
[[nodiscard]] constexpr std::uint32_t max_legs(const RelayConfig& relay) noexcept {
    return 1 + relay.num_sets;
}

} // namespace rtd
