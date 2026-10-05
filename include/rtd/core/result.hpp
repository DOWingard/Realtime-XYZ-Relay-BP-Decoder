#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "rtd/core/confidence.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// Budget of one decode.
struct DecodeLimits {
    // Iterations summed over all legs; unset means no cap. Before each leg the controller computes
    // what remains of the budget: if nothing remains the decode stops, otherwise the leg runs at
    // most min(its own budget, remaining) iterations. A cap of 0 therefore runs no leg at all and
    // returns ê = 0.
    std::optional<std::uint32_t> max_total_iterations;
};

// What one leg did. The per-leg trace lets one uncapped run answer "what would a cap of τ total
// iterations have returned" for every τ: the legs whose cumulative iteration count stays within τ
// are unchanged, and the leg that crosses τ is cut at τ without converging.
struct LegRecord {
    std::uint32_t iterations = 0;
    bool converged = false;
    // This leg's solution replaced the best one found so far (strictly lower weight).
    bool became_best = false;
    // W(ê) of the leg's solution; +∞ when the leg did not converge.
    double weight = 0.0;
};

// Outcome of one decode. The spans point into the decoder's buffers and stay valid until the
// decoder's next decode; copy what you keep.
struct DecodeResult {
    // Some leg converged, i.e. H·ê = σ holds for `hard` on the convergence rows (every row unless
    // the decoder was given a convergence mask).
    bool success = false;
    std::uint32_t iterations = 0;
    std::uint32_t legs_executed = 0;
    // Leg whose solution is returned; empty when no leg converged (then `hard` is leg 0's final ê,
    // or ê = 0 if a cap of 0 let no leg run).
    std::optional<std::uint32_t> best_leg;
    // W(ê) of the returned solution; +∞ when success is false.
    double weight = 0.0;
    std::span<const Bit> hard;
    std::span<const LegRecord> legs;
    // The decode ended because the iteration budget ran out while the stopping rule was
    // unsatisfied, and either a leg was cut short or another leg would have run. Equivalently: the
    // same decode without a cap would have run more iterations than the cap. Legs a selection
    // policy adds for low confidence run after the rule is met, so a cap ending them is not a hit.
    bool cap_hit = false;
    // The returned ê as ascending external column indices (the same solution as `hard`).
    std::span<const index_t> support;
    // Set only by a decoder whose solution sink is a selection policy (SelectingSink). The default
    // member initializer lets designated initializers omit it.
    std::optional<Confidence> confidence = std::nullopt;
};

struct DecodeError {
    enum class Code : std::uint8_t { syndrome_size_mismatch };
    Code code;
    std::size_t expected;
    std::size_t found;
};

[[nodiscard]] constexpr std::string_view to_string(DecodeError::Code code) noexcept {
    switch (code) {
    case DecodeError::Code::syndrome_size_mismatch:
        return "syndrome_size_mismatch";
    }
    return "unknown";
}

} // namespace rtd
