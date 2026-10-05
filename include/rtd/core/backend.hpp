#pragma once

#include <concepts>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "rtd/core/config.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// One leg: up to max_iter min-sum iterations with the memory strengths held constant.
struct LegParams {
    std::uint32_t max_iter = 0;
    AlphaRule alpha = ConstantAlpha{1.0};
    // Apply the memory term Λ = (1 − γ)λ + γM. False means Λ = λ and γ is ignored.
    bool use_memory = false;
};

// Why a backend could not be constructed.
struct BackendError {
    enum class Code : std::uint8_t {
        size_mismatch, // the graph and the priors disagree on n
        unavailable,   // this backend is not implemented or has no device to run on
    };
    Code code;
    std::string detail;
};

[[nodiscard]] constexpr std::string_view to_string(BackendError::Code code) noexcept {
    switch (code) {
    case BackendError::Code::size_mismatch:
        return "size_mismatch";
    case BackendError::Code::unavailable:
        return "unavailable";
    }
    return "unknown";
}

struct LegOutcome {
    bool converged = false;
    std::uint32_t iterations = 0;
    // W(ê) = Σ_{ê_j=1, λ_j finite} λ_j in double, summed in ascending external column order;
    // computed only when the leg converged, +∞ otherwise.
    double weight = 0.0;
};

// The boundary between the relay controller and the machine that runs the iterations.
//
// A backend owns all mutable state of one decoder instance (messages, marginals, memory strengths,
// hard decisions) and executes one leg at a time. The controller owns the leg schedule, the
// best-solution bookkeeping and the stopping rule; it never touches an edge or a marginal. Any
// type with these operations can be driven by RelayDecoder: the CPU backend here, a device backend
// later.
template <class B>
concept LegBackend = std::movable<B> &&
    requires(B b, const B cb, std::span<const Bit> syndrome, std::span<const double> gammas,
             double gamma, const LegParams& params, std::span<double> out,
             std::span<const Bit> mask) {
        { cb.num_rows() } -> std::same_as<index_t>;
        { cb.num_columns() } -> std::same_as<index_t>;
        // Rows on which H·ê = σ must hold for a leg to converge: mask[i] = 1 for a required row,
        // one entry per row; an empty mask requires every row (the default). Kept across decodes.
        { b.set_convergence_rows(mask) } -> std::same_as<void>;
        // Start a decode: copy σ; with init_marginals, set M ← λ (the memory term's first input).
        { b.begin(syndrome, true) } -> std::same_as<void>;
        // Memory strengths for the next leg: one per variable (external order), or one for all.
        { b.set_gamma(gammas) } -> std::same_as<void>;
        { b.set_gamma(gamma) } -> std::same_as<void>;
        // Reset every edge message to its prior, then iterate until H·ê = σ on the convergence
        // rows or max_iter. Marginals carry over from the previous leg.
        { b.run_leg(params) } -> std::same_as<LegOutcome>;
        // Remember the current ê as the solution to return.
        { b.mark_best() } -> std::same_as<void>;
        { b.best_hard() } -> std::same_as<std::span<const Bit>>;
        { b.current_hard() } -> std::same_as<std::span<const Bit>>;
        // The same two solutions as ascending external column indices.
        { b.best_support() } -> std::same_as<std::span<const index_t>>;
        { b.current_support() } -> std::same_as<std::span<const index_t>>;
        // Current marginals M_j as double, external order (tracing and tests).
        { cb.read_marginals(out) } -> std::same_as<void>;
    };

} // namespace rtd
