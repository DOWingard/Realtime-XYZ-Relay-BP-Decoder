#pragma once

// What a selecting solution sink tells the relay controller, and the confidence value it attaches
// to a decode. The rules themselves live in selection.hpp; this header holds only the types the
// controller and DecodeResult need, so that the core's result type does not depend on them.

#include <concepts>
#include <cstdint>
#include <limits>
#include <span>

#include "rtd/core/sink.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// Whether the class gap Δ exists for a decode.
//
// none: no leg converged, so there is no lowest-weight solution ê* to measure from.
// single_class: every solution found has the logical class of ê*. Δ is then undefined, and it is
//   kept as its own state because "no other class was found" is not "the other classes are
//   infinitely unlikely": a decoder that explores little would look certain.
// defined: some solution of another class was found and Δ is finite.
enum class GapState : std::uint8_t { none = 0, single_class = 1, defined = 2 };

// The confidence of one decode, computed from the solutions the sink saw: the first `capacity`
// converged legs, in leg order ("slots"). Two slots with the same support hash hold the same
// solution; a slot is *new* when no earlier slot has its hash. Weights are W(ê) = Σ λ_j over the
// support, λ_j = ln((1 − p_j)/p_j); P(ê) ∝ e^{−W(ê)}.
struct Confidence {
    // Converged legs of the decode, including any beyond the capacity.
    std::uint32_t found = 0;
    // Slots the rules saw (min(found, capacity)), the new slots among them, and the number of
    // logical classes they fall into.
    std::uint32_t seen = 0;
    std::uint32_t distinct = 0;
    std::uint32_t classes = 0;

    // ê*: the earliest slot of minimum weight (a strict-< scan in slot order); its class is the
    // lowest-weight decision. With no slot, the class of the returned ê (leg 0's final estimate).
    std::uint64_t best_class = 0;
    double weight = std::numeric_limits<double>::infinity(); // W(ê*); +∞ when none

    // Δ = W₂ − W(ê*), W₂ = min weight over the slots whose class differs from ê*'s (strict-<
    // scan; second_class is the class of the first slot reaching it). NaN unless defined.
    GapState gap_state = GapState::none;
    double gap = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t second_class = 0;
    // (new slots of ê*'s class) / (new slots); NaN when none.
    double agreement = std::numeric_limits<double>::quiet_NaN();

    // Effort until the first solution: leg + 1 of the first slot and the iterations of every leg
    // up to and including it; both 0 when nothing converged.
    std::uint32_t first_legs = 0;
    std::uint32_t first_iterations = 0;

    // The class-sum decision: Z̃_L = Σ exp(−(W(ê) − W(ê*))) over the new slots of class L in slot
    // order (Z_L·e^{W(ê*)}), the largest Z̃ winning, ties to the class that appeared first;
    // class_sum_top is that Z̃ (NaN when none). The largest-agreement decision: the class with the
    // most new slots, ties to the one whose lightest new slot weighs less, then to the earlier one.
    std::uint64_t class_sum_class = 0;
    double class_sum_top = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t agreement_class = 0;

    // Q_supp^(2) of ê*: split its support into connected components (two faults connected when
    // they flip a common detector), L_c = Σ λ over component c (ascending columns), components in
    // order of their smallest column, q_sum_sq = Σ_c L_c², and Q = sqrt(q_sum_sq) / Λ with
    // Λ = Σ λ over every column of the problem (a constant, as in Lee et al.'s cluster fraction).
    // Larger means the correction's weight sits in fewer, larger clusters. NaN (and 0 components)
    // when there is no ê*, its support is unknown, or the sink has no parity-check matrix.
    double q_supp = std::numeric_limits<double>::quiet_NaN();
    double q_sum_sq = std::numeric_limits<double>::quiet_NaN();
    double q_total = std::numeric_limits<double>::quiet_NaN(); // Λ of the decoded problem
    std::uint32_t components = 0;

    // Detectors of the decoded syndrome that fired, of how many; density = ones / detectors.
    std::uint32_t syndrome_ones = 0;
    std::uint32_t syndrome_rows = 0;
    double density = std::numeric_limits<double>::quiet_NaN();

    // The class the configured selection rule decided, and the leg whose solution represents it
    // (the earliest lightest new slot of that class).
    std::uint64_t decided_class = 0;
    std::uint32_t decided_leg = 0;

    // The configured low-confidence signal as an inverse confidence (larger = less trustworthy):
    // +∞ with no solution, NaN when it has no value for this decode (for the gap: every solution
    // in one class, whose category is gap_state), and whether the decode counts as low.
    double score = std::numeric_limits<double>::quiet_NaN();
    bool low = false;

    // The sink's own stopping rule ended the decode at a converged leg where the relay schedule's
    // rule did not yet hold, so fewer legs ran than the relay rule alone would have run.
    bool stopped_early = false;
    // Legs run after the relay rule was met because the confidence was low.
    std::uint32_t extra_legs = 0;
};

// What the controller knows when a decode ends, handed to a selecting sink.
struct DecodeFacts {
    bool success = false;
    std::uint32_t iterations = 0;
    std::uint32_t legs_executed = 0;
    // The controller's own choice: the lowest-weight converged leg, earliest on ties.
    std::uint32_t best_leg = 0;
    double best_weight = std::numeric_limits<double>::infinity();
    std::span<const index_t> best_support;
    bool stopped_early = false;
    std::uint32_t extra_legs = 0;
    // The syndrome the decode was given: detectors that fired, and its length.
    std::uint32_t syndrome_ones = 0;
    std::uint32_t syndrome_rows = 0;
};

// A selecting sink's answer at the end of a decode: the confidence and, when its rule decided a
// different solution than the controller's lowest-weight one, that solution (views into the sink,
// valid until its next decode).
struct Selection {
    bool replaces = false;
    std::uint32_t leg = 0;
    double weight = 0.0;
    std::span<const index_t> support;
    std::span<const Bit> hard;
    Confidence confidence;
};

// A solution sink that also decides, online, which solution a decode returns and when it stops.
// The controller asks it after every converged leg (the leg has already been reported):
//   stop_requested(): the sink's own stopping rule is satisfied; the decode ends here.
//   low_confidence(): the current solutions give a low confidence value.
//   extra_legs():     how many more legs to run when the relay rule is met with low confidence;
//                     those legs end early once the confidence is no longer low.
// finish() is called once per decode, after the last leg.
template <class S>
concept SelectingSink = SolutionSink<S> && requires(S s, const S cs, const DecodeFacts& facts) {
    { cs.stop_requested() } noexcept -> std::same_as<bool>;
    { cs.low_confidence() } noexcept -> std::same_as<bool>;
    { cs.extra_legs() } noexcept -> std::convertible_to<std::uint32_t>;
    { s.finish(facts) } noexcept -> std::same_as<Selection>;
};

} // namespace rtd
