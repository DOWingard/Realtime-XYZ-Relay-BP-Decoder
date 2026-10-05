#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string_view>

#include "rtd/window/error.hpp"

namespace rtd::window {

// How windows near the two ends of a shot are built.
//
// exact: every window is cut from the true matrix, so the first window (no carried-in faults
//   from round 0) and the last (ending at the noiseless readout) get their own shapes.
// uniform: every window position uses the shape of the second window, the first one that lies
//   wholly inside the bulk; rows past the readout round are zero. One shape per deferral level,
//   the arrangement a fixed hardware pipeline needs, at the price of mis-modelling round 1 and the
//   readout.
enum class Boundary : std::uint8_t { exact, uniform };

// What happens when no leg of a window's decode satisfies the window syndrome.
//
// commit_anyway: commit leg 0's final ê on the committed rounds; its unexplained defects there
//   are dropped.
// defer: commit nothing and decode the same start round again once C more rounds have arrived,
//   with width W + C, up to max_deferrals times.
// flag: as commit_anyway, and mark the stream as having a detected failure.
enum class OnFailure : std::uint8_t { commit_anyway, defer, flag };

[[nodiscard]] std::string_view to_string(Boundary boundary) noexcept;
[[nodiscard]] std::string_view to_string(OnFailure policy) noexcept;
[[nodiscard]] std::optional<Boundary> parse_boundary(std::string_view text) noexcept;
[[nodiscard]] std::optional<OnFailure> parse_on_failure(std::string_view text) noexcept;

// A sliding-window schedule over rounds r = 1 … Rt. Window k starts at round t_k = 1 + k·C,
// spans W rounds, decides the faults whose first round lies in its first C rounds, and keeps the
// other W − C rounds as a buffer that only informs that decision.
struct WindowSpec {
    std::uint32_t width = 0;           // W, rounds
    std::uint32_t commit = 0;          // C, rounds, 1 ≤ C < W
    std::uint32_t converge_rounds = 0; // C′, C ≤ C′ ≤ W: rounds whose rows must satisfy H·ê = σ
    Boundary boundary = Boundary::exact;
    OnFailure on_failure = OnFailure::commit_anyway;
    std::uint32_t max_deferrals = 0; // 0 unless on_failure = defer
    // Iterations of one window decode, summed over its legs (each deferral attempt is a decode of
    // its own); unset means no cap.
    std::optional<std::uint32_t> iteration_cap;
};

// Checks 1 ≤ C < W, C ≤ C′ ≤ W, max_deferrals = 0 unless the policy is defer, and a cap of at
// least one iteration.
[[nodiscard]] std::expected<void, PlanError> validate(const WindowSpec& spec);

} // namespace rtd::window
