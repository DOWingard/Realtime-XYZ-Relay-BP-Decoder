#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rtd::window {

// Why a window specification, a decoding problem or a window plan was rejected. `detail` names
// the offending value, row, column or round, so the error can be logged without re-deriving
// anything.
struct PlanError {
    enum class Code : std::uint8_t {
        // WindowSpec
        commit_out_of_range,     // C = 0 or C ≥ W
        converge_out_of_range,   // C′ outside [C, W]
        deferrals_without_defer, // max_deferrals > 0 under a policy other than defer
        zero_iteration_cap,      // a window that may run no iteration can never converge
        // Problem
        size_mismatch,        // an array's length disagrees with m, n or k
        invalid_matrix,       // H or A is not a valid CSR matrix
        invalid_prior,        // p outside [0, 1)
        too_many_observables, // k > 64: a column's observables no longer fit one 64-bit mask
        // TimeStructure
        no_rows,
        rows_not_grouped_by_round, // detector rounds decrease somewhere
        bad_round_numbering,       // the first detector's round is not 1
        unequal_round_sizes,       // rounds hold different numbers of detectors, or one is missing
        empty_column,              // a fault that flips no detector
        column_spans_rounds,       // a fault touches rounds more than one apart
        // Uniform boundary
        no_bulk_window,    // no window position away from both ends of the shot to copy
        ambiguous_mapping, // two global columns match one shifted bulk column
        // Shape construction
        graph_failed,
    };
    Code code;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(PlanError::Code code) noexcept;

// One line: "<code>: <detail>".
[[nodiscard]] std::string describe(const PlanError& error);

} // namespace rtd::window
