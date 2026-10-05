#include "rtd/window/error.hpp"

#include <format>

namespace rtd::window {

std::string_view to_string(PlanError::Code code) noexcept {
    using enum PlanError::Code;
    switch (code) {
    case commit_out_of_range:
        return "commit_out_of_range";
    case converge_out_of_range:
        return "converge_out_of_range";
    case deferrals_without_defer:
        return "deferrals_without_defer";
    case zero_iteration_cap:
        return "zero_iteration_cap";
    case size_mismatch:
        return "size_mismatch";
    case invalid_matrix:
        return "invalid_matrix";
    case invalid_prior:
        return "invalid_prior";
    case too_many_observables:
        return "too_many_observables";
    case no_rows:
        return "no_rows";
    case rows_not_grouped_by_round:
        return "rows_not_grouped_by_round";
    case bad_round_numbering:
        return "bad_round_numbering";
    case unequal_round_sizes:
        return "unequal_round_sizes";
    case empty_column:
        return "empty_column";
    case column_spans_rounds:
        return "column_spans_rounds";
    case no_bulk_window:
        return "no_bulk_window";
    case ambiguous_mapping:
        return "ambiguous_mapping";
    case graph_failed:
        return "graph_failed";
    }
    return "unknown";
}

std::string describe(const PlanError& error) {
    return std::format("{}: {}", to_string(error.code), error.detail);
}

} // namespace rtd::window
