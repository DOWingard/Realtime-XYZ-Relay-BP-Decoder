#include "rtd/window/time_structure.hpp"

#include <format>
#include <limits>
#include <string>

#include "rtd/core/graph.hpp"

namespace rtd::window {

std::expected<TimeStructure, PlanError> TimeStructure::build(const Problem& problem) {
    using Code = PlanError::Code;
    const auto fail = [](Code code, std::string detail) {
        return std::unexpected(PlanError{.code = code, .detail = std::move(detail)});
    };
    const index_t m = problem.num_rows;
    const index_t n = problem.num_columns;
    const std::span<const std::int32_t> round = problem.detector_round;
    if (m == 0) {
        return fail(Code::no_rows, "the problem has no detectors");
    }
    if (round.size() != m) {
        return fail(Code::size_mismatch,
                    std::format("detector rounds has {} entries, expected {}", round.size(), m));
    }
    if (auto valid = validate_csr(m, n, problem.h_row_ptr, problem.h_col_indices); !valid) {
        return fail(Code::invalid_matrix, std::format("H: {}: {}", to_string(valid.error().code),
                                                      valid.error().detail));
    }
    for (index_t i = 1; i < m; ++i) {
        if (round[i] < round[i - 1]) {
            return fail(Code::rows_not_grouped_by_round,
                        std::format("detector {} has round {} after detector {} with round {}", i,
                                    round[i], i - 1, round[i - 1]));
        }
    }
    if (round[0] != 1) {
        return fail(Code::bad_round_numbering,
                    std::format("the first detector has round {}, expected 1", round[0]));
    }

    // Rounds are non-decreasing from 1, so they are all of size M exactly when row i has round
    // i / M + 1; the first row where that fails names the offending round.
    index_t per_round = 0;
    while (per_round < m && round[per_round] == 1) {
        ++per_round;
    }
    for (index_t i = per_round; i < m; ++i) {
        const auto expected = static_cast<std::int64_t>(i / per_round) + 1;
        if (round[i] != expected) {
            const std::int64_t short_round = round[i] > expected ? expected : round[i];
            return fail(Code::unequal_round_sizes,
                        std::format("round 1 has {} detectors but round {} does not (detector {} "
                                    "has round {}, expected {})",
                                    per_round, short_round, i, round[i], expected));
        }
    }
    if (m % per_round != 0) {
        return fail(Code::unequal_round_sizes,
                    std::format("round 1 has {} detectors but the last round, {}, has {}",
                                per_round, m / per_round + 1, m % per_round));
    }

    TimeStructure time;
    time.detectors_per_round_ = per_round;
    time.rounds_total_ = m / per_round;
    constexpr std::uint32_t unset = std::numeric_limits<std::uint32_t>::max();
    time.first_round_.assign(n, unset);
    std::vector<std::uint32_t> last_round(n, 0);
    for (index_t i = 0; i < m; ++i) {
        const std::uint32_t r = time.round_of_row(i);
        for (index_t e = problem.h_row_ptr[i]; e < problem.h_row_ptr[i + 1]; ++e) {
            const index_t j = problem.h_col_indices[e];
            // Rows are visited in ascending round order, so the first visit sets s(j).
            if (time.first_round_[j] == unset) {
                time.first_round_[j] = r;
            }
            last_round[j] = r;
        }
    }
    for (index_t j = 0; j < n; ++j) {
        if (time.first_round_[j] == unset) {
            return fail(Code::empty_column,
                        std::format("column {} flips no detector, so it has no round", j));
        }
        if (last_round[j] - time.first_round_[j] > 1) {
            return fail(Code::column_spans_rounds,
                        std::format("column {} touches rounds {} and {}, more than one apart", j,
                                    time.first_round_[j], last_round[j]));
        }
    }
    return time;
}

} // namespace rtd::window
