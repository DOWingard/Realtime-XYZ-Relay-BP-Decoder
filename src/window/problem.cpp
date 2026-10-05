#include "rtd/window/problem.hpp"

#include <format>
#include <string>

#include "rtd/core/graph.hpp"

namespace rtd::window {

std::expected<void, PlanError> validate(const Problem& problem) {
    using Code = PlanError::Code;
    const auto fail = [](Code code, std::string detail) {
        return std::unexpected(PlanError{.code = code, .detail = std::move(detail)});
    };
    const auto size_check = [&](std::string_view name, std::size_t found,
                                std::size_t expected) -> std::expected<void, PlanError> {
        if (found != expected) {
            return fail(Code::size_mismatch,
                        std::format("{} has {} entries, expected {}", name, found, expected));
        }
        return {};
    };
    const std::size_t m = problem.num_rows;
    const std::size_t n = problem.num_columns;
    const std::size_t k = problem.num_observables;
    for (const auto& check : {size_check("H row pointer", problem.h_row_ptr.size(), m + 1),
                              size_check("A row pointer", problem.a_row_ptr.size(), k + 1),
                              size_check("priors", problem.priors.size(), n),
                              size_check("detector rounds", problem.detector_round.size(), m)}) {
        if (!check) {
            return check;
        }
    }
    if (auto valid = validate_csr(problem.num_rows, problem.num_columns, problem.h_row_ptr,
                                  problem.h_col_indices);
        !valid) {
        return fail(Code::invalid_matrix, std::format("H: {}: {}", to_string(valid.error().code),
                                                      valid.error().detail));
    }
    if (auto valid = validate_csr(problem.num_observables, problem.num_columns, problem.a_row_ptr,
                                  problem.a_col_indices);
        !valid) {
        return fail(Code::invalid_matrix, std::format("A: {}: {}", to_string(valid.error().code),
                                                      valid.error().detail));
    }
    for (std::size_t j = 0; j < n; ++j) {
        const double p = problem.priors[j];
        // Written so that NaN fails the test.
        if (!(p >= 0.0 && p < 1.0)) { // NOLINT(readability-simplify-boolean-expr)
            return fail(Code::invalid_prior, std::format("p[{}] = {} is not in [0, 1)", j, p));
        }
    }
    return {};
}

} // namespace rtd::window
