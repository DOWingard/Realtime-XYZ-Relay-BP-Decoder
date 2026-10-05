#include "rtd/api/problem.hpp"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

namespace rtd::api {

namespace {

std::unexpected<ApiError> invalid(std::string detail) {
    return std::unexpected(
        ApiError{.code = ApiError::Code::invalid_problem, .detail = std::move(detail)});
}

bool binary(std::span<const Bit> values) noexcept {
    return std::ranges::all_of(values, [](Bit b) { return b <= 1; });
}

} // namespace

Problem::Problem(Token /*token*/, TannerGraph graph, Priors priors,
                 ObservableMatrix observables) noexcept
    : graph_(std::move(graph)), priors_(std::move(priors)), observables_(std::move(observables)) {}

std::expected<std::shared_ptr<const Problem>, ApiError> Problem::create(const ProblemArrays& arrays,
                                                                        const GraphOptions& graph) {
    const index_t m = arrays.num_rows;
    const index_t n = arrays.num_columns;
    const index_t k = arrays.num_observables;
    if (arrays.priors.size() != n) {
        return invalid(std::format("{} priors for {} columns", arrays.priors.size(), n));
    }
    if (!arrays.detector_round.empty() && arrays.detector_round.size() != m) {
        return invalid(
            std::format("{} detector rounds for {} detectors", arrays.detector_round.size(), m));
    }
    if (!arrays.syndrome_bias.empty() &&
        (arrays.syndrome_bias.size() != m || !binary(arrays.syndrome_bias))) {
        return invalid(std::format("the syndrome bias must hold {} values in {{0, 1}}, found {}", m,
                                   arrays.syndrome_bias.size()));
    }
    if (!arrays.observables_bias.empty() &&
        (arrays.observables_bias.size() != k || !binary(arrays.observables_bias))) {
        return invalid(std::format("the observable bias must hold {} values in {{0, 1}}, found {}",
                                   k, arrays.observables_bias.size()));
    }
    auto tanner = TannerGraph::from_csr(m, n, arrays.h_row_ptr, arrays.h_col_indices, graph);
    if (!tanner) {
        return invalid(
            std::format("H: {}: {}", to_string(tanner.error().code), tanner.error().detail));
    }
    auto observables = ObservableMatrix::from_csr(k, n, arrays.a_row_ptr, arrays.a_col_indices);
    if (!observables) {
        return invalid(std::format("A: {}: {}", to_string(observables.error().code),
                                   observables.error().detail));
    }
    auto priors = Priors::from_probabilities(arrays.priors);
    if (!priors) {
        return invalid(std::format("priors: {}: column {}: {}", to_string(priors.error().code),
                                   priors.error().column, priors.error().detail));
    }

    auto problem = std::make_shared<Problem>(Token{}, std::move(*tanner), std::move(*priors),
                                             std::move(*observables));
    problem->h_row_ptr_.assign(arrays.h_row_ptr.begin(), arrays.h_row_ptr.end());
    problem->h_col_indices_.assign(arrays.h_col_indices.begin(), arrays.h_col_indices.end());
    problem->a_row_ptr_.assign(arrays.a_row_ptr.begin(), arrays.a_row_ptr.end());
    problem->a_col_indices_.assign(arrays.a_col_indices.begin(), arrays.a_col_indices.end());
    problem->detector_round_.assign(arrays.detector_round.begin(), arrays.detector_round.end());
    problem->syndrome_bias_.assign(arrays.syndrome_bias.begin(), arrays.syndrome_bias.end());
    problem->observables_bias_.assign(arrays.observables_bias.begin(),
                                      arrays.observables_bias.end());
    // A logical class packs the k observables into one 64-bit word.
    constexpr index_t class_bits = 64;
    if (k <= class_bits) {
        const ObservableMatrix& a = problem->observables_;
        problem->column_classes_.assign(n, 0);
        for (index_t j = 0; j < n; ++j) {
            for (const index_t o : a.column(j)) {
                problem->column_classes_[j] ^= std::uint64_t{1} << o;
            }
        }
    }
    return problem;
}

window::Problem Problem::window_problem() const noexcept {
    return {.num_rows = num_rows(),
            .num_columns = num_columns(),
            .num_observables = num_observables(),
            .h_row_ptr = h_row_ptr_,
            .h_col_indices = h_col_indices_,
            .priors = priors_.probabilities(),
            .a_row_ptr = a_row_ptr_,
            .a_col_indices = a_col_indices_,
            .detector_round = detector_round_};
}

} // namespace rtd::api
