#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>

#include "rtd/api/error.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/observables.hpp"
#include "rtd/core/priors.hpp"
#include "rtd/core/types.hpp"
#include "rtd/window/problem.hpp"

namespace rtd::api {

// The arrays that define a decoding problem, as views the caller keeps alive for the duration of
// Problem::create (which copies them):
//   H ∈ F₂^{m×n}  rows = detectors, columns = fault mechanisms, CSR with strictly increasing
//                  column indices in every row;
//   A ∈ F₂^{k×n}  rows = logical observables, CSR;
//   p ∈ [0, 1)^n  the probability of each fault;
//   ρ ∈ ℤ^m       the round of every detector (only needed for sliding windows; empty otherwise);
//   b_σ ∈ F₂^m, b_ℓ ∈ F₂^k
//                  the syndrome and observable flips of faults that were removed from the problem
//                  because they occur with certainty (p = 1): the problem decodes σ ⊕ b_σ and
//                  reports ℓ̂ ⊕ b_ℓ. Empty when nothing was removed.
struct ProblemArrays {
    index_t num_rows = 0;        // m
    index_t num_columns = 0;     // n
    index_t num_observables = 0; // k
    std::span<const index_t> h_row_ptr;
    std::span<const index_t> h_col_indices;
    std::span<const double> priors;
    std::span<const index_t> a_row_ptr;
    std::span<const index_t> a_col_indices;
    std::span<const std::int32_t> detector_round;
    std::span<const Bit> syndrome_bias;
    std::span<const Bit> observables_bias;
};

// An owned, validated decoding problem: the Tanner graph of H (built with the decoder spec's
// graph options), the priors, the observable matrix and, when given, the detector rounds and the
// biases. Immutable; decoders keep pointers into it, so it is handed out by shared_ptr and every
// decoder built from it holds a reference.
class Problem {
    struct Token {};

public:
    [[nodiscard]] static std::expected<std::shared_ptr<const Problem>, ApiError>
    create(const ProblemArrays& arrays, const GraphOptions& graph);

    // For create() only (the token is private).
    Problem(Token /*token*/, TannerGraph graph, Priors priors,
            ObservableMatrix observables) noexcept;
    Problem(const Problem&) = delete;
    Problem& operator=(const Problem&) = delete;
    Problem(Problem&&) = delete;
    Problem& operator=(Problem&&) = delete;
    ~Problem() = default;

    [[nodiscard]] index_t num_rows() const noexcept { return graph_.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return graph_.num_columns(); }
    [[nodiscard]] index_t num_observables() const noexcept { return observables_.num_rows(); }

    [[nodiscard]] const TannerGraph& graph() const noexcept { return graph_; }
    [[nodiscard]] const Priors& priors() const noexcept { return priors_; }
    [[nodiscard]] const ObservableMatrix& observables() const noexcept { return observables_; }

    [[nodiscard]] bool has_rounds() const noexcept { return !detector_round_.empty(); }
    // The problem as the window layer sees it (views into this object). Meaningful only when
    // has_rounds().
    [[nodiscard]] window::Problem window_problem() const noexcept;

    // Empty when no certain fault was removed.
    [[nodiscard]] std::span<const Bit> syndrome_bias() const noexcept { return syndrome_bias_; }
    [[nodiscard]] std::span<const Bit> observables_bias() const noexcept {
        return observables_bias_;
    }
    // Column j of A as a k-bit mask (bit o = observable o); empty when k > 64, where a logical
    // class no longer fits one word.
    [[nodiscard]] std::span<const std::uint64_t> column_classes() const noexcept {
        return column_classes_;
    }

private:
    TannerGraph graph_;
    Priors priors_;
    ObservableMatrix observables_;
    std::vector<index_t> h_row_ptr_;
    std::vector<index_t> h_col_indices_;
    std::vector<index_t> a_row_ptr_;
    std::vector<index_t> a_col_indices_;
    std::vector<std::int32_t> detector_round_;
    std::vector<Bit> syndrome_bias_;
    std::vector<Bit> observables_bias_;
    std::vector<std::uint64_t> column_classes_;
};

} // namespace rtd::api
