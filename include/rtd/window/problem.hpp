#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "rtd/core/types.hpp"
#include "rtd/window/error.hpp"

namespace rtd::window {

// A decoding problem as the window layer sees it: H ∈ F₂^{m×n} (row i = detector i, column j =
// fault j, H_ij = 1 when fault j flips detector i), the fault probabilities p_j, the observable
// matrix A ∈ F₂^{k×n} (A_oj = 1 when fault j flips logical observable o) and the round
// ρ(i) ∈ {1, …, Rt} of every detector.
//
// Views only: the caller owns the arrays and keeps them alive while building a plan; the plan
// copies what it keeps.
struct Problem {
    index_t num_rows = 0;                         // m
    index_t num_columns = 0;                      // n
    index_t num_observables = 0;                  // k
    std::span<const index_t> h_row_ptr;           // [m + 1]
    std::span<const index_t> h_col_indices;       // [nnz(H)], strictly increasing within a row
    std::span<const double> priors;               // [n], p_j
    std::span<const index_t> a_row_ptr;           // [k + 1]
    std::span<const index_t> a_col_indices;       // [nnz(A)]
    std::span<const std::int32_t> detector_round; // [m], ρ(i)
};

// Checks every array's length, that H and A are valid CSR matrices and that every p_j lies in
// [0, 1) (p = 1 is a certain fault, which belongs in the syndrome, not in the problem).
[[nodiscard]] std::expected<void, PlanError> validate(const Problem& problem);

} // namespace rtd::window
