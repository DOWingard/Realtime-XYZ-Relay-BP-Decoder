#pragma once

#include <expected>
#include <span>

#include "rtd/core/buffer.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// A sparse matrix M ∈ F₂^{r×c} and its product with a vector over GF(2).
//
// Serves as the observable matrix A (ℓ̂ = A·ê gives the logical frame change implied by a
// correction) and, in tests, as H itself (σ = H·e is the syndrome of an injected error).
//
// Stored by column, because the vectors it multiplies are corrections: sparse, with a few hundred
// ones among tens of thousands of entries. The product visits only the columns in their support.
class SparseBinaryMatrix {
public:
    [[nodiscard]] static std::expected<SparseBinaryMatrix, GraphError>
    from_csr(index_t num_rows, index_t num_columns, std::span<const index_t> row_ptr,
             std::span<const index_t> col_indices);

    [[nodiscard]] index_t num_rows() const noexcept { return rows_; }
    [[nodiscard]] index_t num_columns() const noexcept { return cols_; }
    [[nodiscard]] index_t num_nonzeros() const noexcept { return nnz_; }

    // Rows of column j that hold a one, ascending.
    [[nodiscard]] std::span<const index_t> column(index_t j) const noexcept {
        return {row_idx_.data() + col_ptr_[j], col_ptr_[j + 1] - col_ptr_[j]};
    }

    // out = M·x mod 2 for a dense x ∈ {0,1}^c. Preconditions: x.size() == c, out.size() == r.
    void apply(std::span<const Bit> x, std::span<Bit> out) const noexcept;

    // out = M·x mod 2 where x is given by the column indices of its ones (any order, no repeats).
    void apply_support(std::span<const index_t> support, std::span<Bit> out) const noexcept;

private:
    SparseBinaryMatrix() = default;

    index_t rows_ = 0;
    index_t cols_ = 0;
    index_t nnz_ = 0;
    AlignedBuffer<index_t> col_ptr_; // [c+1]
    AlignedBuffer<index_t> row_idx_; // [nnz]
};

using ObservableMatrix = SparseBinaryMatrix;

} // namespace rtd
