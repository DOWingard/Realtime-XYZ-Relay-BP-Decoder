#include "rtd/core/observables.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>

namespace rtd {

static_assert(std::endian::native == std::endian::little,
              "the zero-skipping scan maps byte k of a word to entry j + k");

std::expected<SparseBinaryMatrix, GraphError>
SparseBinaryMatrix::from_csr(index_t num_rows, index_t num_columns,
                             std::span<const index_t> row_ptr,
                             std::span<const index_t> col_indices) {
    if (auto valid = validate_csr(num_rows, num_columns, row_ptr, col_indices); !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    SparseBinaryMatrix a;
    a.rows_ = num_rows;
    a.cols_ = num_columns;
    a.nnz_ = static_cast<index_t>(col_indices.size());
    a.col_ptr_ = AlignedBuffer<index_t>(static_cast<std::size_t>(num_columns) + 1, 0);
    a.row_idx_ = AlignedBuffer<index_t>(col_indices.size());
    for (const index_t j : col_indices) {
        ++a.col_ptr_[j + 1];
    }
    for (index_t j = 0; j < num_columns; ++j) {
        a.col_ptr_[j + 1] += a.col_ptr_[j];
    }
    AlignedBuffer<index_t> fill(num_columns, 0);
    for (index_t i = 0; i < num_rows; ++i) {
        for (index_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            const index_t j = col_indices[k];
            a.row_idx_[a.col_ptr_[j] + fill[j]++] = i;
        }
    }
    return a;
}

void SparseBinaryMatrix::apply(std::span<const Bit> x, std::span<Bit> out) const noexcept {
    std::ranges::fill(out, Bit{0});
    const std::size_t n = x.size();
    std::size_t j = 0;
    // Skip eight zero entries per step; corrections are overwhelmingly zero.
    for (; j + 8 <= n; j += 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, x.data() + j, sizeof word);
        while (word != 0) {
            const auto lane = static_cast<std::size_t>(std::countr_zero(word)) / 8;
            for (const index_t row : column(static_cast<index_t>(j + lane))) {
                out[row] ^= Bit{1};
            }
            word &= ~(std::uint64_t{0xFF} << (lane * 8));
        }
    }
    for (; j < n; ++j) {
        if (x[j] != 0) {
            for (const index_t row : column(static_cast<index_t>(j))) {
                out[row] ^= Bit{1};
            }
        }
    }
}

void SparseBinaryMatrix::apply_support(std::span<const index_t> support,
                                       std::span<Bit> out) const noexcept {
    std::ranges::fill(out, Bit{0});
    for (const index_t j : support) {
        for (const index_t row : column(j)) {
            out[row] ^= Bit{1};
        }
    }
}

} // namespace rtd
