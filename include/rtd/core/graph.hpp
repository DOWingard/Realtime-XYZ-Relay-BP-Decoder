#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rtd/core/buffer.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// Physical order of the edge-message array.
//
// row_major: the edges of each row are contiguous (ordered by internal column) and every row
//   starts on a multiple of row_alignment_slots; the unused slots at the end of a row are padding.
//   The check pass streams the array with unit stride and no index loads; the variable pass
//   gathers through the column view.
// column_blocked: every degree run (below) is a dense block of `degree` planes, plane k holding
//   the k-th edge of each column of the run. The variable pass is unit-stride and vectorises
//   across columns; the check pass gathers through the row view.
enum class EdgeLayout : std::uint8_t { row_major, column_blocked };

// Internal column order. The decoder's arithmetic is independent of it: min and parity are
// order-free and each column's sum is ordered by row index, never by column position.
//
// wavefront: rows are grouped into blocks of block_rows consecutive rows, and each column is
//   assigned to the block holding its last row. An iteration then updates block by block: the
//   check nodes of a block, then every column whose rows are now all updated. Each node sees
//   exactly the inputs of the flooding schedule, so the result is identical, but a column reads
//   messages written moments earlier, from cache, instead of a full pass later. Within a block,
//   columns are sorted by degree (so each degree is one run with a specialised kernel and no
//   data-dependent branch), then by first row.
// degree_classes: one block; columns sorted by degree only.
// natural: one block; the input order.
enum class ColumnOrder : std::uint8_t { wavefront, degree_classes, natural };

struct GraphOptions {
    EdgeLayout layout = EdgeLayout::row_major;
    ColumnOrder column_order = ColumnOrder::wavefront;
    // Rows per block of the wavefront order.
    index_t block_rows = 64;
};

struct GraphError {
    enum class Code : std::uint8_t {
        row_pointer_size,
        row_pointer_start,
        row_pointer_not_monotone,
        row_pointer_end,
        column_out_of_range,
        row_not_strictly_increasing,
        too_large,
        unsupported_options,
    };
    Code code;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(GraphError::Code code) noexcept;

// Checks that (row_ptr, col_indices) is a valid CSR matrix with num_rows rows and num_columns
// columns: offsets start at 0, are monotone and end at col_indices.size(), and each row's column
// indices are < num_columns and strictly increasing.
[[nodiscard]] std::expected<void, GraphError> validate_csr(index_t num_rows, index_t num_columns,
                                                           std::span<const index_t> row_ptr,
                                                           std::span<const index_t> col_indices);

// A maximal range of consecutive internal columns that share one degree.
struct DegreeRun {
    index_t degree;
    index_t col_begin; // internal columns [col_begin, col_end)
    index_t col_end;
    // Column-view offset of col_begin; column c's edges are at edge_begin + (c − col_begin)·degree.
    index_t edge_begin;
    // column_blocked only: slot of (c, k) is slot_begin + k·stride + (c − col_begin).
    index_t slot_begin;
    index_t stride;

    [[nodiscard]] constexpr index_t size() const noexcept { return col_end - col_begin; }
};

// Rows [row_begin, row_end) and the degree runs whose columns become ready once those rows (and
// all earlier ones) have been updated.
struct RowBlock {
    index_t row_begin;
    index_t row_end;
    std::uint32_t run_begin;
    std::uint32_t run_end;
};

// A sequence of message-array slots: either an explicit index list or an arithmetic progression.
class SlotSequence {
public:
    constexpr SlotSequence(const index_t* indices, index_t size) noexcept
        : indices_(indices), size_(size) {}

    [[nodiscard]] static constexpr SlotSequence strided(index_t base, index_t stride,
                                                        index_t size) noexcept {
        SlotSequence s(nullptr, size);
        s.base_ = base;
        s.stride_ = stride;
        return s;
    }

    [[nodiscard]] constexpr index_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr index_t operator[](index_t k) const noexcept {
        return indices_ != nullptr ? indices_[k] : base_ + k * stride_;
    }

private:
    const index_t* indices_;
    index_t size_;
    index_t base_ = 0;
    index_t stride_ = 0;
};

// Raw arrays the row-major kernels run on.
struct RowMajorView {
    index_t num_rows;
    index_t num_columns;
    const index_t* row_slot_begin; // [m+1]; row i's slots are row_slot_begin[i] .. +row_degree(i)
    const index_t* row_ptr;        // [m+1]; row_degree(i) = row_ptr[i+1] − row_ptr[i]
    const index_t* slot_column;    // [slots]; internal column of each slot, num_columns for padding
    const index_t* column_slots;   // [E]; column view: slot of each edge, rows ascending
    const index_t* column_rows;    // [E]; column view: row of each edge, ascending
    std::span<const DegreeRun> runs;
};

// Raw arrays the column-blocked kernels run on.
struct ColumnBlockedView {
    index_t num_rows;
    index_t num_columns;
    const index_t* row_ptr;         // [m+1]
    const index_t* row_edge_slot;   // [E]; slot of each edge in row order
    const index_t* row_edge_column; // [E]; internal column of each edge in row order
    const index_t* column_rows;     // [E]; column view: row of each edge, ascending
    std::span<const DegreeRun> runs;
};

// The Tanner graph of a parity-check matrix H ∈ F₂^{m×n}: one check node per row, one variable
// node per column and one edge per nonzero H_ij. Immutable after construction and shared by
// const reference across every decoder and thread.
//
// Columns carry two numberings: external (the input matrix's) and internal (the physical order,
// see ColumnOrder). Everything a caller sees is external; the kernels work internally.
//
// Invariant the bit-exactness of the decoder rests on: the column view lists each column's edges
// in ascending row order, because the variable update sums incoming messages in that order and
// floating-point addition is not associative. Rows are never permuted.
class TannerGraph {
public:
    // Builds the graph of an m×n matrix given in CSR form. Validates that the offsets start at 0,
    // are monotone and end at col_indices.size(), and that each row's column indices are < n and
    // strictly increasing. Empty rows and columns are legal and counted.
    [[nodiscard]] static std::expected<TannerGraph, GraphError>
    from_csr(index_t num_rows, index_t num_columns, std::span<const index_t> row_ptr,
             std::span<const index_t> col_indices, GraphOptions options = {});

    TannerGraph(TannerGraph&&) noexcept = default;
    TannerGraph& operator=(TannerGraph&&) noexcept = default;
    TannerGraph(const TannerGraph&) = delete;
    TannerGraph& operator=(const TannerGraph&) = delete;
    ~TannerGraph() = default;

    [[nodiscard]] index_t num_rows() const noexcept { return m_; }
    [[nodiscard]] index_t num_columns() const noexcept { return n_; }
    [[nodiscard]] index_t num_edges() const noexcept { return e_; }
    // Length of the message array, including padding.
    [[nodiscard]] index_t num_slots() const noexcept { return slots_; }
    [[nodiscard]] EdgeLayout layout() const noexcept { return options_.layout; }
    [[nodiscard]] ColumnOrder column_order() const noexcept { return options_.column_order; }

    [[nodiscard]] index_t max_row_degree() const noexcept { return max_row_degree_; }
    [[nodiscard]] index_t max_column_degree() const noexcept { return max_col_degree_; }
    [[nodiscard]] index_t empty_rows() const noexcept { return empty_rows_; }
    [[nodiscard]] index_t empty_columns() const noexcept { return empty_cols_; }

    [[nodiscard]] index_t row_degree(index_t row) const noexcept {
        return row_ptr_[row + 1] - row_ptr_[row];
    }
    [[nodiscard]] index_t column_degree(index_t internal_column) const noexcept {
        return col_ptr_[internal_column + 1] - col_ptr_[internal_column];
    }

    // Internal columns of a row's edges, in the same order as row_slots(row).
    [[nodiscard]] std::span<const index_t> row_columns(index_t row) const noexcept;
    [[nodiscard]] SlotSequence row_slots(index_t row) const noexcept;
    // Rows of an internal column's edges, ascending, in the same order as column_slots(column).
    [[nodiscard]] std::span<const index_t> column_rows(index_t internal_column) const noexcept;
    [[nodiscard]] SlotSequence column_slots(index_t internal_column) const noexcept;

    [[nodiscard]] index_t external_column(index_t internal_column) const noexcept {
        return col_external_[internal_column];
    }
    [[nodiscard]] index_t internal_column(index_t external_column) const noexcept {
        return col_internal_[external_column];
    }
    [[nodiscard]] std::span<const index_t> external_columns() const noexcept {
        return col_external_.span();
    }

    [[nodiscard]] std::span<const DegreeRun> runs() const noexcept { return runs_; }
    [[nodiscard]] std::span<const RowBlock> row_blocks() const noexcept { return blocks_; }

    [[nodiscard]] RowMajorView row_major_view() const noexcept;
    [[nodiscard]] ColumnBlockedView column_blocked_view() const noexcept;

private:
    struct ColumnFacts;

    TannerGraph() = default;

    // Construction steps of from_csr, in order.
    [[nodiscard]] ColumnFacts transpose(std::span<const index_t> col_indices,
                                        std::size_t block_rows);
    void order_columns(const ColumnFacts& facts);
    void build_column_view(const ColumnFacts& facts);
    void build_runs(const ColumnFacts& facts);
    [[nodiscard]] std::expected<void, GraphError> lay_out_row_major();
    [[nodiscard]] std::expected<void, GraphError> lay_out_column_blocked();

    index_t m_ = 0;
    index_t n_ = 0;
    index_t e_ = 0;
    index_t slots_ = 0;
    GraphOptions options_{};
    index_t max_row_degree_ = 0;
    index_t max_col_degree_ = 0;
    index_t empty_rows_ = 0;
    index_t empty_cols_ = 0;

    AlignedBuffer<index_t> row_ptr_;        // [m+1] CSR offsets of the input
    AlignedBuffer<index_t> col_ptr_;        // [n+1] column-view offsets, internal order
    AlignedBuffer<index_t> col_rows_;       // [E]   column view: rows, ascending
    AlignedBuffer<index_t> col_slots_;      // [E]   row_major: column view slots
    AlignedBuffer<index_t> row_slot_begin_; // [m+1] row_major: padded row offsets
    AlignedBuffer<index_t> slot_col_;       // [slots] row_major: internal column per slot
    AlignedBuffer<index_t> row_edge_slot_;  // [E]   column_blocked: slot per row edge
    AlignedBuffer<index_t> row_edge_col_;   // [E]   column_blocked: internal column per row edge
    AlignedBuffer<index_t> col_external_;   // [n]   internal → external
    AlignedBuffer<index_t> col_internal_;   // [n]   external → internal
    std::vector<DegreeRun> runs_;
    std::vector<RowBlock> blocks_;
};

} // namespace rtd
