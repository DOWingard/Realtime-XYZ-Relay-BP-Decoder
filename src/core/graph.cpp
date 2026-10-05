#include "rtd/core/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <tuple>
#include <utility>

namespace rtd {

std::string_view to_string(GraphError::Code code) noexcept {
    switch (code) {
    case GraphError::Code::row_pointer_size:
        return "row_pointer_size";
    case GraphError::Code::row_pointer_start:
        return "row_pointer_start";
    case GraphError::Code::row_pointer_not_monotone:
        return "row_pointer_not_monotone";
    case GraphError::Code::row_pointer_end:
        return "row_pointer_end";
    case GraphError::Code::column_out_of_range:
        return "column_out_of_range";
    case GraphError::Code::row_not_strictly_increasing:
        return "row_not_strictly_increasing";
    case GraphError::Code::too_large:
        return "too_large";
    case GraphError::Code::unsupported_options:
        return "unsupported_options";
    }
    return "unknown";
}

namespace {

std::unexpected<GraphError> fail(GraphError::Code code, std::string detail) {
    return std::unexpected(GraphError{.code = code, .detail = std::move(detail)});
}

// Largest slot count we accept; leaves headroom so slot + stride arithmetic cannot wrap.
constexpr std::uint64_t max_slots = std::numeric_limits<index_t>::max() / 2;

} // namespace

std::expected<void, GraphError> validate_csr(index_t num_rows, index_t num_columns,
                                             std::span<const index_t> row_ptr,
                                             std::span<const index_t> col_indices) {
    using Code = GraphError::Code;
    const std::size_t m = num_rows;
    const std::size_t n = num_columns;
    if (row_ptr.size() != m + 1) {
        return fail(Code::row_pointer_size,
                    std::format("row_ptr has {} entries, expected m + 1 = {}", row_ptr.size(), m + 1));
    }
    if (row_ptr[0] != 0) {
        return fail(Code::row_pointer_start, std::format("row_ptr[0] = {}, expected 0", row_ptr[0]));
    }
    for (std::size_t i = 0; i < m; ++i) {
        if (row_ptr[i + 1] < row_ptr[i]) {
            return fail(Code::row_pointer_not_monotone,
                        std::format("row_ptr[{}] = {} < row_ptr[{}] = {}", i + 1, row_ptr[i + 1], i,
                                    row_ptr[i]));
        }
    }
    if (row_ptr[m] != col_indices.size()) {
        return fail(Code::row_pointer_end,
                    std::format("row_ptr[m] = {} but there are {} column indices", row_ptr[m],
                                col_indices.size()));
    }
    const std::size_t e = col_indices.size();
    if (e > max_slots || n > max_slots || m > max_slots) {
        return fail(Code::too_large, std::format("m = {}, n = {}, E = {} exceed 32-bit indexing", m,
                                                 n, e));
    }
    for (std::size_t i = 0; i < m; ++i) {
        const index_t begin = row_ptr[i];
        const index_t end = row_ptr[i + 1];
        for (index_t k = begin; k < end; ++k) {
            const index_t col = col_indices[k];
            if (col >= num_columns) {
                return fail(Code::column_out_of_range,
                            std::format("row {} has column index {} >= n = {}", i, col, n));
            }
            if (k > begin && col <= col_indices[k - 1]) {
                return fail(Code::row_not_strictly_increasing,
                            std::format("row {}: column {} follows column {}", i, col,
                                        col_indices[k - 1]));
            }
        }
    }
    return {};
}

// The input's column view (rows of each external column, ascending) and the facts the column
// order is computed from.
struct TannerGraph::ColumnFacts {
    AlignedBuffer<index_t> degree; // [n]
    AlignedBuffer<index_t> ptr;    // [n+1]
    AlignedBuffer<index_t> rows;   // [E]
    std::size_t block_rows = 1;    // rows per row block

    [[nodiscard]] std::span<const index_t> rows_of(index_t j) const noexcept {
        return {rows.data() + ptr[j], degree[j]};
    }
    // A column is ready once its last row has been updated; a column with no rows is ready at
    // once.
    [[nodiscard]] std::size_t block_of(index_t j) const noexcept {
        return degree[j] == 0 ? 0 : rows[ptr[j + 1] - 1] / block_rows;
    }
};

std::expected<TannerGraph, GraphError>
TannerGraph::from_csr(index_t num_rows, index_t num_columns, std::span<const index_t> row_ptr,
                      std::span<const index_t> col_indices, GraphOptions options) {
    using Code = GraphError::Code;
    if (options.layout == EdgeLayout::column_blocked &&
        options.column_order == ColumnOrder::natural) {
        return fail(Code::unsupported_options,
                    "column_blocked layout needs long degree runs; natural column order has none");
    }
    const bool wavefront = options.column_order == ColumnOrder::wavefront;
    if (wavefront && options.block_rows == 0) {
        return fail(Code::unsupported_options, "wavefront column order needs block_rows >= 1");
    }
    if (auto valid = validate_csr(num_rows, num_columns, row_ptr, col_indices); !valid) {
        return std::unexpected(std::move(valid.error()));
    }

    TannerGraph g;
    g.m_ = num_rows;
    g.n_ = num_columns;
    g.e_ = static_cast<index_t>(col_indices.size());
    g.options_ = options;
    g.row_ptr_ = AlignedBuffer<index_t>(row_ptr.size());
    std::ranges::copy(row_ptr, g.row_ptr_.begin());

    // The wavefront order uses blocks of block_rows rows; the other orders put every row in one.
    const std::size_t m = num_rows;
    const std::size_t block_rows = wavefront ? options.block_rows : std::max<std::size_t>(m, 1);
    const ColumnFacts facts = g.transpose(col_indices, block_rows);
    g.order_columns(facts);
    g.build_column_view(facts);
    g.build_runs(facts);
    auto laid_out = options.layout == EdgeLayout::row_major ? g.lay_out_row_major()
                                                            : g.lay_out_column_blocked();
    if (!laid_out) {
        return std::unexpected(std::move(laid_out.error()));
    }
    return g;
}

TannerGraph::ColumnFacts TannerGraph::transpose(std::span<const index_t> col_indices,
                                                std::size_t block_rows) {
    const std::size_t n = n_;
    ColumnFacts facts{.degree = AlignedBuffer<index_t>(n, 0),
                      .ptr = AlignedBuffer<index_t>(n + 1, 0),
                      .rows = AlignedBuffer<index_t>(col_indices.size()),
                      .block_rows = block_rows};
    for (index_t i = 0; i < m_; ++i) {
        const index_t degree = row_degree(i);
        max_row_degree_ = std::max(max_row_degree_, degree);
        empty_rows_ += degree == 0 ? 1U : 0U;
    }
    for (const index_t j : col_indices) {
        ++facts.degree[j];
    }
    for (std::size_t j = 0; j < n; ++j) {
        facts.ptr[j + 1] = facts.ptr[j] + facts.degree[j];
        max_col_degree_ = std::max(max_col_degree_, facts.degree[j]);
        empty_cols_ += facts.degree[j] == 0 ? 1U : 0U;
    }
    // Visiting rows in ascending order fills each column's list in ascending row order.
    AlignedBuffer<index_t> fill(n, 0);
    for (index_t i = 0; i < m_; ++i) {
        for (index_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            const index_t j = col_indices[k];
            facts.rows[facts.ptr[j] + fill[j]++] = i;
        }
    }
    return facts;
}

void TannerGraph::order_columns(const ColumnFacts& facts) {
    const std::size_t n = n_;
    col_external_ = AlignedBuffer<index_t>(n);
    col_internal_ = AlignedBuffer<index_t>(n);
    for (std::size_t j = 0; j < n; ++j) {
        col_external_[j] = static_cast<index_t>(j);
    }
    const AlignedBuffer<index_t>& degree = facts.degree;
    if (options_.column_order == ColumnOrder::wavefront) {
        // By ready block, then degree, then the column's rows compared lexicographically. The
        // last key puts columns that share rows next to each other, so the variable pass finds
        // their slots in the cache lines the previous column just touched; its first element is
        // the first row, so within a run the columns that reach back furthest come first.
        std::ranges::stable_sort(col_external_, [&](index_t a, index_t b) {
            const auto ka = std::tuple(facts.block_of(a), degree[a]);
            const auto kb = std::tuple(facts.block_of(b), degree[b]);
            if (ka != kb) {
                return ka < kb;
            }
            return std::ranges::lexicographical_compare(facts.rows_of(a), facts.rows_of(b));
        });
    } else if (options_.column_order == ColumnOrder::degree_classes) {
        std::ranges::stable_sort(col_external_,
                                 [&](index_t a, index_t b) { return degree[a] < degree[b]; });
    }
    for (std::size_t c = 0; c < n; ++c) {
        col_internal_[col_external_[c]] = static_cast<index_t>(c);
    }
}

// The column view in internal order: each column's rows, ascending, which is the summation order
// of the variable update.
void TannerGraph::build_column_view(const ColumnFacts& facts) {
    const std::size_t n = n_;
    col_ptr_ = AlignedBuffer<index_t>(n + 1);
    col_rows_ = AlignedBuffer<index_t>(e_);
    col_ptr_[0] = 0;
    for (std::size_t c = 0; c < n; ++c) {
        const std::span<const index_t> rows = facts.rows_of(col_external_[c]);
        std::ranges::copy(rows, col_rows_.begin() + col_ptr_[c]);
        col_ptr_[c + 1] = col_ptr_[c] + static_cast<index_t>(rows.size());
    }
}

// Runs of equal degree within each row block, and each block's row and run ranges.
void TannerGraph::build_runs(const ColumnFacts& facts) {
    const std::size_t m = m_;
    const std::size_t n = n_;
    const std::size_t block_rows = facts.block_rows;
    const std::size_t num_blocks = std::max<std::size_t>((m + block_rows - 1) / block_rows, 1);
    blocks_.reserve(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        blocks_.push_back(RowBlock{.row_begin = static_cast<index_t>(std::min(b * block_rows, m)),
                                   .row_end = static_cast<index_t>(std::min((b + 1) * block_rows, m)),
                                   .run_begin = 0,
                                   .run_end = 0});
    }
    for (std::size_t c = 0; c < n;) {
        const index_t ext = col_external_[c];
        const index_t degree = facts.degree[ext];
        const std::size_t block = facts.block_of(ext);
        std::size_t end = c + 1;
        while (end < n && facts.degree[col_external_[end]] == degree &&
               facts.block_of(col_external_[end]) == block) {
            ++end;
        }
        runs_.push_back(DegreeRun{.degree = degree,
                                  .col_begin = static_cast<index_t>(c),
                                  .col_end = static_cast<index_t>(end),
                                  .edge_begin = col_ptr_[c],
                                  .slot_begin = 0,
                                  .stride = 0});
        blocks_[block].run_end = static_cast<std::uint32_t>(runs_.size());
        c = end;
    }
    // Runs are ordered by block; give empty blocks an empty range at the right place.
    std::uint32_t next_run = 0;
    for (RowBlock& block : blocks_) {
        block.run_begin = next_run;
        block.run_end = std::max(block.run_end, next_run);
        next_run = block.run_end;
    }
}

std::expected<void, GraphError> TannerGraph::lay_out_row_major() {
    const std::size_t m = m_;
    row_slot_begin_ = AlignedBuffer<index_t>(m + 1);
    std::uint64_t slots = 0;
    for (index_t i = 0; i < m_; ++i) {
        row_slot_begin_[i] = static_cast<index_t>(slots);
        slots += round_up(row_degree(i), row_alignment_slots);
    }
    if (slots > max_slots) {
        return fail(GraphError::Code::too_large,
                    std::format("{} padded slots exceed 32-bit indexing", slots));
    }
    row_slot_begin_[m] = static_cast<index_t>(slots);
    slots_ = static_cast<index_t>(slots);
    slot_col_ = AlignedBuffer<index_t>(slots, n_);
    col_slots_ = AlignedBuffer<index_t>(e_);
    // Visiting columns in internal order lays each row's edges out by internal column, so
    // consecutive columns of the variable pass touch adjacent slots of the rows they share.
    AlignedBuffer<index_t> row_fill(m, 0);
    for (index_t c = 0; c < n_; ++c) {
        for (index_t k = col_ptr_[c]; k < col_ptr_[c + 1]; ++k) {
            const index_t row = col_rows_[k];
            const index_t slot = row_slot_begin_[row] + row_fill[row]++;
            col_slots_[k] = slot;
            slot_col_[slot] = c;
        }
    }
    return {};
}

std::expected<void, GraphError> TannerGraph::lay_out_column_blocked() {
    std::uint64_t slots = 0;
    for (DegreeRun& run : runs_) {
        run.stride = static_cast<index_t>(round_up(run.size(), row_alignment_slots));
        run.slot_begin = static_cast<index_t>(slots);
        slots += static_cast<std::uint64_t>(run.stride) * run.degree;
        if (slots > max_slots) {
            return fail(GraphError::Code::too_large,
                        std::format("{} blocked slots exceed 32-bit indexing", slots));
        }
    }
    slots_ = static_cast<index_t>(slots);
    row_edge_slot_ = AlignedBuffer<index_t>(e_);
    row_edge_col_ = AlignedBuffer<index_t>(e_);
    AlignedBuffer<index_t> row_fill(m_, 0);
    for (const DegreeRun& run : runs_) {
        for (index_t c = run.col_begin; c < run.col_end; ++c) {
            for (index_t k = 0; k < run.degree; ++k) {
                const index_t row = col_rows_[col_ptr_[c] + k];
                const index_t edge = row_ptr_[row] + row_fill[row]++;
                row_edge_slot_[edge] = run.slot_begin + (k * run.stride) + (c - run.col_begin);
                row_edge_col_[edge] = c;
            }
        }
    }
    return {};
}

std::span<const index_t> TannerGraph::row_columns(index_t row) const noexcept {
    const index_t degree = row_degree(row);
    if (options_.layout == EdgeLayout::row_major) {
        return {slot_col_.data() + row_slot_begin_[row], degree};
    }
    return {row_edge_col_.data() + row_ptr_[row], degree};
}

SlotSequence TannerGraph::row_slots(index_t row) const noexcept {
    const index_t degree = row_degree(row);
    if (options_.layout == EdgeLayout::row_major) {
        return SlotSequence::strided(row_slot_begin_[row], 1, degree);
    }
    return {row_edge_slot_.data() + row_ptr_[row], degree};
}

std::span<const index_t> TannerGraph::column_rows(index_t internal_column) const noexcept {
    return {col_rows_.data() + col_ptr_[internal_column], column_degree(internal_column)};
}

SlotSequence TannerGraph::column_slots(index_t internal_column) const noexcept {
    const index_t degree = column_degree(internal_column);
    if (options_.layout == EdgeLayout::row_major) {
        return {col_slots_.data() + col_ptr_[internal_column], degree};
    }
    const auto run = std::ranges::upper_bound(runs_, internal_column, {}, &DegreeRun::col_begin) - 1;
    return SlotSequence::strided(run->slot_begin + (internal_column - run->col_begin), run->stride,
                                 degree);
}

RowMajorView TannerGraph::row_major_view() const noexcept {
    return RowMajorView{.num_rows = m_,
                        .num_columns = n_,
                        .row_slot_begin = row_slot_begin_.data(),
                        .row_ptr = row_ptr_.data(),
                        .slot_column = slot_col_.data(),
                        .column_slots = col_slots_.data(),
                        .column_rows = col_rows_.data(),
                        .runs = runs_};
}

ColumnBlockedView TannerGraph::column_blocked_view() const noexcept {
    return ColumnBlockedView{.num_rows = m_,
                             .num_columns = n_,
                             .row_ptr = row_ptr_.data(),
                             .row_edge_slot = row_edge_slot_.data(),
                             .row_edge_column = row_edge_col_.data(),
                             .column_rows = col_rows_.data(),
                             .runs = runs_};
}

} // namespace rtd
