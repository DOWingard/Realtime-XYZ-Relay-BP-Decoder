#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <vector>

#include "rtd/core/graph.hpp"
#include "support.hpp"

namespace {

using namespace rtd;
using test::Csr;

std::expected<TannerGraph, GraphError> build(const Csr& h, GraphOptions options = {}) {
    return TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx, options);
}

const std::vector<GraphOptions> all_options{
    {.layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 2},
    {.layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 64},
    {.layout = EdgeLayout::row_major,
     .column_order = ColumnOrder::degree_classes,
     .block_rows = 64},
    {.layout = EdgeLayout::row_major, .column_order = ColumnOrder::natural, .block_rows = 64},
    {.layout = EdgeLayout::column_blocked, .column_order = ColumnOrder::wavefront, .block_rows = 3},
    {.layout = EdgeLayout::column_blocked,
     .column_order = ColumnOrder::degree_classes,
     .block_rows = 64},
};

TEST(TannerGraph, RepetitionCodeCountsAndDegrees) {
    const Csr h = test::csr_from_dense({{1, 1, 0}, {0, 1, 1}});
    auto g = build(h);
    ASSERT_TRUE(g);
    EXPECT_EQ(g->num_rows(), 2U);
    EXPECT_EQ(g->num_columns(), 3U);
    EXPECT_EQ(g->num_edges(), 4U);
    EXPECT_EQ(g->max_row_degree(), 2U);
    EXPECT_EQ(g->max_column_degree(), 2U);
    EXPECT_EQ(g->empty_rows(), 0U);
    EXPECT_EQ(g->empty_columns(), 0U);
    EXPECT_EQ(g->num_slots(), 2 * row_alignment_slots);
}

TEST(TannerGraph, CountsEmptyRowsAndColumns) {
    const Csr h = test::csr_from_dense({{1, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 0, 0}});
    auto g = build(h);
    ASSERT_TRUE(g);
    EXPECT_EQ(g->empty_rows(), 1U);
    EXPECT_EQ(g->empty_columns(), 2U);
}

// Every layout and order must describe the same matrix: each edge once in the row view and once
// in the column view, the two agreeing on its slot, columns listing rows in ascending order, and
// the internal/external maps being inverse permutations.
TEST(TannerGraph, ViewsDescribeTheSameMatrixInEveryLayout) {
    for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        const Csr h = test::random_csr(40, 150, 0, 7, seed);
        std::set<std::pair<index_t, index_t>> expected;
        for (index_t i = 0; i < h.rows; ++i) {
            for (index_t k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) {
                expected.insert({i, h.col_idx[k]});
            }
        }
        for (const GraphOptions& options : all_options) {
            auto g = build(h, options);
            ASSERT_TRUE(g) << g.error().detail;
            std::set<std::pair<index_t, index_t>> from_rows;
            std::vector<index_t> slot_owner(g->num_slots(), ~index_t{0});
            for (index_t i = 0; i < g->num_rows(); ++i) {
                const auto columns = g->row_columns(i);
                const SlotSequence slots = g->row_slots(i);
                ASSERT_EQ(columns.size(), slots.size());
                for (index_t k = 0; k < slots.size(); ++k) {
                    from_rows.insert({i, g->external_column(columns[k])});
                    ASSERT_LT(slots[k], g->num_slots());
                    EXPECT_EQ(slot_owner[slots[k]], ~index_t{0}) << "slot used twice";
                    slot_owner[slots[k]] = columns[k];
                }
            }
            EXPECT_EQ(from_rows, expected);
            std::set<std::pair<index_t, index_t>> from_columns;
            for (index_t c = 0; c < g->num_columns(); ++c) {
                const auto rows = g->column_rows(c);
                const SlotSequence slots = g->column_slots(c);
                ASSERT_EQ(rows.size(), slots.size());
                EXPECT_TRUE(std::ranges::is_sorted(rows));
                for (index_t k = 0; k < rows.size(); ++k) {
                    from_columns.insert({rows[k], g->external_column(c)});
                    EXPECT_EQ(slot_owner[slots[k]], c) << "row and column views disagree";
                }
                EXPECT_EQ(g->internal_column(g->external_column(c)), c);
            }
            EXPECT_EQ(from_columns, expected);
        }
    }
}

TEST(TannerGraph, RunsAndBlocksPartitionColumnsAndRows) {
    const Csr h = test::random_csr(50, 200, 1, 6, 9);
    for (const GraphOptions& options : all_options) {
        auto g = build(h, options);
        ASSERT_TRUE(g);
        index_t next_column = 0;
        for (const DegreeRun& run : g->runs()) {
            EXPECT_EQ(run.col_begin, next_column);
            EXPECT_LT(run.col_begin, run.col_end);
            for (index_t c = run.col_begin; c < run.col_end; ++c) {
                EXPECT_EQ(g->column_degree(c), run.degree);
            }
            next_column = run.col_end;
        }
        EXPECT_EQ(next_column, g->num_columns());
        index_t next_row = 0;
        std::uint32_t next_run = 0;
        for (const RowBlock& block : g->row_blocks()) {
            EXPECT_EQ(block.row_begin, next_row);
            EXPECT_EQ(block.run_begin, next_run);
            // A column is attached to the block holding its last row.
            for (std::uint32_t r = block.run_begin; r < block.run_end; ++r) {
                const DegreeRun& run = g->runs()[r];
                for (index_t c = run.col_begin; c < run.col_end; ++c) {
                    if (run.degree > 0) {
                        EXPECT_LT(g->column_rows(c).back(), block.row_end);
                        EXPECT_GE(g->column_rows(c).back(), block.row_begin);
                    }
                }
            }
            next_row = block.row_end;
            next_run = block.run_end;
        }
        EXPECT_EQ(next_row, g->num_rows());
        EXPECT_EQ(next_run, g->runs().size());
    }
}

TEST(TannerGraph, DegreeClassesGiveOneRunPerDegree) {
    const Csr h = test::random_csr(30, 300, 1, 5, 4);
    auto g = build(h, {.layout = EdgeLayout::row_major,
                       .column_order = ColumnOrder::degree_classes,
                       .block_rows = 64});
    ASSERT_TRUE(g);
    EXPECT_LE(g->runs().size(), 5U);
    EXPECT_EQ(g->row_blocks().size(), 1U);
}

TEST(TannerGraph, RowMajorRowsAreAlignedAndPaddedWithSentinelColumn) {
    const Csr h = test::random_csr(20, 90, 1, 4, 2);
    auto g = build(h);
    ASSERT_TRUE(g);
    const RowMajorView v = g->row_major_view();
    for (index_t i = 0; i < g->num_rows(); ++i) {
        EXPECT_EQ(v.row_slot_begin[i] % row_alignment_slots, 0U);
        for (index_t s = v.row_slot_begin[i] + g->row_degree(i); s < v.row_slot_begin[i + 1]; ++s) {
            EXPECT_EQ(v.slot_column[s], g->num_columns());
        }
    }
}

TEST(TannerGraph, RejectsMalformedInput) {
    using Code = GraphError::Code;
    const auto code = [](index_t m, index_t n, std::vector<index_t> ptr, std::vector<index_t> idx,
                         GraphOptions options = {}) {
        auto g = TannerGraph::from_csr(m, n, ptr, idx, options);
        return g ? std::optional<Code>{} : std::optional<Code>{g.error().code};
    };
    EXPECT_EQ(code(2, 3, {0, 1}, {0}), Code::row_pointer_size);
    EXPECT_EQ(code(1, 3, {1, 1}, {0}), Code::row_pointer_start);
    EXPECT_EQ(code(2, 3, {0, 2, 1}, {0, 1}), Code::row_pointer_not_monotone);
    EXPECT_EQ(code(1, 3, {0, 2}, {0}), Code::row_pointer_end);
    EXPECT_EQ(code(1, 3, {0, 1}, {3}), Code::column_out_of_range);
    EXPECT_EQ(code(1, 3, {0, 2}, {1, 0}), Code::row_not_strictly_increasing);
    EXPECT_EQ(code(1, 3, {0, 2}, {1, 1}), Code::row_not_strictly_increasing);
    EXPECT_EQ(code(1, 3, {0, 1}, {0}, {EdgeLayout::column_blocked, ColumnOrder::natural, 64}),
              Code::unsupported_options);
    EXPECT_EQ(code(1, 3, {0, 1}, {0}, {EdgeLayout::row_major, ColumnOrder::wavefront, 0}),
              Code::unsupported_options);
    EXPECT_FALSE(code(0, 0, {0}, {}).has_value());
}

} // namespace
