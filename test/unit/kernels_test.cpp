#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "rtd/core/buffer.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/kernels.hpp"
#include "support.hpp"

namespace {

using namespace rtd;
using kernels::Source;

template <class T>
auto bits(T x) {
    if constexpr (sizeof(T) == 4) {
        return std::bit_cast<std::uint32_t>(x);
    } else {
        return std::bit_cast<std::uint64_t>(x);
    }
}

// Values that exercise every branch of the check update: signed zeros, repeated minima,
// saturated magnitudes, NaN and ordinary numbers.
template <class T>
T adversarial(std::mt19937_64& rng) {
    const T max = std::numeric_limits<T>::max();
    switch (rng() % 10) {
    case 0:
        return T{0.0};
    case 1:
        return T{-0.0};
    case 2:
        return max;
    case 3:
        return -max;
    case 4:
        return T{1.5};
    case 5:
        return T{-1.5};
    case 6:
        return std::numeric_limits<T>::quiet_NaN();
    default:
        return static_cast<T>(static_cast<double>(rng() % 2001) / 100.0 - 10.0);
    }
}

template <class T>
class CheckRowTest : public testing::Test {};
using FloatTypes = testing::Types<float, double>;
TYPED_TEST_SUITE(CheckRowTest, FloatTypes);

TYPED_TEST(CheckRowTest, VectorisedMatchesScalarBitForBit) {
    using T = TypeParam;
    using A = FloatArith<T>;
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 2000; ++trial) {
        const auto degree = static_cast<index_t>(1 + rng() % 70);
        const bool syndrome = (rng() & 1U) != 0;
        const T alpha = (rng() & 1U) != 0 ? T{1} : static_cast<T>(0.625);
        const std::size_t padded = round_up(degree, row_alignment_slots);
        AlignedBuffer<T> input(padded, A::max_message());
        for (index_t k = 0; k < degree; ++k) {
            input[k] = adversarial<T>(rng);
        }
        std::vector<T> expected(input.begin(), input.end());
        kernels::check_row_scalar<A, false>(expected.data(), degree, syndrome, alpha,
                                            [&](index_t k) { return input[k]; });
        AlignedBuffer<T> actual = input.clone();
        kernels::check_row_simd<T, false, Source::messages>(actual.data(), nullptr, nullptr,
                                                            degree, syndrome, alpha);
        for (std::size_t k = 0; k < padded; ++k) {
            ASSERT_EQ(bits(actual[k]), bits(expected[k])) << "trial " << trial << " k " << k;
        }
        // Gathered form (column-blocked layout): scattered slots, same answer.
        std::vector<index_t> slots(degree);
        for (index_t k = 0; k < degree; ++k) {
            slots[k] = 3 * k + 1;
        }
        std::vector<T> msg(3 * std::size_t{degree} + 2, T{7});
        for (index_t k = 0; k < degree; ++k) {
            msg[slots[k]] = input[k];
        }
        AlignedBuffer<T> buffer(padded + 16);
        kernels::check_row_gathered_simd<T, false, Source::messages>(
            msg.data(), slots.data(), nullptr, nullptr, degree, syndrome, alpha, buffer.data());
        for (index_t k = 0; k < degree; ++k) {
            ASSERT_EQ(bits(msg[slots[k]]), bits(expected[k])) << "trial " << trial;
        }
    }
}

// σ_i = 0, inputs (+2, -3, +5): parity of negatives is 1, min1 = 2 (at edge 0), min2 = 3.
TEST(CheckRow, HandComputedExample) {
    using A = F32;
    std::vector<float> row{2.0F, -3.0F, 5.0F};
    const std::vector<float> in = row;
    kernels::check_row_scalar<A, true>(row.data(), 3, false, 1.0F,
                                       [&](index_t k) { return in[k]; });
    // Edge 0 is the minimum: gets min2 = 3; sign = parity(1) xor own(0) = negative.
    EXPECT_EQ(row[0], -3.0F);
    // Edge 1: gets min1 = 2; sign = 1 xor 1 = positive.
    EXPECT_EQ(row[1], 2.0F);
    // Edge 2: gets min1 = 2; sign = 1 xor 0 = negative.
    EXPECT_EQ(row[2], -2.0F);
}

TEST(CheckRow, DegreeOneSendsScaledMaximum) {
    using A = F32;
    std::vector<float> row{0.25F};
    kernels::check_row_scalar<A, false>(row.data(), 1, true, 0.5F,
                                        [](index_t) { return 0.25F; });
    EXPECT_EQ(row[0], -(0.5F * std::numeric_limits<float>::max()));
}

TEST(CheckRow, RepeatedMinimumGoesToEveryEdge) {
    using A = F32;
    std::vector<float> row{1.0F, -1.0F, 4.0F};
    const std::vector<float> in = row;
    kernels::check_row_scalar<A, true>(row.data(), 3, false, 1.0F,
                                       [&](index_t k) { return in[k]; });
    EXPECT_EQ(std::fabs(row[0]), 1.0F);
    EXPECT_EQ(std::fabs(row[1]), 1.0F);
    EXPECT_EQ(std::fabs(row[2]), 1.0F);
}

TEST(CheckRows, EmptyRowsAreSkippedAndPaddingKeepsMaximum) {
    const test::Csr h = test::csr_from_dense({{1, 1, 0}, {0, 0, 0}, {0, 1, 1}});
    auto g = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    ASSERT_TRUE(g);
    const RowMajorView v = g->row_major_view();
    const float max = F32::max_message();
    AlignedBuffer<float> msg(g->num_slots() + 64, max);
    AlignedBuffer<float> lambda(g->num_columns() + 64, max);
    lambda[0] = 1.0F;
    lambda[1] = 2.0F;
    lambda[2] = 3.0F;
    const std::vector<Bit> syndrome{1, 0, 0};
    kernels::check_rows<F32, true, Source::priors>(v, syndrome.data(), 1.0F, lambda.data(),
                                                  msg.data(), 0, g->num_rows());
    for (index_t i = 0; i < g->num_rows(); ++i) {
        for (index_t s = v.row_slot_begin[i] + g->row_degree(i); s < v.row_slot_begin[i + 1]; ++s) {
            EXPECT_EQ(msg[s], max) << "padding slot " << s;
        }
    }
    // Row 0 (σ = 1): inputs λ0 = 1, λ1 = 2; both positive, so every output is negative.
    const SlotSequence row0 = g->row_slots(0);
    for (index_t k = 0; k < row0.size(); ++k) {
        EXPECT_LT(msg[row0[k]], 0.0F);
    }
}

// Variable update against the specification on a single column, including the +0 that the
// final suffix adds.
TEST(VariableRun, PrefixSuffixOrderAndSignedZero) {
    const test::Csr h = test::csr_from_dense({{1}, {1}, {1}});
    for (const auto layout : {EdgeLayout::row_major, EdgeLayout::column_blocked}) {
        auto g = TannerGraph::from_csr(
            h.rows, h.cols, h.row_ptr, h.col_idx,
            {.layout = layout, .column_order = ColumnOrder::degree_classes, .block_rows = 64});
        ASSERT_TRUE(g);
        AlignedBuffer<float> msg(g->num_slots() + 64, 0.0F);
        const SlotSequence slots = g->column_slots(0);
        const std::vector<float> eta{-0.0F, 1.0F, -0.0F};
        for (index_t k = 0; k < 3; ++k) {
            msg[slots[k]] = eta[k];
        }
        AlignedBuffer<float> lambda(64, 0.0F);
        lambda[0] = -0.0F;
        AlignedBuffer<float> gamma(64, 0.0F);
        AlignedBuffer<float> marginal(64, 0.0F);
        AlignedBuffer<float> scratch(64);
        AlignedBuffer<index_t> support(8);
        std::vector<std::uint64_t> syndrome_bits(1, 0);
        kernels::HardDecisionSink sink{.support = support.data(),
                                       .count = 0,
                                       .syndrome_bits = syndrome_bits.data(),
                                       .external = g->external_columns().data()};
        const DegreeRun& run = g->runs()[0];
        if (layout == EdgeLayout::row_major) {
            kernels::variable_run<F32, false>(g->row_major_view(), run, run.col_begin, run.col_end,
                                              lambda.data(), gamma.data(), marginal.data(),
                                              msg.data(), scratch.data(), sink);
        } else {
            kernels::variable_run_blocked<F32, false>(
                g->column_blocked_view(), run, run.col_begin, run.col_end, lambda.data(),
                gamma.data(), marginal.data(), msg.data(), scratch.data(), sink);
        }
        // Forward: s = −0; μ1 = −0, s = −0 + −0 = −0; μ2 = −0, s = −0 + 1 = 1; μ3 = 1, s = 1 + −0 = 1.
        // Reverse: t = +0; μ3 = 1 + 0 = 1, t = 0 + −0 = +0; μ2 = −0 + 0 = +0, t = 0 + 1 = 1;
        //          μ1 = −0 + 1 = 1.
        EXPECT_EQ(marginal[0], 1.0F);
        EXPECT_EQ(bits(msg[slots[0]]), bits(1.0F));
        EXPECT_EQ(bits(msg[slots[1]]), bits(0.0F)) << "−0 + +0 must give +0";
        EXPECT_EQ(bits(msg[slots[2]]), bits(1.0F));
        EXPECT_EQ(sink.count, 0U);
    }
}

TEST(Kernels, SyndromeMatchesAndWeight) {
    const test::Csr h = test::csr_from_dense({{1, 1, 0}, {0, 1, 1}});
    auto g = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    ASSERT_TRUE(g);
    EXPECT_TRUE(kernels::syndrome_matches(*g, std::vector<Bit>{0, 1, 0}, std::vector<Bit>{1, 1}));
    EXPECT_FALSE(kernels::syndrome_matches(*g, std::vector<Bit>{1, 0, 0}, std::vector<Bit>{1, 1}));
    const std::vector<double> llr{1.0, std::numeric_limits<double>::infinity(), 2.5};
    EXPECT_EQ(kernels::solution_weight(llr, std::vector<index_t>{0, 1, 2}), 3.5);
}

} // namespace
