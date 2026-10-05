#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "rtd/core/observables.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

TEST(SparseBinaryMatrix, ProductMatchesDefinition) {
    const test::Csr a = test::random_csr(12, 500, 0, 5, 3);
    auto matrix = SparseBinaryMatrix::from_csr(a.rows, a.cols, a.row_ptr, a.col_idx);
    ASSERT_TRUE(matrix);
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<Bit> x(a.cols, 0);
        std::vector<index_t> support;
        for (index_t j = 0; j < a.cols; ++j) {
            if (rng() % 17 == 0) {
                x[j] = 1;
                support.push_back(j);
            }
        }
        const std::vector<Bit> expected = test::syndrome_of(a, x);
        std::vector<Bit> dense(a.rows, 9);
        matrix->apply(x, dense);
        EXPECT_EQ(dense, expected);
        std::vector<Bit> sparse(a.rows, 9);
        std::ranges::shuffle(support, rng);
        matrix->apply_support(support, sparse);
        EXPECT_EQ(sparse, expected);
    }
}

TEST(SparseBinaryMatrix, ZeroVectorGivesZero) {
    const test::Csr a = test::csr_from_dense({{1, 1, 0}, {0, 1, 1}});
    auto matrix = SparseBinaryMatrix::from_csr(a.rows, a.cols, a.row_ptr, a.col_idx);
    ASSERT_TRUE(matrix);
    std::vector<Bit> out(2, 1);
    matrix->apply(std::vector<Bit>(3, 0), out);
    EXPECT_EQ(out, (std::vector<Bit>{0, 0}));
    EXPECT_EQ(matrix->column(1).size(), 2U);
}

TEST(SparseBinaryMatrix, RejectsInvalidCsr) {
    const std::vector<index_t> ptr{0, 1};
    const std::vector<index_t> idx{5};
    EXPECT_FALSE(SparseBinaryMatrix::from_csr(1, 3, ptr, idx));
}

} // namespace
