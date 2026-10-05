#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "rtd/core/gamma.hpp"

namespace {

using namespace rtd;

TEST(ExplicitGammaTable, LegUsesRowLegModuloRows) {
    std::vector<double> table{0.0, 0.1, 1.0, 1.1, 2.0, 2.1};
    auto source = ExplicitGammaTable::create(table, 3, 2);
    ASSERT_TRUE(source);
    std::vector<double> scratch(2);
    // Leg 1 uses row 1; leg 3 wraps to row 0.
    EXPECT_EQ(source->gammas(0, 1, scratch)[0], 1.0);
    EXPECT_EQ(source->gammas(0, 2, scratch)[1], 2.1);
    EXPECT_EQ(source->gammas(0, 3, scratch)[0], 0.0);
    EXPECT_EQ(source->gammas(99, 4, scratch)[1], 1.1);
    EXPECT_TRUE(source->reuses_rows(4));
    EXPECT_FALSE(source->reuses_rows(3));
}

TEST(ExplicitGammaTable, RejectsBadShapesAndValues) {
    EXPECT_FALSE(ExplicitGammaTable::create({}, 0, 2));
    EXPECT_FALSE(ExplicitGammaTable::create({0.1, 0.2, 0.3}, 2, 2));
    EXPECT_FALSE(
        ExplicitGammaTable::create({0.1, std::numeric_limits<double>::quiet_NaN()}, 1, 2));
}

TEST(UniformGammaGenerator, DeterministicInRangeAndIndependentPerStreamAndLeg) {
    auto source = UniformGammaGenerator::create(42, -0.24, 0.66, 5000);
    ASSERT_TRUE(source);
    std::vector<double> a(5000);
    std::vector<double> b(5000);
    std::vector<double> c(5000);
    const std::vector<double> first(source->gammas(3, 1, a).begin(), source->gammas(3, 1, a).end());
    const auto again = source->gammas(3, 1, b);
    EXPECT_TRUE(std::ranges::equal(first, again));
    const auto other_leg = source->gammas(3, 2, c);
    EXPECT_FALSE(std::ranges::equal(first, other_leg));
    const auto other_stream = source->gammas(4, 1, b);
    EXPECT_FALSE(std::ranges::equal(first, other_stream));
    double sum = 0.0;
    for (const double g : first) {
        EXPECT_GE(g, -0.24);
        EXPECT_LT(g, 0.66);
        sum += g;
    }
    // Mean of U[-0.24, 0.66) is 0.21 with standard error 0.26/sqrt(5000) ≈ 0.0037.
    EXPECT_NEAR(sum / 5000.0, 0.21, 0.02);
}

TEST(UniformGammaGenerator, RejectsEmptyInterval) {
    EXPECT_FALSE(UniformGammaGenerator::create(1, 0.5, 0.5, 10));
    EXPECT_FALSE(UniformGammaGenerator::create(1, 0.5, 0.1, 10));
    EXPECT_FALSE(UniformGammaGenerator::create(1, 0.0, 1.0, 0));
}

} // namespace
