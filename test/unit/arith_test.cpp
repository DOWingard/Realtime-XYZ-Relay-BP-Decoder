#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "rtd/core/arith.hpp"

namespace {

using rtd::F32;
using rtd::F64;

template <class A>
class FloatArithTest : public testing::Test {};
using Policies = testing::Types<F32, F64>;
TYPED_TEST_SUITE(FloatArithTest, Policies);

TYPED_TEST(FloatArithTest, InfinitePriorSaturatesToMaxMessage) {
    using A = TypeParam;
    EXPECT_EQ(A::from_llr(std::numeric_limits<double>::infinity()), A::max_message());
    EXPECT_EQ(A::from_llr(2.5), static_cast<typename A::msg_t>(2.5));
}

TYPED_TEST(FloatArithTest, NegativeZeroCountsAsNegative) {
    using A = TypeParam;
    using T = A::msg_t;
    EXPECT_TRUE(A::is_negative(T{-0.0}));
    EXPECT_FALSE(A::is_negative(T{0.0}));
    EXPECT_TRUE(A::is_negative(T{-1.0}));
}

TYPED_TEST(FloatArithTest, WithSignFlipsOnlyTheSignBit) {
    using A = TypeParam;
    using T = A::msg_t;
    EXPECT_TRUE(std::signbit(A::with_sign(T{0.0}, true)));
    EXPECT_EQ(A::with_sign(T{3.0}, true), T{-3.0});
    EXPECT_EQ(A::with_sign(T{3.0}, false), T{3.0});
    EXPECT_EQ(A::magnitude(T{-0.0}), T{0.0});
    EXPECT_FALSE(std::signbit(A::magnitude(T{-0.0})));
}

TYPED_TEST(FloatArithTest, HardDecisionIncludesZero) {
    using A = TypeParam;
    using T = A::acc_t;
    EXPECT_TRUE(A::is_error(T{0.0}));
    EXPECT_TRUE(A::is_error(T{-0.0}));
    EXPECT_TRUE(A::is_error(static_cast<T>(-1e-30)));
    EXPECT_FALSE(A::is_error(static_cast<T>(1e-30)));
    EXPECT_FALSE(A::is_error(std::numeric_limits<T>::quiet_NaN()));
}

template <class T>
using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;

TYPED_TEST(FloatArithTest, MixRoundsEachStepAndKeepsSaturatedPriors) {
    using A = TypeParam;
    using T = A::msg_t;
    const T lambda = static_cast<T>(4.1);
    const T marginal = static_cast<T>(-1.7);
    const T gamma = static_cast<T>(0.3);
    const T keep = T{1} - gamma;
    const T a = lambda * keep;
    const T b = marginal * gamma;
    const T expected = a + b;
    EXPECT_EQ(std::bit_cast<Bits<T>>(A::mix(lambda, marginal, gamma)),
              std::bit_cast<Bits<T>>(expected));
    EXPECT_EQ(A::mix(A::max_message(), marginal, gamma), A::max_message());
}

TEST(FloatArith, ProductsAreNotFusedIntoAdds) {
    // (1 + 2^-23)(1 - 2^-23) = 1 - 2^-46 rounds to 1 in float, so x·y + z with z = -1 is 0 when
    // the product is rounded first and -2^-46 when it is fused. The build must never fuse.
    const float x = 1.0F + 0x1p-23F;
    const float y = 1.0F - 0x1p-23F;
    const float z = -1.0F;
    const float expression = x * y + z;
    EXPECT_EQ(expression, 0.0F);
    EXPECT_EQ(std::fma(x, y, z), -0x1p-46F);
}

} // namespace
