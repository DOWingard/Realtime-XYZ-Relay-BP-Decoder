#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <utility>

#include "rtd/core/buffer.hpp"

namespace {

using rtd::AlignedBuffer;

TEST(AlignedBuffer, IsCacheLineAligned) {
    for (const std::size_t n : {1U, 3U, 17U, 1000U}) {
        const AlignedBuffer<float> b(n);
        EXPECT_EQ(std::bit_cast<std::uintptr_t>(b.data()) % 64, 0U);
        EXPECT_EQ(b.size(), n);
    }
}

TEST(AlignedBuffer, EmptyHoldsNothing) {
    const AlignedBuffer<double> b(0);
    EXPECT_TRUE(b.empty());
    EXPECT_EQ(b.data(), nullptr);
    EXPECT_TRUE(b.span().empty());
}

TEST(AlignedBuffer, FillAndCloneCopyValues) {
    AlignedBuffer<int> a(5, 7);
    for (const int x : a) {
        EXPECT_EQ(x, 7);
    }
    a[2] = 3;
    const AlignedBuffer<int> b = a.clone();
    EXPECT_EQ(b[2], 3);
    EXPECT_NE(a.data(), b.data());
}

TEST(AlignedBuffer, MoveTransfersOwnership) {
    AlignedBuffer<int> a(4, 1);
    const int* p = a.data();
    AlignedBuffer<int> b(std::move(a));
    EXPECT_EQ(b.data(), p);
    EXPECT_EQ(b.size(), 4U);
    EXPECT_EQ(a.size(), 0U); // NOLINT(bugprone-use-after-move): moved-from state is specified
    AlignedBuffer<int> c;
    c = std::move(b);
    EXPECT_EQ(c.data(), p);
}

} // namespace
