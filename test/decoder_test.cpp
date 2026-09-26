#include <gtest/gtest.h>

#include "realtimedecoder/decoder.hpp"

TEST(PlaceholderAdd, AddsTwoPositiveNumbers) {
    EXPECT_EQ(realtimedecoder::placeholder_add(2, 3), 5);
}
