#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "rtd/core/priors.hpp"

namespace {

using rtd::Priors;
using rtd::PriorsError;

TEST(Priors, LogOddsOfEachProbability) {
    const std::vector<double> p{0.5, 0.001, 0.9, 0.0};
    auto priors = Priors::from_probabilities(p);
    ASSERT_TRUE(priors);
    EXPECT_EQ(priors->size(), 4U);
    EXPECT_EQ(priors->llr()[0], 0.0);
    EXPECT_EQ(priors->llr()[1], std::log((1.0 - 0.001) / 0.001));
    EXPECT_LT(priors->llr()[2], 0.0);
    EXPECT_EQ(priors->llr()[3], std::numeric_limits<double>::infinity());
}

TEST(Priors, RejectsCertainFaultsAndNonProbabilities) {
    const auto code = [](std::vector<double> p) {
        auto priors = Priors::from_probabilities(p);
        return priors ? std::optional<PriorsError::Code>{} : priors.error().code;
    };
    EXPECT_EQ(code({0.1, 1.0}), PriorsError::Code::certain_fault);
    EXPECT_EQ(code({-0.1}), PriorsError::Code::not_a_probability);
    EXPECT_EQ(code({1.5}), PriorsError::Code::not_a_probability);
    EXPECT_EQ(code({std::numeric_limits<double>::quiet_NaN()}),
              PriorsError::Code::not_a_probability);
    auto bad = Priors::from_probabilities(std::vector<double>{0.2, 0.3, 1.0});
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().column, 2U);
}

} // namespace
