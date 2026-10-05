#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "rtd/core/config.hpp"

namespace {

using namespace rtd;

TEST(AlphaRule, ConstantAndAdaptiveValues) {
    EXPECT_EQ(alpha_at(ConstantAlpha{0.75}, 0), 0.75);
    EXPECT_EQ(alpha_at(ConstantAlpha{0.75}, 40), 0.75);
    // 1 − 2^(−(t+1)/s): t = 0, s = 1 → 1/2; t = 1 → 3/4.
    EXPECT_EQ(alpha_at(AdaptiveAlpha{1.0}, 0), 0.5);
    EXPECT_EQ(alpha_at(AdaptiveAlpha{1.0}, 1), 0.75);
    EXPECT_EQ(alpha_at(AdaptiveAlpha{2.0}, 1), 0.5);
    EXPECT_TRUE(is_unit_alpha(ConstantAlpha{1.0}));
    EXPECT_FALSE(is_unit_alpha(ConstantAlpha{0.9}));
    EXPECT_FALSE(is_unit_alpha(AdaptiveAlpha{1.0}));
}

TEST(Validate, AcceptsThePaperConfiguration) {
    const MinSumConfig ms{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig rc{.pre_iter = 80, .set_max_iter = 60, .num_sets = 600,
                         .stopping = AfterNConverged{5}};
    EXPECT_TRUE(validate(ms, rc));
    EXPECT_EQ(max_legs(rc), 601U);
}

TEST(Validate, RejectsDegenerateConfigurations) {
    using Code = ConfigError::Code;
    const auto code = [](MinSumConfig ms, RelayConfig rc) {
        auto v = validate(ms, rc);
        return v ? std::optional<Code>{} : v.error().code;
    };
    const RelayConfig plain{.pre_iter = 10, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}};
    EXPECT_FALSE(code({}, plain).has_value());
    EXPECT_EQ(code({.alpha = ConstantAlpha{0.0}, .gamma0 = {}}, plain), Code::invalid_alpha);
    EXPECT_EQ(code({.alpha = ConstantAlpha{-1.0}, .gamma0 = {}}, plain), Code::invalid_alpha);
    EXPECT_EQ(code({.alpha = AdaptiveAlpha{0.0}, .gamma0 = {}}, plain), Code::invalid_alpha);
    EXPECT_EQ(code({.alpha = ConstantAlpha{std::numeric_limits<double>::quiet_NaN()}, .gamma0 = {}},
                   plain),
              Code::invalid_alpha);
    EXPECT_EQ(code({.alpha = ConstantAlpha{1.0}, .gamma0 = std::numeric_limits<double>::infinity()},
                   plain),
              Code::invalid_gamma0);
    RelayConfig relay = plain;
    relay.num_sets = 3;
    relay.set_max_iter = 5;
    EXPECT_EQ(code({}, relay), Code::relay_without_memory);
    relay.set_max_iter = 0;
    EXPECT_EQ(code({.alpha = ConstantAlpha{1.0}, .gamma0 = 0.1}, relay), Code::zero_set_max_iter);
    RelayConfig zero = plain;
    zero.pre_iter = 0;
    EXPECT_EQ(code({}, zero), Code::zero_pre_iter);
    RelayConfig nconv = plain;
    nconv.stopping = AfterNConverged{0};
    EXPECT_EQ(code({}, nconv), Code::zero_stop_count);
}

} // namespace
