// The relay controller against a scripted backend: the leg schedule, the γ rows each leg gets,
// the stopping rules, the best-solution rule and the iteration accounting, with no numerics.

#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "rtd/core/relay.hpp"

namespace {

using namespace rtd;

constexpr double inf = std::numeric_limits<double>::infinity();

class ScriptedBackend {
public:
    ScriptedBackend(index_t rows, index_t columns, std::vector<LegOutcome> script)
        : rows_(rows), columns_(columns), script_(std::move(script)), hard_(columns, 0) {}

    [[nodiscard]] index_t num_rows() const noexcept { return rows_; }
    [[nodiscard]] index_t num_columns() const noexcept { return columns_; }
    void begin(std::span<const Bit> /*syndrome*/, bool init_marginals) noexcept {
        begun = true;
        marginals_initialised = init_marginals;
        leg_ = 0;
        gammas.clear();
        params.clear();
        marked.clear();
    }
    void set_gamma(std::span<const double> g) noexcept { gammas.push_back(g[0]); }
    void set_gamma(double g) noexcept { gammas.push_back(g); }
    [[nodiscard]] LegOutcome run_leg(const LegParams& p) noexcept {
        params.push_back(p);
        return script_.at(leg_++);
    }
    void mark_best() noexcept {
        marked.push_back(leg_ - 1);
        hard_[0] = static_cast<Bit>(leg_ - 1);
    }
    [[nodiscard]] std::span<const Bit> best_hard() noexcept { return hard_; }
    [[nodiscard]] std::span<const Bit> current_hard() noexcept { return hard_; }
    [[nodiscard]] static std::span<const index_t> best_support() noexcept { return {}; }
    [[nodiscard]] static std::span<const index_t> current_support() noexcept { return {}; }
    void set_convergence_rows(std::span<const Bit> /*mask*/) noexcept {}
    void read_marginals(std::span<double> /*out*/) const noexcept {}

    bool begun = false;
    bool marginals_initialised = false;
    std::vector<double> gammas;
    std::vector<LegParams> params;
    std::vector<std::uint32_t> marked;

private:
    index_t rows_;
    index_t columns_;
    std::vector<LegOutcome> script_;
    std::vector<Bit> hard_;
    std::uint32_t leg_ = 0;
};

static_assert(LegBackend<ScriptedBackend>);

LegOutcome converged(std::uint32_t iterations, double weight) {
    return {.converged = true, .iterations = iterations, .weight = weight};
}
LegOutcome failed(std::uint32_t iterations) {
    return {.converged = false, .iterations = iterations, .weight = inf};
}

// γ rows tagged by index so the test can see which row each leg received.
ExplicitGammaTable table(std::size_t rows) {
    std::vector<double> values;
    for (std::size_t r = 0; r < rows; ++r) {
        values.push_back(static_cast<double>(r) + 0.5);
        values.push_back(0.0);
    }
    return *ExplicitGammaTable::create(values, rows, 2);
}

const MinSumConfig memory{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
const std::vector<Bit> syndrome{1, 0, 1};

TEST(RelayDecoder, StopsAfterLegZeroWhenItConvergesUnderAfterLeg0) {
    const ExplicitGammaTable gammas = table(4);
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {converged(7, 2.0)}), memory,
        {.pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterLeg0{}}, &gammas);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    EXPECT_TRUE(r->success);
    EXPECT_EQ(r->iterations, 7U);
    EXPECT_EQ(r->legs_executed, 1U);
    EXPECT_EQ(r->best_leg, 0U);
    EXPECT_EQ(r->weight, 2.0);
    EXPECT_TRUE(decoder->backend().marginals_initialised);
    EXPECT_EQ(decoder->backend().params[0].max_iter, 10U);
    EXPECT_TRUE(decoder->backend().params[0].use_memory);
    EXPECT_EQ(decoder->backend().gammas, std::vector<double>{0.125});
}

TEST(RelayDecoder, AfterLeg0WithoutConvergenceRunsEveryLeg) {
    const ExplicitGammaTable gammas = table(2);
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {failed(10), converged(3, 5.0), converged(4, 1.0), failed(5)}),
        memory, {.pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterLeg0{}},
        &gammas);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->legs_executed, 4U);
    EXPECT_EQ(r->iterations, 22U);
    EXPECT_EQ(r->best_leg, 2U);
    EXPECT_EQ(r->weight, 1.0);
    // Leg r uses row r mod 2: legs 1, 2, 3 get rows 1, 0, 1.
    EXPECT_EQ(decoder->backend().gammas, (std::vector<double>{0.125, 1.5, 0.5, 1.5}));
    EXPECT_EQ(decoder->backend().marked, (std::vector<std::uint32_t>{0, 1, 2}));
    EXPECT_EQ(decoder->backend().params[1].max_iter, 5U);
    ASSERT_EQ(r->legs.size(), 4U);
    EXPECT_FALSE(r->legs[0].converged);
    EXPECT_TRUE(r->legs[1].became_best);
    EXPECT_TRUE(r->legs[2].became_best);
    EXPECT_FALSE(r->legs[3].became_best);
}

TEST(RelayDecoder, AfterNConvergedCountsLegZeroAndKeepsTheEarliestOfEqualWeights) {
    const ExplicitGammaTable gammas = table(8);
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2,
                        {converged(9, 3.0), failed(5), converged(2, 3.0), converged(1, 0.5)}),
        memory,
        {.pre_iter = 10, .set_max_iter = 5, .num_sets = 6, .stopping = AfterNConverged{2}},
        &gammas);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    // Leg 0 and leg 2 converge (count reaches 2); leg 2 ties leg 0's weight and does not replace it.
    EXPECT_EQ(r->legs_executed, 3U);
    EXPECT_EQ(r->iterations, 16U);
    EXPECT_EQ(r->best_leg, 0U);
    EXPECT_EQ(r->hard[0], 0);
    EXPECT_FALSE(r->legs[2].became_best);
}

TEST(RelayDecoder, NoConvergenceReturnsLegZerosCorrection) {
    const ExplicitGammaTable gammas = table(3);
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {failed(10), failed(5), failed(5)}), memory,
        {.pre_iter = 10, .set_max_iter = 5, .num_sets = 2, .stopping = AfterNConverged{1}},
        &gammas);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    EXPECT_FALSE(r->success);
    EXPECT_FALSE(r->best_leg.has_value());
    EXPECT_EQ(r->weight, inf);
    EXPECT_EQ(r->iterations, 20U);
    EXPECT_EQ(r->hard[0], 0);
    EXPECT_EQ(decoder->backend().marked, std::vector<std::uint32_t>{0});
}

TEST(RelayDecoder, AllLegsNeverStopsEarly) {
    const ExplicitGammaTable gammas = table(3);
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {converged(1, 4.0), converged(1, 3.0), converged(1, 2.0)}), memory,
        {.pre_iter = 10, .set_max_iter = 5, .num_sets = 2, .stopping = AllLegs{}}, &gammas);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    EXPECT_EQ(r->legs_executed, 3U);
    EXPECT_EQ(r->best_leg, 2U);
}

TEST(RelayDecoder, PlainMinSumHasNoMemoryAndOneLeg) {
    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {failed(30)}), MinSumConfig{},
        {.pre_iter = 30, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}, nullptr);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    EXPECT_FALSE(decoder->backend().marginals_initialised);
    EXPECT_FALSE(decoder->backend().params[0].use_memory);
    EXPECT_TRUE(decoder->backend().gammas.empty());
    EXPECT_EQ(r->legs_executed, 1U);
}

TEST(RelayDecoder, RejectsMissingOrMismatchedGammaSourceAndWrongSyndromeSize) {
    const RelayConfig relay{.pre_iter = 10, .set_max_iter = 5, .num_sets = 2,
                            .stopping = AfterLeg0{}};
    auto missing = RelayDecoder<ScriptedBackend>::create(ScriptedBackend(3, 2, {}), memory, relay,
                                                         nullptr);
    ASSERT_FALSE(missing);
    EXPECT_EQ(missing.error().code, ConfigError::Code::missing_gamma_source);
    const ExplicitGammaTable wide = *ExplicitGammaTable::create({0.1, 0.2, 0.3}, 1, 3);
    auto mismatched =
        RelayDecoder<ScriptedBackend>::create(ScriptedBackend(3, 2, {}), memory, relay, &wide);
    ASSERT_FALSE(mismatched);
    EXPECT_EQ(mismatched.error().code, ConfigError::Code::gamma_width_mismatch);

    auto decoder = RelayDecoder<ScriptedBackend>::create(
        ScriptedBackend(3, 2, {failed(1)}), MinSumConfig{},
        {.pre_iter = 1, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}, nullptr);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(std::vector<Bit>{1, 0});
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().code, DecodeError::Code::syndrome_size_mismatch);
    EXPECT_FALSE(decoder->backend().begun);
}

} // namespace
