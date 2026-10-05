// The optimised decoder against the naive reference on random problems, in every layout, column
// order, executor and number format; and structural properties every decode must satisfy.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/kernels.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

struct Problem {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::vector<Bit>> syndromes;
};

// A random code with awkward priors: a few p = 0 columns (λ = +∞), a few p > 1/2 (λ < 0), and
// syndromes of errors drawn from the priors (so some are zero).
Problem random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots) {
    Problem problem{.h = test::random_csr(m, n, 1, 6, seed), .p = {}, .syndromes = {}};
    std::mt19937_64 rng(seed * 7919);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        const double u = unit(rng);
        if (u < 0.03) {
            problem.p.push_back(0.0);
        } else if (u < 0.06) {
            problem.p.push_back(0.6);
        } else {
            problem.p.push_back(0.002 + 0.1 * unit(rng));
        }
    }
    for (std::size_t s = 0; s < shots; ++s) {
        std::vector<Bit> error(n, 0);
        for (index_t j = 0; j < n; ++j) {
            error[j] = unit(rng) < problem.p[j] ? 1 : 0;
        }
        problem.syndromes.push_back(test::syndrome_of(problem.h, error));
    }
    return problem;
}

struct Scenario {
    const char* name;
    MinSumConfig min_sum;
    RelayConfig relay;
};

const std::vector<Scenario> scenarios{
    {.name = "min_sum",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "normalised",
     .min_sum = {.alpha = ConstantAlpha{0.8}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{2.5}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "mem_bp",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.2},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "relay_nconv",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 12, .set_max_iter = 8, .num_sets = 15, .stopping = AfterNConverged{3}}},
    {.name = "relay_all_adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 6, .set_max_iter = 6, .num_sets = 8, .stopping = AllLegs{}}},
    {.name = "relay_after_leg0",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = -0.1},
     .relay = {.pre_iter = 5, .set_max_iter = 5, .num_sets = 6, .stopping = AfterLeg0{}}},
};

template <class T>
std::uint64_t bits(T x) {
    if constexpr (sizeof(T) == 4) {
        return std::bit_cast<std::uint32_t>(x);
    } else {
        return std::bit_cast<std::uint64_t>(x);
    }
}

// What the comparisons exercised, so a vacuous agreement (say, nothing ever converging) fails.
struct Coverage {
    std::size_t decodes = 0;
    std::size_t converged = 0;
    std::size_t failed = 0;
    std::size_t multi_leg = 0;
    std::size_t best_after_leg0 = 0;
};

template <class A, class MakeExecutor>
void compare_with_reference(const Problem& problem, const GraphOptions& options,
                            MakeExecutor make_executor, const std::string& label,
                            Coverage& coverage) {
    using Exec = decltype(make_executor());
    using T = A::msg_t;
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, options);
    ASSERT_TRUE(graph) << graph.error().detail;
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(5, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    test::ReferenceDecoder<T> reference(problem.h, priors->llr());
    std::vector<double> marginals(problem.h.cols);

    for (const Scenario& scenario : scenarios) {
        auto backend = CpuBackend<A, Exec>::create(*graph, *priors, make_executor());
        ASSERT_TRUE(backend);
        auto decoder = RelayDecoder<CpuBackend<A, Exec>>::create(
            std::move(*backend), scenario.min_sum, scenario.relay, &*gammas);
        ASSERT_TRUE(decoder) << decoder.error().detail;
        for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
            const auto& syndrome = problem.syndromes[s];
            const test::ReferenceResult want =
                reference.decode(syndrome, scenario.min_sum, scenario.relay, &*gammas, s);
            auto got = decoder->decode(syndrome, s);
            ASSERT_TRUE(got);
            const std::string where = std::format("{} {} shot {}", label, scenario.name, s);
            ASSERT_EQ(got->success, want.success) << where;
            ASSERT_EQ(got->iterations, want.iterations) << where;
            ASSERT_EQ(got->legs_executed, want.legs) << where;
            ASSERT_EQ(got->best_leg, want.best_leg) << where;
            ASSERT_EQ(std::bit_cast<std::uint64_t>(got->weight),
                      std::bit_cast<std::uint64_t>(want.weight))
                << where;
            ASSERT_TRUE(std::ranges::equal(got->hard, want.hard)) << where;
            for (std::uint32_t leg = 0; leg < got->legs_executed; ++leg) {
                ASSERT_EQ(got->legs[leg].iterations, want.leg_iterations[leg]) << where;
                ASSERT_EQ(got->legs[leg].converged, want.leg_converged[leg]) << where;
            }
            decoder->backend().read_marginals(marginals);
            for (index_t j = 0; j < problem.h.cols; ++j) {
                ASSERT_EQ(bits(static_cast<T>(marginals[j])), bits(static_cast<T>(want.marginals[j])))
                    << where << " column " << j;
            }

            ++coverage.decodes;
            coverage.converged += got->success ? 1U : 0U;
            coverage.failed += got->success ? 0U : 1U;
            coverage.multi_leg += got->legs_executed > 1 ? 1U : 0U;
            coverage.best_after_leg0 += got->best_leg.value_or(0) > 0 ? 1U : 0U;

            // Properties of every result.
            if (got->success) {
                EXPECT_TRUE(kernels::syndrome_matches(*graph, got->hard, syndrome)) << where;
            }
            std::uint32_t total = 0;
            for (const LegRecord& leg : got->legs) {
                total += leg.iterations;
                if (leg.converged) {
                    EXPECT_LE(got->weight, leg.weight) << where;
                }
            }
            EXPECT_EQ(total, got->iterations) << where;
        }
    }
}

struct Configuration {
    GraphOptions options;
    unsigned team;
};

const std::vector<Configuration> configurations{
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 4},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::degree_classes,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::natural,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::column_blocked,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 5},
     .team = 0},
    {.options = {.layout = EdgeLayout::column_blocked,
                 .column_order = ColumnOrder::degree_classes,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 4},
     .team = 3},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::natural,
                 .block_rows = 64},
     .team = 2},
    {.options = {.layout = EdgeLayout::column_blocked,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 3},
     .team = 4},
};

class AgainstReference : public testing::TestWithParam<std::size_t> {};

TEST_P(AgainstReference, FloatAndDoubleMatchBitForBit) {
    const Configuration& config = configurations[GetParam()];
    Coverage coverage;
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const Problem problem = random_problem(seed, 36, 150, 12);
        const std::string label = std::format("config {} seed {}", GetParam(), seed);
        if (config.team == 0) {
            const auto serial = [] { return Serial{}; };
            compare_with_reference<F32>(problem, config.options, serial, label, coverage);
            compare_with_reference<F64>(problem, config.options, serial, label, coverage);
        } else {
            const auto team = [&] { return Team(config.team); };
            compare_with_reference<F32>(problem, config.options, team, label, coverage);
            compare_with_reference<F64>(problem, config.options, team, label, coverage);
        }
    }
    EXPECT_GT(coverage.converged, coverage.decodes / 10);
    EXPECT_GT(coverage.failed, 0U);
    EXPECT_GT(coverage.multi_leg, coverage.decodes / 10);
    EXPECT_GT(coverage.best_after_leg0, 0U);
    std::cout << std::format("[ coverage ] {} decodes: {} converged, {} failed, {} multi-leg, {} "
                             "best after leg 0\n",
                             coverage.decodes, coverage.converged, coverage.failed,
                             coverage.multi_leg, coverage.best_after_leg0);
}

INSTANTIATE_TEST_SUITE_P(AllConfigurations, AgainstReference,
                         testing::Range<std::size_t>(0, configurations.size()));

TEST(Decode, ZeroSyndromeDecodesToZeroInOneIteration) {
    const test::Csr h = test::random_csr(20, 80, 1, 5, 3);
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    auto priors = Priors::from_probabilities(std::vector<double>(h.cols, 0.01));
    ASSERT_TRUE(graph && priors);
    auto decoder = CpuRelayDecoder<F32>::create(
        *CpuBackend<F32>::create(*graph, *priors), MinSumConfig{},
        {.pre_iter = 50, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}, nullptr);
    ASSERT_TRUE(decoder);
    auto r = decoder->decode(std::vector<Bit>(h.rows, 0));
    ASSERT_TRUE(r);
    EXPECT_TRUE(r->success);
    EXPECT_EQ(r->iterations, 1U);
    EXPECT_EQ(r->weight, 0.0);
    EXPECT_TRUE(std::ranges::all_of(r->hard, [](Bit b) { return b == 0; }));
}

// The 3-bit repetition code with weight-2 checks: each single-fault syndrome has that fault as
// its unique most likely explanation.
TEST(Decode, RepetitionCodeSingleFaults) {
    const test::Csr h = test::csr_from_dense({{1, 1, 0}, {0, 1, 1}});
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    auto priors = Priors::from_probabilities(std::vector<double>{0.003, 0.003, 0.003});
    ASSERT_TRUE(graph && priors);
    auto decoder = CpuRelayDecoder<F32>::create(
        *CpuBackend<F32>::create(*graph, *priors), MinSumConfig{},
        {.pre_iter = 10, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}, nullptr);
    ASSERT_TRUE(decoder);
    const std::vector<std::pair<std::vector<Bit>, std::vector<Bit>>> cases{
        {{1, 0}, {1, 0, 0}}, {{1, 1}, {0, 1, 0}}, {{0, 1}, {0, 0, 1}}, {{0, 0}, {0, 0, 0}}};
    for (const auto& [syndrome, error] : cases) {
        auto r = decoder->decode(syndrome);
        ASSERT_TRUE(r);
        EXPECT_TRUE(r->success);
        EXPECT_TRUE(std::ranges::equal(r->hard, error));
    }
}

TEST(Decode, TracerSeesEveryIteration) {
    const test::Csr h = test::random_csr(15, 60, 1, 4, 8);
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    auto priors = Priors::from_probabilities(std::vector<double>(h.cols, 0.05));
    ASSERT_TRUE(graph && priors);
    auto backend = CpuBackend<F32, Serial, MarginalTracer>::create(*graph, *priors);
    ASSERT_TRUE(backend);
    auto decoder = RelayDecoder<CpuBackend<F32, Serial, MarginalTracer>>::create(
        std::move(*backend), MinSumConfig{},
        {.pre_iter = 7, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}, nullptr);
    ASSERT_TRUE(decoder);
    std::vector<Bit> syndrome(h.rows, 0);
    syndrome.at(0) = 1;
    auto r = decoder->decode(syndrome);
    ASSERT_TRUE(r);
    const auto& steps = decoder->backend().tracer().steps();
    ASSERT_EQ(steps.size(), r->iterations);
    std::vector<double> final_marginals(h.cols);
    decoder->backend().read_marginals(final_marginals);
    EXPECT_EQ(steps.back().marginals, final_marginals);
}

} // namespace
