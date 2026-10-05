// The convergence mask: a leg stops as soon as H·ê = σ holds on the masked rows.
//
// An empty or all-ones mask must leave every decode bit-identical to the default. A partial mask
// only weakens the stopping test, so along the part of the decode where both runs are in the same
// state it stops no later, and whatever it returns satisfies the masked rows.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/kernels.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

struct Problem {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::vector<Bit>> syndromes;
};

Problem random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots) {
    Problem problem{.h = test::random_csr(m, n, 1, 6, seed), .p = {}, .syndromes = {}};
    std::mt19937_64 rng(seed * 7121);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        // Mostly small rates, with a few p = 0 columns (λ = +∞) and p > 1/2 columns (λ < 0).
        const double u = unit(rng);
        double p = 0.6;
        if (u < 0.03) {
            p = 0.0;
        } else if (u >= 0.06) {
            p = 0.002 + 0.1 * unit(rng);
        }
        problem.p.push_back(p);
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
    {.name = "adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{2.5}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "relay_nconv",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 12, .set_max_iter = 8, .num_sets = 15, .stopping = AfterNConverged{3}}},
    {.name = "relay_all",
     .min_sum = {.alpha = ConstantAlpha{0.9}, .gamma0 = 0.125},
     .relay = {.pre_iter = 6, .set_max_iter = 6, .num_sets = 8, .stopping = AllLegs{}}},
};

struct Configuration {
    GraphOptions options;
    unsigned team;
};

const std::vector<Configuration> configurations{
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 4},
     .team = 0},
    {.options = {.layout = EdgeLayout::column_blocked,
                 .column_order = ColumnOrder::degree_classes,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::natural,
                 .block_rows = 64},
     .team = 0},
    {.options = {.layout = EdgeLayout::row_major,
                 .column_order = ColumnOrder::wavefront,
                 .block_rows = 4},
     .team = 3},
};

// H·ê ⊕ σ, row by row, straight from the CSR.
std::vector<Bit> residual(const test::Csr& h, std::span<const Bit> hard,
                          std::span<const Bit> syndrome) {
    std::vector<Bit> r = test::syndrome_of(h, hard);
    for (index_t i = 0; i < h.rows; ++i) {
        r[i] ^= syndrome[i];
    }
    return r;
}

double naive_weight(std::span<const double> llr, std::span<const Bit> hard) {
    double w = 0.0;
    for (std::size_t j = 0; j < hard.size(); ++j) {
        if (hard[j] != 0 && std::isfinite(llr[j])) {
            w += llr[j];
        }
    }
    return w;
}

// Everything observable about a decode, copied out of the decoder's buffers.
struct Snapshot {
    DecodeResult result;
    std::vector<LegRecord> legs;
    std::vector<Bit> hard;
    std::vector<index_t> support;
    std::vector<double> marginals;
};

template <class Decoder>
Snapshot decode(Decoder& decoder, std::span<const Bit> syndrome, std::uint64_t stream) {
    auto r = decoder.decode(syndrome, stream);
    EXPECT_TRUE(r);
    Snapshot snap{.result = *r,
                  .legs = {r->legs.begin(), r->legs.end()},
                  .hard = {r->hard.begin(), r->hard.end()},
                  .support = {r->support.begin(), r->support.end()},
                  .marginals = std::vector<double>(decoder.num_columns())};
    decoder.backend().read_marginals(snap.marginals);
    return snap;
}

void expect_identical(const Snapshot& a, const Snapshot& b, const std::string& where) {
    EXPECT_EQ(a.result.success, b.result.success) << where;
    EXPECT_EQ(a.result.iterations, b.result.iterations) << where;
    EXPECT_EQ(a.result.legs_executed, b.result.legs_executed) << where;
    EXPECT_EQ(a.result.best_leg, b.result.best_leg) << where;
    EXPECT_EQ(bits_of(a.result.weight), bits_of(b.result.weight)) << where;
    EXPECT_EQ(a.result.cap_hit, b.result.cap_hit) << where;
    EXPECT_EQ(a.hard, b.hard) << where;
    EXPECT_EQ(a.support, b.support) << where;
    ASSERT_EQ(a.legs.size(), b.legs.size()) << where;
    for (std::size_t i = 0; i < a.legs.size(); ++i) {
        EXPECT_EQ(a.legs[i].iterations, b.legs[i].iterations) << where;
        EXPECT_EQ(a.legs[i].converged, b.legs[i].converged) << where;
        EXPECT_EQ(a.legs[i].became_best, b.legs[i].became_best) << where;
        EXPECT_EQ(bits_of(a.legs[i].weight), bits_of(b.legs[i].weight)) << where;
    }
    ASSERT_EQ(a.marginals.size(), b.marginals.size()) << where;
    for (std::size_t j = 0; j < a.marginals.size(); ++j) {
        ASSERT_EQ(bits_of(a.marginals[j]), bits_of(b.marginals[j])) << where << " column " << j;
    }
}

struct MaskCoverage {
    std::size_t decodes = 0;
    std::size_t earlier = 0;          // the masked run stopped some leg strictly earlier
    std::size_t masked_only = 0;      // converged under the mask but not without it
    std::size_t violates_full = 0;    // masked answer leaves a masked-out row unsatisfied
};

// One syndrome decoded with and without the partial mask.
void check_partial(const Snapshot& got, const Snapshot& want, const test::Csr& h,
                   std::span<const Bit> syndrome, std::span<const Bit> mask,
                   std::span<const double> llr, bool single_leg, const std::string& where,
                   MaskCoverage& coverage) {
    ++coverage.decodes;
    // Along the common prefix both runs are in the same state, so the weaker test fires no
    // later; once some leg stops earlier the runs diverge and nothing more follows.
    const std::size_t common = std::min(got.legs.size(), want.legs.size());
    for (std::size_t leg = 0; leg < common; ++leg) {
        const LegRecord& g = got.legs[leg];
        const LegRecord& w = want.legs[leg];
        ASSERT_LE(g.iterations, w.iterations) << where << " leg " << leg;
        if (w.converged) {
            ASSERT_TRUE(g.converged) << where << " leg " << leg;
        }
        if (g.iterations < w.iterations) {
            ASSERT_TRUE(g.converged) << where << " leg " << leg;
            ++coverage.earlier;
            break;
        }
    }
    if (single_leg) {
        EXPECT_LE(got.result.iterations, want.result.iterations) << where;
    }
    if (want.result.success) {
        EXPECT_TRUE(got.result.success) << where;
    }
    if (got.result.success) {
        coverage.masked_only += want.result.success ? 0U : 1U;
        const std::vector<Bit> r = residual(h, got.hard, syndrome);
        bool outside = false;
        for (index_t i = 0; i < h.rows; ++i) {
            if (mask[i] != 0) {
                ASSERT_EQ(r[i], 0) << where << " row " << i;
            } else {
                outside = outside || r[i] != 0;
            }
        }
        coverage.violates_full += outside ? 1U : 0U;
        const double weight = naive_weight(llr, got.hard);
        EXPECT_EQ(bits_of(got.result.weight), bits_of(weight)) << where;
    }
    EXPECT_TRUE(std::ranges::is_sorted(got.support)) << where;
    std::vector<Bit> dense(h.cols, 0);
    for (const index_t j : got.support) {
        dense[j] = 1;
    }
    EXPECT_EQ(dense, got.hard) << where;
}

template <class A, class Exec, class MakeExecutor>
void check_masks(const Problem& problem, const GraphOptions& options, MakeExecutor make_executor,
                 const std::string& label, MaskCoverage& coverage) {
    using Decoder = RelayDecoder<CpuBackend<A, Exec>>;
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, options);
    ASSERT_TRUE(graph) << graph.error().detail;
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(5, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    const index_t m = problem.h.rows;
    const std::vector<Bit> all_ones(m, 1);
    // Requires the first half of the rows and every third row after it.
    std::vector<Bit> partial(m, 0);
    for (index_t i = 0; i < m; ++i) {
        partial[i] = (i < m / 2 || i % 3 == 0) ? 1 : 0;
    }

    for (const Scenario& scenario : scenarios) {
        const auto make = [&] {
            auto d = Decoder::create(*CpuBackend<A, Exec>::create(*graph, *priors, make_executor()),
                                     scenario.min_sum, scenario.relay, &*gammas);
            EXPECT_TRUE(d);
            return std::move(*d);
        };
        Decoder plain = make();
        Decoder empty = make();
        Decoder ones = make();
        Decoder masked = make();
        empty.set_convergence_rows({});
        ones.set_convergence_rows(all_ones);
        masked.set_convergence_rows(partial);

        for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
            const auto& syndrome = problem.syndromes[s];
            const std::string where = std::format("{} {} shot {}", label, scenario.name, s);
            const Snapshot want = decode(plain, syndrome, s);
            expect_identical(decode(empty, syndrome, s), want, where + " empty mask");
            expect_identical(decode(ones, syndrome, s), want, where + " all-ones mask");
            check_partial(decode(masked, syndrome, s), want, problem.h, syndrome, partial,
                          priors->llr(), scenario.relay.num_sets == 0, where, coverage);
        }
    }
}

TEST(ConvergenceRows, EmptyAndAllOnesMatchTheDefaultAndPartialStopsNoLater) {
    MaskCoverage coverage;
    for (std::size_t c = 0; c < configurations.size(); ++c) {
        const Configuration& config = configurations[c];
        for (std::uint64_t seed = 1; seed <= 2; ++seed) {
            const Problem problem = random_problem(seed, 36, 150, 12);
            const std::string label = std::format("config {} seed {}", c, seed);
            if (config.team == 0) {
                const auto serial = [] { return Serial{}; };
                check_masks<F32, Serial>(problem, config.options, serial, label + " f32",
                                         coverage);
                check_masks<F64, Serial>(problem, config.options, serial, label + " f64",
                                         coverage);
            } else {
                const auto team = [&] { return Team(config.team); };
                check_masks<F32, Team>(problem, config.options, team, label + " f32", coverage);
                check_masks<F64, Team>(problem, config.options, team, label + " f64", coverage);
            }
        }
    }
    // The comparisons are not vacuous: the mask did change when and what the decoder returned.
    EXPECT_GT(coverage.earlier, coverage.decodes / 20);
    EXPECT_GT(coverage.masked_only, 0U);
    EXPECT_GT(coverage.violates_full, 0U);
}

TEST(ConvergenceRows, NoRequiredRowConvergesOnTheFirstIteration) {
    const Problem problem = random_problem(9, 30, 120, 6);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx);
    auto priors = Priors::from_probabilities(problem.p);
    auto gammas = UniformGammaGenerator::create(2, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(graph && priors && gammas);
    auto decoder = CpuRelayDecoder<F32>::create(
        *CpuBackend<F32>::create(*graph, *priors), {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.1},
        {.pre_iter = 20, .set_max_iter = 5, .num_sets = 4, .stopping = AfterNConverged{3}},
        &*gammas);
    ASSERT_TRUE(decoder);
    decoder->set_convergence_rows(std::vector<Bit>(problem.h.rows, 0));
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        auto r = decoder->decode(problem.syndromes[s], s);
        ASSERT_TRUE(r);
        EXPECT_TRUE(r->success);
        EXPECT_EQ(r->legs_executed, 3U);
        EXPECT_EQ(r->iterations, 3U);
        for (const LegRecord& leg : r->legs) {
            EXPECT_TRUE(leg.converged);
            EXPECT_EQ(leg.iterations, 1U);
        }
    }
}

TEST(ConvergenceRows, MaskPersistsAcrossDecodesAndCanBeCleared) {
    const Problem problem = random_problem(3, 36, 150, 12);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx);
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(graph && priors);
    const RelayConfig relay{
        .pre_iter = 30, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}};
    auto make = [&] {
        return *CpuRelayDecoder<F32>::create(*CpuBackend<F32>::create(*graph, *priors),
                                             MinSumConfig{}, relay, nullptr);
    };
    auto plain = make();
    auto toggled = make();
    auto fixed = make();
    std::vector<Bit> first_rows(problem.h.rows, 0);
    std::fill_n(first_rows.begin(), problem.h.rows / 3, Bit{1});
    fixed.set_convergence_rows(first_rows);
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        const auto& syndrome = problem.syndromes[s];
        const std::string where = std::format("shot {}", s);
        toggled.set_convergence_rows(first_rows);
        expect_identical(decode(toggled, syndrome, s), decode(fixed, syndrome, s), where);
        toggled.set_convergence_rows({});
        expect_identical(decode(toggled, syndrome, s), decode(plain, syndrome, s), where);
    }
}

// Entries past the end of a short mask count as required, entries past row m are ignored.
TEST(ConvergenceRows, ShortAndLongMasksArePaddedWithRequiredRows) {
    const Problem problem = random_problem(6, 36, 150, 12);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx);
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(graph && priors);
    const RelayConfig relay{
        .pre_iter = 30, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}};
    auto make = [&] {
        return *CpuRelayDecoder<F64>::create(*CpuBackend<F64>::create(*graph, *priors),
                                             MinSumConfig{}, relay, nullptr);
    };
    const index_t m = problem.h.rows;
    std::vector<Bit> full(m, 1);
    for (index_t i = 0; i < 10; ++i) {
        full[i] = 0;
    }
    const std::vector<Bit> short_mask(full.begin(), full.begin() + 12);
    std::vector<Bit> long_mask = full;
    long_mask.resize(m + 70, 0);
    auto a = make();
    auto b = make();
    auto c = make();
    a.set_convergence_rows(full);
    b.set_convergence_rows(short_mask);
    c.set_convergence_rows(long_mask);
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        const Snapshot want = decode(a, problem.syndromes[s], s);
        expect_identical(decode(b, problem.syndromes[s], s), want, "short");
        expect_identical(decode(c, problem.syndromes[s], s), want, "long");
    }
}

} // namespace
