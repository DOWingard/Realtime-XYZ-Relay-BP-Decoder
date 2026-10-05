// Solution sinks: the relay controller reports every converged leg, and RecordingSink reduces
// each solution to (leg, cumulative iterations, weight, logical class, hash, size).
//
// The events are checked against the decode's own per-leg trace, the class and hash against a
// naive recomputation from the dense ê, and a decoder with a sink against one without, bit for bit.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/sink.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

// What a windowed decoder requires of its inner decoder.
template <class D>
concept SyndromeDecoder = requires(D d, const D cd, std::span<const Bit> s, std::uint64_t stream,
                                   DecodeLimits lim, std::span<const Bit> mask) {
    { cd.num_rows() } -> std::same_as<index_t>;
    { cd.num_columns() } -> std::same_as<index_t>;
    { d.set_convergence_rows(mask) } -> std::same_as<void>;
    { d.decode(s, stream, lim) } -> std::same_as<std::expected<DecodeResult, DecodeError>>;
};

static_assert(SyndromeDecoder<CpuRelayDecoder<F32>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F64, Team>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F32, Serial, RecordingSink>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F64, Team, RecordingSink>>);

// ---- The hash and the class, by hand ------------------------------------------------------------

TEST(SolutionHash, KnownAnswers) {
    // SplitMix64's first output from state 0 is the published 0xE220A8397B1DCDAF; the other
    // values were computed independently with arbitrary-precision integers masked to 64 bits.
    EXPECT_EQ(splitmix64_hash(0), 0xE220A8397B1DCDAFULL);
    EXPECT_EQ(solution_hash({}), 0ULL);
    const std::vector<index_t> one{0};
    EXPECT_EQ(solution_hash(one), 0xE220A8397B1DCDAFULL);
    const std::vector<index_t> three{3};
    EXPECT_EQ(solution_hash(three), 0x1D0B14E4DB018FEDULL);
    const std::vector<index_t> pair{0, 1};
    EXPECT_EQ(solution_hash(pair), 0x08B4FDA8C892B50EULL);
    const std::vector<index_t> spread{1, 5, 70000};
    EXPECT_EQ(solution_hash(spread), 0xEB4E9F1BB8E43FCEULL);
    const std::vector<index_t> last{71279};
    EXPECT_EQ(solution_hash(last), 0xC166250C59D49BCAULL);
    static_assert(splitmix64_hash(0) == 0xE220A8397B1DCDAFULL);
}

TEST(SolutionHash, ClassIsTheXorOfTheColumnMasks) {
    const std::vector<std::uint64_t> masks{0b0001, 0b0010, 0b0110, 0b1000, 0b0011};
    const std::vector<index_t> support{0, 2, 4};
    EXPECT_EQ(solution_class(masks, support), 0b0001ULL ^ 0b0110ULL ^ 0b0011ULL);
    EXPECT_EQ(solution_class(masks, {}), 0ULL);
}

// ---- RecordingSink on its own ------------------------------------------------------------------

TEST(RecordingSink, ValidatesItsCapacity) {
    const std::vector<std::uint64_t> masks(4, 1);
    EXPECT_FALSE(RecordingSink::create(0, masks));
    EXPECT_FALSE(RecordingSink::create(RecordingSink::max_capacity + 1, masks));
    auto bad = RecordingSink::create(0, masks);
    ASSERT_FALSE(bad);
    EXPECT_EQ(bad.error().code, SinkError::Code::invalid_capacity);
    EXPECT_EQ(to_string(bad.error().code), "invalid_capacity");
    EXPECT_TRUE(RecordingSink::create(1, masks));
    EXPECT_TRUE(RecordingSink::create(RecordingSink::max_capacity, masks));
    EXPECT_THROW(RecordingSink(0, masks), std::invalid_argument);
    EXPECT_THROW(RecordingSink(21, masks), std::invalid_argument);
    EXPECT_NO_THROW(RecordingSink(20, masks));
}

TEST(RecordingSink, StoresUpToItsCapacityAndCountsTheRest) {
    const std::vector<std::uint64_t> masks{1, 2, 4, 8, 16, 32};
    RecordingSink sink(3, masks);
    const std::vector<std::vector<index_t>> supports{{0}, {1, 2}, {}, {3, 4, 5}, {0, 5}};
    sink.on_decode_begin();
    std::uint32_t cumulative = 0;
    for (std::uint32_t k = 0; k < supports.size(); ++k) {
        cumulative += 7;
        sink.on_solution(SolutionEvent{.leg = 2 * k,
                                       .cumulative_iterations = cumulative,
                                       .weight = 0.5 * k,
                                       .support = supports[k]});
    }
    EXPECT_EQ(sink.found(), 5U);
    EXPECT_EQ(sink.overflow(), 2U);
    ASSERT_EQ(sink.records().size(), 3U);
    for (std::uint32_t k = 0; k < 3; ++k) {
        const SolutionRecord& r = sink.records()[k];
        EXPECT_EQ(r.leg, 2 * k);
        EXPECT_EQ(r.cumulative_iterations, 7 * (k + 1));
        EXPECT_EQ(r.weight, 0.5 * k);
        EXPECT_EQ(r.size, supports[k].size());
        EXPECT_EQ(r.logical_class, solution_class(masks, supports[k]));
        EXPECT_EQ(r.hash, solution_hash(supports[k]));
    }
    // A new decode starts from nothing.
    sink.on_decode_begin();
    EXPECT_EQ(sink.found(), 0U);
    EXPECT_EQ(sink.overflow(), 0U);
    EXPECT_TRUE(sink.records().empty());
    sink.on_solution(SolutionEvent{
        .leg = 0, .cumulative_iterations = 1, .weight = 0.0, .support = supports[3]});
    ASSERT_EQ(sink.records().size(), 1U);
    EXPECT_EQ(sink.records()[0].logical_class, 8ULL ^ 16ULL ^ 32ULL);
    EXPECT_EQ(sink.class_of(supports[1]), 6ULL);
}

// ---- Sinks on real decodes ----------------------------------------------------------------------

// Keeps every event with a copy of its support.
struct CaptureSink {
    static constexpr bool enabled = true;
    struct Event {
        std::uint32_t leg;
        std::uint32_t cumulative_iterations;
        double weight;
        std::vector<index_t> support;
    };
    std::vector<Event> events;
    std::size_t begins = 0;

    void on_decode_begin() noexcept {
        events.clear();
        ++begins;
    }
    // Allocation failure in a test double ends the test binary, which is all a test needs.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void on_solution(const SolutionEvent& event) noexcept {
        events.push_back(Event{.leg = event.leg,
                               .cumulative_iterations = event.cumulative_iterations,
                               .weight = event.weight,
                               .support = {event.support.begin(), event.support.end()}});
    }
};

static_assert(SolutionSink<CaptureSink>);

struct Problem {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::vector<Bit>> syndromes;
    std::vector<std::uint64_t> column_class;
};

Problem random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots) {
    Problem problem{
        .h = test::random_csr(m, n, 1, 6, seed), .p = {}, .syndromes = {}, .column_class = {}};
    std::mt19937_64 rng(seed * 31337);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        const double u = unit(rng);
        // Low enough rates that most decodes converge, often in several legs; a few p = 0
        // columns (λ = +∞) and p > 1/2 columns (λ < 0).
        double p = 0.6;
        if (u < 0.03) {
            p = 0.0;
        } else if (u >= 0.05) {
            p = 0.002 + 0.03 * unit(rng);
        }
        problem.p.push_back(p);
        // A 24-observable mask, as for the gross code, with most columns flipping none.
        problem.column_class.push_back(unit(rng) < 0.2 ? (rng() & 0xFFFFFFULL) : 0);
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
    {.name = "relay_nconv",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 12, .set_max_iter = 8, .num_sets = 15, .stopping = AfterNConverged{5}}},
    {.name = "relay_all",
     .min_sum = {.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 6, .set_max_iter = 6, .num_sets = 30, .stopping = AllLegs{}}},
    {.name = "relay_after_leg0",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = -0.1},
     .relay = {.pre_iter = 5, .set_max_iter = 5, .num_sets = 6, .stopping = AfterLeg0{}}},
};

std::vector<Bit> dense_of(std::span<const index_t> support, index_t n) {
    std::vector<Bit> e(n, 0);
    for (const index_t j : support) {
        e[j] = 1;
    }
    return e;
}

// The class and hash straight from their definitions, scanning the dense ê.
std::uint64_t naive_class(const std::vector<std::uint64_t>& masks, const std::vector<Bit>& e) {
    std::uint64_t c = 0;
    for (std::size_t j = 0; j < e.size(); ++j) {
        if (e[j] != 0) {
            c ^= masks[j];
        }
    }
    return c;
}

std::uint64_t naive_hash(const std::vector<Bit>& e) {
    std::uint64_t h = 0;
    for (std::size_t j = 0; j < e.size(); ++j) {
        if (e[j] == 0) {
            continue;
        }
        std::uint64_t x = h ^ j;
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
        h = x ^ (x >> 31U);
    }
    return h;
}

double naive_weight(std::span<const double> llr, const std::vector<Bit>& e) {
    double w = 0.0;
    for (std::size_t j = 0; j < e.size(); ++j) {
        if (e[j] != 0 && std::isfinite(llr[j])) {
            w += llr[j];
        }
    }
    return w;
}

void expect_same_result(const DecodeResult& a, const DecodeResult& b, const std::string& where) {
    EXPECT_EQ(a.success, b.success) << where;
    EXPECT_EQ(a.iterations, b.iterations) << where;
    EXPECT_EQ(a.legs_executed, b.legs_executed) << where;
    EXPECT_EQ(a.best_leg, b.best_leg) << where;
    EXPECT_EQ(bits_of(a.weight), bits_of(b.weight)) << where;
    EXPECT_EQ(a.cap_hit, b.cap_hit) << where;
    EXPECT_TRUE(std::ranges::equal(a.hard, b.hard)) << where;
    EXPECT_TRUE(std::ranges::equal(a.support, b.support)) << where;
    ASSERT_EQ(a.legs.size(), b.legs.size()) << where;
    for (std::size_t i = 0; i < a.legs.size(); ++i) {
        EXPECT_EQ(a.legs[i].iterations, b.legs[i].iterations) << where;
        EXPECT_EQ(a.legs[i].converged, b.legs[i].converged) << where;
        EXPECT_EQ(a.legs[i].became_best, b.legs[i].became_best) << where;
        EXPECT_EQ(bits_of(a.legs[i].weight), bits_of(b.legs[i].weight)) << where;
    }
}

struct SinkCoverage {
    std::size_t decodes = 0;
    std::size_t events = 0;
    std::size_t multi_solution = 0;
    std::size_t overflowed = 0;
    std::size_t repeated_hash = 0;
    std::size_t nonzero_class = 0;
};

template <class A, class Exec, class MakeExecutor>
void check_sinks(const Problem& problem, const GraphOptions& options, MakeExecutor make_executor,
                 std::uint32_t capacity, const std::string& label, SinkCoverage& coverage) {
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, options);
    ASSERT_TRUE(graph) << graph.error().detail;
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(8, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    const index_t n = problem.h.cols;
    std::vector<double> m_plain(n);
    std::vector<double> m_recorded(n);

    for (const Scenario& scenario : scenarios) {
        auto plain = RelayDecoder<CpuBackend<A, Exec>>::create(
            *CpuBackend<A, Exec>::create(*graph, *priors, make_executor()), scenario.min_sum,
            scenario.relay, &*gammas);
        auto captured = RelayDecoder<CpuBackend<A, Exec>, CaptureSink>::create(
            *CpuBackend<A, Exec>::create(*graph, *priors, make_executor()), scenario.min_sum,
            scenario.relay, &*gammas);
        auto recorded = RelayDecoder<CpuBackend<A, Exec>, RecordingSink>::create(
            *CpuBackend<A, Exec>::create(*graph, *priors, make_executor()), scenario.min_sum,
            scenario.relay, &*gammas, RecordingSink(capacity, problem.column_class));
        ASSERT_TRUE(plain && captured && recorded);

        for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
            const auto& syndrome = problem.syndromes[s];
            const std::string where = std::format("{} {} shot {}", label, scenario.name, s);
            auto a = plain->decode(syndrome, s);
            auto b = captured->decode(syndrome, s);
            auto c = recorded->decode(syndrome, s);
            ASSERT_TRUE(a && b && c);
            // Listening changes nothing.
            expect_same_result(*b, *a, where + " capture");
            expect_same_result(*c, *a, where + " recording");
            plain->backend().read_marginals(m_plain);
            recorded->backend().read_marginals(m_recorded);
            EXPECT_EQ(m_plain, m_recorded) << where;

            // One event per converged leg, in leg order, matching the per-leg trace.
            const std::vector<CaptureSink::Event>& events = captured->sink().events;
            std::vector<std::uint32_t> converged_legs;
            std::vector<std::uint32_t> cumulative;
            std::uint32_t total = 0;
            for (std::uint32_t leg = 0; leg < a->legs.size(); ++leg) {
                total += a->legs[leg].iterations;
                if (a->legs[leg].converged) {
                    converged_legs.push_back(leg);
                    cumulative.push_back(total);
                }
            }
            ASSERT_EQ(events.size(), converged_legs.size()) << where;
            std::vector<std::uint64_t> hashes;
            for (std::size_t k = 0; k < events.size(); ++k) {
                const CaptureSink::Event& e = events[k];
                const std::uint32_t leg = converged_legs[k];
                EXPECT_EQ(e.leg, leg) << where;
                EXPECT_EQ(e.cumulative_iterations, cumulative[k]) << where;
                EXPECT_EQ(bits_of(e.weight), bits_of(a->legs[leg].weight)) << where;
                EXPECT_TRUE(std::ranges::is_sorted(e.support)) << where;
                const std::vector<Bit> dense = dense_of(e.support, n);
                EXPECT_EQ(test::syndrome_of(problem.h, dense), syndrome) << where << " leg " << leg;
                EXPECT_EQ(bits_of(e.weight), bits_of(naive_weight(priors->llr(), dense))) << where;
                hashes.push_back(naive_hash(dense));
                if (a->best_leg == leg) {
                    EXPECT_TRUE(std::ranges::equal(e.support, a->support)) << where;
                }
            }

            // The recording sink's reduction of the same events.
            const RecordingSink& sink = recorded->sink();
            EXPECT_EQ(sink.found(), events.size()) << where;
            EXPECT_EQ(sink.overflow(), events.size() > capacity ? events.size() - capacity : 0U)
                << where;
            ASSERT_EQ(sink.records().size(), std::min<std::size_t>(events.size(), capacity))
                << where;
            for (std::size_t k = 0; k < sink.records().size(); ++k) {
                const SolutionRecord& r = sink.records()[k];
                const std::vector<Bit> dense = dense_of(events[k].support, n);
                EXPECT_EQ(r.leg, events[k].leg) << where;
                EXPECT_EQ(r.cumulative_iterations, events[k].cumulative_iterations) << where;
                EXPECT_EQ(bits_of(r.weight), bits_of(events[k].weight)) << where;
                EXPECT_EQ(r.size, events[k].support.size()) << where;
                EXPECT_EQ(r.logical_class, naive_class(problem.column_class, dense)) << where;
                EXPECT_EQ(r.hash, hashes[k]) << where;
                coverage.nonzero_class += r.logical_class != 0 ? 1U : 0U;
            }
            // The class of the returned ê, which exists also when nothing converged.
            EXPECT_EQ(sink.class_of(c->support),
                      naive_class(problem.column_class,
                                  std::vector<Bit>(c->hard.begin(), c->hard.end())))
                << where;

            std::ranges::sort(hashes);
            coverage.repeated_hash +=
                std::ranges::adjacent_find(hashes) != hashes.end() ? 1U : 0U;
            ++coverage.decodes;
            coverage.events += events.size();
            coverage.multi_solution += events.size() > 1 ? 1U : 0U;
            coverage.overflowed += sink.overflow() > 0 ? 1U : 0U;
        }
    }
}

TEST(SolutionSinks, EventsMatchTheLegTraceAndRecordsMatchANaiveRecomputation) {
    SinkCoverage coverage;
    const GraphOptions wavefront{
        .layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 4};
    const GraphOptions blocked{.layout = EdgeLayout::column_blocked,
                               .column_order = ColumnOrder::wavefront,
                               .block_rows = 5};
    const auto serial = [] { return Serial{}; };
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const Problem problem = random_problem(seed, 36, 150, 12);
        const std::string label = std::format("seed {}", seed);
        check_sinks<F32, Serial>(problem, wavefront, serial, RecordingSink::max_capacity,
                                 label + " f32", coverage);
        check_sinks<F64, Serial>(problem, blocked, serial, 3, label + " f64 capacity 3",
                                 coverage);
    }
    std::cout << std::format("[ coverage ] {} decodes, {} events, {} with several solutions, {} "
                             "overflowed, {} with a repeated solution, {} nonzero classes\n",
                             coverage.decodes, coverage.events, coverage.multi_solution,
                             coverage.overflowed, coverage.repeated_hash, coverage.nonzero_class);
    EXPECT_GT(coverage.multi_solution, coverage.decodes / 10);
    EXPECT_GT(coverage.overflowed, 0U);
    EXPECT_GT(coverage.repeated_hash, 0U);
    EXPECT_GT(coverage.nonzero_class, 0U);
}

TEST(SolutionSinks, TeamDecodesReportTheSameEvents) {
    SinkCoverage coverage;
    const GraphOptions wavefront{
        .layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 4};
    const Problem problem = random_problem(4, 36, 150, 8);
    const auto team = [] { return Team(3); };
    check_sinks<F32, Team>(problem, wavefront, team, 5, "team f32", coverage);
    EXPECT_GT(coverage.events, 0U);
}

// Under a cap the sink sees exactly the legs that ran and converged, never past the budget.
TEST(SolutionSinks, CappedDecodesReportOnlyTheLegsThatRan) {
    const Problem problem = random_problem(7, 36, 150, 12);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx);
    auto priors = Priors::from_probabilities(problem.p);
    auto gammas = UniformGammaGenerator::create(4, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(graph && priors && gammas);
    const Scenario& scenario = scenarios[2];
    auto decoder = CpuRelayDecoder<F32, Serial, RecordingSink>::create(
        *CpuBackend<F32>::create(*graph, *priors), scenario.min_sum, scenario.relay, &*gammas,
        RecordingSink(RecordingSink::max_capacity, problem.column_class));
    ASSERT_TRUE(decoder);
    std::size_t checked = 0;
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        for (const std::uint32_t cap : {0U, 1U, 7U, 20U, 45U, 100U}) {
            auto r = decoder->decode(problem.syndromes[s], s, {.max_total_iterations = cap});
            ASSERT_TRUE(r);
            std::uint32_t converged = 0;
            for (const LegRecord& leg : r->legs) {
                converged += leg.converged ? 1U : 0U;
            }
            const RecordingSink& sink = decoder->sink();
            EXPECT_EQ(sink.found(), converged) << "shot " << s << " cap " << cap;
            for (const SolutionRecord& record : sink.records()) {
                EXPECT_LE(record.cumulative_iterations, cap);
                EXPECT_LT(record.leg, r->legs_executed);
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 20U);
}

TEST(SolutionSinks, RecordingSinkMustCoverEveryColumn) {
    const test::Csr h = test::csr_from_dense({{1, 1, 0}, {0, 1, 1}});
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    auto priors = Priors::from_probabilities(std::vector<double>{0.01, 0.01, 0.01});
    ASSERT_TRUE(graph && priors);
    const std::vector<std::uint64_t> two{1, 2};
    const RelayConfig relay{
        .pre_iter = 5, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}};
    auto rejected = CpuRelayDecoder<F32, Serial, RecordingSink>::create(
        *CpuBackend<F32>::create(*graph, *priors), MinSumConfig{}, relay, nullptr,
        RecordingSink(1, two));
    ASSERT_FALSE(rejected);
    EXPECT_NE(rejected.error().detail.find("solution sink"), std::string::npos);
    const std::vector<std::uint64_t> three{1, 2, 4};
    auto accepted = CpuRelayDecoder<F32, Serial, RecordingSink>::create(
        *CpuBackend<F32>::create(*graph, *priors), MinSumConfig{}, relay, nullptr,
        RecordingSink(1, three));
    ASSERT_TRUE(accepted);
    auto r = accepted->decode(std::vector<Bit>{1, 1});
    ASSERT_TRUE(r);
    ASSERT_EQ(accepted->sink().records().size(), 1U);
    EXPECT_EQ(accepted->sink().records()[0].logical_class, 2ULL);
    EXPECT_EQ(accepted->sink().records()[0].size, 1U);
    EXPECT_EQ(accepted->sink().records()[0].hash, solution_hash(std::vector<index_t>{1}));
}

} // namespace
