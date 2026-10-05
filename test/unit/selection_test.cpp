// Selection policies and confidence values.
//
// The rules of SelectionState are checked against worked examples and against a naive
// restatement of their definitions (plain loops over every slot, no incremental state), on
// hand-made slot sequences and on the solutions of real relay decodes. Inside the relay
// controller, the default policy must leave every decode bit for bit as it was; early stops must
// end a decode exactly where a replay of the full recording says; a class-level rule must return
// the lightest solution of the class it decides; a low-confidence decode must run its extra legs.
// The window layer's low-confidence actions and the per-window history are tested with scripted
// inner decoders and hand-built confidences.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/selection.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/shots.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/history.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"
#include "support.hpp"
#include "unit/allocation_counter.hpp"
#include "window_inner_support.hpp"

namespace {

using namespace rtd;
namespace fs = std::filesystem;

const fs::path fixture_root = RTD_FIXTURE_DIR;
constexpr double inf = std::numeric_limits<double>::infinity();

std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

// Equal as doubles bit for bit, NaN included.
bool same(double a, double b) { return bits_of(a) == bits_of(b); }

// ---- A naive restatement of the rules ---------------------------------------------------------

struct Slot {
    std::uint32_t leg = 0;
    std::uint32_t iterations = 0;
    double weight = 0.0;
    std::uint64_t klass = 0;
    std::uint64_t hash = 0;
    std::vector<index_t> support;
};

struct Expected {
    GapState state = GapState::none;
    std::size_t best = 0;
    double gap = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t second = 0;
    std::uint32_t distinct = 0;
    std::uint32_t distinct_best = 0;
    std::uint32_t classes = 0;
    std::uint64_t class_sum_class = 0;
    double class_sum_top = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t agreement_class = 0;
};

// Every value from the first `used` slots, each by a full scan in slot order.
Expected naive(const std::vector<Slot>& slots, std::size_t used) {
    Expected e;
    if (used == 0) {
        return e;
    }
    double best_w = inf;
    for (std::size_t q = 0; q < used; ++q) {
        if (slots[q].weight < best_w) {
            best_w = slots[q].weight;
            e.best = q;
        }
    }
    const std::uint64_t top = slots[e.best].klass;
    std::vector<bool> fresh(used, false);
    std::vector<std::uint64_t> order;
    for (std::size_t q = 0; q < used; ++q) {
        fresh[q] = std::none_of(slots.begin(), slots.begin() + static_cast<std::ptrdiff_t>(q),
                                [&](const Slot& s) { return s.hash == slots[q].hash; });
        if (std::ranges::find(order, slots[q].klass) == order.end()) {
            order.push_back(slots[q].klass);
        }
        e.distinct += fresh[q] ? 1U : 0U;
        e.distinct_best += fresh[q] && slots[q].klass == top ? 1U : 0U;
    }
    e.classes = static_cast<std::uint32_t>(order.size());
    double w2 = inf;
    std::optional<std::uint64_t> c2;
    for (std::size_t q = 0; q < used; ++q) {
        if (slots[q].klass != top && slots[q].weight < w2) {
            w2 = slots[q].weight;
            c2 = slots[q].klass;
        }
    }
    if (c2) {
        e.state = GapState::defined;
        e.gap = w2 - best_w;
        e.second = *c2;
    } else {
        e.state = GapState::single_class;
    }
    // Class sums and counts per class in order of first appearance.
    std::optional<std::uint64_t> sum_choice;
    double sum_top = -inf;
    std::optional<std::uint64_t> count_choice;
    std::uint32_t count_top = 0;
    double light_top = inf;
    for (const std::uint64_t c : order) {
        double acc = 0.0;
        std::uint32_t count = 0;
        double light = inf;
        for (std::size_t q = 0; q < used; ++q) {
            if (fresh[q] && slots[q].klass == c) {
                const double d = slots[q].weight - best_w;
                acc = acc + std::exp(-d);
                ++count;
                light = std::min(light, slots[q].weight);
            }
        }
        if (!sum_choice || acc > sum_top) {
            sum_choice = c;
            sum_top = acc;
        }
        if (!count_choice || count > count_top || (count == count_top && light < light_top)) {
            count_choice = c;
            count_top = count;
            light_top = light;
        }
    }
    e.class_sum_class = *sum_choice;
    e.class_sum_top = sum_top;
    e.agreement_class = *count_choice;
    return e;
}

SolutionRecord record_of(const Slot& s) {
    return SolutionRecord{.leg = s.leg,
                          .cumulative_iterations = s.iterations,
                          .weight = s.weight,
                          .logical_class = s.klass,
                          .hash = s.hash,
                          .size = static_cast<std::uint32_t>(s.support.size())};
}

// Slots with (weight, class, hash) and empty supports; legs 0, 2, 4, … and iterations 100, 200, ….
std::vector<Slot>
slots_of(std::initializer_list<std::tuple<double, std::uint64_t, std::uint64_t>> list) {
    std::vector<Slot> slots;
    std::uint32_t q = 0;
    for (const auto& [w, c, h] : list) {
        slots.push_back(Slot{.leg = 2 * q,
                             .iterations = 100 * (q + 1),
                             .weight = w,
                             .klass = c,
                             .hash = h,
                             .support = {}});
        ++q;
    }
    return slots;
}

// A state over n columns without a graph (no Q_supp), fed with the given slots.
struct Fed {
    std::vector<std::uint64_t> classes;
    std::vector<double> llr;
    SelectionState state;
};

Fed feed(const SelectionConfig& config, const std::vector<Slot>& slots, index_t n = 8) {
    Fed fed{.classes = std::vector<std::uint64_t>(n, 0),
            .llr = std::vector<double>(n, 1.0),
            .state = *SelectionState::create(config, {}, {}, nullptr)};
    auto state = SelectionState::create(config, fed.classes, fed.llr, nullptr);
    EXPECT_TRUE(state);
    fed.state = std::move(*state);
    fed.state.begin();
    for (const Slot& s : slots) {
        fed.state.add_record(record_of(s), s.support);
    }
    return fed;
}

Confidence finish(SelectionState& state) { return state.finish(DecodeFacts{}).confidence; }

constexpr std::uint64_t A = 0b01;
constexpr std::uint64_t B = 0b10;
constexpr std::uint64_t C = 0b11;

// ---- Configuration -----------------------------------------------------------------------------

TEST(SelectionConfig, ValidatesEveryParameter) {
    EXPECT_TRUE(validate(SelectionConfig{}));
    SelectionConfig c;
    c.capacity = 0;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_capacity);
    c.capacity = 21;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_capacity);
    c = SelectionConfig{};
    c.stop = StopRule::agree;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_stop);
    c.stop_count = 2;
    EXPECT_TRUE(validate(c));
    c.stop = StopRule::gap_extend;
    c.stop_count = 21;
    c.capacity = 20;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_stop);
    c.stop_count = 5;
    c.stop_gap = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_stop);
    c = SelectionConfig{};
    c.signal = ConfidenceSignal::gap;
    c.threshold = inf;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_threshold);
    c = SelectionConfig{};
    c.extra_legs = 3;
    EXPECT_EQ(validate(c).error().code, SelectionError::Code::invalid_threshold);
    EXPECT_EQ(to_string(SelectionError::Code::missing_graph), "missing_graph");
}

TEST(SelectionConfig, NamesRoundTrip) {
    for (const SelectionRule r : {SelectionRule::lowest_weight, SelectionRule::class_sum,
                                  SelectionRule::largest_agreement}) {
        EXPECT_EQ(parse_selection_rule(to_string(r)), r);
    }
    for (const StopRule r : {StopRule::fixed, StopRule::agree, StopRule::agree_distinct,
                             StopRule::gap, StopRule::gap_extend}) {
        EXPECT_EQ(parse_stop_rule(to_string(r)), r);
    }
    for (const ConfidenceSignal s :
         {ConfidenceSignal::none, ConfidenceSignal::gap, ConfidenceSignal::agreement,
          ConfidenceSignal::weight, ConfidenceSignal::first_legs,
          ConfidenceSignal::first_iterations, ConfidenceSignal::q_supp}) {
        EXPECT_EQ(parse_confidence_signal(to_string(s)), s);
    }
    EXPECT_FALSE(parse_selection_rule("median"));
    EXPECT_FALSE(parse_stop_rule(""));
    EXPECT_EQ(to_string(GapState::single_class), "single_class");
}

TEST(SelectionConfig, QSuppNeedsAGraphAndSizesMustAgree) {
    std::vector<std::uint64_t> classes(4, 0);
    std::vector<double> llr(3, 1.0);
    EXPECT_EQ(SelectionState::create({}, classes, llr, nullptr).error().code,
              SelectionError::Code::size_mismatch);
    llr.push_back(1.0);
    SelectionConfig c;
    c.signal = ConfidenceSignal::q_supp;
    EXPECT_EQ(SelectionState::create(c, classes, llr, nullptr).error().code,
              SelectionError::Code::missing_graph);
}

// ---- The rules on hand-made slots
// ----------------------------------------------------------------

TEST(SelectionRules, NoSolutionIsItsOwnCategory) {
    Fed fed = feed({}, {});
    const Selection s = fed.state.finish(DecodeFacts{});
    EXPECT_EQ(s.confidence.gap_state, GapState::none);
    EXPECT_EQ(s.confidence.weight, inf);
    EXPECT_TRUE(std::isnan(s.confidence.gap));
    EXPECT_TRUE(std::isnan(s.confidence.agreement));
    EXPECT_EQ(s.confidence.first_legs, 0U);
    EXPECT_FALSE(s.replaces);
}

TEST(SelectionRules, ASingleClassIsUndefinedNeverInfinite) {
    Fed fed = feed({}, slots_of({{4.0, A, 1}, {4.0, A, 1}, {6.0, A, 2}}));
    const Confidence c = finish(fed.state);
    EXPECT_EQ(c.gap_state, GapState::single_class);
    EXPECT_TRUE(std::isnan(c.gap));
    EXPECT_EQ(c.distinct, 2U);
    EXPECT_EQ(c.agreement, 1.0);
    EXPECT_EQ(c.first_legs, 1U);
    EXPECT_EQ(c.first_iterations, 100U);
    EXPECT_EQ(c.found, 3U);
    EXPECT_EQ(c.seen, 3U);
    // Under the gap signal Δ has no value, so neither has the score; `low` follows the
    // single-class setting. With no solution the score is +∞ and the decode is low.
    for (const bool single_low : {true, false}) {
        SelectionConfig config;
        config.signal = ConfidenceSignal::gap;
        config.threshold = 1.0;
        config.single_class_is_low = single_low;
        Fed gapped = feed(config, slots_of({{4.0, A, 1}, {6.0, A, 2}}));
        const Confidence g = finish(gapped.state);
        EXPECT_TRUE(std::isnan(g.score)) << single_low;
        EXPECT_EQ(g.low, single_low);
        Fed empty = feed(config, {});
        const Confidence e = finish(empty.state);
        EXPECT_EQ(e.score, inf);
        EXPECT_TRUE(e.low);
    }
}

TEST(SelectionRules, WorkedExample) {
    // ê* is the first slot of weight 3 (slot 1); the runner-up class B reaches 3.5 first at slot 2.
    const auto slots = slots_of(
        {{5.0, A, 10}, {3.0, A, 11}, {3.5, B, 12}, {3.5, C, 13}, {3.0, A, 11}, {7.0, B, 14}});
    Fed fed = feed({}, slots);
    const Confidence c = finish(fed.state);
    EXPECT_EQ(c.gap_state, GapState::defined);
    EXPECT_EQ(c.best_class, A);
    EXPECT_EQ(c.weight, 3.0);
    EXPECT_EQ(c.gap, 0.5);
    EXPECT_EQ(c.second_class, B);
    EXPECT_EQ(c.distinct, 5U);
    EXPECT_EQ(c.classes, 3U);
    EXPECT_EQ(c.agreement, 2.0 / 5.0);
    const double za = 0.0 + std::exp(-(5.0 - 3.0)) + std::exp(-(3.0 - 3.0));
    EXPECT_EQ(c.class_sum_class, A);
    EXPECT_EQ(c.class_sum_top, za);
    EXPECT_EQ(c.agreement_class, A); // A and B both have 2; A's lightest (3.0) beats B's (3.5)
    EXPECT_EQ(c.decided_class, A);
    EXPECT_EQ(c.decided_leg, 2U);
    const Expected e = naive(slots, slots.size());
    EXPECT_EQ(e.gap, c.gap);
    EXPECT_EQ(e.class_sum_top, c.class_sum_top);
}

TEST(SelectionRules, ClassSumTiesGoToTheEarlierClass) {
    Fed fed = feed({}, slots_of({{2.0, B, 1}, {2.0, A, 2}}));
    const Confidence c = finish(fed.state);
    EXPECT_EQ(c.class_sum_class, B);
    EXPECT_EQ(c.best_class, B);
}

TEST(SelectionRules, LargestAgreementTiesGoToTheLighterThenTheEarlierClass) {
    Fed lighter = feed({}, slots_of({{4.0, B, 1}, {3.0, A, 2}, {5.0, B, 3}, {6.0, A, 4}}));
    EXPECT_EQ(finish(lighter.state).agreement_class, A);
    Fed earlier = feed({}, slots_of({{4.0, B, 1}, {4.0, A, 2}}));
    EXPECT_EQ(finish(earlier.state).agreement_class, B);
    Fed more = feed({}, slots_of({{1.0, A, 1}, {4.0, B, 2}, {5.0, B, 3}}));
    EXPECT_EQ(finish(more.state).agreement_class, B);
}

TEST(SelectionRules, ARepeatedSupportCountsOnce) {
    // Class B appears three times but as one solution: class sum and agreement see one.
    Fed fed = feed({}, slots_of({{3.0, A, 1}, {3.2, B, 2}, {3.2, B, 2}, {3.2, B, 2}, {4.0, A, 3}}));
    const Confidence c = finish(fed.state);
    EXPECT_EQ(c.distinct, 3U);
    EXPECT_EQ(c.agreement_class, A);
    EXPECT_EQ(c.agreement, 2.0 / 3.0);
}

TEST(SelectionRules, CapacityBoundsTheSlotsTheRulesSee) {
    SelectionConfig config;
    config.capacity = 2;
    Fed fed = feed(config, slots_of({{3.0, A, 1}, {4.0, B, 2}, {1.0, B, 3}}));
    const Confidence c = finish(fed.state);
    EXPECT_EQ(c.found, 3U);
    EXPECT_EQ(c.seen, 2U);
    EXPECT_EQ(c.best_class, A); // the lighter third slot is beyond the capacity
    EXPECT_EQ(c.gap, 1.0);
}

// Every value against the naive restatement, on random slot sequences with repeats, ties and up
// to five classes, for every prefix length.
TEST(SelectionRules, AgreeWithTheNaiveDefinitionsOnRandomSlots) {
    std::mt19937_64 rng(2026);
    std::size_t defined = 0;
    std::size_t differing_decisions = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + (rng() % 20);
        std::vector<Slot> slots;
        for (std::size_t q = 0; q < n; ++q) {
            Slot s;
            s.leg = static_cast<std::uint32_t>(3 * q + (rng() % 3));
            s.iterations = static_cast<std::uint32_t>(50 * (q + 1));
            if (q > 0 && rng() % 4 == 0) {
                const Slot& twin = slots[rng() % q];
                s.weight = twin.weight;
                s.klass = twin.klass;
                s.hash = twin.hash;
            } else {
                s.weight = static_cast<double>(rng() % 12) * 0.25 + 10.0;
                s.klass = rng() % 5;
                s.hash = rng();
            }
            slots.push_back(s);
        }
        for (std::uint32_t cap = 1; cap <= n; ++cap) {
            SelectionConfig config;
            config.capacity = cap;
            Fed fed = feed(config, slots);
            const Confidence c = finish(fed.state);
            const Expected e = naive(slots, cap);
            const std::string where = std::format("trial {} cap {}", trial, cap);
            ASSERT_EQ(c.gap_state, e.state) << where;
            EXPECT_EQ(c.best_class, slots[e.best].klass) << where;
            EXPECT_TRUE(same(c.weight, slots[e.best].weight)) << where;
            EXPECT_TRUE(same(c.gap, e.gap)) << where;
            if (e.state == GapState::defined) {
                EXPECT_EQ(c.second_class, e.second) << where;
                ++defined;
            }
            EXPECT_EQ(c.distinct, e.distinct) << where;
            EXPECT_EQ(c.classes, e.classes) << where;
            EXPECT_TRUE(same(c.agreement, static_cast<double>(e.distinct_best) /
                                              static_cast<double>(e.distinct)))
                << where;
            EXPECT_EQ(c.class_sum_class, e.class_sum_class) << where;
            EXPECT_TRUE(same(c.class_sum_top, e.class_sum_top)) << where;
            EXPECT_EQ(c.agreement_class, e.agreement_class) << where;
            EXPECT_EQ(c.first_legs, slots[0].leg + 1) << where;
            differing_decisions += c.class_sum_class != c.best_class ? 1 : 0;
        }
    }
    EXPECT_GT(defined, 500U);
    EXPECT_GT(differing_decisions, 20U);
}

// ---- Stopping rules
// --------------------------------------------------------------------------------

// The slot after which the rule first holds, feeding the slots one by one.
std::optional<std::size_t> fires_at(const SelectionConfig& config, const std::vector<Slot>& slots) {
    std::vector<std::uint64_t> classes(8, 0);
    std::vector<double> llr(8, 1.0);
    auto state = SelectionState::create(config, classes, llr, nullptr);
    EXPECT_TRUE(state);
    state->begin();
    for (std::size_t q = 0; q < slots.size(); ++q) {
        state->add_record(record_of(slots[q]), {});
        if (state->stop_requested()) {
            return q;
        }
    }
    return std::nullopt;
}

TEST(SelectionStops, AgreementCountsLegsOrDistinctSolutions) {
    const auto slots = slots_of({{3.0, A, 1}, {3.0, A, 1}, {4.0, B, 2}, {5.0, A, 3}});
    SelectionConfig legs;
    legs.stop = StopRule::agree;
    legs.stop_count = 2;
    EXPECT_EQ(fires_at(legs, slots), 1U); // the repeat counts as a converged leg
    SelectionConfig distinct = legs;
    distinct.stop = StopRule::agree_distinct;
    EXPECT_EQ(fires_at(distinct, slots), 3U);
    distinct.stop_count = 3;
    EXPECT_EQ(fires_at(distinct, slots), std::nullopt);
    // The class that must agree is ê*'s, which changes when a lighter solution arrives.
    const auto moving = slots_of({{3.0, A, 1}, {2.0, B, 2}, {4.0, A, 3}, {5.0, B, 4}});
    EXPECT_EQ(fires_at(legs, moving), 3U);
}

TEST(SelectionStops, GapAndGapExtend) {
    const auto slots = slots_of({{3.0, A, 1}, {3.4, B, 2}, {2.0, A, 3}, {4.0, C, 4}});
    SelectionConfig gap;
    gap.stop = StopRule::gap;
    gap.stop_gap = 1.0;
    EXPECT_EQ(fires_at(gap, slots), 2U); // Δ = 3.4 − 2.0 ≥ 1 after slot 2 (IEEE ≥)
    gap.stop_gap = 1.4; // 3.4 − 2.0 rounds to the double nearest 1.4: equal, so it fires
    EXPECT_EQ(fires_at(gap, slots), 2U);
    gap.stop_gap = 1.5;
    EXPECT_EQ(fires_at(gap, slots), std::nullopt);
    SelectionConfig extend;
    extend.stop = StopRule::gap_extend;
    extend.stop_count = 1;
    extend.stop_gap = 1.0;
    EXPECT_EQ(fires_at(extend, slots), 0U); // a single class after n0 = 1 slot
    extend.stop_count = 2;
    EXPECT_EQ(fires_at(extend, slots), 2U);
    extend.stop_gap = 0.0;
    EXPECT_EQ(fires_at(extend, slots), 1U); // t = 0 is fixed(n0)
}

// ---- Support clusters
// ------------------------------------------------------------------------------

TEST(SupportClusters, ComponentsAndTheClusterFraction) {
    // Columns 0-1 share row 0, 1-3 share row 2 (0, 1, 3 connected); 2 alone; 4 alone.
    const test::Csr h =
        test::csr_from_dense({{1, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 1, 0}, {0, 0, 0, 0, 1}});
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    ASSERT_TRUE(graph);
    const std::vector<double> llr{1.0, 2.0, 4.0, inf, 0.5};
    auto clusters = SupportClusters::create(*graph, llr);
    ASSERT_TRUE(clusters);
    EXPECT_EQ(clusters->total(), 1.0 + 2.0 + 4.0 + 0.5);
    const std::vector<index_t> support{0, 1, 2, 3, 4};
    const SupportClusters::Terms t = clusters->terms(support);
    EXPECT_EQ(t.components, 3U);
    // Components {0, 1, 3} (λ 1 + 2, the infinite λ skipped), {2}, {4}, in that order.
    EXPECT_EQ(t.sum_sq, ((0.0 + 3.0 * 3.0) + 4.0 * 4.0) + 0.5 * 0.5);
    EXPECT_EQ(clusters->q2(t), std::sqrt(t.sum_sq) / 7.5);
    // The rows are released: a second call on another support is independent.
    const std::vector<index_t> pair{2, 4};
    EXPECT_EQ(clusters->terms(pair).components, 2U);
    EXPECT_EQ(clusters->terms({}).components, 0U);
    EXPECT_EQ(clusters->terms({}).sum_sq, 0.0);
    std::vector<double> short_llr(4, 1.0);
    EXPECT_FALSE(SupportClusters::create(*graph, short_llr));
}

TEST(SupportClusters, MergeThroughALaterColumn) {
    // 0 and 2 are apart until column 3 touches both of their rows.
    const test::Csr h = test::csr_from_dense({{1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, 1}});
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    ASSERT_TRUE(graph);
    const std::vector<double> llr{1.0, 1.0, 1.0, 1.0};
    auto clusters = SupportClusters::create(*graph, llr);
    ASSERT_TRUE(clusters);
    const std::vector<index_t> support{0, 1, 2, 3};
    const auto t = clusters->terms(support);
    EXPECT_EQ(t.components, 2U);
    EXPECT_EQ(t.sum_sq, 9.0 + 1.0);
}

// ---- Inside the relay controller
// --------------------------------------------------------------------

// Keeps every converged leg with its support.
struct CaptureSink {
    static constexpr bool enabled = true;
    std::vector<Slot> slots;
    const std::vector<std::uint64_t>* classes = nullptr;

    void on_decode_begin() noexcept { slots.clear(); }
    // Allocation failure in a test double ends the test binary, which is all a test needs.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void on_solution(const SolutionEvent& event) noexcept {
        slots.push_back(Slot{.leg = event.leg,
                             .iterations = event.cumulative_iterations,
                             .weight = event.weight,
                             .klass = solution_class(*classes, event.support),
                             .hash = solution_hash(event.support),
                             .support = {event.support.begin(), event.support.end()}});
    }
};

struct Random {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::uint64_t> classes;
    std::vector<std::vector<Bit>> syndromes;
};

// Columns carry a logical mask with probability class_fraction, drawn from the bits of class_mask.
Random random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots, double rate,
                      double class_fraction = 0.25, std::uint64_t class_mask = 0x7) {
    Random r{.h = test::random_csr(m, n, 1, 5, seed), .p = {}, .classes = {}, .syndromes = {}};
    std::mt19937_64 rng(seed * 7919);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        r.p.push_back(0.004 + rate * unit(rng));
        r.classes.push_back(unit(rng) < class_fraction ? (rng() & class_mask) : 0);
    }
    for (std::size_t s = 0; s < shots; ++s) {
        std::vector<Bit> e(n, 0);
        for (index_t j = 0; j < n; ++j) {
            e[j] = unit(rng) < r.p[j] ? 1 : 0;
        }
        r.syndromes.push_back(test::syndrome_of(r.h, e));
    }
    return r;
}

struct Built {
    TannerGraph graph;
    Priors priors;
    UniformGammaGenerator gammas;
};

Built build(const Random& r) {
    auto graph = TannerGraph::from_csr(r.h.rows, r.h.cols, r.h.row_ptr, r.h.col_idx);
    auto priors = Priors::from_probabilities(r.p);
    auto gammas = UniformGammaGenerator::create(11, -0.24, 0.66, r.h.cols);
    EXPECT_TRUE(graph && priors && gammas);
    return Built{.graph = std::move(*graph), .priors = std::move(*priors), .gammas = *gammas};
}

const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};

RelayConfig relay(std::uint32_t count, std::uint32_t sets = 40) {
    return RelayConfig{
        .pre_iter = 10, .set_max_iter = 8, .num_sets = sets, .stopping = AfterNConverged{count}};
}

template <class Sink> auto decoder_with(const Built& b, const RelayConfig& r, Sink sink) {
    auto made = CpuRelayDecoder<F32, Serial, Sink>::create(
        *CpuBackend<F32>::create(b.graph, b.priors), min_sum, r, &b.gammas, std::move(sink));
    EXPECT_TRUE(made);
    return std::move(*made);
}

SelectionSink<NoSink> selecting(const SelectionConfig& config, const Random& r, const Built& b) {
    auto state = SelectionState::create(config, r.classes, b.priors.llr(), &b.graph);
    EXPECT_TRUE(state) << (state ? "" : state.error().detail);
    return SelectionSink<NoSink>(std::move(*state));
}

// The default policy changes nothing, and its confidence equals a replay of the same decode's
// solutions (with supports, so Q_supp too) through a fresh state and the naive definitions.
TEST(SelectionDecoder, TheDefaultPolicyChangesNothingAndMatchesAReplay) {
    const Random r = random_problem(5, 60, 110, 150, 0.03);
    const Built b = build(r);
    for (const std::uint32_t s : {1U, 5U, 20U}) {
        auto plain = decoder_with(b, relay(s), NoSink{});
        auto capture = decoder_with(b, relay(s), CaptureSink{.slots = {}, .classes = &r.classes});
        auto policy = decoder_with(b, relay(s), selecting(SelectionConfig{}, r, b));
        auto clusters = SupportClusters::create(b.graph, b.priors.llr());
        ASSERT_TRUE(clusters);
        std::size_t multi = 0;
        for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
            const auto a = plain.decode(r.syndromes[shot], shot);
            const auto c = capture.decode(r.syndromes[shot], shot);
            const auto p = policy.decode(r.syndromes[shot], shot);
            ASSERT_TRUE(a && c && p);
            const std::string where = std::format("S {} shot {}", s, shot);
            EXPECT_FALSE(a->confidence.has_value());
            ASSERT_TRUE(p->confidence.has_value());
            EXPECT_EQ(a->success, p->success) << where;
            EXPECT_EQ(a->iterations, p->iterations) << where;
            EXPECT_EQ(a->legs_executed, p->legs_executed) << where;
            EXPECT_EQ(a->best_leg, p->best_leg) << where;
            EXPECT_TRUE(same(a->weight, p->weight)) << where;
            EXPECT_TRUE(std::ranges::equal(a->support, p->support)) << where;
            EXPECT_TRUE(std::ranges::equal(a->hard, p->hard)) << where;

            const std::vector<Slot>& slots = capture.sink().slots;
            const Confidence& got = *p->confidence;
            const Expected e = naive(slots, std::min<std::size_t>(slots.size(), 20));
            ASSERT_EQ(got.gap_state, e.state) << where;
            EXPECT_EQ(got.found, slots.size()) << where;
            if (slots.empty()) {
                EXPECT_EQ(got.best_class, solution_class(r.classes, a->support)) << where;
                continue;
            }
            multi += slots.size() > 1 ? 1U : 0U;
            EXPECT_TRUE(same(got.gap, e.gap)) << where;
            EXPECT_TRUE(same(got.class_sum_top, e.class_sum_top)) << where;
            EXPECT_EQ(got.class_sum_class, e.class_sum_class) << where;
            EXPECT_EQ(got.agreement_class, e.agreement_class) << where;
            EXPECT_EQ(got.decided_leg, slots[e.best].leg) << where;
            EXPECT_EQ(got.syndrome_rows, r.h.rows) << where;
            EXPECT_EQ(got.syndrome_ones,
                      static_cast<std::uint32_t>(std::ranges::count(r.syndromes[shot], 1)));
            // Q_supp of ê* straight from its support.
            const SupportClusters::Terms t = clusters->terms(slots[e.best].support);
            EXPECT_TRUE(same(got.q_sum_sq, t.sum_sq)) << where;
            EXPECT_TRUE(same(got.q_supp, clusters->q2(t))) << where;
            EXPECT_EQ(got.components, t.components) << where;
            // The same solutions replayed through a fresh state give the same values.
            auto replay =
                SelectionState::create(SelectionConfig{}, r.classes, b.priors.llr(), &b.graph);
            ASSERT_TRUE(replay);
            replay->begin();
            for (const Slot& slot : slots) {
                replay->add_record(record_of(slot), slot.support);
            }
            DecodeFacts facts;
            facts.syndrome_ones = got.syndrome_ones;
            facts.syndrome_rows = got.syndrome_rows;
            const Confidence again = replay->finish(facts).confidence;
            EXPECT_TRUE(same(again.q_supp, got.q_supp)) << where;
            EXPECT_TRUE(same(again.gap, got.gap)) << where;
            EXPECT_TRUE(same(again.agreement, got.agreement)) << where;
        }
        if (s > 1) {
            EXPECT_GT(multi, 30U);
        }
    }
}

// An early stop ends the decode exactly where the replay of a full recording says: after the
// first slot q where the rule holds, with iterations = iterations[q] and legs = leg[q] + 1; the
// returned solution is the lightest of the first q + 1 slots.
TEST(SelectionDecoder, EarlyStopsEndWhereTheReplaySays) {
    const Random r = random_problem(9, 60, 110, 200, 0.04);
    const Built b = build(r);
    constexpr std::uint32_t s_max = 12;
    auto capture =
        decoder_with(b, relay(s_max, 60), CaptureSink{.slots = {}, .classes = &r.classes});
    std::vector<SelectionConfig> rules;
    for (const auto& [rule, count, t] :
         {std::tuple{StopRule::agree, 3U, 0.0}, std::tuple{StopRule::agree_distinct, 2U, 0.0},
          std::tuple{StopRule::gap, 0U, 1.5}, std::tuple{StopRule::gap_extend, 3U, 2.0}}) {
        SelectionConfig c;
        c.capacity = s_max;
        c.stop = rule;
        c.stop_count = count;
        c.stop_gap = t;
        rules.push_back(c);
    }
    std::size_t early = 0;
    for (const SelectionConfig& config : rules) {
        auto policy = decoder_with(b, relay(s_max, 60), selecting(config, r, b));
        for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
            const auto full = capture.decode(r.syndromes[shot], shot);
            const auto p = policy.decode(r.syndromes[shot], shot);
            ASSERT_TRUE(full && p);
            const std::string where = std::format("{} shot {}", to_string(config.stop), shot);
            const std::vector<Slot>& slots = capture.sink().slots;
            const std::optional<std::size_t> q = fires_at(config, slots);
            ASSERT_TRUE(p->confidence);
            // Early only when the rule fired before the relay rule's s_max-th solution.
            EXPECT_EQ(p->confidence->stopped_early, q.has_value() && *q + 1 < s_max) << where;
            if (!q) {
                // Never fired: the relay rule (or the end of the schedule) ended both decodes.
                EXPECT_EQ(p->iterations, full->iterations) << where;
                EXPECT_EQ(p->legs_executed, full->legs_executed) << where;
                continue;
            }
            ++early;
            EXPECT_EQ(p->iterations, slots[*q].iterations) << where;
            EXPECT_EQ(p->legs_executed, slots[*q].leg + 1) << where;
            const Expected e = naive(slots, *q + 1);
            EXPECT_EQ(p->best_leg, slots[e.best].leg) << where;
            EXPECT_TRUE(same(p->weight, slots[e.best].weight)) << where;
            EXPECT_TRUE(std::ranges::equal(p->support, slots[e.best].support)) << where;
        }
    }
    EXPECT_GT(early, 150U);
}

// Under the class-sum and largest-agreement rules the decoder returns the lightest solution of
// the decided class, with a dense ê that matches it.
TEST(SelectionDecoder, ClassRulesReturnTheDecidedClassesLightestSolution) {
    // Two classes, so that several solutions often share one and the rules disagree.
    const Random r = random_problem(13, 50, 100, 250, 0.06, 0.5, 0x1);
    const Built b = build(r);
    auto capture = decoder_with(b, relay(20, 60), CaptureSink{.slots = {}, .classes = &r.classes});
    std::size_t replaced = 0;
    for (const SelectionRule rule : {SelectionRule::class_sum, SelectionRule::largest_agreement}) {
        SelectionConfig config;
        config.rule = rule;
        auto policy = decoder_with(b, relay(20, 60), selecting(config, r, b));
        for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
            const auto full = capture.decode(r.syndromes[shot], shot);
            const auto p = policy.decode(r.syndromes[shot], shot);
            ASSERT_TRUE(full && p);
            const std::vector<Slot>& slots = capture.sink().slots;
            if (slots.empty()) {
                EXPECT_FALSE(p->success);
                continue;
            }
            const Expected e = naive(slots, std::min<std::size_t>(slots.size(), 20));
            const std::uint64_t decided =
                rule == SelectionRule::class_sum ? e.class_sum_class : e.agreement_class;
            // The lightest slot of the decided class, earliest on ties.
            std::size_t pick = slots.size();
            for (std::size_t q = 0; q < std::min<std::size_t>(slots.size(), 20); ++q) {
                if (slots[q].klass == decided &&
                    (pick == slots.size() || slots[q].weight < slots[pick].weight)) {
                    pick = q;
                }
            }
            ASSERT_LT(pick, slots.size());
            const std::string where = std::format("{} shot {}", to_string(rule), shot);
            EXPECT_EQ(p->confidence->decided_class, decided) << where;
            EXPECT_EQ(p->best_leg, slots[pick].leg) << where;
            EXPECT_TRUE(same(p->weight, slots[pick].weight)) << where;
            EXPECT_TRUE(std::ranges::equal(p->support, slots[pick].support)) << where;
            std::vector<Bit> dense(r.h.cols, 0);
            for (const index_t j : slots[pick].support) {
                dense[j] = 1;
            }
            EXPECT_TRUE(std::ranges::equal(p->hard, dense)) << where;
            EXPECT_EQ(solution_class(r.classes, p->support), decided) << where;
            replaced += slots[pick].leg != *full->best_leg ? 1U : 0U;
        }
    }
    // Fixed seeds: a handful of decodes where the rules overrule the lowest weight, each checked
    // above.
    EXPECT_GT(replaced, 0U);
}

// A low-confidence decode runs up to extra_legs more legs after the relay rule is met; a
// confident one runs none.
TEST(SelectionDecoder, LowConfidenceDecodesRunExtraLegs) {
    const Random r = random_problem(17, 60, 110, 200, 0.05);
    const Built b = build(r);
    auto plain = decoder_with(b, relay(2, 80), NoSink{});
    // Low confidence = a correction heavier than the median one: about half the decodes extend,
    // and a lighter solution found by an extra leg can end the extension.
    std::vector<double> weights;
    for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
        const auto a = plain.decode(r.syndromes[shot], shot);
        ASSERT_TRUE(a);
        if (a->success) {
            weights.push_back(a->weight);
        }
    }
    ASSERT_FALSE(weights.empty());
    std::ranges::nth_element(weights,
                             weights.begin() + static_cast<std::ptrdiff_t>(weights.size() / 2));
    SelectionConfig config;
    config.capacity = 20;
    config.signal = ConfidenceSignal::weight;
    config.threshold = weights[weights.size() / 2];
    config.extra_legs = 6;
    auto policy = decoder_with(b, relay(2, 80), selecting(config, r, b));
    std::size_t extended = 0;
    for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
        const auto a = plain.decode(r.syndromes[shot], shot);
        const auto p = policy.decode(r.syndromes[shot], shot);
        ASSERT_TRUE(a && p);
        const Confidence& c = *p->confidence;
        const std::string where = std::format("shot {}", shot);
        // The first legs are identical: the extension only appends legs.
        ASSERT_GE(p->legs_executed, a->legs_executed) << where;
        EXPECT_EQ(p->legs_executed - a->legs_executed, c.extra_legs) << where;
        EXPECT_LE(c.extra_legs, 6U) << where;
        for (std::size_t leg = 0; leg < a->legs.size(); ++leg) {
            EXPECT_EQ(p->legs[leg].iterations, a->legs[leg].iterations) << where;
        }
        if (c.extra_legs > 0) {
            ++extended;
        } else if (a->success && a->legs_executed < 81) {
            // No extension: the decode was confident when the relay rule was met.
            EXPECT_FALSE(c.low && c.found == 2) << where;
        }
    }
    EXPECT_GT(extended, 10U);
}

// A sink whose confidence is low until it has seen `restore_at` solutions: the controller's side
// of the extension, independent of any rule.
struct ScriptedSink {
    static constexpr bool enabled = true;
    std::uint32_t restore_at = 4;
    std::uint32_t budget = 30;
    std::vector<std::uint32_t> legs; // the converged legs of the current decode

    void on_decode_begin() noexcept { legs.clear(); }
    // Allocation failure in a test double ends the test binary, which is all a test needs.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void on_solution(const SolutionEvent& event) noexcept { legs.push_back(event.leg); }
    [[nodiscard]] static bool stop_requested() noexcept { return false; }
    [[nodiscard]] bool low_confidence() const noexcept { return legs.size() < restore_at; }
    [[nodiscard]] std::uint32_t extra_legs() const noexcept { return budget; }
    [[nodiscard]] Selection finish(const DecodeFacts& facts) const noexcept {
        Selection out;
        out.confidence.extra_legs = facts.extra_legs;
        out.confidence.low = low_confidence();
        return out;
    }
};
static_assert(SelectingSink<ScriptedSink>);

// The extension ends at the solution that restores the confidence, or after its budget of legs,
// or at the end of the schedule.
TEST(SelectionDecoder, TheExtensionEndsWhenConfidenceIsRestored) {
    const Random r = random_problem(17, 60, 110, 200, 0.05);
    const Built b = build(r);
    constexpr std::uint32_t sets = 80;
    // A budget of 30 legs usually finds the two missing solutions; a budget of 1 never can.
    for (const std::uint32_t budget : {30U, 1U}) {
        auto policy = decoder_with(b, relay(2, sets),
                                   ScriptedSink{.restore_at = 4, .budget = budget, .legs = {}});
        std::size_t restored = 0;
        std::size_t exhausted = 0;
        for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
            const auto p = policy.decode(r.syndromes[shot], shot);
            ASSERT_TRUE(p);
            const std::vector<std::uint32_t>& legs = policy.sink().legs;
            const std::string where = std::format("budget {} shot {}", budget, shot);
            if (legs.size() < 2) {
                EXPECT_EQ(p->confidence->extra_legs, 0U) << where; // the relay rule was not met
                continue;
            }
            const std::uint32_t last = p->legs_executed - 1;
            if (legs.size() >= 4) {
                ASSERT_EQ(legs.size(), 4U) << where;
                EXPECT_EQ(last, legs[3]) << where;
                EXPECT_FALSE(p->confidence->low) << where;
                ++restored;
            } else {
                EXPECT_EQ(last, std::min(legs[1] + budget, sets)) << where;
                EXPECT_TRUE(p->confidence->low) << where;
                ++exhausted;
            }
            EXPECT_EQ(p->confidence->extra_legs, last - legs[1]) << where;
        }
        EXPECT_GT(budget == 1 ? exhausted : restored, 5U) << "budget " << budget;
        if (budget == 1) {
            EXPECT_EQ(restored, 0U);
        }
    }
}

// With a relay rule of S converged legs an extension adds at most one solution per leg, and the
// rules hold `capacity` of them: a decoder given more extra legs than the rules have free slots
// runs only as many as there are slots, so the rules see every solution and the confidence
// describes the returned one. Capacity 4 with 10 extra legs decodes exactly like 2 extra legs.
TEST(SelectionDecoder, TheExtensionNeverRunsPastTheRulesCapacity) {
    const Random r = random_problem(17, 60, 110, 200, 0.05);
    const Built b = build(r);
    // W* > −1 always holds: every decode that meets the relay rule is low and extends.
    const auto config = [](std::uint32_t capacity, std::uint32_t extra) {
        SelectionConfig c;
        c.capacity = capacity;
        c.signal = ConfidenceSignal::weight;
        c.threshold = -1.0;
        c.extra_legs = extra;
        return c;
    };
    auto tight = decoder_with(b, relay(2, 80), selecting(config(4, 10), r, b));
    auto roomy = decoder_with(b, relay(2, 80), selecting(config(20, 2), r, b));
    std::size_t filled = 0;
    std::size_t extended = 0;
    for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
        const auto t = tight.decode(r.syndromes[shot], shot);
        const auto w = roomy.decode(r.syndromes[shot], shot);
        ASSERT_TRUE(t && w);
        const std::string where = std::format("shot {}", shot);
        const Confidence& c = *t->confidence;
        EXPECT_LE(c.found, 4U) << where;
        EXPECT_EQ(c.seen, c.found) << where;
        EXPECT_LE(c.extra_legs, 2U) << where;
        EXPECT_EQ(c.extra_legs, w->confidence->extra_legs) << where;
        EXPECT_EQ(t->legs_executed, w->legs_executed) << where;
        EXPECT_EQ(t->iterations, w->iterations) << where;
        EXPECT_EQ(t->best_leg, w->best_leg) << where;
        if (t->success) {
            // ê* of the lowest-weight rule is the solution the controller returns.
            EXPECT_TRUE(same(c.weight, t->weight)) << where;
            EXPECT_EQ(c.best_class, solution_class(r.classes, t->support)) << where;
        }
        extended += c.extra_legs > 0 ? 1U : 0U;
        filled += c.found == 4 ? 1U : 0U;
    }
    EXPECT_GT(extended, 50U);
    EXPECT_GT(filled, 0U); // extensions whose two legs both converged, filling every slot
}

// A cap that ends a low-confidence extension is not a cap hit, because the relay rule was met
// before the extension began; the same cap one iteration before the rule is met is one.
TEST(SelectionDecoder, ACapEndingAnExtensionIsNotACapHit) {
    const Random r = random_problem(17, 60, 110, 200, 0.05);
    const Built b = build(r);
    constexpr std::uint32_t sets = 80;
    // Confidence is never restored: every decode that meets the relay rule extends by 30 legs.
    auto policy = decoder_with(b, relay(2, sets),
                               ScriptedSink{.restore_at = 100, .budget = 30, .legs = {}});
    std::size_t checked = 0;
    for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
        const auto full = policy.decode(r.syndromes[shot], shot);
        ASSERT_TRUE(full);
        const std::vector<std::uint32_t> legs = policy.sink().legs;
        if (legs.size() < 2 || legs[1] >= sets || full->legs_executed == legs[1] + 1) {
            continue; // the rule was never met, or no extension leg ran
        }
        EXPECT_FALSE(full->cap_hit);
        // Iterations up to and including the leg that met the rule.
        std::uint32_t met = 0;
        for (std::uint32_t leg = 0; leg <= legs[1]; ++leg) {
            met += full->legs[leg].iterations;
        }
        const std::string where = std::format("shot {}", shot);
        // At `met` no extension leg may start; at met + 3 the first one is cut short.
        for (const std::uint32_t cap : {met, met + 3}) {
            const auto p = policy.decode(r.syndromes[shot], shot,
                                         DecodeLimits{.max_total_iterations = cap});
            ASSERT_TRUE(p);
            EXPECT_FALSE(p->cap_hit) << where << " cap " << cap;
            EXPECT_LE(p->iterations, cap) << where;
            EXPECT_GE(p->legs_executed, legs[1] + 1) << where;
            EXPECT_EQ(p->confidence->extra_legs, p->legs_executed - 1 - legs[1]) << where;
        }
        const auto short_of_rule = policy.decode(r.syndromes[shot], shot,
                                                 DecodeLimits{.max_total_iterations = met - 1});
        ASSERT_TRUE(short_of_rule);
        EXPECT_TRUE(short_of_rule->cap_hit) << where;
        EXPECT_EQ(short_of_rule->confidence->extra_legs, 0U) << where;
        ++checked;
    }
    EXPECT_GT(checked, 10U);
}

// With a selecting sink the decode path still allocates nothing.
TEST(SelectionDecoder, DecodingAllocatesNothing) {
    if constexpr (!window::test::allocations_counted) {
        GTEST_SKIP() << "allocations are not counted under sanitizers";
    }
    const Random r = random_problem(21, 60, 110, 40, 0.05);
    const Built b = build(r);
    SelectionConfig config;
    config.rule = SelectionRule::class_sum;
    config.stop = StopRule::gap_extend;
    config.stop_count = 2;
    config.stop_gap = 2.0;
    config.signal = ConfidenceSignal::q_supp;
    config.threshold = 0.01;
    config.extra_legs = 3;
    auto policy = decoder_with(b, relay(5, 60), selecting(config, r, b));
    ASSERT_TRUE(policy.decode(r.syndromes[0], 0));
    const std::uint64_t before = window::test::allocations();
    for (std::size_t shot = 0; shot < r.syndromes.size(); ++shot) {
        ASSERT_TRUE(policy.decode(r.syndromes[shot], shot));
    }
    EXPECT_EQ(window::test::allocations(), before);
}

// ---- The window layer
// --------------------------------------------------------------------------------

// An exact inner decoder whose decodes carry a confidence that is low for chosen (window, attempt).
class LowInner {
public:
    LowInner(const window::Shape& shape, const std::vector<window::test::StreamKey>* low)
        : exact_(shape), low_(low) {}

    [[nodiscard]] index_t num_rows() const noexcept { return exact_.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return exact_.num_columns(); }
    void set_convergence_rows(std::span<const Bit> mask) noexcept {
        exact_.set_convergence_rows(mask);
    }
    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> syndrome, std::uint64_t stream, DecodeLimits limits) {
        auto result = exact_.decode(syndrome, stream, limits);
        if (result) {
            const window::test::StreamKey key = window::test::split_stream(stream);
            Confidence c;
            c.gap_state = result->success ? GapState::defined : GapState::none;
            c.low = std::ranges::find(*low_, key) != low_->end();
            result->confidence = c;
        }
        return result;
    }

private:
    window::test::BruteForceInner exact_;
    const std::vector<window::test::StreamKey>* low_;
};

static_assert(window::SyndromeDecoder<LowInner>);

TEST(SelectionWindows, LowConfidenceWindowsAreDeferredOrFlagged) {
    auto artifact = io::load_artifact(fixture_root / "toy_rep3");
    ASSERT_TRUE(artifact);
    const window::ArtifactProblem source(*artifact);
    const auto spec = [](window::OnFailure on_failure, std::uint32_t deferrals) {
        return window::WindowSpec{.width = 3,
                                  .commit = 1,
                                  .converge_rounds = 3,
                                  .boundary = window::Boundary::exact,
                                  .on_failure = on_failure,
                                  .max_deferrals = deferrals,
                                  .iteration_cap = std::nullopt};
    };
    auto defer_plan =
        window::WindowPlan::build(source.problem(), spec(window::OnFailure::defer, 2));
    auto flag_plan = window::WindowPlan::build(source.problem(), spec(window::OnFailure::flag, 0));
    ASSERT_TRUE(defer_plan && flag_plan);
    const std::uint64_t shot = 3;
    const auto key = [&](std::uint32_t w, std::uint32_t a) {
        return window::test::StreamKey{.shot = shot, .window = w, .attempt = a};
    };
    // Window 1 is low at attempts 0 and 1 and confident at 2; window 3 is low at every attempt.
    const std::vector<window::test::StreamKey> low{key(1, 0), key(1, 1), key(3, 0), key(3, 1),
                                                   key(3, 2)};
    const std::vector<Bit> zero(source.problem().num_rows, 0);

    auto make = [&](const window::WindowPlan& plan) {
        return window::StreamDecoder<LowInner>::create(
            plan, [&](const window::Shape& shape) { return LowInner(shape, &low); });
    };
    auto defer = make(*defer_plan);
    ASSERT_TRUE(defer);
    // Defer needs the plan's deferral attempts.
    auto flagging = make(*flag_plan);
    ASSERT_TRUE(flagging);
    EXPECT_EQ(flagging->set_on_low_confidence(window::OnLowConfidence::defer).error().code,
              window::StreamError::Code::invalid_low_confidence_action);
    ASSERT_TRUE(flagging->set_on_low_confidence(window::OnLowConfidence::flag));
    ASSERT_TRUE(defer->set_on_low_confidence(window::OnLowConfidence::defer));

    std::vector<window::WindowRecord> deferred_records;
    ASSERT_TRUE(window::decode_shot(*defer, zero, shot, [&](const window::Commit& c) {
        deferred_records.push_back(c.record);
    }));
    ASSERT_GE(deferred_records.size(), 4U);
    EXPECT_EQ(deferred_records[0].attempts, 1U);
    EXPECT_FALSE(deferred_records[0].low_confidence);
    EXPECT_EQ(deferred_records[1].attempts, 3U);
    EXPECT_EQ(deferred_records[1].low_confidence_deferrals, 2U);
    EXPECT_FALSE(deferred_records[1].low_confidence);
    EXPECT_FALSE(deferred_records[1].flagged);
    ASSERT_TRUE(deferred_records[1].confidence);
    // Window 3 runs out of attempts (or reaches a final placement) while still low: committed,
    // flagged.
    const window::WindowRecord& w3 = deferred_records[3];
    if (w3.attempts > 0) {
        EXPECT_TRUE(w3.low_confidence);
        EXPECT_TRUE(w3.flagged);
    }
    EXPECT_TRUE(defer->flagged() || w3.attempts == 0);

    std::vector<window::WindowRecord> flagged_records;
    ASSERT_TRUE(window::decode_shot(*flagging, zero, shot, [&](const window::Commit& c) {
        flagged_records.push_back(c.record);
    }));
    EXPECT_EQ(flagged_records[1].attempts, 1U);
    EXPECT_TRUE(flagged_records[1].low_confidence);
    EXPECT_TRUE(flagged_records[1].flagged);
    EXPECT_FALSE(flagged_records[0].flagged);
    EXPECT_TRUE(flagging->flagged());
    EXPECT_EQ(window::to_string(window::OnLowConfidence::flag), "flag");
    EXPECT_EQ(window::parse_on_low_confidence("defer"), window::OnLowConfidence::defer);
    EXPECT_FALSE(window::parse_on_low_confidence("abort"));
}

// ---- The history over the last L windows
// -------------------------------------------------------------

Confidence conf(GapState state, double gap, double agreement, double weight, double sum_sq,
                double total) {
    Confidence c;
    c.gap_state = state;
    c.gap = gap;
    c.agreement = agreement;
    c.weight = weight;
    c.first_legs = 2;
    c.first_iterations = 30;
    c.q_sum_sq = sum_sq;
    c.q_total = total;
    c.syndrome_ones = 3;
    c.syndrome_rows = 10;
    return c;
}

TEST(SignalHistory, CombinesTheLastLWindows) {
    using window::HistorySignal;
    const std::vector<std::uint32_t> lengths{1, 2, 3};
    const std::vector<HistorySignal> signals{HistorySignal::gap,     HistorySignal::agreement,
                                             HistorySignal::weight,  HistorySignal::q_supp,
                                             HistorySignal::density, HistorySignal::first_legs};
    auto history = window::SignalHistory::create(lengths, signals, nullptr, {}, 0);
    ASSERT_TRUE(history);
    history->reset();
    history->push(conf(GapState::defined, 2.0, 0.5, 10.0, 4.0, 100.0), true, {});
    EXPECT_EQ(history->value(0, 0), 2.0);
    EXPECT_EQ(history->state(2), GapState::defined);
    history->push(conf(GapState::single_class, std::nan(""), 1.0, 7.0, 9.0, 50.0), true, {});
    EXPECT_TRUE(std::isnan(history->value(0, 0))); // last window alone: undefined
    EXPECT_EQ(history->state(0), GapState::single_class);
    EXPECT_EQ(history->value(0, 1), 2.0);                // min over the defined ones
    EXPECT_EQ(history->value(1, 1), 0.5 * 1.0);          // product, oldest first
    EXPECT_EQ(history->value(2, 1), (0.0 + 10.0) + 7.0); // Σ W*
    EXPECT_EQ(history->value(3, 1), std::sqrt(0.0 + 4.0 + 9.0) / (0.0 + 100.0 + 50.0));
    EXPECT_EQ(history->value(4, 1), 6.0 / 20.0);
    EXPECT_EQ(history->value(5, 1), 4.0);
    // A window with no solution makes every decode signal over it unusable; a position that ran
    // no decode contributes nothing.
    history->push(conf(GapState::none, std::nan(""), std::nan(""), inf, std::nan(""), 1.0), true,
                  {});
    EXPECT_EQ(history->state(0), GapState::none);
    EXPECT_EQ(history->state(2), GapState::none);
    EXPECT_EQ(history->value(2, 2), inf);
    history->push(std::nullopt, false, {});
    EXPECT_EQ(history->state(0), GapState::none); // nothing decoded in the last window
    EXPECT_EQ(history->state(1), GapState::none); // the none window is still in range
    history->reset();
    EXPECT_TRUE(std::isnan(history->value(0, 0)));

    EXPECT_FALSE(window::SignalHistory::create({}, signals, nullptr, {}, 0));
    const std::vector<std::uint32_t> twice{2, 2};
    EXPECT_FALSE(window::SignalHistory::create(twice, signals, nullptr, {}, 0));
    const std::vector<HistorySignal> commits{HistorySignal::commit_q_supp};
    EXPECT_FALSE(window::SignalHistory::create(lengths, commits, nullptr, {}, 4));
    for (const HistorySignal s : signals) {
        EXPECT_EQ(window::parse_history_signal(window::to_string(s)), s);
    }
}

TEST(SignalHistory, CommitSignalsUseTheUnionOfTheCommits) {
    using window::HistorySignal;
    const test::Csr h = test::csr_from_dense({{1, 1, 0, 0}, {0, 0, 1, 1}, {0, 1, 1, 0}});
    auto graph = TannerGraph::from_csr(h.rows, h.cols, h.row_ptr, h.col_idx);
    ASSERT_TRUE(graph);
    const std::vector<double> llr{1.0, 2.0, 3.0, 4.0};
    const std::vector<std::uint32_t> lengths{1, 2};
    const std::vector<HistorySignal> signals{HistorySignal::commit_weight,
                                             HistorySignal::commit_q_supp};
    auto history = window::SignalHistory::create(lengths, signals, &*graph, llr, 2);
    ASSERT_TRUE(history);
    history->reset();
    const std::vector<index_t> first{2, 3};
    const std::vector<index_t> second{0};
    history->push(std::nullopt, true, first);
    EXPECT_EQ(history->value(0, 0), 7.0);
    history->push(std::nullopt, true, second);
    EXPECT_EQ(history->value(0, 0), 1.0);
    EXPECT_EQ(history->value(0, 1), ((0.0 + 1.0) + 3.0) + 4.0); // ascending over {0, 2, 3}
    // {0} and {2, 3} are separate components: sqrt(1 + 49) / 10.
    EXPECT_EQ(history->value(1, 1), std::sqrt(1.0 + 49.0) / 10.0);
}

} // namespace
