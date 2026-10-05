// The total-iteration budget of a decode (DecodeLimits).
//
// A capped decode must equal what the uncapped decode had found after `cap` iterations: the legs
// whose cumulative count stays within the cap are unchanged, the leg that crosses it is cut at the
// cap without converging, and no further leg starts. The tests derive that prediction from the
// uncapped per-leg trace and compare, first against a model backend with scripted legs (every
// edge case of the rule), then against the real CPU backend on random problems.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

constexpr double inf = std::numeric_limits<double>::infinity();

std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

// ---- Model backend ------------------------------------------------------------------------------

// What leg r of a decode does: it converges at iteration `converges_at` with solution `support`
// if its budget allows, and otherwise runs its whole budget without converging.
struct LegPlan {
    std::optional<std::uint32_t> converges_at;
    double weight = inf;
    std::vector<index_t> support;
};

// Responds to the iteration budget the way a real backend does, so a capped decode can be checked
// against the same script decoded without a cap.
class ModelBackend {
public:
    ModelBackend(index_t columns, std::vector<LegPlan> plan)
        : columns_(columns), plan_(std::move(plan)), best_dense_(columns, 0),
          current_dense_(columns, 0) {}

    [[nodiscard]] static index_t num_rows() noexcept { return 2; }
    [[nodiscard]] index_t num_columns() const noexcept { return columns_; }
    void set_convergence_rows(std::span<const Bit> /*mask*/) noexcept {}
    void begin(std::span<const Bit> /*syndrome*/, bool /*init_marginals*/) noexcept {
        leg_ = 0;
        budgets.clear();
        current_.clear();
        best_.clear();
    }
    void set_gamma(std::span<const double> /*gammas*/) noexcept {}
    void set_gamma(double /*gamma*/) noexcept {}
    [[nodiscard]] LegOutcome run_leg(const LegParams& params) noexcept {
        budgets.push_back(params.max_iter);
        const LegPlan& leg = plan_.at(leg_);
        ++leg_;
        if (leg.converges_at && *leg.converges_at <= params.max_iter) {
            current_ = leg.support;
            return {.converged = true, .iterations = *leg.converges_at, .weight = leg.weight};
        }
        // An unconverged leg's final ê depends on how long it ran.
        current_ = {std::min(leg_ - 1, params.max_iter) % columns_};
        if (const index_t other = (leg_ + params.max_iter) % columns_; other != current_[0]) {
            current_.push_back(other);
        }
        std::ranges::sort(current_);
        return {.converged = false, .iterations = params.max_iter, .weight = inf};
    }
    void mark_best() noexcept { best_ = current_; }
    [[nodiscard]] std::span<const Bit> best_hard() noexcept { return dense(best_, best_dense_); }
    [[nodiscard]] std::span<const Bit> current_hard() noexcept {
        return dense(current_, current_dense_);
    }
    [[nodiscard]] std::span<const index_t> best_support() const noexcept { return best_; }
    [[nodiscard]] std::span<const index_t> current_support() noexcept { return current_; }
    void read_marginals(std::span<double> /*out*/) const noexcept {}

    std::vector<std::uint32_t> budgets; // max_iter handed to each leg of the last decode

private:
    static std::span<const Bit> dense(const std::vector<index_t>& support,
                                      std::vector<Bit>& out) noexcept {
        std::ranges::fill(out, Bit{0});
        for (const index_t j : support) {
            out[j] = 1;
        }
        return out;
    }

    index_t columns_;
    std::vector<LegPlan> plan_;
    std::vector<index_t> current_;
    std::vector<index_t> best_;
    std::vector<Bit> best_dense_;
    std::vector<Bit> current_dense_;
    std::uint32_t leg_ = 0;
};

static_assert(LegBackend<ModelBackend>);

LegPlan converges(std::uint32_t at, double weight, std::vector<index_t> support) {
    return {.converges_at = at, .weight = weight, .support = std::move(support)};
}
LegPlan never() { return {}; }

// Tags every γ row so that a relay leg can run; the model ignores the values.
ExplicitGammaTable gamma_table(std::size_t width) {
    return *ExplicitGammaTable::create(std::vector<double>(width, 0.25), 1, width);
}

constexpr index_t model_columns = 16;
const MinSumConfig memory{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
const std::vector<Bit> model_syndrome{1, 0};

using ModelDecoder = RelayDecoder<ModelBackend>;

ModelDecoder model_decoder(std::vector<LegPlan> plan, const RelayConfig& relay,
                           const GammaSource* gammas) {
    auto decoder = ModelDecoder::create(ModelBackend(model_columns, std::move(plan)), memory,
                                        relay, gammas);
    EXPECT_TRUE(decoder);
    return std::move(*decoder);
}

DecodeResult capped(ModelDecoder& decoder, std::optional<std::uint32_t> cap) {
    auto r = decoder.decode(model_syndrome, 0, DecodeLimits{.max_total_iterations = cap});
    EXPECT_TRUE(r);
    return *r;
}

std::vector<index_t> copy(std::span<const index_t> s) { return {s.begin(), s.end()}; }

// ---- Prediction from the uncapped trace --------------------------------------------------------

struct Prediction {
    std::vector<LegRecord> legs;
    std::uint32_t iterations = 0;
    std::optional<std::uint32_t> best_leg;
    double weight = inf;
    bool cap_hit = false;
    bool last_cut = false; // the last leg was cut short by the cap
};

Prediction predict(std::span<const LegRecord> uncapped, std::uint32_t cap) {
    Prediction p;
    std::uint32_t total = 0;
    for (const LegRecord& leg : uncapped) {
        total += leg.iterations;
    }
    p.cap_hit = total > cap;
    for (const LegRecord& leg : uncapped) {
        if (p.iterations == cap) {
            break;
        }
        const std::uint32_t remaining = cap - p.iterations;
        LegRecord record = leg;
        if (leg.iterations > remaining) {
            record = LegRecord{
                .iterations = remaining, .converged = false, .became_best = false, .weight = inf};
            p.last_cut = true;
        }
        p.legs.push_back(record);
        p.iterations += record.iterations;
        if (record.converged && record.weight < p.weight) {
            p.weight = record.weight;
            p.best_leg = static_cast<std::uint32_t>(p.legs.size() - 1);
        }
        if (p.last_cut) {
            break;
        }
    }
    return p;
}

void expect_matches(const DecodeResult& got, const Prediction& want, const std::string& where) {
    EXPECT_EQ(got.legs_executed, want.legs.size()) << where;
    EXPECT_EQ(got.iterations, want.iterations) << where;
    EXPECT_EQ(got.best_leg, want.best_leg) << where;
    EXPECT_EQ(got.success, want.best_leg.has_value()) << where;
    EXPECT_EQ(bits_of(got.weight), bits_of(want.weight)) << where;
    EXPECT_EQ(got.cap_hit, want.cap_hit) << where;
    ASSERT_EQ(got.legs.size(), want.legs.size()) << where;
    for (std::size_t i = 0; i < want.legs.size(); ++i) {
        EXPECT_EQ(got.legs[i].iterations, want.legs[i].iterations) << where << " leg " << i;
        EXPECT_EQ(got.legs[i].converged, want.legs[i].converged) << where << " leg " << i;
        EXPECT_EQ(got.legs[i].became_best, want.legs[i].became_best) << where << " leg " << i;
        EXPECT_EQ(bits_of(got.legs[i].weight), bits_of(want.legs[i].weight))
            << where << " leg " << i;
    }
}

// ---- Edge cases on the model --------------------------------------------------------------------

TEST(DecodeLimits, NoCapIsTheOldPath) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig relay{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 4, .stopping = AfterNConverged{2}};
    const std::vector<LegPlan> plan{never(), converges(3, 4.0, {1, 2}), never(),
                                    converges(5, 2.0, {3}), never()};
    ModelDecoder a = model_decoder(plan, relay, &gammas);
    ModelDecoder b = model_decoder(plan, relay, &gammas);
    auto plain = a.decode(model_syndrome, 7);
    ASSERT_TRUE(plain);
    const DecodeResult limited = capped(b, std::nullopt);
    EXPECT_FALSE(plain->cap_hit);
    EXPECT_EQ(plain->iterations, 10U + 3 + 5 + 5);
    EXPECT_EQ(plain->legs_executed, 4U);
    EXPECT_EQ(plain->best_leg, 3U);
    EXPECT_EQ(copy(plain->support), (std::vector<index_t>{3}));
    expect_matches(limited, predict(plain->legs, plain->iterations), "no cap");
    EXPECT_EQ(a.backend().budgets, (std::vector<std::uint32_t>{10, 5, 5, 5}));
    EXPECT_EQ(b.backend().budgets, a.backend().budgets);
}

TEST(DecodeLimits, CapBelowPreIterCutsLegZero) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig relay{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterNConverged{1}};
    ModelDecoder decoder = model_decoder(
        {converges(7, 1.0, {4}), converges(1, 0.5, {5}), never(), never()}, relay, &gammas);
    const DecodeResult r = capped(decoder, 6);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.cap_hit);
    EXPECT_EQ(r.iterations, 6U);
    EXPECT_EQ(r.legs_executed, 1U);
    EXPECT_FALSE(r.legs[0].converged);
    EXPECT_EQ(r.weight, inf);
    EXPECT_EQ(decoder.backend().budgets, std::vector<std::uint32_t>{6});
    // Leg 0's final ê after the 6 iterations it ran is what comes back.
    EXPECT_EQ(copy(r.support), (std::vector<index_t>{0, 7}));
    EXPECT_EQ(r.hard[0], 1);
    EXPECT_EQ(r.hard[7], 1);

    // A cap that still lets leg 0 converge changes nothing.
    const DecodeResult enough = capped(decoder, 7);
    EXPECT_TRUE(enough.success);
    EXPECT_FALSE(enough.cap_hit);
    EXPECT_EQ(enough.iterations, 7U);
    EXPECT_EQ(copy(enough.support), std::vector<index_t>{4});
}

TEST(DecodeLimits, LegZeroConvergingAtTheCapWithMoreLegsToRun) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig relay{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterNConverged{2}};
    ModelDecoder decoder = model_decoder(
        {converges(7, 1.0, {4}), converges(2, 0.5, {5}), never(), never()}, relay, &gammas);
    // The budget is spent exactly as leg 0 converges; leg 1 would have run.
    const DecodeResult r = capped(decoder, 7);
    EXPECT_TRUE(r.success);
    EXPECT_TRUE(r.cap_hit);
    EXPECT_EQ(r.best_leg, 0U);
    EXPECT_EQ(r.legs_executed, 1U);
    EXPECT_EQ(copy(r.support), std::vector<index_t>{4});
    // One more iteration is not enough for leg 1 (it needs 2): cut short, earlier solution kept.
    const DecodeResult cut = capped(decoder, 8);
    EXPECT_TRUE(cut.success);
    EXPECT_TRUE(cut.cap_hit);
    EXPECT_EQ(cut.legs_executed, 2U);
    EXPECT_EQ(cut.iterations, 8U);
    EXPECT_EQ(cut.best_leg, 0U);
    EXPECT_EQ(decoder.backend().budgets, (std::vector<std::uint32_t>{8, 1}));
    // Two more: leg 1 converges on the last allowed iteration and the stopping rule is met.
    const DecodeResult done = capped(decoder, 9);
    EXPECT_TRUE(done.success);
    EXPECT_FALSE(done.cap_hit);
    EXPECT_EQ(done.best_leg, 1U);
    EXPECT_EQ(done.iterations, 9U);
    EXPECT_EQ(copy(done.support), std::vector<index_t>{5});
}

TEST(DecodeLimits, CapExactlyAtALegBoundary) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig three{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterNConverged{1}};
    const std::vector<LegPlan> plan{never(), never(), converges(2, 3.0, {9}), never()};
    ModelDecoder decoder = model_decoder(plan, three, &gammas);
    // Legs 0 and 1 use exactly the budget; leg 2 would have run.
    const DecodeResult r = capped(decoder, 15);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.cap_hit);
    EXPECT_EQ(r.legs_executed, 2U);
    EXPECT_EQ(r.iterations, 15U);
    EXPECT_EQ(decoder.backend().budgets, (std::vector<std::uint32_t>{10, 5}));
    // Leg 0 ran its full budget, so its final ê is the uncapped one.
    EXPECT_EQ(copy(r.support), (std::vector<index_t>{0, 11}));

    // The same boundary at the end of the schedule: nothing more would have run.
    const RelayConfig one{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 1, .stopping = AfterNConverged{1}};
    ModelDecoder last = model_decoder({never(), never()}, one, &gammas);
    const DecodeResult end = capped(last, 15);
    EXPECT_FALSE(end.success);
    EXPECT_FALSE(end.cap_hit);
    EXPECT_EQ(end.legs_executed, 2U);

    // A leg that converges on the boundary but leaves the stopping rule unsatisfied.
    const RelayConfig nconv{
        .pre_iter = 10, .set_max_iter = 5, .num_sets = 3, .stopping = AfterNConverged{3}};
    ModelDecoder more = model_decoder(
        {converges(4, 2.0, {1}), converges(5, 1.0, {2}), converges(1, 0.1, {3}), never()},
        nconv, &gammas);
    const DecodeResult boundary = capped(more, 9);
    EXPECT_TRUE(boundary.success);
    EXPECT_TRUE(boundary.cap_hit);
    EXPECT_EQ(boundary.best_leg, 1U);
    EXPECT_EQ(boundary.legs_executed, 2U);
    EXPECT_EQ(copy(boundary.support), std::vector<index_t>{2});
}

TEST(DecodeLimits, CapHitWithoutAnySolutionReturnsLegZerosFinalCorrection) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig relay{
        .pre_iter = 4, .set_max_iter = 3, .num_sets = 5, .stopping = AfterLeg0{}};
    ModelDecoder decoder =
        model_decoder({never(), never(), never(), never(), never(), never()}, relay, &gammas);
    const DecodeResult r = capped(decoder, 9);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.cap_hit);
    EXPECT_EQ(r.iterations, 9U);
    EXPECT_EQ(r.legs_executed, 3U);
    EXPECT_EQ(decoder.backend().budgets, (std::vector<std::uint32_t>{4, 3, 2}));
    EXPECT_EQ(copy(r.support), (std::vector<index_t>{0, 5}));
    // The whole schedule is 4 + 5·3 = 19 iterations.
    EXPECT_TRUE(capped(decoder, 18).cap_hit);
    EXPECT_FALSE(capped(decoder, 19).cap_hit);
    EXPECT_FALSE(capped(decoder, 20).cap_hit);
    EXPECT_EQ(capped(decoder, 20).iterations, 19U);
}

TEST(DecodeLimits, ZeroCapRunsNoLeg) {
    const ExplicitGammaTable gammas = gamma_table(model_columns);
    const RelayConfig relay{
        .pre_iter = 4, .set_max_iter = 3, .num_sets = 1, .stopping = AfterLeg0{}};
    ModelDecoder decoder = model_decoder({converges(1, 1.0, {2}), never()}, relay, &gammas);
    const DecodeResult r = capped(decoder, 0);
    EXPECT_FALSE(r.success);
    EXPECT_TRUE(r.cap_hit);
    EXPECT_EQ(r.iterations, 0U);
    EXPECT_EQ(r.legs_executed, 0U);
    EXPECT_TRUE(r.legs.empty());
    EXPECT_TRUE(r.support.empty());
    EXPECT_EQ(r.weight, inf);
    EXPECT_TRUE(std::ranges::all_of(r.hard, [](Bit b) { return b == 0; }));
    EXPECT_TRUE(decoder.backend().budgets.empty());
}

// Random scripts under every stopping rule and every cap from 0 to past the end.
TEST(DecodeLimits, ModelDecodesEqualThePredictionFromTheUncappedTrace) {
    std::mt19937_64 rng(2024);
    std::uniform_int_distribution<std::uint32_t> coin(0, 2);
    std::uniform_int_distribution<std::uint32_t> weight_level(1, 4); // few levels: many ties
    std::size_t cut_cases = 0;
    std::size_t boundary_cases = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const std::uint32_t pre = 1 + static_cast<std::uint32_t>(rng() % 6);
        const std::uint32_t set = 1 + static_cast<std::uint32_t>(rng() % 4);
        const auto sets = static_cast<std::uint32_t>(rng() % 6);
        StoppingRule stopping = AfterLeg0{};
        switch (rng() % 3) {
        case 0:
            break;
        case 1:
            stopping = AfterNConverged{1 + static_cast<std::uint32_t>(rng() % 3)};
            break;
        default:
            stopping = AllLegs{};
            break;
        }
        std::vector<LegPlan> plan;
        for (std::uint32_t leg = 0; leg <= sets; ++leg) {
            const std::uint32_t budget = leg == 0 ? pre : set;
            if (coin(rng) == 0) {
                plan.push_back(never());
            } else {
                const auto at = 1 + static_cast<std::uint32_t>(rng() % budget);
                plan.push_back(converges(at, weight_level(rng), {leg % model_columns}));
            }
        }
        const ExplicitGammaTable gammas = gamma_table(model_columns);
        const RelayConfig relay{
            .pre_iter = pre, .set_max_iter = set, .num_sets = sets, .stopping = stopping};
        ModelDecoder decoder = model_decoder(plan, relay, &gammas);
        const DecodeResult full = capped(decoder, std::nullopt);
        const std::vector<LegRecord> trace(full.legs.begin(), full.legs.end());
        const std::uint32_t total = full.iterations;
        for (std::uint32_t cap = 0; cap <= total + 2; ++cap) {
            const std::string where = std::format("trial {} cap {} of {}", trial, cap, total);
            const Prediction want = predict(trace, cap);
            const DecodeResult got = capped(decoder, cap);
            expect_matches(got, want, where);
            // Each leg got min(its budget, what remained).
            std::uint32_t used = 0;
            const std::vector<std::uint32_t>& budgets = decoder.backend().budgets;
            ASSERT_EQ(budgets.size(), got.legs_executed) << where;
            for (std::size_t leg = 0; leg < budgets.size(); ++leg) {
                const std::uint32_t own = leg == 0 ? pre : set;
                EXPECT_EQ(budgets[leg], std::min(own, cap - used)) << where << " leg " << leg;
                used += got.legs[leg].iterations;
            }
            if (want.best_leg) {
                EXPECT_EQ(copy(got.support), plan[*want.best_leg].support) << where;
            }
            cut_cases += want.last_cut ? 1U : 0U;
            boundary_cases += (want.cap_hit && !want.last_cut) ? 1U : 0U;
        }
    }
    EXPECT_GT(cut_cases, 100U);
    EXPECT_GT(boundary_cases, 100U);
}

// ---- The real backend ---------------------------------------------------------------------------

// Keeps each converged leg's support so a capped decode's answer can be checked against it.
struct SupportCapture {
    static constexpr bool enabled = true;
    std::vector<std::vector<index_t>> by_leg; // empty for legs that did not converge

    void on_decode_begin() noexcept { by_leg.clear(); }
    // Allocation failure in a test double ends the test binary, which is all a test needs.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void on_solution(const SolutionEvent& event) noexcept {
        by_leg.resize(std::max<std::size_t>(by_leg.size(), event.leg + std::size_t{1}));
        by_leg[event.leg].assign(event.support.begin(), event.support.end());
    }
};

struct Problem {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::vector<Bit>> syndromes;
};

Problem random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots) {
    Problem problem{.h = test::random_csr(m, n, 1, 6, seed), .p = {}, .syndromes = {}};
    std::mt19937_64 rng(seed * 104729);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        problem.p.push_back(unit(rng) < 0.04 ? 0.6 : 0.002 + 0.1 * unit(rng));
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
     .relay = {.pre_iter = 12, .set_max_iter = 8, .num_sets = 15, .stopping = AfterNConverged{3}}},
    {.name = "relay_all_adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 6, .set_max_iter = 6, .num_sets = 8, .stopping = AllLegs{}}},
    {.name = "relay_after_leg0",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = -0.1},
     .relay = {.pre_iter = 5, .set_max_iter = 5, .num_sets = 6, .stopping = AfterLeg0{}}},
};

// Every cap worth testing for one uncapped trace: 0 and 1, both sides of every leg boundary and of
// the end, and a few in between.
std::set<std::uint32_t> interesting_caps(std::span<const LegRecord> trace, std::mt19937_64& rng) {
    std::set<std::uint32_t> caps{0, 1, 2};
    std::uint32_t cumulative = 0;
    for (const LegRecord& leg : trace) {
        cumulative += leg.iterations;
        for (const std::uint32_t c : {cumulative - 1, cumulative, cumulative + 1}) {
            caps.insert(c);
        }
    }
    for (int k = 0; k < 4; ++k) {
        caps.insert(static_cast<std::uint32_t>(rng() % (cumulative + 3)));
    }
    return caps;
}

struct CapCoverage {
    std::size_t decodes = 0;
    std::size_t cut_with_solution = 0;
    std::size_t cut_without_solution = 0;
    std::size_t leg0_cut = 0;
    std::size_t boundary = 0;
};

template <class A, class Exec, class MakeExecutor>
void check_caps_against_trace(const Problem& problem, const GraphOptions& options,
                              MakeExecutor make_executor, const std::string& label,
                              CapCoverage& coverage) {
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, options);
    ASSERT_TRUE(graph) << graph.error().detail;
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(11, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    std::mt19937_64 rng(99);
    std::vector<double> m_full(problem.h.cols);
    std::vector<double> m_capped(problem.h.cols);

    for (const Scenario& scenario : scenarios) {
        auto reference = RelayDecoder<CpuBackend<A, Exec>, SupportCapture>::create(
            *CpuBackend<A, Exec>::create(*graph, *priors, make_executor()), scenario.min_sum,
            scenario.relay, &*gammas);
        auto decoder = RelayDecoder<CpuBackend<A, Exec>>::create(
            *CpuBackend<A, Exec>::create(*graph, *priors, make_executor()), scenario.min_sum,
            scenario.relay, &*gammas);
        ASSERT_TRUE(reference && decoder);
        for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
            const auto& syndrome = problem.syndromes[s];
            auto full = reference->decode(syndrome, s);
            ASSERT_TRUE(full);
            ASSERT_FALSE(full->cap_hit);
            const std::vector<LegRecord> trace(full->legs.begin(), full->legs.end());
            const std::vector<std::vector<index_t>> supports = reference->sink().by_leg;
            const std::vector<index_t> leg0_final = copy(full->support);
            reference->backend().read_marginals(m_full);

            for (const std::uint32_t cap : interesting_caps(trace, rng)) {
                const std::string where = std::format("{} {} shot {} cap {} of {}", label,
                                                      scenario.name, s, cap, full->iterations);
                auto got = decoder->decode(syndrome, s, DecodeLimits{.max_total_iterations = cap});
                ASSERT_TRUE(got) << where;
                const Prediction want = predict(trace, cap);
                expect_matches(*got, want, where);
                ++coverage.decodes;
                if (want.best_leg) {
                    EXPECT_EQ(copy(got->support), supports.at(*want.best_leg)) << where;
                } else if (!want.legs.empty() && want.legs[0].iterations == trace[0].iterations &&
                           !full->success) {
                    // Leg 0 ran its full course and nothing converged in either run.
                    EXPECT_EQ(copy(got->support), leg0_final) << where;
                }
                // The dense and sparse forms of the answer agree.
                std::vector<Bit> dense(problem.h.cols, 0);
                for (const index_t j : got->support) {
                    dense[j] = 1;
                }
                EXPECT_TRUE(std::ranges::equal(got->hard, dense)) << where;
                if (cap >= full->iterations) {
                    decoder->backend().read_marginals(m_capped);
                    EXPECT_EQ(m_capped, m_full) << where;
                    EXPECT_EQ(copy(got->support), copy(full->support)) << where;
                }
                if (want.last_cut) {
                    coverage.leg0_cut += want.legs.size() == 1 ? 1U : 0U;
                    coverage.cut_with_solution += want.best_leg ? 1U : 0U;
                    coverage.cut_without_solution += want.best_leg ? 0U : 1U;
                } else if (want.cap_hit) {
                    ++coverage.boundary;
                }
            }
        }
    }
}

TEST(DecodeLimits, CpuDecodesEqualThePredictionFromTheUncappedTrace) {
    CapCoverage coverage;
    const GraphOptions wavefront{
        .layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 4};
    const GraphOptions blocked{.layout = EdgeLayout::column_blocked,
                               .column_order = ColumnOrder::degree_classes,
                               .block_rows = 64};
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const Problem problem = random_problem(seed, 36, 150, 10);
        const auto serial = [] { return Serial{}; };
        const std::string label = std::format("seed {}", seed);
        check_caps_against_trace<F32, Serial>(problem, wavefront, serial, label + " f32",
                                              coverage);
        check_caps_against_trace<F64, Serial>(problem, blocked, serial, label + " f64 blocked",
                                              coverage);
    }
    EXPECT_GT(coverage.cut_with_solution, 10U);
    EXPECT_GT(coverage.cut_without_solution, 10U);
    EXPECT_GT(coverage.leg0_cut, 10U);
    EXPECT_GT(coverage.boundary, 10U);
}

TEST(DecodeLimits, TeamDecodesEqualThePredictionFromTheUncappedTrace) {
    CapCoverage coverage;
    const GraphOptions wavefront{
        .layout = EdgeLayout::row_major, .column_order = ColumnOrder::wavefront, .block_rows = 4};
    const Problem problem = random_problem(4, 36, 150, 6);
    const auto team = [] { return Team(3); };
    check_caps_against_trace<F32, Team>(problem, wavefront, team, "team f32", coverage);
    EXPECT_GT(coverage.decodes, 100U);
}

// When leg 0 is cut and nothing converged, the answer is leg 0's ê after `cap` iterations: the
// same computation as a decoder whose leg 0 has exactly `cap` iterations and no relay legs.
TEST(DecodeLimits, CutLegZeroEqualsAShorterLegZero) {
    const Problem problem = random_problem(5, 36, 150, 12);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx);
    auto priors = Priors::from_probabilities(problem.p);
    auto gammas = UniformGammaGenerator::create(3, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(graph && priors && gammas);
    const MinSumConfig min_sum{.alpha = AdaptiveAlpha{2.0}, .gamma0 = 0.2};
    auto decoder = CpuRelayDecoder<F64>::create(
        *CpuBackend<F64>::create(*graph, *priors), min_sum,
        {.pre_iter = 40, .set_max_iter = 10, .num_sets = 5, .stopping = AfterNConverged{2}},
        &*gammas);
    ASSERT_TRUE(decoder);
    std::size_t compared = 0;
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        for (const std::uint32_t cap : {1U, 2U, 5U, 13U, 39U}) {
            auto got = decoder->decode(problem.syndromes[s], s, {.max_total_iterations = cap});
            ASSERT_TRUE(got);
            if (got->success) {
                continue;
            }
            auto shorter = CpuRelayDecoder<F64>::create(
                *CpuBackend<F64>::create(*graph, *priors), min_sum,
                {.pre_iter = cap, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}},
                nullptr);
            ASSERT_TRUE(shorter);
            auto want = shorter->decode(problem.syndromes[s], s);
            ASSERT_TRUE(want);
            EXPECT_TRUE(got->cap_hit);
            EXPECT_EQ(copy(got->support), copy(want->support)) << "shot " << s << " cap " << cap;
            ++compared;
        }
    }
    EXPECT_GT(compared, 5U);
}

} // namespace
