// Window plans of random time-structured problems under random specifications, checked against
// the definitions (see window_support.hpp): the partition of the columns, the committed-round
// rows, merging, the detector rule, the identity limit, shape sharing in the bulk and the
// uniform boundary's column mapping.

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <limits>
#include <random>
#include <vector>

#include "rtd/window/plan.hpp"
#include "window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;

constexpr std::uint64_t seeds = 400;

struct Draw {
    test::OwnedProblem problem;
    WindowSpec spec;
};

std::uint32_t uniform_int(std::mt19937_64& rng, std::uint32_t lo, std::uint32_t hi) {
    return std::uniform_int_distribution<std::uint32_t>(lo, hi)(rng);
}

// A random spec with 1 ≤ C < W ≤ max_width, C ≤ C′ ≤ W and a random policy (with up to three
// deferrals when it is defer).
WindowSpec random_spec(std::mt19937_64& rng, std::uint32_t max_width, Boundary boundary) {
    WindowSpec spec;
    spec.width = uniform_int(rng, 2, max_width);
    spec.commit = uniform_int(rng, 1, spec.width - 1);
    spec.converge_rounds = uniform_int(rng, spec.commit, spec.width);
    spec.boundary = boundary;
    spec.on_failure = static_cast<OnFailure>(uniform_int(rng, 0, 2));
    spec.max_deferrals = spec.on_failure == OnFailure::defer ? uniform_int(rng, 0, 3) : 0;
    return spec;
}

std::string describe_spec(const WindowSpec& s) {
    return std::format("W={} C={} C'={} {} {} D={}", s.width, s.commit, s.converge_rounds,
                       to_string(s.boundary), to_string(s.on_failure), s.max_deferrals);
}

// Whether a uniform plan exists: t_1 + W_a ≤ R for the widest attempt.
bool uniform_fits(const WindowSpec& s, std::uint32_t rounds_total) {
    return 1 + s.commit + s.width + s.max_deferrals * s.commit <= rounds_total - 1;
}

TEST(WindowProperties, RandomExactPlansSatisfyTheInvariants) {
    for (std::uint64_t seed = 0; seed < seeds; ++seed) {
        std::mt19937_64 rng(seed);
        const index_t rounds = uniform_int(rng, 2, 9);
        const auto problem = test::random_problem(seed, rounds, uniform_int(rng, 1, 4),
                                                  uniform_int(rng, 1, 6));
        const WindowSpec spec = random_spec(rng, rounds + 2, Boundary::exact);
        SCOPED_TRACE(std::format("seed {}, Rt = {}, {} columns, {}", seed, rounds,
                                 problem.columns.size(), describe_spec(spec)));
        auto plan = WindowPlan::build(problem.view(), spec);
        ASSERT_TRUE(plan) << describe(plan.error());
        test::check_exact_plan(problem.view(), spec, *plan);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
    }
}

TEST(WindowProperties, WidthCoveringTheShotReproducesTheProblem) {
    for (std::uint64_t seed = 0; seed < seeds; ++seed) {
        std::mt19937_64 rng(seed + 1000);
        const index_t rounds = uniform_int(rng, 1, 8);
        const auto problem = test::random_problem(seed + 1000, rounds, uniform_int(rng, 1, 4),
                                                  uniform_int(rng, 1, 6));
        const Problem view = problem.view();
        WindowSpec spec = random_spec(rng, 12, Boundary::exact);
        spec.width = std::max({spec.width, std::uint32_t{rounds}, std::uint32_t{2}});
        spec.converge_rounds = spec.width;
        SCOPED_TRACE(std::format("seed {}, Rt = {}, {}", seed, rounds, describe_spec(spec)));
        auto plan = WindowPlan::build(view, spec);
        ASSERT_TRUE(plan) << describe(plan.error());
        ASSERT_EQ(plan->num_positions(), 1U);
        ASSERT_EQ(plan->schedule().size(), 1U) << "a final window has no deferral attempts";
        ASSERT_EQ(plan->shapes().size(), 1U);
        const Shape& shape = plan->shapes()[0];
        EXPECT_TRUE(std::ranges::equal(shape.row_ptr(), view.h_row_ptr));
        EXPECT_TRUE(std::ranges::equal(shape.col_indices(), view.h_col_indices));
        EXPECT_TRUE(test::bitwise_equal(shape.priors().probabilities(), view.priors));
        EXPECT_TRUE(std::ranges::equal(shape.commit_class(), problem.classes));
        EXPECT_EQ(shape.committed_columns(), view.num_columns);
        EXPECT_EQ(shape.merged_columns(), 0U);
        EXPECT_TRUE(std::ranges::all_of(shape.converge(), [](Bit b) { return b == 1; }));
        const Placement& only = plan->schedule()[0];
        for (index_t j = 0; j < view.num_columns; ++j) {
            ASSERT_EQ(only.columns[j], j);
        }
    }
}

TEST(WindowProperties, BulkWindowsShareAShapeOnTranslationInvariantProblems) {
    for (std::uint64_t seed = 0; seed < seeds; ++seed) {
        std::mt19937_64 rng(seed + 2000);
        const index_t rounds = uniform_int(rng, 5, 16);
        const auto problem = test::translation_invariant_problem(
            seed + 2000, rounds, uniform_int(rng, 1, 4), uniform_int(rng, 1, 6));
        const Problem view = problem.view();
        const WindowSpec spec = random_spec(rng, 6, Boundary::exact);
        SCOPED_TRACE(std::format("seed {}, Rt = {}, {}", seed, rounds, describe_spec(spec)));
        auto plan = WindowPlan::build(view, spec);
        ASSERT_TRUE(plan) << describe(plan.error());
        test::check_exact_plan(view, spec, *plan);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
        // Every non-final placement of one attempt level looks the same: nothing in the problem
        // distinguishes one start round from another away from the readout.
        constexpr std::uint32_t unset = std::numeric_limits<std::uint32_t>::max();
        std::vector<std::uint32_t> bulk_shape(spec.max_deferrals + 1, unset);
        for (const Placement& p : plan->schedule()) {
            if (p.final) {
                continue;
            }
            if (bulk_shape[p.attempt] == unset) {
                bulk_shape[p.attempt] = p.shape;
            }
            EXPECT_EQ(p.shape, bulk_shape[p.attempt])
                << "window " << p.window << " attempt " << p.attempt;
        }

        if (!uniform_fits(spec, rounds)) {
            continue;
        }
        WindowSpec uniform_spec = spec;
        uniform_spec.boundary = Boundary::uniform;
        auto uniform = WindowPlan::build(view, uniform_spec);
        ASSERT_TRUE(uniform) << describe(uniform.error());
        test::check_uniform_plan(view, uniform_spec, *uniform, *plan);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
        // Where the exact window is a bulk window, the uniform placement maps onto the same
        // global columns, merged groups included (their representatives shift together).
        for (std::uint32_t k = 0; k < plan->num_positions(); ++k) {
            for (std::uint32_t a = 0; a < plan->attempts(k); ++a) {
                const Placement* exact = plan->placement(k, a);
                ASSERT_NE(exact, nullptr) << "position " << k << " attempt " << a;
                if (!exact->final) {
                    const Placement* mapped = uniform->placement(k, a);
                    ASSERT_NE(mapped, nullptr) << "position " << k << " attempt " << a;
                    EXPECT_EQ(mapped->columns, exact->columns)
                        << "position " << k << " attempt " << a;
                }
            }
        }
        // Away from the ends every committed bulk column has a real counterpart; only positions
        // whose rounds pass the readout can have virtual committed columns.
        for (const Placement& p : uniform->schedule()) {
            if (p.first_round + spec.commit - 1 < rounds) {
                EXPECT_EQ(p.virtual_committed, 0U) << "position " << p.window;
            }
        }
    }
}

TEST(WindowProperties, RandomUniformPlansMapColumnsByShiftedSupport) {
    std::uint64_t built = 0;
    for (std::uint64_t seed = 0; seed < seeds; ++seed) {
        std::mt19937_64 rng(seed + 3000);
        const index_t rounds = uniform_int(rng, 2, 14);
        const auto problem = test::random_problem(seed + 3000, rounds, uniform_int(rng, 1, 4),
                                                  uniform_int(rng, 1, 6));
        const Problem view = problem.view();
        const WindowSpec spec = random_spec(rng, 6, Boundary::uniform);
        SCOPED_TRACE(std::format("seed {}, Rt = {}, {}", seed, rounds, describe_spec(spec)));
        auto plan = WindowPlan::build(view, spec);
        if (!uniform_fits(spec, rounds)) {
            ASSERT_FALSE(plan);
            EXPECT_EQ(plan.error().code, PlanError::Code::no_bulk_window);
            continue;
        }
        ASSERT_TRUE(plan) << describe(plan.error());
        WindowSpec exact_spec = spec;
        exact_spec.boundary = Boundary::exact;
        auto exact = WindowPlan::build(view, exact_spec);
        ASSERT_TRUE(exact) << describe(exact.error());
        test::check_uniform_plan(view, spec, *plan, *exact);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
        ++built;
    }
    EXPECT_GT(built, seeds / 10) << "too few feasible uniform specs to test anything";
}

} // namespace
