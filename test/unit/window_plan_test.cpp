// Window plans on hand-built problems whose windows are worked out below, on the committed bb18
// fixture, and (when available) on the gross-code artifacts.

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <vector>

#include "rtd/io/artifact.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
using Code = PlanError::Code;
namespace fs = std::filesystem;

WindowSpec spec(std::uint32_t width, std::uint32_t commit, std::uint32_t converge,
                Boundary boundary = Boundary::exact,
                OnFailure on_failure = OnFailure::commit_anyway, std::uint32_t deferrals = 0) {
    return {.width = width,
            .commit = commit,
            .converge_rounds = converge,
            .boundary = boundary,
            .on_failure = on_failure,
            .max_deferrals = deferrals,
            .iteration_cap = std::nullopt};
}

// Rt = 4 rounds of M = 2 detectors (round r = rows 2r − 2, 2r − 1).
//   column: 0      1    2      3      4      5    6      7    8    9
//   rows:   {2,4}  {0}  {0,2}  {2,5}  {1,3}  {4}  {5,7}  {6}  {3}  {7}
//   s(j):   2      1    1      2      1      3    3      4    2    4
// With W = 2, C = 1 the first window (rounds 1–2) cuts columns 0 and 3 to the same row {2}, so
// they merge; column 8 keeps row {3} alone.
test::OwnedProblem hand_problem() {
    test::OwnedProblem p;
    p.rounds_total = 4;
    p.per_round = 2;
    p.num_observables = 3;
    p.columns = {{2, 4}, {0}, {0, 2}, {2, 5}, {1, 3}, {4}, {5, 7}, {6}, {3}, {7}};
    p.priors = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
    p.classes = {1, 2, 4, 3, 5, 6, 7, 0, 1, 2};
    p.finish();
    return p;
}

TEST(WindowPlanExact, HandWorkedWindows) {
    const auto problem = hand_problem();
    const WindowSpec s = spec(2, 1, 2);
    auto plan = WindowPlan::build(problem.view(), s);
    ASSERT_TRUE(plan) << describe(plan.error());
    ASSERT_EQ(plan->num_positions(), 3U);
    ASSERT_EQ(plan->shapes().size(), 3U);
    ASSERT_EQ(plan->schedule().size(), 3U);

    // Window 0: rounds 1–2, columns {0+3 merged, 1, 2, 4, 8}; commits s = 1: {1, 2, 4}.
    const Placement* w0 = plan->placement(0, 0);
    ASSERT_NE(w0, nullptr);
    EXPECT_EQ(w0->first_round, 1U);
    EXPECT_EQ(w0->rounds, 2U);
    EXPECT_EQ(w0->commit_rounds, 1U);
    EXPECT_FALSE(w0->final);
    EXPECT_EQ(w0->columns, (std::vector<index_t>{0, 1, 2, 4, 8}));
    EXPECT_EQ(w0->members_ptr, (std::vector<index_t>{0, 2, 3, 4, 5, 6}));
    EXPECT_EQ(w0->members, (std::vector<index_t>{0, 3, 1, 2, 4, 8}));
    const Shape& s0 = plan->shape_of(*w0);
    EXPECT_TRUE(std::ranges::equal(s0.row_ptr(), std::vector<index_t>{0, 2, 3, 5, 7}));
    EXPECT_TRUE(std::ranges::equal(s0.col_indices(), std::vector<index_t>{1, 2, 3, 0, 2, 3, 4}));
    EXPECT_TRUE(std::ranges::equal(s0.commit(), std::vector<Bit>{0, 1, 1, 1, 0}));
    EXPECT_TRUE(std::ranges::equal(s0.commit_class(), std::vector<std::uint64_t>{0, 2, 4, 5, 0}));
    EXPECT_TRUE(std::ranges::equal(s0.converge(), std::vector<Bit>{1, 1, 1, 1}));
    EXPECT_EQ(s0.merged_columns(), 1U);
    EXPECT_EQ(s0.committed_columns(), 3U);
    const double p0 = 0.01;
    const double p3 = 0.04;
    const double merged = p0 * (1.0 - p3) + p3 * (1.0 - p0);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(s0.priors().probabilities()[0]),
              std::bit_cast<std::uint64_t>(merged));
    EXPECT_EQ(s0.priors().probabilities()[4], 0.09);
    EXPECT_TRUE(std::ranges::equal(s0.column_rows(0), std::vector<index_t>{2}));
    EXPECT_TRUE(std::ranges::equal(s0.column_rows(3), std::vector<index_t>{1, 3}));

    // Window 1: rounds 2–3, columns {0, 3, 5, 6, 8} (5 and 6 cut to {4} and {5}: no merge);
    // commits s = 2: {0, 3, 8}. Column 2 (s = 1) touches row 2 but was committed by window 0.
    const Placement* w1 = plan->placement(1, 0);
    ASSERT_NE(w1, nullptr);
    EXPECT_EQ(w1->first_round, 2U);
    EXPECT_EQ(w1->first_row, 2U);
    EXPECT_FALSE(w1->final);
    EXPECT_EQ(w1->columns, (std::vector<index_t>{0, 3, 5, 6, 8}));
    const Shape& s1 = plan->shape_of(*w1);
    EXPECT_TRUE(std::ranges::equal(s1.row_ptr(), std::vector<index_t>{0, 2, 3, 5, 7}));
    EXPECT_TRUE(std::ranges::equal(s1.col_indices(), std::vector<index_t>{0, 1, 4, 0, 2, 1, 3}));
    EXPECT_TRUE(std::ranges::equal(s1.commit(), std::vector<Bit>{1, 1, 0, 0, 1}));
    EXPECT_TRUE(std::ranges::equal(s1.commit_class(), std::vector<std::uint64_t>{1, 3, 0, 0, 1}));
    EXPECT_EQ(s1.merged_columns(), 0U);

    // Window 2: rounds 3–4 reach the readout: final, commits everything left.
    const Placement* w2 = plan->placement(2, 0);
    ASSERT_NE(w2, nullptr);
    EXPECT_EQ(w2->first_round, 3U);
    EXPECT_EQ(w2->rounds, 2U);
    EXPECT_EQ(w2->commit_rounds, 2U);
    EXPECT_TRUE(w2->final);
    EXPECT_EQ(w2->columns, (std::vector<index_t>{5, 6, 7, 9}));
    const Shape& s2 = plan->shape_of(*w2);
    EXPECT_TRUE(std::ranges::equal(s2.row_ptr(), std::vector<index_t>{0, 1, 2, 3, 5}));
    EXPECT_TRUE(std::ranges::equal(s2.col_indices(), std::vector<index_t>{0, 1, 2, 1, 3}));
    EXPECT_TRUE(std::ranges::equal(s2.commit(), std::vector<Bit>{1, 1, 1, 1}));

    EXPECT_EQ(plan->placement(3, 0), nullptr);
    EXPECT_EQ(plan->placement(0, 1), nullptr);
    test::check_exact_plan(problem.view(), s, *plan);

    const PlanStats stats = plan->stats();
    EXPECT_EQ(stats.shapes, 3U);
    EXPECT_EQ(stats.placements, 3U);
    EXPECT_EQ(stats.positions, 3U);
    EXPECT_EQ(stats.merged_columns, 1U);
    EXPECT_EQ(stats.virtual_committed, 0U);
    EXPECT_GT(stats.memory_bytes, 0U);
}

TEST(WindowPlanExact, ConvergenceRowsFollowConvergeRounds) {
    const auto problem = hand_problem();
    auto plan = WindowPlan::build(problem.view(), spec(3, 1, 1));
    ASSERT_TRUE(plan) << describe(plan.error());
    const Placement* first = plan->placement(0, 0);
    ASSERT_NE(first, nullptr);
    const Shape& head = plan->shape_of(*first);
    EXPECT_TRUE(std::ranges::equal(head.converge(), std::vector<Bit>{1, 1, 0, 0, 0, 0}));
    const Placement* last = plan->placement(plan->num_positions() - 1, 0);
    ASSERT_NE(last, nullptr);
    ASSERT_TRUE(last->final);
    EXPECT_TRUE(
        std::ranges::all_of(plan->shape_of(*last).converge(), [](Bit b) { return b == 1; }));
    test::check_exact_plan(problem.view(), spec(3, 1, 1), *plan);
}

TEST(WindowPlanExact, DeferralAttemptsWidenUntilTheReadout) {
    const auto problem = hand_problem();
    const WindowSpec s = spec(2, 1, 1, Boundary::exact, OnFailure::defer, 5);
    auto plan = WindowPlan::build(problem.view(), s);
    ASSERT_TRUE(plan) << describe(plan.error());
    // Window 0 (t = 1): widths 2, 3 non-final, width 4 reaches round 4: three attempts.
    ASSERT_EQ(plan->attempts(0), 3U);
    const Placement* second = plan->placement(0, 1);
    const Placement* third = plan->placement(0, 2);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(second->rounds, 3U);
    EXPECT_FALSE(second->final);
    EXPECT_TRUE(third->final);
    EXPECT_EQ(third->rounds, 4U);
    // Attempt 1 converges on C′ + C = 2 rounds and still commits only round 1.
    const Shape& widened = plan->shape_of(*second);
    EXPECT_EQ(std::ranges::count(widened.converge(), Bit{1}), 4);
    EXPECT_EQ(second->commit_rounds, 1U);
    EXPECT_EQ(plan->attempts(1), 2U);
    EXPECT_EQ(plan->attempts(2), 1U);
    test::check_exact_plan(problem.view(), s, *plan);
}

TEST(WindowPlanExact, WidthCoveringTheShotIsTheWholeProblem) {
    const auto problem = hand_problem();
    const Problem view = problem.view();
    for (const std::uint32_t width : {4U, 5U, 9U}) {
        auto plan = WindowPlan::build(view, spec(width, 1, width));
        ASSERT_TRUE(plan) << describe(plan.error());
        ASSERT_EQ(plan->num_positions(), 1U);
        ASSERT_EQ(plan->shapes().size(), 1U);
        const Placement* only = plan->placement(0, 0);
        ASSERT_NE(only, nullptr);
        EXPECT_TRUE(only->final);
        EXPECT_EQ(only->rounds, 4U);
        const Shape& shape = plan->shapes()[0];
        EXPECT_TRUE(std::ranges::equal(shape.row_ptr(), view.h_row_ptr));
        EXPECT_TRUE(std::ranges::equal(shape.col_indices(), view.h_col_indices));
        EXPECT_TRUE(test::bitwise_equal(shape.priors().probabilities(), view.priors));
        EXPECT_TRUE(std::ranges::equal(shape.commit_class(), problem.classes));
        EXPECT_TRUE(std::ranges::all_of(shape.commit(), [](Bit b) { return b == 1; }));
        EXPECT_TRUE(std::ranges::all_of(shape.converge(), [](Bit b) { return b == 1; }));
        std::vector<index_t> identity(view.num_columns);
        std::ranges::iota(identity, index_t{0});
        EXPECT_EQ(only->columns, identity);
    }
}

// Rt = 6 rounds of one detector. Round r holds a_r = {r − 1, r} (r ≤ 5) and b_r = {r − 1}, listed
// a1 b1 a2 b2 … a5 b5 b6 (indices 0 … 10). W = 2, C = 1: the bulk shape is placement (1, 0),
// rounds 2–3, with columns a2, b2 and the merged pair a3 + b3 (both cut to row 2).
test::OwnedProblem ladder_problem() {
    test::OwnedProblem p;
    p.rounds_total = 6;
    p.per_round = 1;
    p.num_observables = 1;
    for (index_t r = 1; r <= 6; ++r) {
        if (r <= 5) {
            p.columns.push_back({r - 1, r});
            p.priors.push_back(0.01);
            p.classes.push_back(1);
        }
        p.columns.push_back({r - 1});
        p.priors.push_back(0.02);
        p.classes.push_back(0);
    }
    p.finish();
    return p;
}

TEST(WindowPlanUniform, HandWorkedMapping) {
    const auto problem = ladder_problem();
    const WindowSpec s = spec(2, 1, 2, Boundary::uniform);
    auto plan = WindowPlan::build(problem.view(), s);
    ASSERT_TRUE(plan) << describe(plan.error());
    ASSERT_EQ(plan->shapes().size(), 1U);
    ASSERT_EQ(plan->num_positions(), 6U);
    const Shape& bulk = plan->shapes()[0];
    EXPECT_EQ(bulk.num_columns(), 3U);
    EXPECT_EQ(bulk.merged_columns(), 1U);
    EXPECT_TRUE(std::ranges::equal(bulk.commit(), std::vector<Bit>{1, 1, 0}));
    const std::vector<std::vector<index_t>> expected{
        {0, 1, 2},                           // k = 0: a1, b1, a2
        {2, 3, 4},                           // k = 1: itself
        {4, 5, 6},                           // k = 2
        {6, 7, 8},                           // k = 3
        {8, 9, virtual_column},              // k = 4: a5, b5; a6 does not exist
        {virtual_column, 10, virtual_column} // k = 5: a6 missing, b6
    };
    for (std::uint32_t k = 0; k < 6; ++k) {
        const Placement* p = plan->placement(k, 0);
        ASSERT_NE(p, nullptr) << "position " << k;
        EXPECT_EQ(p->columns, expected[k]) << "position " << k;
        EXPECT_EQ(p->first_round, k + 1);
        EXPECT_EQ(p->rounds, 2U);
        EXPECT_EQ(p->commit_rounds, 1U);
        EXPECT_FALSE(p->final);
        EXPECT_EQ(p->virtual_committed, k == 5 ? 1U : 0U);
    }
    EXPECT_EQ(plan->stats().virtual_committed, 1U);
    auto exact = WindowPlan::build(problem.view(), spec(2, 1, 2));
    ASSERT_TRUE(exact) << describe(exact.error());
    test::check_uniform_plan(problem.view(), s, *plan, *exact);
    test::check_exact_plan(problem.view(), spec(2, 1, 2), *exact);
}

TEST(WindowPlanUniform, OneShapePerDeferralLevel) {
    const auto problem = ladder_problem();
    const WindowSpec s = spec(2, 1, 1, Boundary::uniform, OnFailure::defer, 1);
    auto plan = WindowPlan::build(problem.view(), s);
    ASSERT_TRUE(plan) << describe(plan.error());
    ASSERT_EQ(plan->shapes().size(), 2U);
    EXPECT_EQ(plan->shapes()[1].rounds(), 3U);
    EXPECT_EQ(plan->attempts(0), 2U);
    const Placement* last_deferral = plan->placement(5, 1);
    ASSERT_NE(last_deferral, nullptr);
    EXPECT_EQ(last_deferral->shape, 1U);
    auto exact = WindowPlan::build(problem.view(), spec(2, 1, 1, Boundary::exact,
                                                        OnFailure::defer, 1));
    ASSERT_TRUE(exact) << describe(exact.error());
    test::check_uniform_plan(problem.view(), s, *plan, *exact);
}

TEST(WindowPlanUniform, RejectsAProblemWithoutABulkWindow) {
    // Rt = 4: t_1 + W = 2 + 2 > R = 3.
    const auto problem = hand_problem();
    auto plan = WindowPlan::build(problem.view(), spec(2, 1, 2, Boundary::uniform));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::no_bulk_window);
    // Rt = 6 (R = 5), W = 2, C = 1: one deferral fits (t_1 + W_1 = 2 + 3 = 5 <= R), two do not
    // (2 + 4 > 5).
    const auto ladder = ladder_problem();
    EXPECT_TRUE(WindowPlan::build(ladder.view(),
                                  spec(2, 1, 1, Boundary::uniform, OnFailure::defer, 1)));
    plan = WindowPlan::build(ladder.view(), spec(2, 1, 1, Boundary::uniform, OnFailure::defer, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::no_bulk_window);
}

TEST(WindowPlanUniform, RejectsAnAmbiguousMapping) {
    auto problem = ladder_problem();
    problem.columns.push_back({1}); // a second copy of b2
    problem.priors.push_back(0.03);
    problem.classes.push_back(0);
    problem.finish();
    auto plan = WindowPlan::build(problem.view(), spec(2, 1, 2, Boundary::uniform));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::ambiguous_mapping);
    // The exact boundary never maps, so identical columns are simply two columns there.
    EXPECT_TRUE(WindowPlan::build(problem.view(), spec(2, 1, 2)));
}

TEST(WindowPlan, RejectsInvalidInputs) {
    const auto problem = hand_problem();
    auto plan = WindowPlan::build(problem.view(), spec(2, 2, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::commit_out_of_range);
    plan = WindowPlan::build(problem.view(), spec(3, 1, 4));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::converge_out_of_range);

    auto spanning = hand_problem();
    spanning.columns[0] = {2, 6}; // rounds 2 and 4
    spanning.finish();
    plan = WindowPlan::build(spanning.view(), spec(2, 1, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::column_spans_rounds);

    auto shuffled_rows = hand_problem();
    shuffled_rows.rounds = {1, 2, 1, 2, 3, 3, 4, 4};
    plan = WindowPlan::build(shuffled_rows.view(), spec(2, 1, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::rows_not_grouped_by_round);

    auto many = hand_problem();
    many.num_observables = 65;
    many.a_row_ptr.assign(66, 0);
    many.a_col.clear();
    plan = WindowPlan::build(many.view(), spec(2, 1, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::too_many_observables);

    auto certain = hand_problem();
    certain.priors[4] = 1.0;
    plan = WindowPlan::build(certain.view(), spec(2, 1, 2));
    ASSERT_FALSE(plan);
    EXPECT_EQ(plan.error().code, Code::invalid_prior);
}

TEST(WindowPlan, BuildsGraphsWithTheRequestedOptions) {
    const auto problem = hand_problem();
    const GraphOptions options{.layout = EdgeLayout::column_blocked,
                               .column_order = ColumnOrder::degree_classes,
                               .block_rows = 64};
    auto plan = WindowPlan::build(problem.view(), spec(2, 1, 2), options);
    ASSERT_TRUE(plan) << describe(plan.error());
    for (const Shape& shape : plan->shapes()) {
        EXPECT_EQ(shape.graph().layout(), EdgeLayout::column_blocked);
        EXPECT_EQ(shape.graph().column_order(), ColumnOrder::degree_classes);
    }
    // A combination the graph refuses is reported, not ignored.
    auto refused = WindowPlan::build(problem.view(), spec(2, 1, 2),
                                     {.layout = EdgeLayout::column_blocked,
                                      .column_order = ColumnOrder::natural,
                                      .block_rows = 64});
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().code, Code::graph_failed);
    // Moving the plan keeps the shapes (and so any decoder's pointers into them) in place.
    const TannerGraph* graph = &plan->shapes()[0].graph();
    WindowPlan moved = std::move(*plan);
    EXPECT_EQ(&moved.shapes()[0].graph(), graph);
}

// ---- The committed bb18 fixture: Rt = 4 rounds of 18 detectors. ----------------------------

const fs::path bb18 = fs::path(RTD_FIXTURE_DIR) / "bb18_choi" / "artifact";

TEST(WindowPlanArtifact, Bb18ExactPlansSatisfyTheInvariants) {
    auto artifact = io::load_artifact(bb18);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const ArtifactProblem source(*artifact);
    const Problem problem = source.problem();
    for (const auto& [w, c] : {std::pair{2U, 1U}, std::pair{3U, 1U}, std::pair{3U, 2U},
                               std::pair{4U, 1U}, std::pair{6U, 2U}}) {
        for (const std::uint32_t converge : {c, w}) {
            const WindowSpec s = spec(w, c, converge, Boundary::exact, OnFailure::defer, 2);
            auto plan = WindowPlan::build(problem, s);
            ASSERT_TRUE(plan) << describe(plan.error());
            test::check_exact_plan(problem, s, *plan);
        }
    }
    // Uniform needs t_1 + W ≤ R = 3, impossible with W ≥ 2.
    auto uniform = WindowPlan::build(problem, spec(2, 1, 2, Boundary::uniform));
    ASSERT_FALSE(uniform);
    EXPECT_EQ(uniform.error().code, Code::no_bulk_window);
}

TEST(WindowPlanArtifact, Bb18WholeShotWindowIsTheArtifact) {
    auto artifact = io::load_artifact(bb18);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const ArtifactProblem source(*artifact);
    const Problem problem = source.problem();
    auto plan = WindowPlan::build(problem, spec(4, 1, 4));
    ASSERT_TRUE(plan) << describe(plan.error());
    ASSERT_EQ(plan->shapes().size(), 1U);
    const Shape& shape = plan->shapes()[0];
    EXPECT_TRUE(std::ranges::equal(shape.row_ptr(), problem.h_row_ptr));
    EXPECT_TRUE(std::ranges::equal(shape.col_indices(), problem.h_col_indices));
    EXPECT_TRUE(test::bitwise_equal(shape.priors().probabilities(), problem.priors));
    EXPECT_TRUE(test::bitwise_equal(shape.priors().llr(), artifact->priors.llr()));
    // The rebuilt CSR is the file's.
    auto h_indices = io::read_npy<index_t>(bb18 / "H_indices.npy");
    ASSERT_TRUE(h_indices);
    EXPECT_TRUE(std::ranges::equal(problem.h_col_indices, h_indices->span()));
    auto a_indices = io::read_npy<index_t>(bb18 / "A_indices.npy");
    ASSERT_TRUE(a_indices);
    EXPECT_TRUE(std::ranges::equal(problem.a_col_indices, a_indices->span()));
}

// ---- Gross-code artifacts (not committed). -------------------------------------------------

// The R = 12 artifact named by RTD_GROSS_ARTIFACT (as for the gross goldens).
TEST(WindowPlanArtifact, GrossR12ExactInvariants) {
    const char* dir = std::getenv("RTD_GROSS_ARTIFACT");
    if (dir == nullptr) {
        GTEST_SKIP() << "set RTD_GROSS_ARTIFACT to run";
    }
    auto artifact = io::load_artifact(dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const ArtifactProblem source(*artifact);
    const Problem problem = source.problem();
    const WindowSpec s = spec(12, 8, 12);
    auto plan = WindowPlan::build(problem, s);
    ASSERT_TRUE(plan) << describe(plan.error());
    EXPECT_EQ(plan->rounds_total(), 13U);
    EXPECT_EQ(plan->detectors_per_round(), 144U);
    EXPECT_EQ(plan->num_positions(), 2U);
    test::check_exact_plan(problem, s, *plan);
    for (const Shape& shape : plan->shapes()) {
        std::cout << std::format("[ window   ] R=12 (12,8) shape {}: {} x {}, {} edges, {} merged\n",
                                 shape.index(), shape.num_rows(), shape.num_columns(),
                                 shape.num_edges(), shape.merged_columns());
    }
    // Short windows with deferral attempts and fewer convergence rounds.
    for (const WindowSpec& other : {spec(3, 1, 2, Boundary::exact, OnFailure::defer, 2),
                                    spec(6, 4, 4, Boundary::exact, OnFailure::flag)}) {
        auto more = WindowPlan::build(problem, other);
        ASSERT_TRUE(more) << describe(more.error());
        test::check_exact_plan(problem, other, *more);
    }
    // t_1 + W = 9 + 12 exceeds R = 12: no bulk window to copy.
    auto uniform = WindowPlan::build(problem, spec(12, 8, 12, Boundary::uniform));
    ASSERT_FALSE(uniform);
    EXPECT_EQ(uniform.error().code, Code::no_bulk_window);
}

// The R = 48 artifact at data/artifacts/gross_choi_p0.003_r48 (or RTD_GROSS_ARTIFACT_R48).
fs::path r48_artifact() {
    if (const char* dir = std::getenv("RTD_GROSS_ARTIFACT_R48"); dir != nullptr) {
        return dir;
    }
    return fs::path(RTD_FIXTURE_DIR) / ".." / ".." / "data" / "artifacts" /
           "gross_choi_p0.003_r48";
}

TEST(WindowPlanArtifact, GrossR48ShapesByContent) {
    const fs::path dir = r48_artifact();
    if (!fs::exists(dir / "manifest.json")) {
        GTEST_SKIP() << "no R = 48 artifact at " << dir;
    }
    auto artifact = io::load_artifact(dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const ArtifactProblem source(*artifact);
    const Problem problem = source.problem();

    const WindowSpec exact_spec = spec(12, 8, 12);
    const auto start = std::chrono::steady_clock::now();
    auto exact = WindowPlan::build(problem, exact_spec);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_TRUE(exact) << describe(exact.error());
    EXPECT_EQ(exact->rounds_total(), 49U);
    EXPECT_EQ(exact->num_positions(), 6U);
    EXPECT_EQ(exact->shapes().size(), 3U) << "head, bulk and tail";
    for (std::uint32_t k = 1; k + 1 < exact->num_positions(); ++k) {
        const Placement* bulk = exact->placement(k, 0);
        ASSERT_NE(bulk, nullptr) << "bulk window " << k;
        EXPECT_EQ(bulk->shape, 1U) << "bulk window " << k;
    }
    std::cout << std::format("[ window   ] R=48 exact (12,8): {} shapes, built in {:.3f} s, "
                             "~{:.1f} MB\n",
                             exact->shapes().size(), seconds,
                             static_cast<double>(exact->stats().memory_bytes) / 1e6);
    test::check_exact_plan(problem, exact_spec, *exact);

    const WindowSpec uniform_spec = spec(12, 8, 12, Boundary::uniform);
    auto uniform = WindowPlan::build(problem, uniform_spec);
    ASSERT_TRUE(uniform) << describe(uniform.error());
    EXPECT_EQ(uniform->shapes().size(), 1U);
    EXPECT_EQ(uniform->num_positions(), 7U);
    test::check_uniform_plan(problem, uniform_spec, *uniform, *exact);
    std::cout << std::format("[ window   ] R=48 uniform (12,8): {} virtual committed columns\n",
                             uniform->stats().virtual_committed);
}

} // namespace
