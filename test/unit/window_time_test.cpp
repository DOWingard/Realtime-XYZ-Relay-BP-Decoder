// Window specifications, problem validation and the time structure of a problem: s(j), the
// rounds, and every rejection.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "rtd/window/problem.hpp"
#include "rtd/window/spec.hpp"
#include "rtd/window/time_structure.hpp"
#include "window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
using Code = PlanError::Code;

WindowSpec spec(std::uint32_t width, std::uint32_t commit, std::uint32_t converge) {
    return {.width = width,
            .commit = commit,
            .converge_rounds = converge,
            .boundary = Boundary::exact,
            .on_failure = OnFailure::commit_anyway,
            .max_deferrals = 0,
            .iteration_cap = std::nullopt};
}

// Rt = 3 rounds of M = 2 detectors; columns chosen so that s = 1, 1, 2, 2, 3.
test::OwnedProblem small_problem() {
    test::OwnedProblem p;
    p.rounds_total = 3;
    p.per_round = 2;
    p.num_observables = 2;
    p.columns = {{0, 2}, {1}, {3, 5}, {2, 4}, {5}};
    p.priors = {0.01, 0.02, 0.03, 0.04, 0.05};
    p.classes = {1, 0, 2, 3, 0};
    p.finish();
    return p;
}

TEST(WindowSpec, AcceptsTheValidRange) {
    EXPECT_TRUE(validate(spec(2, 1, 1)));
    EXPECT_TRUE(validate(spec(2, 1, 2)));
    EXPECT_TRUE(validate(spec(12, 8, 8)));
    EXPECT_TRUE(validate(spec(12, 8, 12)));
    WindowSpec deferring = spec(12, 8, 12);
    deferring.on_failure = OnFailure::defer;
    deferring.max_deferrals = 3;
    deferring.iteration_cap = 600;
    EXPECT_TRUE(validate(deferring));
}

TEST(WindowSpec, RejectsCommitOutsideOneToWidth) {
    for (const auto& [w, c] : {std::pair{3U, 0U}, std::pair{3U, 3U}, std::pair{3U, 4U},
                               std::pair{0U, 0U}, std::pair{1U, 1U}}) {
        const auto valid = validate(spec(w, c, c));
        ASSERT_FALSE(valid) << w << ", " << c;
        EXPECT_EQ(valid.error().code, Code::commit_out_of_range);
        EXPECT_FALSE(valid.error().detail.empty());
    }
}

TEST(WindowSpec, RejectsConvergeRoundsOutsideCommitToWidth) {
    for (const std::uint32_t converge : {0U, 1U, 7U}) {
        const auto valid = validate(spec(6, 2, converge));
        ASSERT_FALSE(valid) << converge;
        EXPECT_EQ(valid.error().code, Code::converge_out_of_range);
    }
}

TEST(WindowSpec, RejectsDeferralsWithoutDeferAndAZeroCap) {
    WindowSpec s = spec(4, 2, 4);
    s.max_deferrals = 1;
    for (const OnFailure policy : {OnFailure::commit_anyway, OnFailure::flag}) {
        s.on_failure = policy;
        const auto valid = validate(s);
        ASSERT_FALSE(valid);
        EXPECT_EQ(valid.error().code, Code::deferrals_without_defer);
    }
    WindowSpec capped = spec(4, 2, 4);
    capped.iteration_cap = 0;
    const auto valid = validate(capped);
    ASSERT_FALSE(valid);
    EXPECT_EQ(valid.error().code, Code::zero_iteration_cap);
}

TEST(WindowSpec, NamesRoundTrip) {
    for (const Boundary b : {Boundary::exact, Boundary::uniform}) {
        EXPECT_EQ(parse_boundary(to_string(b)), b);
    }
    for (const OnFailure f : {OnFailure::commit_anyway, OnFailure::defer, OnFailure::flag}) {
        EXPECT_EQ(parse_on_failure(to_string(f)), f);
    }
    EXPECT_FALSE(parse_boundary("Exact"));
    EXPECT_FALSE(parse_on_failure("commit-anyway"));
    EXPECT_EQ(describe(PlanError{.code = Code::no_bulk_window, .detail = "why"}),
              "no_bulk_window: why");
}

TEST(ProblemValidation, AcceptsAWellFormedProblem) {
    const auto p = small_problem();
    EXPECT_TRUE(validate(p.view()));
}

TEST(ProblemValidation, RejectsWrongSizesMatricesAndPriors) {
    const auto p = small_problem();
    {
        Problem bad = p.view();
        bad.priors = bad.priors.first(4);
        const auto valid = validate(bad);
        ASSERT_FALSE(valid);
        EXPECT_EQ(valid.error().code, Code::size_mismatch);
    }
    {
        Problem bad = p.view();
        bad.detector_round = bad.detector_round.first(5);
        const auto valid = validate(bad);
        ASSERT_FALSE(valid);
        EXPECT_EQ(valid.error().code, Code::size_mismatch);
    }
    {
        auto broken = p;
        broken.h_col[1] = 99; // column out of range
        const auto valid = validate(broken.view());
        ASSERT_FALSE(valid);
        EXPECT_EQ(valid.error().code, Code::invalid_matrix);
    }
    {
        auto broken = p;
        std::swap(broken.a_col[0], broken.a_col[1]); // a row no longer strictly increasing
        const auto valid = validate(broken.view());
        ASSERT_FALSE(valid);
        EXPECT_EQ(valid.error().code, Code::invalid_matrix);
    }
    for (const double prior : {1.0, -0.1, std::nan("")}) {
        auto broken = p;
        broken.priors[3] = prior;
        const auto valid = validate(broken.view());
        ASSERT_FALSE(valid) << prior;
        EXPECT_EQ(valid.error().code, Code::invalid_prior);
    }
}

TEST(TimeStructure, FindsRoundsAndFirstRounds) {
    const auto p = small_problem();
    auto time = TimeStructure::build(p.view());
    ASSERT_TRUE(time) << describe(time.error());
    EXPECT_EQ(time->rounds_total(), 3U);
    EXPECT_EQ(time->detectors_per_round(), 2U);
    EXPECT_EQ(time->num_columns(), 5U);
    const std::vector<std::uint32_t> expected{1, 1, 2, 2, 3};
    EXPECT_TRUE(std::ranges::equal(time->first_rounds(), expected));
    EXPECT_EQ(time->first_round(3), 2U);
    EXPECT_EQ(time->first_row_of_round(3), 4U);
    EXPECT_EQ(time->round_of_row(3), 2U);
    EXPECT_EQ(time->round_of_row(4), 3U);
}

TEST(TimeStructure, OneDetectorPerRoundAndOneRound) {
    test::OwnedProblem p;
    p.rounds_total = 1;
    p.per_round = 3;
    p.num_observables = 1;
    p.columns = {{0}, {1, 2}};
    p.priors = {0.1, 0.2};
    p.classes = {1, 0};
    p.finish();
    auto time = TimeStructure::build(p.view());
    ASSERT_TRUE(time) << describe(time.error());
    EXPECT_EQ(time->rounds_total(), 1U);
    EXPECT_EQ(time->detectors_per_round(), 3U);
}

TEST(TimeStructure, RejectsRowsNotGroupedByRound) {
    auto p = small_problem();
    std::swap(p.rounds[1], p.rounds[2]); // rounds 1, 2, 1, 2, 3, 3
    const auto time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::rows_not_grouped_by_round);
    EXPECT_NE(time.error().detail.find("detector 2"), std::string::npos) << time.error().detail;
}

TEST(TimeStructure, RejectsRoundsOfDifferentSizes) {
    auto p = small_problem();
    p.rounds = {1, 1, 2, 3, 3, 3}; // round 2 has one detector, round 3 three
    auto time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::unequal_round_sizes);

    p.rounds = {1, 1, 2, 2, 4, 4}; // round 3 missing
    time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::unequal_round_sizes);

    p.rounds = {1, 1, 2, 2, 2, 3}; // the last round is short
    time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::unequal_round_sizes);
}

TEST(TimeStructure, RejectsRoundsNotStartingAtOne) {
    auto p = small_problem();
    p.rounds = {0, 0, 1, 1, 2, 2};
    const auto time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::bad_round_numbering);
}

TEST(TimeStructure, RejectsAFaultSpanningThreeRounds) {
    auto p = small_problem();
    p.columns[0] = {0, 5}; // rounds 1 and 3
    p.finish();
    const auto time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::column_spans_rounds);
    EXPECT_NE(time.error().detail.find("column 0"), std::string::npos) << time.error().detail;
}

TEST(TimeStructure, RejectsAnEmptyColumnAndAnEmptyProblem) {
    auto p = small_problem();
    p.columns[2].clear();
    p.finish();
    auto time = TimeStructure::build(p.view());
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::empty_column);

    const std::vector<index_t> row_ptr{0};
    const Problem empty{.num_rows = 0,
                        .num_columns = 0,
                        .num_observables = 0,
                        .h_row_ptr = row_ptr,
                        .h_col_indices = {},
                        .priors = {},
                        .a_row_ptr = row_ptr,
                        .a_col_indices = {},
                        .detector_round = {}};
    time = TimeStructure::build(empty);
    ASSERT_FALSE(time);
    EXPECT_EQ(time.error().code, Code::no_rows);
}

} // namespace
