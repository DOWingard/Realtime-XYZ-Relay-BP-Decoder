#pragma once

// Test helpers for the window layer: owned time-structured problems (hand-built or random) and
// independent checkers that re-derive every window from the global problem and compare it with
// a plan. The checkers use only the definitions (s(j) as the smallest round a fault touches,
// windows as row and column ranges, Lee et al.'s detector rule), never the plan builder's code.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <functional>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <vector>

#include "rtd/core/types.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/problem.hpp"

namespace rtd::window::test {

// A problem given column by column, with the CSR arrays a Problem views.
struct OwnedProblem {
    index_t rounds_total = 0;
    index_t per_round = 0;
    index_t num_observables = 0;
    std::vector<std::vector<index_t>> columns; // rows of each column, ascending
    std::vector<double> priors;
    std::vector<std::uint64_t> classes; // observable mask of each column
    std::vector<std::int32_t> rounds;   // per row
    std::vector<index_t> h_row_ptr;
    std::vector<index_t> h_col;
    std::vector<index_t> a_row_ptr;
    std::vector<index_t> a_col;

    // Builds the CSR arrays and the natural round array from `columns` and `classes`.
    void finish() {
        const index_t m = rounds_total * per_round;
        rounds.resize(m);
        for (index_t i = 0; i < m; ++i) {
            rounds[i] = static_cast<std::int32_t>(i / per_round + 1);
        }
        std::vector<std::vector<index_t>> rows(m);
        for (index_t j = 0; j < columns.size(); ++j) {
            for (const index_t i : columns[j]) {
                rows[i].push_back(j);
            }
        }
        h_row_ptr.assign(1, 0);
        h_col.clear();
        for (const auto& row : rows) {
            h_col.insert(h_col.end(), row.begin(), row.end());
            h_row_ptr.push_back(static_cast<index_t>(h_col.size()));
        }
        a_row_ptr.assign(1, 0);
        a_col.clear();
        for (index_t o = 0; o < num_observables; ++o) {
            for (index_t j = 0; j < columns.size(); ++j) {
                if (((classes[j] >> o) & 1U) != 0) {
                    a_col.push_back(j);
                }
            }
            a_row_ptr.push_back(static_cast<index_t>(a_col.size()));
        }
    }

    [[nodiscard]] Problem view() const {
        return {.num_rows = rounds_total * per_round,
                .num_columns = static_cast<index_t>(columns.size()),
                .num_observables = num_observables,
                .h_row_ptr = h_row_ptr,
                .h_col_indices = h_col,
                .priors = priors,
                .a_row_ptr = a_row_ptr,
                .a_col_indices = a_col,
                .detector_round = rounds};
    }
};

// Supports of every column of a Problem, rows ascending.
inline std::vector<std::vector<index_t>> column_supports(const Problem& p) {
    std::vector<std::vector<index_t>> cols(p.num_columns);
    for (index_t i = 0; i < p.num_rows; ++i) {
        for (index_t e = p.h_row_ptr[i]; e < p.h_row_ptr[i + 1]; ++e) {
            cols[p.h_col_indices[e]].push_back(i);
        }
    }
    return cols;
}

inline std::vector<std::uint64_t> column_classes(const Problem& p) {
    std::vector<std::uint64_t> classes(p.num_columns, 0);
    for (index_t o = 0; o < p.num_observables; ++o) {
        for (index_t e = p.a_row_ptr[o]; e < p.a_row_ptr[o + 1]; ++e) {
            classes[p.a_col_indices[e]] |= std::uint64_t{1} << o;
        }
    }
    return classes;
}

// A random subset of `size` distinct rows of round r (1-based), ascending.
inline std::vector<index_t> random_rows(std::mt19937_64& rng, index_t per_round, index_t round,
                                        index_t size) {
    std::vector<index_t> all(per_round);
    std::ranges::iota(all, (round - 1) * per_round);
    std::ranges::shuffle(all, rng);
    all.resize(std::min(size, per_round));
    std::ranges::sort(all);
    return all;
}

// The mask with the lowest `observables` bits set.
inline std::uint64_t all_observables(index_t observables) {
    return observables >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << observables) - 1;
}

inline double random_prior(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double u = unit(rng);
    if (u < 0.05) {
        return 0.0; // λ = +∞: a fault that cannot happen is still a legal column
    }
    return 0.001 + 0.3 * unit(rng);
}

// A random problem with Rt rounds of M rows and faults in rounds s and (sometimes) s + 1. Some
// faults share their round-s rows with another fault of the same round, so windows cut there
// must merge them; no two faults share their whole support. Column order is shuffled so that
// local order cannot come from round order by accident.
inline OwnedProblem random_problem(std::uint64_t seed, index_t rounds_total, index_t per_round,
                                   index_t observables) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<index_t> per_round_faults(1, 6);
    std::uniform_int_distribution<index_t> part(1, 3);
    std::bernoulli_distribution coin(0.6);
    std::bernoulli_distribution copy_head(0.35);
    std::uniform_int_distribution<std::uint64_t> mask(0, all_observables(observables));
    OwnedProblem p;
    p.rounds_total = rounds_total;
    p.per_round = per_round;
    p.num_observables = observables;
    std::set<std::vector<index_t>> seen;
    for (index_t r = 1; r <= rounds_total; ++r) {
        std::vector<std::vector<index_t>> heads;
        const index_t faults = per_round_faults(rng);
        for (index_t f = 0; f < faults * 3 && heads.size() < faults; ++f) {
            std::vector<index_t> head = !heads.empty() && copy_head(rng)
                                            ? heads[std::uniform_int_distribution<std::size_t>(
                                                  0, heads.size() - 1)(rng)]
                                            : random_rows(rng, per_round, r, part(rng));
            std::vector<index_t> rows = head;
            if (r < rounds_total && coin(rng)) {
                const auto tail = random_rows(rng, per_round, r + 1, part(rng));
                rows.insert(rows.end(), tail.begin(), tail.end());
            }
            if (!seen.insert(rows).second) {
                continue;
            }
            heads.push_back(head);
            p.columns.push_back(rows);
            p.priors.push_back(random_prior(rng));
            p.classes.push_back(mask(rng));
        }
    }
    std::vector<index_t> order(p.columns.size());
    std::ranges::iota(order, index_t{0});
    std::ranges::shuffle(order, rng);
    OwnedProblem shuffled = p;
    for (index_t j = 0; j < order.size(); ++j) {
        shuffled.columns[j] = p.columns[order[j]];
        shuffled.priors[j] = p.priors[order[j]];
        shuffled.classes[j] = p.classes[order[j]];
    }
    shuffled.finish();
    return shuffled;
}

// A problem invariant under translation in time: the same faults, priors and observables in
// every round, listed round by round in the same order. In the readout round Rt a fault keeps
// only its round-Rt rows; faults that become empty or identical to an earlier one are dropped.
inline OwnedProblem translation_invariant_problem(std::uint64_t seed, index_t rounds_total,
                                                  index_t per_round, index_t observables) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<index_t> part(1, 3);
    std::uniform_int_distribution<index_t> count(2, 7);
    std::bernoulli_distribution coin(0.7);
    std::bernoulli_distribution copy_head(0.35);
    std::uniform_int_distribution<std::uint64_t> mask(0, all_observables(observables));
    // Template faults as (rows in the fault's first round, rows in the next), offsets 0 … M − 1.
    struct Template {
        std::vector<index_t> head;
        std::vector<index_t> tail;
        double prior;
        std::uint64_t mask;
    };
    std::vector<Template> templates;
    std::set<std::pair<std::vector<index_t>, std::vector<index_t>>> seen;
    const index_t wanted = count(rng);
    for (index_t f = 0; f < wanted * 3 && templates.size() < wanted; ++f) {
        std::vector<index_t> head = !templates.empty() && copy_head(rng)
                                        ? templates[std::uniform_int_distribution<std::size_t>(
                                                        0, templates.size() - 1)(rng)]
                                              .head
                                        : random_rows(rng, per_round, 1, part(rng));
        std::vector<index_t> tail = coin(rng) ? random_rows(rng, per_round, 1, part(rng))
                                              : std::vector<index_t>{};
        if (!seen.insert({head, tail}).second) {
            continue;
        }
        templates.push_back(
            {.head = head, .tail = tail, .prior = random_prior(rng), .mask = mask(rng)});
    }
    OwnedProblem p;
    p.rounds_total = rounds_total;
    p.per_round = per_round;
    p.num_observables = observables;
    for (index_t r = 1; r <= rounds_total; ++r) {
        std::set<std::vector<index_t>> in_round;
        for (const Template& t : templates) {
            std::vector<index_t> rows;
            rows.reserve(t.head.size() + t.tail.size());
            for (const index_t i : t.head) {
                rows.push_back(i + (r - 1) * per_round);
            }
            if (r < rounds_total) {
                for (const index_t i : t.tail) {
                    rows.push_back(i + r * per_round);
                }
            }
            if (!in_round.insert(rows).second) {
                continue;
            }
            p.columns.push_back(rows);
            p.priors.push_back(t.prior);
            p.classes.push_back(t.mask);
        }
    }
    p.finish();
    return p;
}

// The window rule stated through detectors (Lee et al.): window k's columns are the faults that
// touch a detector of rounds [t_k, t_k + W) and are not yet committed; it commits those among
// them that touch a detector of rounds [t_k, t_k + C) (all of them in the final window).
struct DetectorRuleWindow {
    std::vector<index_t> columns;
    std::vector<index_t> committed;
};

inline std::vector<DetectorRuleWindow>
detector_rule(const std::vector<std::vector<index_t>>& supports, index_t per_round,
              std::uint32_t rounds_total, std::uint32_t width, std::uint32_t commit) {
    std::vector<DetectorRuleWindow> windows;
    std::vector<bool> done(supports.size(), false);
    for (std::uint32_t t = 1;; t += commit) {
        const bool final = t + width - 1 >= rounds_total;
        const std::uint32_t end = final ? rounds_total + 1 : t + width;
        DetectorRuleWindow w;
        for (index_t j = 0; j < supports.size(); ++j) {
            if (done[j]) {
                continue;
            }
            bool touches = false;
            bool touches_commit = false;
            for (const index_t i : supports[j]) {
                const std::uint32_t r = i / per_round + 1;
                touches = touches || (r >= t && r < end);
                touches_commit = touches_commit || (r >= t && r < t + commit);
            }
            if (!touches) {
                continue;
            }
            w.columns.push_back(j);
            if (final || touches_commit) {
                w.committed.push_back(j);
            }
        }
        for (const index_t j : w.committed) {
            done[j] = true;
        }
        windows.push_back(std::move(w));
        if (final) {
            return windows;
        }
    }
}

inline bool bitwise_equal(std::span<const double> a, std::span<const double> b) {
    return std::ranges::equal(a, b, [](double x, double y) {
        return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
    });
}

inline bool same_content(const Shape& a, const Shape& b) {
    return a.num_rows() == b.num_rows() && std::ranges::equal(a.row_ptr(), b.row_ptr()) &&
           std::ranges::equal(a.col_indices(), b.col_indices()) &&
           bitwise_equal(a.priors().probabilities(), b.priors().probabilities()) &&
           std::ranges::equal(a.commit(), b.commit()) &&
           std::ranges::equal(a.commit_class(), b.commit_class()) &&
           std::ranges::equal(a.converge(), b.converge());
}

// Facts about the global problem that the checkers derive from its definition.
struct Reference {
    std::vector<std::vector<index_t>> supports;
    std::vector<std::uint64_t> classes;
    index_t per_round = 0;
    std::uint32_t rounds_total = 0;

    Reference(const Problem& problem, const WindowPlan& plan)
        : supports(column_supports(problem)), classes(column_classes(problem)),
          per_round(plan.detectors_per_round()), rounds_total(plan.rounds_total()) {}

    // s(j): the smallest round among the rows of column j.
    [[nodiscard]] std::uint32_t first_round(index_t j) const {
        return supports[j].front() / per_round + 1;
    }
};

// What a placement (k, a) of an exact plan must be.
struct ExpectedPlacement {
    std::uint32_t t;      // first round
    std::uint32_t rounds; // rounds covered
    std::uint32_t last;   // last round
    bool final;
};

inline ExpectedPlacement expected_placement(const WindowSpec& spec, std::uint32_t rounds_total,
                                            std::uint32_t k, std::uint32_t a) {
    const std::uint32_t t = 1 + k * spec.commit;
    const std::uint32_t width = spec.width + a * spec.commit;
    const bool final = t + width - 1 >= rounds_total;
    const std::uint32_t rounds = final ? rounds_total - t + 1 : width;
    return {.t = t, .rounds = rounds, .last = t + rounds - 1, .final = final};
}

// The placement's schedule fields, and that its members are exactly the faults with
// t ≤ s(j) ≤ last.
inline void check_placement_frame(const WindowSpec& spec, const WindowPlan& plan,
                                  const Reference& ref, const Placement& pl,
                                  const ExpectedPlacement& e) {
    ASSERT_EQ(pl.first_round, e.t);
    ASSERT_EQ(pl.first_row, (e.t - 1) * ref.per_round);
    ASSERT_EQ(pl.final, e.final);
    ASSERT_EQ(pl.rounds, e.rounds);
    ASSERT_EQ(pl.commit_rounds, e.final ? e.rounds : spec.commit);
    ASSERT_EQ(plan.placement(pl.window, pl.attempt), &pl);
    ASSERT_LT(pl.shape, plan.shapes().size());
    const Shape& shape = plan.shape_of(pl);
    ASSERT_EQ(shape.rounds(), e.rounds);
    ASSERT_EQ(shape.num_rows(), e.rounds * ref.per_round);
    ASSERT_EQ(pl.columns.size(), shape.num_columns());
    ASSERT_EQ(pl.members_ptr.size(), std::size_t{shape.num_columns()} + 1);
    ASSERT_EQ(shape.graph().num_rows(), shape.num_rows());
    ASSERT_EQ(shape.graph().num_columns(), shape.num_columns());
    ASSERT_EQ(shape.graph().num_edges(), shape.num_edges());

    std::vector<index_t> members(pl.members.begin(), pl.members.end());
    std::ranges::sort(members);
    std::vector<index_t> expected;
    for (index_t j = 0; j < ref.supports.size(); ++j) {
        if (ref.first_round(j) >= e.t && ref.first_round(j) <= e.last) {
            expected.push_back(j);
        }
    }
    ASSERT_EQ(members, expected);
}

// Rows of column j inside the placement's rows, as local rows.
inline std::vector<index_t> local_rows_of(const Reference& ref, const Placement& pl,
                                          const ExpectedPlacement& e, index_t j) {
    std::vector<index_t> rows;
    for (const index_t i : ref.supports[j]) {
        if (i < e.last * ref.per_round) {
            rows.push_back(i - pl.first_row);
        }
    }
    return rows;
}

// Each local column: its members (a group only in the last round of a non-final window, with
// identical rows inside the window), its rows, its prior folded over the members, its commit
// flag and class. Attempt-0 commits are recorded in committed_at.
inline void check_local_columns(const Problem& problem, const WindowSpec& spec,
                                const Shape& shape, const Reference& ref, const Placement& pl,
                                const ExpectedPlacement& e, std::vector<int>& committed_at) {
    index_t merged = 0;
    for (index_t l = 0; l < shape.num_columns(); ++l) {
        const std::span<const index_t> members(pl.members.data() + pl.members_ptr[l],
                                               pl.members_ptr[l + 1] - pl.members_ptr[l]);
        ASSERT_FALSE(members.empty());
        ASSERT_TRUE(std::ranges::is_sorted(members));
        const index_t rep = members.front();
        ASSERT_EQ(pl.columns[l], rep);
        ASSERT_TRUE(l == 0 || pl.columns[l - 1] < rep) << "local columns ascend by representative";
        const auto rows = local_rows_of(ref, pl, e, rep);
        ASSERT_TRUE(std::ranges::equal(shape.column_rows(l), rows));
        double p = problem.priors[rep];
        for (std::size_t m = 1; m < members.size(); ++m) {
            ASSERT_FALSE(e.final) << "no merging in a final window";
            ASSERT_EQ(ref.first_round(members[m]), e.last) << "merged columns lie in the last round";
            ASSERT_EQ(local_rows_of(ref, pl, e, members[m]), rows);
            const double q = problem.priors[members[m]];
            p = p * (1.0 - q) + q * (1.0 - p);
        }
        merged += members.size() > 1 ? 1U : 0U;
        ASSERT_EQ(std::bit_cast<std::uint64_t>(shape.priors().probabilities()[l]),
                  std::bit_cast<std::uint64_t>(p));
        const bool commit = e.final || ref.first_round(rep) < e.t + spec.commit;
        ASSERT_TRUE(!commit || members.size() == 1) << "merged columns are never committed";
        ASSERT_EQ(shape.commit()[l], commit ? 1 : 0);
        ASSERT_EQ(shape.commit_class()[l], commit ? ref.classes[rep] : 0);
        if (commit && pl.attempt == 0) {
            ASSERT_EQ(committed_at[rep], -1) << "column " << rep << " committed twice";
            committed_at[rep] = static_cast<int>(pl.window);
        }
    }
    ASSERT_EQ(shape.merged_columns(), merged);
}

// Merging is complete (no two local columns of the cut round keep the same rows), the CSR and
// the column view describe one matrix, and the convergence rows are the first C′ + a·C rounds
// (all rows of a final window).
inline void check_local_matrix(const WindowSpec& spec, const Shape& shape, const Reference& ref,
                               const Placement& pl, const ExpectedPlacement& e) {
    std::set<std::vector<index_t>> cut_supports;
    for (index_t l = 0; l < shape.num_columns(); ++l) {
        if (!e.final && ref.first_round(pl.columns[l]) == e.last) {
            const auto rows = shape.column_rows(l);
            ASSERT_TRUE(cut_supports.emplace(rows.begin(), rows.end()).second) << "column " << l;
        }
    }
    std::vector<std::vector<index_t>> from_csr(shape.num_columns());
    for (index_t i = 0; i < shape.num_rows(); ++i) {
        const auto row = shape.col_indices().subspan(
            shape.row_ptr()[i], shape.row_ptr()[i + 1] - shape.row_ptr()[i]);
        ASSERT_TRUE(std::ranges::adjacent_find(row, std::greater_equal<>()) == row.end());
        for (const index_t l : row) {
            from_csr[l].push_back(i);
        }
    }
    for (index_t l = 0; l < shape.num_columns(); ++l) {
        ASSERT_TRUE(std::ranges::equal(from_csr[l], shape.column_rows(l)));
    }
    const std::uint32_t converge =
        e.final ? e.rounds : spec.converge_rounds + pl.attempt * spec.commit;
    for (index_t i = 0; i < shape.num_rows(); ++i) {
        ASSERT_EQ(shape.converge()[i], i < converge * ref.per_round ? 1 : 0) << "row " << i;
    }
}

// The attempt-0 commit sets partition the columns, a row of a committed round of window k is
// touched only by columns committed at k or before, and the detector rule builds the same
// column and commit sets.
inline void check_commit_sets(const Problem& problem, const WindowSpec& spec,
                              const WindowPlan& plan, const Reference& ref,
                              const std::vector<int>& committed_at) {
    for (index_t j = 0; j < committed_at.size(); ++j) {
        ASSERT_GE(committed_at[j], 0) << "column " << j << " never committed";
    }
    const auto lee =
        detector_rule(ref.supports, ref.per_round, ref.rounds_total, spec.width, spec.commit);
    ASSERT_EQ(lee.size(), plan.num_positions());
    for (std::uint32_t k = 0; k < plan.num_positions(); ++k) {
        const Placement* first = plan.placement(k, 0);
        ASSERT_NE(first, nullptr) << "window " << k;
        const Placement& pl = *first;
        const index_t end = pl.first_row + pl.commit_rounds * ref.per_round;
        for (index_t i = pl.first_row; i < end; ++i) {
            for (index_t e = problem.h_row_ptr[i]; e < problem.h_row_ptr[i + 1]; ++e) {
                ASSERT_LE(committed_at[problem.h_col_indices[e]], static_cast<int>(k))
                    << "row " << i << " of window " << k;
            }
        }
        std::vector<index_t> columns(pl.members.begin(), pl.members.end());
        std::ranges::sort(columns);
        std::vector<index_t> committed;
        const Shape& shape = plan.shape_of(pl);
        for (index_t l = 0; l < shape.num_columns(); ++l) {
            if (shape.commit()[l] != 0) {
                committed.push_back(pl.columns[l]);
            }
        }
        EXPECT_EQ(columns, lee[k].columns) << "window " << k;
        EXPECT_EQ(committed, lee[k].committed) << "window " << k;
    }
}

// Every property of an exact-boundary plan that follows from the definitions: each placement's
// rows, columns, merges, priors, commit flags, classes and convergence rows; the attempt
// structure; the partition of the columns by the attempt-0 commit sets; that a committed round's
// rows are touched only by columns committed there or earlier; agreement with the detector
// rule; and shape enumeration by content in order of first use.
inline void check_exact_plan(const Problem& problem, const WindowSpec& spec,
                             const WindowPlan& plan) {
    const Reference ref(problem, plan);
    ASSERT_EQ(std::size_t{ref.per_round} * ref.rounds_total, problem.num_rows);
    const std::uint32_t max_attempt = spec.on_failure == OnFailure::defer ? spec.max_deferrals : 0;
    const std::uint32_t expected_windows =
        ref.rounds_total <= spec.width
            ? 1
            : 1 + (ref.rounds_total - spec.width + spec.commit - 1) / spec.commit;
    ASSERT_EQ(plan.num_positions(), expected_windows);

    std::vector<int> committed_at(problem.num_columns, -1);
    std::uint32_t shapes_seen = 0;
    for (const Placement& pl : plan.schedule()) {
        SCOPED_TRACE(testing::Message() << "window " << pl.window << " attempt " << pl.attempt);
        const ExpectedPlacement e =
            expected_placement(spec, ref.rounds_total, pl.window, pl.attempt);
        ASSERT_LE(pl.shape, shapes_seen) << "shapes are numbered in order of first use";
        shapes_seen = std::max(shapes_seen, pl.shape + 1);
        check_placement_frame(spec, plan, ref, pl, e);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
        check_local_columns(problem, spec, plan.shape_of(pl), ref, pl, e, committed_at);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
        check_local_matrix(spec, plan.shape_of(pl), ref, pl, e);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
    }
    ASSERT_EQ(shapes_seen, plan.shapes().size());

    for (std::uint32_t k = 0; k < plan.num_positions(); ++k) {
        std::uint32_t attempts = 1;
        while (attempts <= max_attempt &&
               !expected_placement(spec, ref.rounds_total, k, attempts - 1).final) {
            ++attempts;
        }
        ASSERT_EQ(plan.attempts(k), attempts) << "window " << k;
    }
    ASSERT_EQ(plan.placement(plan.num_positions(), 0), nullptr);

    check_commit_sets(problem, spec, plan, ref, committed_at);

    for (std::size_t x = 0; x < plan.shapes().size(); ++x) {
        ASSERT_EQ(plan.shapes()[x].index(), x);
        for (std::size_t y = x + 1; y < plan.shapes().size(); ++y) {
            ASSERT_FALSE(same_content(plan.shapes()[x], plan.shapes()[y])) << x << " = " << y;
        }
    }
}

// Every property of a uniform-boundary plan: the schedule, that shape a is the exact shape of
// placement (1, a) (given by `exact`, an exact plan with the same spec), and that each mapped
// column's support is the bulk representative's shifted by (t_k − t_1)·M rows, while a virtual
// column has no such counterpart.
inline void check_uniform_plan(const Problem& problem, const WindowSpec& spec,
                               const WindowPlan& plan, const WindowPlan& exact) {
    const auto supports = column_supports(problem);
    std::map<std::vector<index_t>, std::vector<index_t>> by_support;
    for (index_t j = 0; j < supports.size(); ++j) {
        by_support[supports[j]].push_back(j);
    }
    const index_t per_round = plan.detectors_per_round();
    const std::uint32_t rounds_total = plan.rounds_total();
    const std::uint32_t max_attempt = spec.on_failure == OnFailure::defer ? spec.max_deferrals : 0;
    const std::uint32_t positions = (rounds_total + spec.commit - 1) / spec.commit;
    ASSERT_EQ(plan.num_positions(), positions);
    ASSERT_EQ(plan.shapes().size(), max_attempt + 1);
    ASSERT_EQ(plan.schedule().size(), std::size_t{positions} * (max_attempt + 1));
    for (std::uint32_t a = 0; a <= max_attempt; ++a) {
        const Placement* bulk = exact.placement(1, a);
        ASSERT_NE(bulk, nullptr);
        ASSERT_FALSE(bulk->final);
        ASSERT_TRUE(same_content(plan.shapes()[a], exact.shape_of(*bulk)));
        const Placement* mapped = plan.placement(1, a);
        ASSERT_NE(mapped, nullptr);
        ASSERT_EQ(mapped->columns, bulk->columns);
    }
    for (const Placement& pl : plan.schedule()) {
        const std::uint32_t k = pl.window;
        const std::uint32_t a = pl.attempt;
        const std::uint32_t t = 1 + k * spec.commit;
        ASSERT_EQ(plan.placement(k, a), &pl);
        ASSERT_EQ(pl.shape, a);
        ASSERT_EQ(pl.first_round, t);
        ASSERT_EQ(pl.first_row, (t - 1) * per_round);
        ASSERT_EQ(pl.rounds, spec.width + a * spec.commit);
        ASSERT_EQ(pl.commit_rounds, spec.commit);
        ASSERT_FALSE(pl.final);
        ASSERT_TRUE(pl.members.empty());
        const Shape& shape = plan.shape_of(pl);
        const Placement* bulk = exact.placement(1, a);
        ASSERT_NE(bulk, nullptr);
        const std::vector<index_t>& reps = bulk->columns;
        ASSERT_EQ(pl.columns.size(), reps.size());
        const std::int64_t shift =
            (std::int64_t{k} - 1) * std::int64_t{spec.commit} * std::int64_t{per_round};
        index_t virtual_committed = 0;
        for (std::size_t l = 0; l < reps.size(); ++l) {
            std::vector<index_t> target;
            bool in_range = true;
            for (const index_t i : supports[reps[l]]) {
                const std::int64_t shifted = std::int64_t{i} + shift;
                in_range = in_range && shifted >= 0 && shifted < std::int64_t{problem.num_rows};
                target.push_back(static_cast<index_t>(shifted));
            }
            const auto found = in_range ? by_support.find(target) : by_support.end();
            if (found == by_support.end()) {
                ASSERT_EQ(pl.columns[l], virtual_column) << "position " << k << " column " << l;
                if (shape.commit()[l] != 0) {
                    ++virtual_committed;
                }
            } else {
                ASSERT_EQ(found->second.size(), 1U);
                ASSERT_EQ(pl.columns[l], found->second.front())
                    << "position " << k << " column " << l;
            }
        }
        ASSERT_EQ(pl.virtual_committed, virtual_committed);
    }
}

} // namespace rtd::window::test
