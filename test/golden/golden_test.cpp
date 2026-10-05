// Bit-exact agreement with the reference decoder on identical inputs: for every shot of every
// golden set, the correction ê, the convergence flag, the iteration count, the solution weight,
// the per-leg record and (where the reference exposes them) the final marginals must be equal —
// not close, equal.

#include <gtest/gtest.h>

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"

namespace {

using namespace rtd;
namespace fs = std::filesystem;

const fs::path fixture_root = RTD_FIXTURE_DIR;

// Reference "unique best" flag of each leg, derived from our trace: set on the returned leg when
// no later converged leg reached the same weight.
std::vector<bool> unique_best_flags(const DecodeResult& r) {
    std::vector<bool> flags(r.legs_executed, false);
    if (!r.best_leg) {
        return flags;
    }
    bool unique = true;
    for (std::uint32_t leg = *r.best_leg + 1; leg < r.legs_executed; ++leg) {
        if (r.legs[leg].converged && r.legs[leg].weight == r.weight) {
            unique = false;
        }
    }
    flags[*r.best_leg] = unique;
    return flags;
}

struct Tally {
    std::size_t shots = 0;
    std::size_t mismatched_shots = 0;
};

template <class A, class Exec>
Tally check_golden(const io::Artifact& artifact, const io::Golden& golden, Exec executor) {
    Tally tally;
    auto backend = CpuBackend<A, Exec>::create(artifact.graph, artifact.priors, std::move(executor));
    EXPECT_TRUE(backend.has_value());
    if (!backend) {
        return tally;
    }
    const GammaSource* gammas = golden.gammas ? &*golden.gammas : nullptr;
    auto decoder = RelayDecoder<CpuBackend<A, Exec>>::create(std::move(*backend), golden.min_sum,
                                                            golden.relay, gammas);
    EXPECT_TRUE(decoder.has_value()) << (decoder ? "" : decoder.error().detail);
    if (!decoder) {
        return tally;
    }
    const index_t n = artifact.num_columns();
    std::vector<double> marginals(n);
    const std::size_t posterior_shots = golden.posterior ? golden.posterior->rows() : 0;

    for (std::size_t s = 0; s < golden.count(); ++s) {
        auto result = decoder->decode(golden.detectors.row(s));
        EXPECT_TRUE(result.has_value());
        if (!result) {
            return tally;
        }
        ++tally.shots;
        bool ok = true;
        const auto expected_hard = golden.decoding.row(s);
        ok &= std::ranges::equal(result->hard, expected_hard);
        ok &= result->success == (golden.success.data[s] != 0);
        ok &= static_cast<std::int64_t>(result->iterations) == golden.iterations.data[s];
        ok &= std::bit_cast<std::uint64_t>(result->weight) ==
              std::bit_cast<std::uint64_t>(golden.weight.data[s]);
        if (golden.legs_ptr) {
            const auto begin = static_cast<std::size_t>(golden.legs_ptr->data[s]);
            const auto end = static_cast<std::size_t>(golden.legs_ptr->data[s + 1]);
            ok &= result->legs_executed == end - begin;
            if (result->legs_executed == end - begin) {
                const std::vector<bool> flags = unique_best_flags(*result);
                for (std::size_t leg = 0; leg < end - begin; ++leg) {
                    const LegRecord& record = result->legs[leg];
                    ok &= std::cmp_equal(record.iterations,
                                         golden.leg_iterations->data[begin + leg]);
                    ok &= record.converged == (golden.leg_converged->data[begin + leg] != 0);
                    ok &= flags[leg] == (golden.leg_unique_best->data[begin + leg] != 0);
                }
            }
        }
        // The reference reports the marginals as they were when its returned leg ended; ours
        // are those of the last leg run. They coincide when the returned leg is the last one.
        const bool last_leg_returned =
            result->legs_executed == 1 ||
            (result->best_leg && *result->best_leg + 1 == result->legs_executed);
        if (s < posterior_shots && last_leg_returned) {
            decoder->backend().read_marginals(marginals);
            const auto expected = golden.posterior->row(s);
            for (index_t j = 0; j < n; ++j) {
                ok &= std::bit_cast<std::uint64_t>(marginals[j]) ==
                      std::bit_cast<std::uint64_t>(expected[j]);
            }
        }
        if (!ok) {
            ++tally.mismatched_shots;
            if (tally.mismatched_shots <= 5) {
                ADD_FAILURE() << std::format(
                    "{} shot {}: success {} vs {}, iterations {} vs {}, weight {} vs {}, legs {}",
                    golden.directory.filename().string(), s, result->success,
                    golden.success.data[s], result->iterations, golden.iterations.data[s],
                    result->weight, golden.weight.data[s], result->legs_executed);
            }
        }
    }
    return tally;
}

template <class Exec>
Tally dispatch_float(const io::Artifact& artifact, const io::Golden& golden, Exec executor) {
    if (golden.float_type == "f32") {
        return check_golden<F32>(artifact, golden, std::move(executor));
    }
    if (golden.float_type == "f64") {
        return check_golden<F64>(artifact, golden, std::move(executor));
    }
    ADD_FAILURE() << "unknown float type " << golden.float_type;
    return {};
}

struct Case {
    std::string golden;
    ColumnOrder order;
    unsigned team; // 0 = Serial
    EdgeLayout layout = EdgeLayout::row_major;
};

std::string_view order_name(ColumnOrder order) {
    switch (order) {
    case ColumnOrder::wavefront:
        return "wavefront";
    case ColumnOrder::degree_classes:
        return "degree";
    case ColumnOrder::natural:
        return "natural";
    }
    return "unknown";
}

std::string case_name(const testing::TestParamInfo<Case>& info) {
    return std::format("{}_{}_{}_{}", info.param.golden,
                       info.param.layout == EdgeLayout::row_major ? "rows" : "blocked",
                       order_name(info.param.order),
                       info.param.team == 0 ? std::string("serial")
                                            : std::format("team{}", info.param.team));
}

class Bb18Golden : public testing::TestWithParam<Case> {};

TEST_P(Bb18Golden, MatchesReferenceBitForBit) {
    const Case& c = GetParam();
    io::ArtifactOptions options;
    options.graph.column_order = c.order;
    options.graph.layout = c.layout;
    // Small blocks, so the 72-row fixture has several of them and the team splits rows.
    options.graph.block_rows = 8;
    auto artifact = io::load_artifact(fixture_root / "bb18_choi" / "artifact", options);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto golden = io::load_golden(fixture_root / "bb18_choi" / c.golden, artifact->num_detectors(),
                                  artifact->num_columns());
    ASSERT_TRUE(golden) << io::describe(golden.error());
    ASSERT_GT(golden->count(), 0U);
    const Tally tally = c.team == 0 ? dispatch_float(*artifact, *golden, Serial{})
                                    : dispatch_float(*artifact, *golden, Team(c.team));
    EXPECT_EQ(tally.shots, golden->count());
    EXPECT_EQ(tally.mismatched_shots, 0U);
}

std::vector<Case> bb18_cases() {
    const std::vector<std::string> goldens{
        "min_sum_f32",       "min_sum_f64",       "mem_bp_f32",      "min_sum_adaptive_f32",
        "relay_f32",         "relay_f64",         "relay_all_f32",   "relay_preiter_f32",
        "single_columns_f32"};
    std::vector<Case> cases;
    for (const auto& g : goldens) {
        cases.push_back({.golden = g, .order = ColumnOrder::wavefront, .team = 0});
        cases.push_back({.golden = g, .order = ColumnOrder::degree_classes, .team = 0});
        cases.push_back({.golden = g, .order = ColumnOrder::natural, .team = 0});
        cases.push_back({.golden = g, .order = ColumnOrder::wavefront, .team = 3});
        cases.push_back({.golden = g, .order = ColumnOrder::degree_classes, .team = 2});
        cases.push_back({.golden = g,
                         .order = ColumnOrder::wavefront,
                         .team = 0,
                         .layout = EdgeLayout::column_blocked});
        cases.push_back({.golden = g,
                         .order = ColumnOrder::degree_classes,
                         .team = 0,
                         .layout = EdgeLayout::column_blocked});
        cases.push_back({.golden = g,
                         .order = ColumnOrder::wavefront,
                         .team = 3,
                         .layout = EdgeLayout::column_blocked});
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Fixture, Bb18Golden, testing::ValuesIn(bb18_cases()), case_name);

// Gross-code goldens are large and not committed. Point RTD_GROSS_ARTIFACT at the artifact
// directory and RTD_GROSS_GOLDEN at a directory of golden sets to run them.
TEST(GrossGolden, MatchesReferenceBitForBit) {
    const char* artifact_dir = std::getenv("RTD_GROSS_ARTIFACT");
    const char* golden_root = std::getenv("RTD_GROSS_GOLDEN");
    if (artifact_dir == nullptr || golden_root == nullptr) {
        GTEST_SKIP() << "set RTD_GROSS_ARTIFACT and RTD_GROSS_GOLDEN to run";
    }
    auto artifact = io::load_artifact(artifact_dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    std::vector<fs::path> sets;
    for (const auto& entry : fs::directory_iterator(golden_root)) {
        if (fs::exists(entry.path() / "manifest.json")) {
            sets.push_back(entry.path());
        }
    }
    std::ranges::sort(sets);
    ASSERT_FALSE(sets.empty());
    for (const fs::path& dir : sets) {
        auto golden = io::load_golden(dir, artifact->num_detectors(), artifact->num_columns());
        ASSERT_TRUE(golden) << io::describe(golden.error());
        const Tally serial = dispatch_float(*artifact, *golden, Serial{});
        EXPECT_EQ(serial.mismatched_shots, 0U) << dir.filename();
        const Tally team = dispatch_float(*artifact, *golden, Team(3));
        EXPECT_EQ(team.mismatched_shots, 0U) << dir.filename() << " (team of 3)";
        std::cout << std::format("[ gross    ] {}: {} shots, {} mismatched (serial), {} (team)\n",
                                 dir.filename().string(), serial.shots, serial.mismatched_shots,
                                 team.mismatched_shots);
    }
}

} // namespace
