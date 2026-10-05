// The batch driver's building blocks: spec parsing, CPU lists, statistics, the logger, and an
// end-to-end batch run on the committed fixture checked against its goldens.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/harness/batch.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/harness/system.hpp"
#include "rtd/harness/writer.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/shots.hpp"
#include "harness_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::harness;
using json = nlohmann::json;
namespace fs = std::filesystem;

using test::fixture_root;
using test::write_shots;

json valid_spec() {
    return json::parse(R"({
        "version": 2,
        "policy": "f32",
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "team", "threads": 3},
        "alpha": {"rule": "adaptive", "scaling": 2.0},
        "gamma0": 0.125,
        "pre_iter": 80,
        "set_max_iter": 60,
        "num_sets": 600,
        "stopping": {"rule": "after_n_converged", "count": 5},
        "gamma_source": {"type": "explicit", "path": "tables/gammas.npy"},
        "window": {"mode": "whole_shot"}
    })");
}

TEST(Spec, ParsesEveryField) {
    auto spec = parse_spec(valid_spec(), "/configs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_EQ(spec->policy, Policy::f32);
    EXPECT_EQ(spec->backend, BackendKind::cpu);
    EXPECT_EQ(spec->graph.layout, EdgeLayout::row_major);
    EXPECT_EQ(spec->graph.column_order, ColumnOrder::wavefront);
    EXPECT_EQ(spec->graph.block_rows, 64U);
    EXPECT_EQ(spec->executor.team_threads, 3U);
    ASSERT_TRUE(std::holds_alternative<AdaptiveAlpha>(spec->min_sum.alpha));
    EXPECT_EQ(std::get<AdaptiveAlpha>(spec->min_sum.alpha).scaling, 2.0);
    EXPECT_EQ(spec->min_sum.gamma0, 0.125);
    EXPECT_EQ(spec->relay.pre_iter, 80U);
    EXPECT_EQ(spec->relay.set_max_iter, 60U);
    EXPECT_EQ(spec->relay.num_sets, 600U);
    ASSERT_TRUE(std::holds_alternative<AfterNConverged>(spec->relay.stopping));
    EXPECT_EQ(std::get<AfterNConverged>(spec->relay.stopping).count, 5U);
    EXPECT_EQ(spec->gamma.kind, GammaSpec::Kind::explicit_table);
    EXPECT_EQ(spec->gamma.table, fs::path("/configs/tables/gammas.npy"));
    EXPECT_EQ(spec->window.mode, WindowSettings::Mode::whole_shot);
    EXPECT_FALSE(spec->window.is_sliding());
    EXPECT_EQ(spec->raw, valid_spec());
}

TEST(Spec, EveryTopLevelFieldIsRequired) {
    const json full = valid_spec();
    for (const auto& [key, value] : full.items()) {
        json doc = full;
        doc.erase(key);
        auto spec = parse_spec(doc, ".");
        ASSERT_FALSE(spec) << "accepted a spec without " << key;
        EXPECT_EQ(spec.error().field, key);
    }
}

TEST(Spec, NestedFieldsAreNamedByPath) {
    json doc = valid_spec();
    doc["executor"].erase("threads");
    auto spec = parse_spec(doc, ".");
    ASSERT_FALSE(spec);
    EXPECT_EQ(spec.error().field, "executor.threads");

    doc = valid_spec();
    doc["stopping"] = {{"rule", "sometimes"}};
    spec = parse_spec(doc, ".");
    ASSERT_FALSE(spec);
    EXPECT_EQ(spec.error().field, "stopping.rule");
    EXPECT_NE(spec.error().problem.find("after_leg0"), std::string::npos);
}

TEST(Spec, RejectsWrongTypesAndRanges) {
    const std::vector<std::pair<std::string, json>> bad{
        {"pre_iter", -1},        {"pre_iter", 1.5},          {"block_rows", "64"},
        {"policy", "f16"},       {"gamma0", "0.1"},          {"num_sets", 1LL << 40},
        {"layout", "diagonal"},  {"executor", json::array()}};
    for (const auto& [key, value] : bad) {
        json doc = valid_spec();
        doc[key] = value;
        auto spec = parse_spec(doc, ".");
        EXPECT_FALSE(spec) << key << " = " << value.dump();
    }
}

TEST(Spec, RejectsDegenerateConfigurations) {
    json doc = valid_spec();
    doc["gamma_source"] = {{"type", "none"}};
    auto spec = parse_spec(doc, ".");
    ASSERT_FALSE(spec);
    EXPECT_EQ(spec.error().field, "gamma_source");

    doc = valid_spec();
    doc["gamma0"] = nullptr; // relay legs without a memory term
    spec = parse_spec(doc, ".");
    ASSERT_FALSE(spec);
    EXPECT_EQ(spec.error().field, "relay_without_memory");

    doc = valid_spec();
    doc["executor"] = {{"type", "team"}, {"threads", 0}};
    EXPECT_FALSE(parse_spec(doc, "."));
}

TEST(Spec, PlainMinSumNeedsNoGammaSource) {
    json doc = valid_spec();
    doc["gamma0"] = nullptr;
    doc["num_sets"] = 0;
    doc["stopping"] = {{"rule", "after_leg0"}};
    doc["gamma_source"] = {{"type", "none"}};
    doc["executor"] = {{"type", "serial"}};
    doc["alpha"] = {{"rule", "constant"}, {"value", 1.0}};
    auto spec = parse_spec(doc, ".");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_FALSE(spec->min_sum.gamma0);
    EXPECT_EQ(spec->executor.team_threads, 0U);
}

TEST(System, ParsesCpuLists) {
    auto cpus = parse_cpu_list("0-2,6,8-9");
    ASSERT_TRUE(cpus);
    EXPECT_EQ(*cpus, (std::vector<unsigned>{0, 1, 2, 6, 8, 9}));
    cpus = parse_cpu_list("3,1,3");
    ASSERT_TRUE(cpus);
    EXPECT_EQ(*cpus, (std::vector<unsigned>{1, 3}));
    for (const char* bad : {"", "a", "2-1", "1,,2", "-3", "4-"}) {
        EXPECT_FALSE(parse_cpu_list(bad)) << bad;
    }
}

TEST(Statistics, WilsonInterval) {
    const Proportion none = wilson(0, 100);
    EXPECT_EQ(none.estimate, 0.0);
    EXPECT_EQ(none.low, 0.0);
    // Upper Wilson bound of 0/n is z²/(n + z²).
    const double z2 = 1.959963984540054 * 1.959963984540054;
    EXPECT_NEAR(none.high, z2 / (100 + z2), 1e-15);
    const Proportion half = wilson(50, 100);
    EXPECT_NEAR(half.low, 0.40383, 1e-5);
    EXPECT_NEAR(half.high, 0.59617, 1e-5);
}

TEST(Statistics, PerCycleRateInvertsTheBlockRate) {
    EXPECT_EQ(ler_per_cycle(0.0, 12), 0.0);
    EXPECT_EQ(ler_per_cycle(1.0, 12), 1.0);
    // Forward model: the block fails in each cycle with probability p, independently.
    const double p = 7e-6;
    const std::uint32_t rounds = 12;
    const double block = 1.0 - std::pow(1.0 - p, rounds);
    EXPECT_NEAR(*ler_per_cycle(block, rounds) / p, 1.0, 1e-10);
    EXPECT_FALSE(ler_per_cycle(-0.1, 12));
    EXPECT_FALSE(ler_per_cycle(1.1, 12));
    EXPECT_FALSE(ler_per_cycle(0.1, 0));
}

TEST(Statistics, PerQubitRateInvertsTheBlockRate) {
    EXPECT_EQ(ler_per_qubit_per_cycle(0.0, 12, 12), 0.0);
    // One qubit: p = (1 − (1 − 2P)^(1/R)) / 2.
    const double one = *ler_per_qubit_per_cycle(0.1, 1, 5);
    EXPECT_NEAR(one, (1.0 - std::pow(0.8, 0.2)) / 2.0, 1e-15);
    // Forward model: each of k qubits flips per cycle with probability p, independently.
    const double p = 7e-6;
    const std::uint32_t k = 12;
    const std::uint32_t rounds = 12;
    const double qubit = (1.0 - std::pow(1.0 - 2.0 * p, rounds)) / 2.0;
    const double block = 1.0 - std::pow(1.0 - qubit, k);
    EXPECT_NEAR(*ler_per_qubit_per_cycle(block, k, rounds) / p, 1.0, 1e-9);
    EXPECT_FALSE(ler_per_qubit_per_cycle(1.0, 12, 12));
}

TEST(Statistics, SummaryReportsBothPerCycleConventions) {
    ShotResults results;
    results.count = 4;
    results.success = {1, 1, 1, 0};
    results.logical_failure = {0, 1, 0, 0};
    results.iterations = {10, 20, 30, 40};
    results.legs = {1, 1, 2, 3};
    results.decode_ns = {1000, 2000, 3000, 4000};
    results.wall_seconds = 2.0;

    const nlohmann::json both = summarize(results, 12U, 12U);
    EXPECT_EQ(both["block_error"]["count"], 1);
    EXPECT_EQ(both["not_converged"]["count"], 1);
    EXPECT_EQ(both["not_converged_but_correct"], 1);
    EXPECT_DOUBLE_EQ(both["iterations"]["max"].get<double>(), 40.0);
    EXPECT_DOUBLE_EQ(both["decode_time_us"]["p50"].get<double>(), 2.0);
    EXPECT_DOUBLE_EQ(both["ler_per_cycle"]["rate"].get<double>(), *ler_per_cycle(0.25, 12));
    EXPECT_DOUBLE_EQ(both["ler_per_qubit_per_cycle"]["rate"].get<double>(),
                     *ler_per_qubit_per_cycle(0.25, 12, 12));

    const nlohmann::json no_rounds = summarize(results, 12U, std::nullopt);
    EXPECT_FALSE(no_rounds.contains("ler_per_cycle"));
    EXPECT_FALSE(no_rounds.contains("ler_per_qubit_per_cycle"));
    const nlohmann::json no_k = summarize(results, std::nullopt, 12U);
    EXPECT_TRUE(no_k.contains("ler_per_cycle"));
    EXPECT_FALSE(no_k.contains("ler_per_qubit_per_cycle"));
}

TEST(Statistics, NearestRankQuantiles) {
    const std::vector<std::uint32_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::span<const std::uint32_t> sorted(values);
    EXPECT_EQ(nearest_rank(sorted, 0.0), 1U);
    EXPECT_EQ(nearest_rank(sorted, 0.5), 5U);
    EXPECT_EQ(nearest_rank(sorted, 0.9), 9U);
    EXPECT_EQ(nearest_rank(sorted, 0.91), 10U);
    EXPECT_EQ(nearest_rank(sorted, 1.0), 10U);
}

TEST(Logger, WritesOneJsonObjectPerLine) {
    std::ostringstream out;
    JsonLogger logger(out, Level::info, "abc123");
    logger.debug("ctx", "hidden");
    logger.info("ctx", "hello", {{"shots", 7}});
    logger.error("io", "failed", "disk full", {{"path", "/x"}});
    std::istringstream lines(out.str());
    std::string line;
    std::vector<json> parsed;
    while (std::getline(lines, line)) {
        parsed.push_back(json::parse(line));
    }
    ASSERT_EQ(parsed.size(), 2U);
    for (const json& entry : parsed) {
        for (const char* key : {"timestamp", "level", "context", "message", "run_id"}) {
            EXPECT_TRUE(entry.contains(key)) << key;
        }
        EXPECT_EQ(entry["run_id"], "abc123");
    }
    EXPECT_EQ(parsed[0]["level"], "info");
    EXPECT_EQ(parsed[0]["shots"], 7);
    EXPECT_EQ(parsed[1]["level"], "error");
    EXPECT_EQ(parsed[1]["error"], "disk full");
    EXPECT_EQ(parsed[1]["path"], "/x");
    EXPECT_TRUE(parsed[1].contains("stack"));
}

struct BatchCase {
    std::string golden;
    unsigned workers;
    unsigned team;                  // 0 = Serial
    std::uint32_t record_solutions; // recording must leave every golden output unchanged
    bool save_supports;
};

class BatchGolden : public testing::TestWithParam<BatchCase> {};

TEST_P(BatchGolden, ReproducesTheGoldens) {
    const BatchCase& c = GetParam();
    auto artifact = io::load_artifact(fixture_root / "bb18_choi" / "artifact");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto golden = io::load_golden(fixture_root / "bb18_choi" / c.golden, artifact->num_detectors(),
                                  artifact->num_columns());
    ASSERT_TRUE(golden) << io::describe(golden.error());

    const fs::path root =
        fs::path(testing::TempDir()) / std::format("rtd_batch_{}_{}_{}_{}_{}", c.golden, c.workers,
                                                   c.team, c.record_solutions, c.save_supports);
    fs::remove_all(root);
    const fs::path shots_dir = write_shots(*artifact, *golden, root / "shots");
    auto shots = io::load_shots(shots_dir, *artifact, false);
    ASSERT_TRUE(shots) << io::describe(shots.error());

    DecoderSpec spec;
    spec.policy = golden->float_type == "f64" ? Policy::f64 : Policy::f32;
    spec.executor.team_threads = c.team;
    spec.min_sum = golden->min_sum;
    spec.relay = golden->relay;
    const GammaSource* gammas = golden->gammas ? &*golden->gammas : nullptr;

    std::ostringstream log;
    JsonLogger logger(log, Level::warn, "test");
    RunOptions options{.first = 0,
                       .count = golden->count(),
                       .workers = c.workers,
                       .cpus = {},
                       .warmup = 2,
                       .save_decodings = true,
                       .record_solutions = c.record_solutions,
                       .save_solution_supports = c.save_supports,
                       .save_commits = false};
    auto runner = BatchRunner::create(*artifact, *shots, spec, gammas, options, logger);
    ASSERT_TRUE(runner) << runner.error().message;
    auto results = runner->run();
    ASSERT_TRUE(results) << results.error().message;

    const index_t n = artifact->num_columns();
    const index_t k = artifact->num_observables();
    std::vector<Bit> expected_prediction(k);
    std::size_t mismatches = 0;
    for (std::size_t s = 0; s < golden->count(); ++s) {
        bool ok = results->success[s] == golden->success.data[s];
        ok &= std::cmp_equal(results->iterations[s], golden->iterations.data[s]);
        ok &= std::bit_cast<std::uint64_t>(results->weight[s]) ==
              std::bit_cast<std::uint64_t>(golden->weight.data[s]);
        const auto decoding = golden->decoding.row(s);
        ok &= std::ranges::equal(decoding, std::span(results->decodings).subspan(s * n, n));
        artifact->observables.apply(decoding, expected_prediction);
        ok &= std::ranges::equal(expected_prediction, std::span(results->predicted).subspan(s * k, k));
        ok &= (results->logical_failure[s] != 0) ==
              std::ranges::any_of(expected_prediction, [](Bit b) { return b != 0; });
        mismatches += ok ? 0U : 1U;
    }
    EXPECT_EQ(mismatches, 0U);

    const json summary = summarize(*results, 8, 3);
    EXPECT_EQ(summary["shots"], golden->count());
    fs::create_directories(root / "out");
    const auto written = write_results(root / "out", *results, {{"summary", summary}});
    ASSERT_TRUE(written) << io::describe(written.error());
    for (const char* file : {"success.npy", "iterations.npy", "legs.npy", "best_leg.npy",
                             "weight.npy", "decode_ns.npy", "predicted_observables.npy",
                             "logical_failure.npy", "decodings.npy", "run.json"}) {
        EXPECT_TRUE(fs::exists(root / "out" / file)) << file;
    }
    auto iterations = io::read_npy<std::uint32_t>(root / "out" / "iterations.npy", 1);
    ASSERT_TRUE(iterations);
    EXPECT_TRUE(std::ranges::equal(iterations->span(), results->iterations));
    EXPECT_EQ(log.str(), "") << "unexpected warnings";
    fs::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(Fixture, BatchGolden,
                         testing::Values(BatchCase{"min_sum_f32", 1, 0, 0, false},
                                         BatchCase{"relay_f32", 3, 0, 0, false},
                                         BatchCase{"relay_f64", 2, 2, 0, false},
                                         BatchCase{"relay_all_f32", 1, 3, 0, false},
                                         BatchCase{"min_sum_f32", 2, 0, 1, true},
                                         BatchCase{"relay_f32", 3, 0, 20, true},
                                         BatchCase{"relay_all_f32", 2, 0, 7, true}),
                         [](const testing::TestParamInfo<BatchCase>& param_info) {
                             const BatchCase& c = param_info.param;
                             return std::format("{}_w{}_t{}_rec{}{}", c.golden, c.workers, c.team,
                                                c.record_solutions, c.save_supports ? "s" : "");
                         });

TEST(Batch, RejectsBadRanges) {
    auto artifact = io::load_artifact(fixture_root / "bb18_choi" / "artifact");
    ASSERT_TRUE(artifact);
    auto golden = io::load_golden(fixture_root / "bb18_choi" / "min_sum_f32",
                                  artifact->num_detectors(), artifact->num_columns());
    ASSERT_TRUE(golden);
    const fs::path root = fs::path(testing::TempDir()) / "rtd_batch_ranges";
    fs::remove_all(root);
    auto shots = io::load_shots(write_shots(*artifact, *golden, root), *artifact, false);
    ASSERT_TRUE(shots);
    DecoderSpec spec;
    spec.min_sum = golden->min_sum;
    spec.relay = golden->relay;
    std::ostringstream log;
    JsonLogger logger(log, Level::error, "test");
    const auto create = [&](std::size_t first, std::size_t count, unsigned workers,
                            std::vector<unsigned> cpus = {}) {
        RunOptions options;
        options.first = first;
        options.count = count;
        options.workers = workers;
        options.cpus = std::move(cpus);
        return BatchRunner::create(*artifact, *shots, spec, nullptr, std::move(options), logger);
    };
    EXPECT_FALSE(create(0, 101, 1));
    EXPECT_FALSE(create(99, 2, 1));
    EXPECT_FALSE(create(0, 0, 1));
    EXPECT_FALSE(create(0, 10, 0));
    EXPECT_FALSE(create(0, 10, 2, {0}));
    EXPECT_TRUE(create(99, 1, 1));
    fs::remove_all(root);
}

} // namespace
