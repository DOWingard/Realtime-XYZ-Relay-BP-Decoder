// The rtd_decode command line, run in process: spec version checks, a sliding spec decoded by
// window, the recording flags, and an end-to-end run on the bb18 R = 9 fixture with and without
// recording.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"

namespace {

using namespace rtd;
using json = nlohmann::json;
namespace fs = std::filesystem;

const fs::path artifact_dir = test::fixture_root / "bb18_choi_r9" / "artifact";
const fs::path shots_dir = test::fixture_root / "bb18_choi_r9" / "shots";

using test::CliRun;
using test::first_error;
using test::logged;
using test::read_all;
using test::read_json;
using test::run_cli;

json relay_spec() {
    return json::parse(R"({
        "version": 2,
        "policy": "f32",
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "serial"},
        "alpha": {"rule": "constant", "value": 1.0},
        "gamma0": 0.125,
        "pre_iter": 20,
        "set_max_iter": 15,
        "num_sets": 40,
        "stopping": {"rule": "after_n_converged", "count": 3},
        "gamma_source": {"type": "uniform", "seed": 7, "low": -0.24, "high": 0.66},
        "window": {"mode": "whole_shot"}
    })");
}

fs::path write_spec(const fs::path& dir, const std::string& name, const json& spec) {
    const fs::path file = dir / name;
    std::ofstream(file) << spec.dump(2);
    return file;
}

std::vector<std::string> decode_args(const fs::path& config, const fs::path& out) {
    return {"--artifact", artifact_dir.string(), "--shots", shots_dir.string(),
            "--config",   config.string(),       "--out",   out.string()};
}

TEST(DecodeCli, HelpListsTheRecordingFlags) {
    const CliRun run = run_cli({"--help"});
    EXPECT_EQ(run.code, 0);
    for (const char* flag :
         {"--record-solutions", "--save-solution-supports", "--save-commits", "version 2"}) {
        EXPECT_NE(run.out.find(flag), std::string::npos) << flag;
    }
}

TEST(DecodeCli, AVersionOneSpecIsRejectedWithTheUpgrade) {
    const fs::path dir = test::scratch_dir("rtd_cli_v1");
    json spec = relay_spec();
    spec.erase("version");
    spec.erase("window");
    const fs::path out = dir / "out";
    const CliRun run = run_cli(decode_args(write_spec(dir, "v1.json", spec), out));
    EXPECT_EQ(run.code, 2);
    const json error = first_error(run);
    ASSERT_FALSE(error.is_null());
    EXPECT_EQ(error.at("context"), "config");
    EXPECT_EQ(error.at("field"), "version");
    const std::string text = error.at("error");
    EXPECT_NE(text.find(R"("window": {"mode": "whole_shot"})"), std::string::npos) << text;
    EXPECT_FALSE(fs::exists(out));
    fs::remove_all(dir);
}

TEST(DecodeCli, ASlidingSpecIsDecodedWindowByWindow) {
    const fs::path dir = test::scratch_dir("rtd_cli_sliding");
    const fs::path out = dir / "out";
    const fs::path spec =
        test::fixture_root / "bb18_choi_r9" / "window_4_2_exact_commit_anyway" / "spec.json";
    std::vector<std::string> args = decode_args(spec, out);
    args.insert(args.end(), {"--count", "2"});
    const CliRun run = run_cli(args);
    ASSERT_EQ(run.code, 0);
    EXPECT_TRUE(logged(run, "window plan built"));
    const json record = read_json(out / "run.json");
    EXPECT_EQ(record.at("window").at("mode"), "sliding");
    const auto windows = record.at("plan").at("positions").get<std::size_t>();
    EXPECT_GT(windows, 1U);
    test::read_array<std::uint32_t>(out / "win_iterations.npy", {2, windows});
    EXPECT_EQ(read_all<std::int32_t>(out / "best_leg.npy"), (std::vector<std::int32_t>{-1, -1}));
    fs::remove_all(dir);
}

TEST(DecodeCli, RecordingFlagsAreValidated) {
    const fs::path dir = test::scratch_dir("rtd_cli_flags");
    const fs::path config = write_spec(dir, "spec.json", relay_spec());
    const fs::path out = dir / "out";
    const std::vector<std::pair<std::vector<std::string>, std::string>> cases{
        {{"--record-solutions", "21"}, "[0, 20]"},
        {{"--record-solutions", "x"}, "non-negative integer"},
        {{"--record-solutions", "-1"}, "non-negative integer"},
        {{"--record-solutions"}, "needs a value"},
        {{"--save-solution-supports"}, "--record-solutions N with N > 0"},
        {{"--record-solutions", "0", "--save-solution-supports"}, "N > 0"},
        {{"--save-commits"}, "sliding"},
    };
    for (const auto& [extra, expected] : cases) {
        std::vector<std::string> args = decode_args(config, out);
        args.insert(args.end(), extra.begin(), extra.end());
        const CliRun run = run_cli(args);
        EXPECT_EQ(run.code, 2) << extra.front();
        const json error = first_error(run);
        ASSERT_FALSE(error.is_null()) << extra.front();
        const std::string text = error.at("error");
        EXPECT_NE(text.find(expected), std::string::npos) << text;
        EXPECT_FALSE(fs::exists(out)) << extra.front();
    }
    fs::remove_all(dir);
}

TEST(DecodeCli, RecordsSolutionsEndToEnd) {
    const fs::path dir = test::scratch_dir("rtd_cli_record");
    const fs::path config = write_spec(dir, "spec.json", relay_spec());
    const fs::path plain = dir / "plain";
    const fs::path recorded = dir / "recorded";

    // Shots 10 … 49 of the file, decoded without and with recording.
    std::vector<std::string> plain_args = decode_args(config, plain);
    plain_args.insert(plain_args.end(), {"--first", "10", "--count", "40"});
    const CliRun off = run_cli(plain_args);
    ASSERT_EQ(off.code, 0);
    std::vector<std::string> args = decode_args(config, recorded);
    args.insert(args.end(), {"--first", "10", "--count", "40", "--record-solutions", "4",
                             "--save-solution-supports", "--workers", "2", "--warmup", "1"});
    const CliRun on = run_cli(args);
    ASSERT_EQ(on.code, 0);
    EXPECT_TRUE(logged(on, "solution recording on"));
    EXPECT_TRUE(logged(on, "solutions recorded"));
    EXPECT_FALSE(logged(off, "solution recording on"));

    // Every per-shot array is unchanged by recording.
    for (const char* name : {"success.npy", "predicted_observables.npy", "logical_failure.npy"}) {
        EXPECT_EQ(read_all<std::uint8_t>(plain / name), read_all<std::uint8_t>(recorded / name))
            << name;
    }
    for (const char* name : {"iterations.npy", "legs.npy"}) {
        EXPECT_EQ(read_all<std::uint32_t>(plain / name), read_all<std::uint32_t>(recorded / name))
            << name;
    }
    EXPECT_EQ(read_all<std::int32_t>(plain / "best_leg.npy"),
              read_all<std::int32_t>(recorded / "best_leg.npy"));
    const auto weight_bits = [](const std::vector<double>& w) {
        std::vector<std::uint64_t> bits;
        bits.reserve(w.size());
        for (const double x : w) {
            bits.push_back(std::bit_cast<std::uint64_t>(x));
        }
        return bits;
    };
    EXPECT_EQ(weight_bits(read_all<double>(plain / "weight.npy")),
              weight_bits(read_all<double>(recorded / "weight.npy")));
    EXPECT_FALSE(fs::exists(plain / "sol_count.npy"));
    EXPECT_FALSE(fs::exists(plain / "solsup_idx.npy"));

    const std::size_t shots = 40;
    const std::vector<std::uint32_t> count =
        test::read_array<std::uint32_t>(recorded / "sol_count.npy", {shots, 1});
    const std::vector<std::uint32_t> size =
        test::read_array<std::uint32_t>(recorded / "sol_size.npy", {shots, 1, 4});
    const std::vector<std::uint64_t> ptr =
        test::read_array<std::uint64_t>(recorded / "solsup_ptr.npy", {(shots * 4) + 1});
    ASSERT_EQ(size.size() + 1, ptr.size());
    for (std::size_t q = 0; q < size.size(); ++q) {
        EXPECT_EQ(ptr[q + 1] - ptr[q], size[q]);
    }
    test::read_array<std::uint32_t>(recorded / "solsup_idx.npy", {ptr.back()});
    for (const char* name : {"sol_class.npy", "sol_hash.npy"}) {
        test::read_array<std::uint64_t>(recorded / name, {shots, 1, 4});
    }
    test::read_array<std::uint64_t>(recorded / "returned_class.npy", {shots, 1});
    test::read_array<double>(recorded / "sol_weight.npy", {shots, 1, 4});

    const json plain_run = read_json(plain / "run.json");
    const json run = read_json(recorded / "run.json");
    EXPECT_EQ(run.at("window"), json({{"mode", "whole_shot"}}));
    EXPECT_EQ(run.at("decoder").at("version"), 2);
    EXPECT_EQ(run.at("recording").at("record_solutions"), 4);
    EXPECT_EQ(run.at("recording").at("save_solution_supports"), true);
    EXPECT_EQ(run.at("recording").at("save_commits"), false);
    EXPECT_EQ(run.at("summary").at("solutions").at("slots"), 4);
    EXPECT_EQ(plain_run.at("recording").at("record_solutions"), 0);
    EXPECT_FALSE(plain_run.at("summary").contains("solutions"));
    // The summary means the same with and without recording.
    for (const char* key : {"shots", "block_error", "not_converged", "iterations", "legs"}) {
        EXPECT_EQ(run.at("summary").at(key), plain_run.at("summary").at(key)) << key;
    }
    std::uint64_t found = 0;
    for (const std::uint32_t c : count) {
        found += c;
    }
    EXPECT_GT(found, 0U);
    fs::remove_all(dir);
}

} // namespace
