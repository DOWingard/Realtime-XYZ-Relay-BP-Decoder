// rtd_decode with a fixed-point number format: the spec's "arithmetic" field and its rules, and
// end-to-end runs that must reproduce the integer emulator's goldens (sliding windows through the
// golden's own spec, and the whole-shot decode against the W ≥ Rt golden).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

#include "harness_support.hpp"
#include "rtd/harness/spec.hpp"

namespace {

using namespace rtd;
using namespace rtd::harness;
using namespace rtd::test;
using json = nlohmann::json;
namespace fs = std::filesystem;

const fs::path repo_root = fs::path(RTD_FIXTURE_DIR).parent_path().parent_path();

json fixed_spec() {
    return json::parse(R"({
        "version": 2,
        "arithmetic": "int4.2.8",
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "serial"},
        "alpha": {"rule": "adaptive", "scaling": 1.0},
        "gamma0": 0.125,
        "pre_iter": 80,
        "set_max_iter": 60,
        "num_sets": 600,
        "stopping": {"rule": "after_n_converged", "count": 5},
        "gamma_source": {"type": "uniform", "seed": 1, "low": -0.24, "high": 0.66},
        "window": {"mode": "whole_shot"}
    })");
}

SpecError rejection(const json& doc) {
    auto spec = parse_spec(doc, "/specs");
    EXPECT_FALSE(spec) << "accepted " << doc.dump();
    return spec ? SpecError{} : spec.error();
}

TEST(SpecArithmetic, NamesEveryCompiledFormat) {
    for (const auto& [name, policy] : {std::pair{"f32", Policy::f32},
                                       std::pair{"f64", Policy::f64},
                                       std::pair{"int4.2.8", Policy::int4_2_8},
                                       std::pair{"int5.2.8", Policy::int5_2_8},
                                       std::pair{"int6.2.8", Policy::int6_2_8}}) {
        json doc = fixed_spec();
        doc["arithmetic"] = name;
        auto spec = parse_spec(doc, "/specs");
        ASSERT_TRUE(spec) << name << ": " << spec.error().field << ": " << spec.error().problem;
        EXPECT_EQ(spec->policy, policy);
        EXPECT_EQ(to_string(policy), name);
        EXPECT_EQ(is_fixed_point(policy), std::string(name).starts_with("int"));
    }
}

TEST(SpecArithmetic, PolicyStillWorksAlone) {
    json doc = fixed_spec();
    doc.erase("arithmetic");
    doc["policy"] = "f64";
    doc["alpha"] = {{"rule", "constant"}, {"value", 0.8}};
    auto spec = parse_spec(doc, "/specs");
    ASSERT_TRUE(spec) << spec.error().problem;
    EXPECT_EQ(spec->policy, Policy::f64);
    doc["policy"] = "int4.2.8"; // fixed-point formats are named by "arithmetic" only
    EXPECT_EQ(rejection(doc).field, "policy");
}

TEST(SpecArithmetic, TheFormatIsNamedExactlyOnce) {
    json both = fixed_spec();
    both["policy"] = "f32";
    EXPECT_EQ(rejection(both).field, "arithmetic");
    json neither = fixed_spec();
    neither.erase("arithmetic");
    const SpecError error = rejection(neither);
    EXPECT_EQ(error.field, "policy");
    EXPECT_TRUE(error.problem.contains("arithmetic"));
    json unknown = fixed_spec();
    unknown["arithmetic"] = "int8.2.8";
    EXPECT_EQ(rejection(unknown).field, "arithmetic");
}

TEST(SpecArithmetic, FixedPointNeedsTheCpuAndAShiftScaling) {
    json cuda = fixed_spec();
    cuda["backend"] = "cuda";
    EXPECT_EQ(rejection(cuda).field, "backend");
    json scaled = fixed_spec();
    scaled["alpha"] = {{"rule", "constant"}, {"value", 0.8}};
    EXPECT_EQ(rejection(scaled).field, "alpha");
    json slow = fixed_spec();
    slow["alpha"] = {{"rule", "adaptive"}, {"scaling", 2.0}};
    EXPECT_EQ(rejection(slow).field, "alpha");
    json shift = fixed_spec();
    shift["alpha"] = {{"rule", "constant"}, {"value", 0.875}};
    EXPECT_TRUE(parse_spec(shift, "/specs"));
    json unit = fixed_spec();
    unit["alpha"] = {{"rule", "constant"}, {"value", 1.0}};
    EXPECT_TRUE(parse_spec(unit, "/specs"));
}

// ---- End to end against the emulator's goldens --------------------------------------------

// Test binaries of several builds may run these decodes at once, and scratch_dir() clears a
// directory of the same name, so each process writes under its own.
fs::path own_scratch_dir(const std::string& name) {
    return scratch_dir(std::format("{}_{}", name, ::getpid()));
}

std::vector<fs::path> fixed_goldens() {
    std::vector<fs::path> dirs;
    const fs::path root = fixture_root / "bb18_choi_r9";
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.path().filename().string().starts_with("fixed_") &&
            fs::exists(entry.path() / "manifest.json")) {
            dirs.push_back(entry.path());
        }
    }
    std::ranges::sort(dirs);
    return dirs;
}

template <io::NpyElement T>
void expect_rows(const fs::path& golden_file, const fs::path& ours_file, std::size_t count,
                 std::size_t column = 0, std::size_t stride = 0) {
    const auto golden = read_all<T>(golden_file);
    const auto ours = read_all<T>(ours_file);
    const std::size_t width = golden.size() / std::max<std::size_t>(
                                                   read_json(golden_file.parent_path() /
                                                             "manifest.json")
                                                       .at("count")
                                                       .get<std::size_t>(),
                                                   1);
    std::size_t mismatches = 0;
    for (std::size_t s = 0; s < count; ++s) {
        const std::size_t ours_width = stride == 0 ? width : 1;
        for (std::size_t c = 0; c < ours_width; ++c) {
            const T expected = golden[(s * width) + (stride == 0 ? c : column)];
            const T got = ours[(s * ours_width) + c];
            bool equal = expected == got;
            if constexpr (std::same_as<T, double>) {
                equal = std::bit_cast<std::uint64_t>(expected) == std::bit_cast<std::uint64_t>(got);
            }
            if (!equal && ++mismatches <= 5) {
                ADD_FAILURE() << std::format("{} shot {} column {}: golden {} ours {}",
                                             ours_file.filename().string(), s, c, expected, got);
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << ours_file;
}

std::size_t decode_count() {
#ifdef __OPTIMIZE__
    return 40;
#else
    return 6;
#endif
}

TEST(FixedRtdDecode, SlidingRunsReproduceTheEmulator) {
    std::size_t runs = 0;
    for (const fs::path& dir : fixed_goldens()) {
        const json manifest = read_json(dir / "manifest.json");
        const json spec = read_json(dir / "spec.json");
        const std::uint32_t width = manifest.at("window").at("width").get<std::uint32_t>();
        if (width >= manifest.at("rounds").get<std::uint32_t>() + 1) {
            continue; // the whole-shot golden has its own test
        }
        SCOPED_TRACE(dir.string());
        const fs::path out = own_scratch_dir("rtd_fixed_" + dir.filename().string());
        const std::size_t count = decode_count();
        const CliRun run = run_cli(
            {"--artifact", (repo_root / manifest.at("artifact").at("path").get<std::string>()).string(),
             "--shots", dir.string(), "--config", (dir / "spec.json").string(), "--out", out.string(),
             "--count", std::to_string(count), "--workers", "2", "--save-commits"});
        ASSERT_EQ(run.code, 0) << first_error(run).dump();
        for (const char* name : {"win_iterations", "win_legs", "win_unexplained", "win_virtual"}) {
            expect_rows<std::uint32_t>(dir / std::format("{}.npy", name),
                                       out / std::format("{}.npy", name), count);
        }
        for (const char* name : {"win_attempts", "win_converged", "win_cap_hit", "win_flagged",
                                 "predicted_observables", "logical_failure", "flagged",
                                 "success"}) {
            expect_rows<std::uint8_t>(dir / std::format("{}.npy", name),
                                      out / std::format("{}.npy", name), count);
        }
        for (const char* name : {"win_weight", "win_committed_weight", "weight"}) {
            expect_rows<double>(dir / std::format("{}.npy", name),
                                out / std::format("{}.npy", name), count);
        }
        for (const char* name : {"iterations", "legs"}) {
            expect_rows<std::uint32_t>(dir / std::format("{}.npy", name),
                                       out / std::format("{}.npy", name), count);
        }
        const auto golden_ptr = read_all<std::uint64_t>(dir / "commit_ptr.npy");
        const auto golden_faults = read_all<std::uint32_t>(dir / "commit_faults.npy");
        const auto ours_ptr = read_all<std::uint64_t>(out / "commit_ptr.npy");
        const auto ours_faults = read_all<std::uint32_t>(out / "commit_faults.npy");
        const std::size_t cells = ours_ptr.size() - 1;
        ASSERT_LE(cells + 1, golden_ptr.size());
        for (std::size_t i = 0; i < cells; ++i) {
            EXPECT_TRUE(std::ranges::equal(
                std::span(golden_faults).subspan(golden_ptr[i], golden_ptr[i + 1] - golden_ptr[i]),
                std::span(ours_faults).subspan(ours_ptr[i], ours_ptr[i + 1] - ours_ptr[i])))
                << "commits of cell " << i;
        }
        const json record = read_json(out / "run.json");
        EXPECT_EQ(record.at("decoder").at("arithmetic"), spec.at("arithmetic"));
        fs::remove_all(out);
        ++runs;
    }
    EXPECT_EQ(runs, 4U);
}

TEST(FixedRtdDecode, WholeShotRunReproducesTheOneWindowGolden) {
    const fs::path dir = fixture_root / "bb18_choi_r9" / "fixed_int4_whole_shot";
    ASSERT_TRUE(fs::exists(dir / "manifest.json"));
    const json manifest = read_json(dir / "manifest.json");
    ASSERT_EQ(manifest.at("num_positions"), 1);
    json spec = read_json(dir / "spec.json");
    spec["window"] = {{"mode", "whole_shot"}};
    spec["gamma_source"] = {{"type", "explicit"}, {"path", (dir / "gammas" / "shape_0.npy").string()}};
    const fs::path out = own_scratch_dir("rtd_fixed_whole_shot");
    const fs::path config = out / "spec.json";
    std::ofstream(config) << spec.dump(2);
    const std::size_t count = decode_count();
    const CliRun run = run_cli(
        {"--artifact", (repo_root / manifest.at("artifact").at("path").get<std::string>()).string(),
         "--shots", dir.string(), "--config", config.string(), "--out", (out / "run").string(),
         "--count", std::to_string(count)});
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    // One window holding the whole problem: per-window records are the whole-shot ones.
    expect_rows<std::uint32_t>(dir / "win_iterations.npy", out / "run" / "iterations.npy", count, 0, 1);
    expect_rows<std::uint32_t>(dir / "win_legs.npy", out / "run" / "legs.npy", count, 0, 1);
    expect_rows<std::uint8_t>(dir / "win_converged.npy", out / "run" / "success.npy", count, 0, 1);
    expect_rows<double>(dir / "win_weight.npy", out / "run" / "weight.npy", count, 0, 1);
    expect_rows<std::uint8_t>(dir / "predicted_observables.npy",
                              out / "run" / "predicted_observables.npy", count);
    expect_rows<std::uint8_t>(dir / "logical_failure.npy", out / "run" / "logical_failure.npy",
                              count);
    fs::remove_all(out);
}

} // namespace
