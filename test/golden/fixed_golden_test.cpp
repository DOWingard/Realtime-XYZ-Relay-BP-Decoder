// Golden tests of the fixed-point decoder: every golden written by the Python integer emulator
// (rtd.fixed_golden) must be reproduced array by array by the C++ stream decoder with the
// fixed-point relay decoder inside: per-window iterations, legs, attempts, convergence, cap hits,
// weights (bitwise), unexplained detectors, flags, virtual commits, the committed faults of every
// window, and per shot the frame, the logical failure, the flag, success, iterations, legs and
// weight.
//
// The committed fixture goldens (bb18, R = 9) always run; gross-code goldens run when
// RTD_FIXED_GOLDEN_DIR names a directory holding them (e.g. data/golden/fixed).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/decoder_fixed.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/sha256.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
namespace fs = std::filesystem;
using Json = nlohmann::json;

const fs::path fixture_root = RTD_FIXTURE_DIR;
const fs::path repo_root = fs::path(RTD_FIXTURE_DIR).parent_path().parent_path();

constexpr bool sanitized =
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    true;
#elifdef __has_feature
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
    true;
#else
    false;
#endif
#else
    false;
#endif
constexpr bool optimised =
#ifdef __OPTIMIZE__
    true;
#else
    false;
#endif

// Optimised builds decode every shot; debug and sanitizer builds the first few, which still
// reach every window position. RTD_FIXED_GOLDEN_SHOTS = N overrides: the first N, 0 = all.
std::size_t shots_to_decode(std::size_t shots) {
    if (const char* limit = std::getenv("RTD_FIXED_GOLDEN_SHOTS"); limit != nullptr) {
        const auto n = static_cast<std::size_t>(std::stoull(limit));
        return n == 0 ? shots : std::min(shots, n);
    }
    return optimised && !sanitized ? shots : std::min<std::size_t>(shots, 6);
}

struct Settings {
    std::string arithmetic;
    MinSumConfig min_sum;
    RelayConfig relay;
    WindowSpec window;
};

std::expected<Settings, std::string> settings_of(const Json& manifest) {
    const Json& d = manifest.at("decoder");
    Settings s;
    s.arithmetic = d.at("arithmetic").get<std::string>();
    if (!d.at("gamma0").is_null()) {
        s.min_sum.gamma0 = d.at("gamma0").get<double>();
    }
    const Json& alpha = d.at("alpha");
    if (alpha.at("rule") == "constant") {
        s.min_sum.alpha = ConstantAlpha{alpha.at("value").get<double>()};
    } else if (alpha.at("rule") == "adaptive") {
        s.min_sum.alpha = AdaptiveAlpha{alpha.at("scaling").get<double>()};
    } else {
        return std::unexpected("unknown alpha rule " + alpha.dump());
    }
    s.relay.pre_iter = d.at("pre_iter").get<std::uint32_t>();
    s.relay.set_max_iter = d.at("set_max_iter").get<std::uint32_t>();
    s.relay.num_sets = d.at("num_sets").get<std::uint32_t>();
    const Json& stopping = d.at("stopping");
    if (stopping.at("rule") == "after_n_converged") {
        s.relay.stopping = AfterNConverged{stopping.at("count").get<std::uint32_t>()};
    } else if (stopping.at("rule") == "after_leg0") {
        s.relay.stopping = AfterLeg0{};
    } else if (stopping.at("rule") == "all_legs") {
        s.relay.stopping = AllLegs{};
    } else {
        return std::unexpected("unknown stopping rule " + stopping.dump());
    }
    const Json& w = manifest.at("window");
    const auto boundary = parse_boundary(w.at("boundary").get<std::string>());
    const auto policy = parse_on_failure(w.at("on_failure").get<std::string>());
    if (!boundary || !policy) {
        return std::unexpected("unknown boundary or policy in " + w.dump());
    }
    s.window = WindowSpec{.width = w.at("width").get<std::uint32_t>(),
                          .commit = w.at("commit").get<std::uint32_t>(),
                          .converge_rounds = w.at("converge_rounds").get<std::uint32_t>(),
                          .boundary = *boundary,
                          .on_failure = *policy,
                          .max_deferrals = w.at("max_deferrals").get<std::uint32_t>(),
                          .iteration_cap = std::nullopt};
    if (!w.at("iteration_cap").is_null()) {
        s.window.iteration_cap = w.at("iteration_cap").get<std::uint32_t>();
    }
    return s;
}

// What the C++ stream decoder produced for the decoded shots, in the goldens' layouts.
struct Produced {
    std::size_t shots = 0;
    std::size_t windows = 0;
    std::size_t observables = 0;
    std::size_t decoded = 0; // shots 0 .. decoded − 1
    std::map<std::string, std::vector<std::uint32_t>> u32;
    std::map<std::string, std::vector<std::uint8_t>> u8;
    std::map<std::string, std::vector<double>> f64;
    std::vector<std::vector<std::uint32_t>> commits; // [S·K]

    void size(std::size_t s, std::size_t k, std::size_t o) {
        shots = s;
        windows = k;
        observables = o;
        for (const char* name : {"win_iterations", "win_legs", "win_unexplained", "win_virtual"}) {
            u32[name].assign(s * k, 0);
        }
        for (const char* name : {"win_attempts", "win_converged", "win_cap_hit", "win_flagged"}) {
            u8[name].assign(s * k, 0);
        }
        f64["win_weight"].assign(s * k, 0.0);
        f64["win_committed_weight"].assign(s * k, 0.0);
        commits.assign(s * k, {});
        u8["predicted_observables"].assign(s * o, 0);
        for (const char* name : {"logical_failure", "flagged", "success"}) {
            u8[name].assign(s, 0);
        }
        u32["iterations"].assign(s, 0);
        u32["legs"].assign(s, 0);
        f64["weight"].assign(s, 0.0);
    }

    // Values per shot of an array with this name, or 0 when it is not a per-shot array.
    [[nodiscard]] std::size_t width(const std::string& name) const {
        if (name.starts_with("win_")) {
            return windows;
        }
        return name == "predicted_observables" ? observables : 1;
    }
};

template <class T>
bool same(T a, T b) {
    if constexpr (std::same_as<T, double>) {
        return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
    } else {
        return a == b;
    }
}

template <io::NpyElement T>
void compare_array(const fs::path& file, const std::vector<T>& ours, const Produced& p) {
    const std::string name = file.stem().string();
    auto golden = io::read_npy<T>(file);
    ASSERT_TRUE(golden) << name << ": " << io::describe(golden.error());
    const std::size_t width = p.width(name);
    ASSERT_EQ(golden->size(), p.shots * width) << name;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < p.decoded * width; ++i) {
        if (!same(golden->data[i], ours[i])) {
            if (++mismatches <= 5) {
                ADD_FAILURE() << std::format("{} shot {} column {}: golden {} ours {}", name,
                                             i / width, i % width, golden->data[i], ours[i]);
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << name;
}

void compare_commits(const fs::path& dir, const Produced& p) {
    auto ptr = io::read_npy<std::uint64_t>(dir / "commit_ptr.npy", 1);
    auto faults = io::read_npy<std::uint32_t>(dir / "commit_faults.npy", 1);
    ASSERT_TRUE(ptr && faults);
    ASSERT_EQ(ptr->size(), (p.shots * p.windows) + 1);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < p.decoded * p.windows; ++i) {
        const std::span<const std::uint32_t> expected(faults->data.data() + ptr->data[i],
                                                      ptr->data[i + 1] - ptr->data[i]);
        if (!std::ranges::equal(expected, p.commits[i]) && ++mismatches <= 5) {
            ADD_FAILURE() << std::format("commits of shot {} window {}: golden {} ours {}",
                                         i / p.windows, i % p.windows, expected.size(),
                                         p.commits[i].size());
        }
    }
    EXPECT_EQ(mismatches, 0U) << "commit_faults";
}

// Every array of the golden directory, by name; one this test cannot produce is a failure.
void compare_directory(const fs::path& dir, const Produced& p) {
    const std::set<std::string> required{
        "win_iterations", "win_legs",        "win_attempts",    "win_converged",
        "win_cap_hit",    "win_weight",      "win_committed_weight", "win_unexplained",
        "win_flagged",    "win_virtual",     "commit_ptr",      "commit_faults",
        "predicted_observables", "logical_failure", "flagged", "success",
        "iterations",     "legs",            "weight"};
    std::set<std::string> seen;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".npy") {
            continue;
        }
        const std::string name = entry.path().stem().string();
        seen.insert(name);
        if (name == "detectors" || name == "observables" || name == "commit_faults") {
            continue;
        }
        if (name == "commit_ptr") {
            compare_commits(dir, p);
        } else if (p.u32.contains(name)) {
            compare_array(entry.path(), p.u32.at(name), p);
        } else if (p.u8.contains(name)) {
            compare_array(entry.path(), p.u8.at(name), p);
        } else if (p.f64.contains(name)) {
            compare_array(entry.path(), p.f64.at(name), p);
        } else {
            ADD_FAILURE() << "golden array " << name << " is not compared";
        }
    }
    for (const std::string& name : required) {
        EXPECT_TRUE(seen.contains(name)) << "golden has no " << name << ".npy";
    }
}

template <class A, class Exec>
void decode_golden(const WindowPlan& plan, const Settings& settings,
                   std::span<const ExplicitGammaTable> tables, const io::NpyArray<Bit>& detectors,
                   const io::NpyArray<Bit>& truth, std::uint64_t first,
                   const std::function<Exec()>& executor, Produced& out) {
    using Relay = CpuRelayDecoder<A, Exec>;
    auto decoder = StreamDecoder<Relay>::create(
        plan, [&](const Shape& shape) -> std::expected<Relay, std::string> {
            auto backend = CpuBackend<A, Exec>::create(shape.graph(), shape.priors(), executor());
            if (!backend) {
                return std::unexpected(backend.error().detail);
            }
            auto relay = Relay::create(std::move(*backend), settings.min_sum, settings.relay,
                                       settings.relay.num_sets > 0 ? &tables[shape.index()]
                                                                   : nullptr);
            if (!relay) {
                return std::unexpected(relay.error().detail);
            }
            return std::move(*relay);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());
    const std::size_t k = out.windows;
    const std::size_t o = out.observables;
    for (std::size_t s = 0; s < out.decoded; ++s) {
        const auto summary = decode_shot(*decoder, detectors.row(s), first + s, [&](const Commit& c) {
            const std::size_t i = (s * k) + c.window;
            const WindowRecord& r = c.record;
            out.u32["win_iterations"][i] = r.iterations;
            out.u32["win_legs"][i] = r.legs;
            out.u32["win_unexplained"][i] = r.unexplained;
            out.u32["win_virtual"][i] = r.virtual_commits;
            out.u8["win_attempts"][i] = static_cast<std::uint8_t>(r.attempts);
            out.u8["win_converged"][i] = static_cast<std::uint8_t>(r.converged);
            out.u8["win_cap_hit"][i] = static_cast<std::uint8_t>(r.cap_hit);
            out.u8["win_flagged"][i] = static_cast<std::uint8_t>(r.flagged);
            out.f64["win_weight"][i] = r.weight;
            out.f64["win_committed_weight"][i] = r.committed_weight;
            out.commits[i].assign(c.faults.begin(), c.faults.end());
        });
        ASSERT_TRUE(summary) << "shot " << s << ": " << describe(summary.error());
        bool failure = false;
        for (std::size_t b = 0; b < o; ++b) {
            const Bit predicted = decoder->frame()[b];
            out.u8["predicted_observables"][(s * o) + b] = predicted;
            failure = failure || predicted != truth.row(s)[b];
        }
        out.u8["logical_failure"][s] = static_cast<std::uint8_t>(failure);
        out.u8["flagged"][s] = static_cast<std::uint8_t>(summary->flagged);
        out.u8["success"][s] = static_cast<std::uint8_t>(summary->success);
        out.u32["iterations"][s] = summary->iterations;
        out.u32["legs"][s] = summary->legs;
        out.f64["weight"][s] = summary->weight;
    }
}

template <class Exec>
void dispatch(const std::string& arithmetic, const WindowPlan& plan, const Settings& settings,
              std::span<const ExplicitGammaTable> tables, const io::NpyArray<Bit>& detectors,
              const io::NpyArray<Bit>& truth, std::uint64_t first,
              const std::function<Exec()>& executor, Produced& out) {
    if (arithmetic == Int4_2_8::name) {
        decode_golden<Int4_2_8>(plan, settings, tables, detectors, truth, first, executor, out);
    } else if (arithmetic == Int5_2_8::name) {
        decode_golden<Int5_2_8>(plan, settings, tables, detectors, truth, first, executor, out);
    } else if (arithmetic == Int6_2_8::name) {
        decode_golden<Int6_2_8>(plan, settings, tables, detectors, truth, first, executor, out);
    } else {
        ADD_FAILURE() << "no compiled fixed-point format " << arithmetic;
    }
}

// Reproduces one golden directory; returns the number of shots decoded.
std::size_t check_golden(const fs::path& dir, bool team) {
    std::ifstream in(dir / "manifest.json");
    const Json manifest = Json::parse(in);
    const fs::path artifact_dir = repo_root / manifest.at("artifact").at("path").get<std::string>();
    auto artifact = io::load_artifact(artifact_dir);
    EXPECT_TRUE(artifact) << (artifact ? "" : io::describe(artifact.error()));
    if (!artifact) {
        return 0;
    }
    EXPECT_EQ(io::sha256_file(artifact_dir / "manifest.json").value_or(""),
              manifest.at("artifact").at("manifest_sha256").get<std::string>())
        << "the golden was made from another artifact";
    const auto settings = settings_of(manifest);
    EXPECT_TRUE(settings) << (settings ? "" : settings.error());
    if (!settings) {
        return 0;
    }
    const ArtifactProblem source(*artifact);
    auto plan = WindowPlan::build(source.problem(), settings->window);
    EXPECT_TRUE(plan) << (plan ? "" : describe(plan.error()));
    if (!plan) {
        return 0;
    }
    EXPECT_EQ(plan->shapes().size(), manifest.at("num_shapes").get<std::size_t>());
    EXPECT_EQ(plan->num_positions(), manifest.at("num_positions").get<std::size_t>());

    const auto rows = manifest.at("gamma").at("rows").get<std::size_t>();
    std::vector<ExplicitGammaTable> tables;
    for (const Shape& shape : plan->shapes()) {
        const std::array<std::optional<std::size_t>, 2> extents{rows,
                                                                 std::size_t{shape.num_columns()}};
        auto table = io::read_npy<double>(
            dir / "gammas" / std::format("shape_{}.npy", shape.index()), 2, extents);
        EXPECT_TRUE(table) << (table ? "" : io::describe(table.error()));
        if (!table) {
            return 0;
        }
        auto gammas = ExplicitGammaTable::create(
            std::vector<double>(table->span().begin(), table->span().end()), rows,
            shape.num_columns());
        EXPECT_TRUE(gammas);
        if (!gammas) {
            return 0;
        }
        tables.push_back(std::move(*gammas));
    }
    auto detectors = io::read_npy<Bit>(dir / "detectors.npy", 2);
    auto truth = io::read_npy<Bit>(dir / "observables.npy", 2);
    EXPECT_TRUE(detectors && truth);
    if (!detectors || !truth) {
        return 0;
    }
    Produced produced;
    produced.size(detectors->rows(), plan->num_positions(), artifact->num_observables());
    produced.decoded = shots_to_decode(detectors->rows());
    const auto first = manifest.at("first").get<std::uint64_t>();
    if (team) {
        produced.decoded = std::min<std::size_t>(produced.decoded, 3);
        const std::function<Team()> make = [] { return Team(2); };
        dispatch(settings->arithmetic, *plan, *settings, tables, *detectors, *truth, first, make,
                 produced);
    } else {
        const std::function<Serial()> make = [] { return Serial{}; };
        dispatch(settings->arithmetic, *plan, *settings, tables, *detectors, *truth, first, make,
                 produced);
    }
    if (testing::Test::HasFatalFailure()) {
        return 0;
    }
    compare_directory(dir, produced);
    return produced.decoded;
}

std::vector<fs::path> fixture_goldens() {
    std::vector<fs::path> dirs;
    for (const char* fixture : {"bb18_choi", "bb18_choi_r9"}) {
        const fs::path root = fixture_root / fixture;
        if (!fs::exists(root)) {
            continue;
        }
        for (const auto& entry : fs::directory_iterator(root)) {
            if (entry.path().filename().string().starts_with("fixed_") &&
                fs::exists(entry.path() / "manifest.json")) {
                dirs.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(dirs);
    return dirs;
}

class FixedGolden : public testing::TestWithParam<fs::path> {};

TEST_P(FixedGolden, StreamDecoderReproducesTheEmulator) {
    EXPECT_GT(check_golden(GetParam(), false), 0U);
}

TEST_P(FixedGolden, TeamExecutorReproducesTheEmulator) {
    EXPECT_GT(check_golden(GetParam(), true), 0U);
}

std::string golden_name(const testing::TestParamInfo<fs::path>& info) {
    std::string name = info.param.filename().string();
    std::ranges::replace_if(
        name, [](char c) { return std::isalnum(static_cast<unsigned char>(c)) == 0; }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Fixture, FixedGolden, testing::ValuesIn(fixture_goldens()), golden_name);

TEST(FixedGoldenFixtures, ArePresent) { EXPECT_EQ(fixture_goldens().size(), 5U); }

TEST(FixedGoldenExternal, StreamDecoderReproducesEveryGolden) {
    const char* root = std::getenv("RTD_FIXED_GOLDEN_DIR");
    if (root == nullptr) {
        GTEST_SKIP() << "set RTD_FIXED_GOLDEN_DIR to run";
    }
    std::vector<fs::path> dirs;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_directory() && fs::exists(entry.path() / "manifest.json") &&
            fs::exists(entry.path() / "win_iterations.npy")) {
            dirs.push_back(entry.path());
        }
    }
    std::ranges::sort(dirs);
    ASSERT_FALSE(dirs.empty()) << "no fixed-point golden below " << root;
    for (const fs::path& dir : dirs) {
        SCOPED_TRACE(dir.string());
        const std::size_t decoded = check_golden(dir, false);
        std::cout << std::format("[ fixed    ] {}: {} shots\n", dir.string(), decoded);
        EXPECT_GT(decoded, 0U);
    }
}

} // namespace
