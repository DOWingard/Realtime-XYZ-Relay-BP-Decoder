// rtd_decode in sliding-window mode, run in process through its command line.
//
// Golden: every array of every window golden that the Python windowing reference wrote for the
// bb18 fixtures (relay_bp inside, explicit γ tables per shape) is reproduced by rtd_decode, name by
// name, with its dtype and shape; the window events it logs add up to the golden's records.
// Identity: a window at least as wide as the shot (W ≥ Rt) is the whole-shot problem, so a sliding
// run must reproduce the whole-shot goldens and rtd_decode's own whole-shot runs, including the
// uniform γ draws (window 0, attempt 0 of shot s uses stream s).
// Also: the rejections at startup, the shape and dtype of every output, recorded solutions against
// a recomputation from the plan's commit classes, committed corrections against the syndrome,
// team executors against serial ones, and the rate limit on window event lines.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/shots.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"

namespace {

using namespace rtd;
using json = nlohmann::json;
namespace fs = std::filesystem;
using test::CliRun;
using test::first_error;
using test::logged;
using test::read_all;
using test::read_array;
using test::read_json;
using test::run_cli;

// Golden manifests name their artifact relative to the repository root.
const fs::path repo_root = test::fixture_root.parent_path().parent_path();
const fs::path r9 = test::fixture_root / "bb18_choi_r9";

// Optimised builds decode every shot. Unoptimised and sanitizer builds, which run the decoder 5 to
// 20 times slower, decode subsets that reach the same code paths. RTD_WINDOW_GOLDEN_SHOTS=N
// overrides both: the first N shots, or every shot for N = 0.
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
constexpr bool every_shot = optimised && !sanitized;

std::optional<std::size_t> shot_limit() {
    if (const char* limit = std::getenv("RTD_WINDOW_GOLDEN_SHOTS"); limit != nullptr) {
        return static_cast<std::size_t>(std::stoull(limit));
    }
    return std::nullopt;
}

// Shots to decode out of `available`: all of them in optimised builds, `slow` otherwise.
std::size_t shots_for(std::size_t available, std::size_t slow) {
    if (const auto limit = shot_limit()) {
        return *limit == 0 ? available : std::min(available, *limit);
    }
    return every_shot ? available : std::min(available, slow);
}

std::vector<std::string> decode_args(const fs::path& artifact, const fs::path& shots,
                                     const fs::path& config, const fs::path& out) {
    return {"--artifact", artifact.string(), "--shots", shots.string(),
            "--config",   config.string(),   "--out",   out.string()};
}

fs::path write_spec(const fs::path& dir, const std::string& name, const json& spec) {
    const fs::path file = dir / name;
    std::ofstream(file) << spec.dump(2);
    return file;
}

json sliding_window(std::uint32_t width, std::uint32_t commit, std::uint32_t converge,
                    const std::string& boundary, const std::string& policy, std::uint32_t deferrals,
                    std::optional<std::uint32_t> cap = std::nullopt) {
    return {{"mode", "sliding"},          {"width", width},
            {"commit", commit},           {"converge_rounds", converge},
            {"boundary", boundary},       {"on_failure", policy},
            {"max_deferrals", deferrals}, {"iteration_cap", cap ? json(*cap) : json(nullptr)}};
}

// A Relay-BP spec with uniform γ draws, small enough for unoptimised builds.
json relay_spec(json window, unsigned team = 0) {
    return {{"version", 2},
            {"policy", "f32"},
            {"backend", "cpu"},
            {"layout", "row_major"},
            {"column_order", "wavefront"},
            {"block_rows", 64},
            {"executor",
             team == 0 ? json{{"type", "serial"}} : json{{"type", "team"}, {"threads", team}}},
            {"alpha", {{"rule", "constant"}, {"value", 1.0}}},
            {"gamma0", 0.125},
            {"pre_iter", 20},
            {"set_max_iter", 15},
            {"num_sets", 40},
            {"stopping", {{"rule", "after_n_converged"}, {"count", 3}}},
            {"gamma_source", {{"type", "uniform"}, {"seed", 7}, {"low", -0.24}, {"high", 0.66}}},
            {"window", std::move(window)}};
}

// ---- Raw .npy comparison ----------------------------------------------------------------------

// An .npy file's header and bytes, compared row by row (so doubles compare bitwise).
struct RawNpy {
    io::NpyHeader header;
    std::vector<char> bytes;

    [[nodiscard]] std::size_t row_bytes() const {
        std::size_t items = 1;
        for (std::size_t d = 1; d < header.shape.size(); ++d) {
            items *= header.shape[d];
        }
        return items * io::item_size(header.dtype);
    }
    [[nodiscard]] std::span<const char> rows(std::size_t first, std::size_t count) const {
        return std::span<const char>(bytes).subspan(header.data_offset + (first * row_bytes()),
                                                    count * row_bytes());
    }
};

std::optional<RawNpy> read_raw(const fs::path& file) {
    auto header = io::read_npy_header(file);
    EXPECT_TRUE(header) << file << ": " << (header ? "" : io::describe(header.error()));
    if (!header) {
        return std::nullopt;
    }
    std::ifstream in(file, std::ios::binary);
    RawNpy raw{.header = std::move(*header),
               .bytes = std::vector<char>(std::istreambuf_iterator<char>(in),
                                          std::istreambuf_iterator<char>())};
    return raw;
}

// Rows [first, first + count) of a golden array against our output of those `count` shots: the
// same dtype, the same shape past the first axis, and identical bytes.
void expect_rows_equal(const fs::path& golden_file, const fs::path& ours_file, std::size_t first,
                       std::size_t count) {
    const std::string name = golden_file.stem().string();
    ASSERT_TRUE(fs::exists(ours_file)) << "rtd_decode wrote no " << name;
    const auto golden = read_raw(golden_file);
    const auto ours = read_raw(ours_file);
    ASSERT_TRUE(golden && ours) << name;
    EXPECT_EQ(io::to_string(ours->header.dtype), io::to_string(golden->header.dtype)) << name;
    std::vector<std::size_t> shape = golden->header.shape;
    ASSERT_FALSE(shape.empty()) << name;
    ASSERT_LE(first + count, shape[0]) << name;
    shape[0] = count;
    ASSERT_EQ(ours->header.shape, shape) << name;
    const auto expected = golden->rows(first, count);
    const auto got = ours->rows(0, count);
    if (!std::ranges::equal(expected, got)) {
        const std::size_t row_bytes = golden->row_bytes();
        for (std::size_t r = 0; r < count; ++r) {
            if (!std::ranges::equal(expected.subspan(r * row_bytes, row_bytes),
                                    got.subspan(r * row_bytes, row_bytes))) {
                ADD_FAILURE() << std::format("{}: shot {} differs", name, first + r);
                break;
            }
        }
    }
}

// The committed faults of cells [first·K, (first + count)·K) against ours.
void expect_commits_equal(const fs::path& golden_dir, const fs::path& out, std::size_t first,
                          std::size_t count, std::size_t windows) {
    const auto golden_ptr = read_all<std::uint64_t>(golden_dir / "commit_ptr.npy");
    const auto golden_faults = read_all<std::uint32_t>(golden_dir / "commit_faults.npy");
    const auto ptr = read_all<std::uint64_t>(out / "commit_ptr.npy");
    const auto faults = read_all<std::uint32_t>(out / "commit_faults.npy");
    ASSERT_EQ(ptr.size(), (count * windows) + 1);
    ASSERT_EQ(ptr.back(), faults.size());
    std::size_t mismatches = 0;
    for (std::size_t c = 0; c < count * windows; ++c) {
        const std::size_t g = (first * windows) + c;
        const std::span<const std::uint32_t> expected(golden_faults.data() + golden_ptr[g],
                                                      golden_ptr[g + 1] - golden_ptr[g]);
        const std::span<const std::uint32_t> got(faults.data() + ptr[c], ptr[c + 1] - ptr[c]);
        if (!std::ranges::equal(expected, got) && ++mismatches <= 3) {
            ADD_FAILURE() << std::format(
                "commit_faults shot {} window {}: golden {} faults, ours {}", first + (c / windows),
                c % windows, expected.size(), got.size());
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

// Totals of the "(run total)" window event lines of a run, by event.
std::map<std::string, std::uint64_t> event_totals(const CliRun& run) {
    std::map<std::string, std::uint64_t> totals;
    for (const json& line : run.log) {
        if (line.at("context") == "window" && line.contains("total")) {
            totals[line.at("event").get<std::string>()] += line.at("total").get<std::uint64_t>();
        }
    }
    return totals;
}

// ---- Window goldens through rtd_decode --------------------------------------------------------

std::vector<fs::path> window_goldens() {
    std::vector<fs::path> dirs;
    for (const char* fixture : {"bb18_choi", "bb18_choi_r9"}) {
        for (const auto& entry : fs::directory_iterator(test::fixture_root / fixture)) {
            if (entry.path().filename().string().starts_with("window_") &&
                fs::exists(entry.path() / "manifest.json")) {
                dirs.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(dirs);
    return dirs;
}

// Contiguous shot ranges [first, first + count) of a golden to decode: every shot, or the first
// two plus the first two with a window that did not converge, that was deferred, and that an
// earlier final window decided.
std::vector<std::pair<std::size_t, std::size_t>> golden_ranges(const fs::path& dir,
                                                               std::size_t shots) {
    if (every_shot || shot_limit()) {
        return {{0, shots_for(shots, shots)}};
    }
    const auto converged = io::read_npy<std::uint8_t>(dir / "win_converged.npy", 2);
    const auto attempts = io::read_npy<std::uint8_t>(dir / "win_attempts.npy", 2);
    EXPECT_TRUE(converged && attempts);
    std::set<std::size_t> picked{0, 1};
    if (converged && attempts) {
        const auto kinds = std::to_array<bool (*)(std::uint8_t, std::uint8_t)>({
            [](std::uint8_t c, std::uint8_t /*a*/) { return c == 0; },
            [](std::uint8_t /*c*/, std::uint8_t a) { return a > 1; },
            [](std::uint8_t /*c*/, std::uint8_t a) { return a == 0; },
        });
        for (const auto kind : kinds) {
            std::size_t found = 0;
            for (std::size_t s = 0; s < shots && found < 2; ++s) {
                const auto c = converged->row(s);
                const auto a = attempts->row(s);
                for (std::size_t w = 0; w < c.size(); ++w) {
                    if (kind(c[w], a[w])) {
                        picked.insert(s);
                        ++found;
                        break;
                    }
                }
            }
        }
    }
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (const std::size_t s : picked) {
        if (!ranges.empty() && ranges.back().first + ranges.back().second == s) {
            ++ranges.back().second;
        } else {
            ranges.emplace_back(s, 1);
        }
    }
    return ranges;
}

// Window events a run of these golden rows must report: every non-converged decoded window (as
// deferral_limit under defer, not_converged otherwise) and every converged window that needed a
// deferral.
std::map<std::string, std::uint64_t> expected_events(const fs::path& dir, const json& window,
                                                     std::size_t first, std::size_t count) {
    const auto converged = read_all<std::uint8_t>(dir / "win_converged.npy");
    const auto attempts = read_all<std::uint8_t>(dir / "win_attempts.npy");
    const std::size_t windows =
        converged.size() / read_all<std::uint8_t>(dir / "flagged.npy").size();
    std::map<std::string, std::uint64_t> expected;
    const bool defer = window.at("on_failure") == "defer";
    for (std::size_t q = first * windows; q < (first + count) * windows; ++q) {
        if (attempts[q] == 0) {
            continue;
        }
        if (converged[q] == 0) {
            ++expected[defer ? "deferral_limit" : "not_converged"];
        } else if (attempts[q] > 1) {
            ++expected["deferred"];
        }
    }
    return expected;
}

class SlidingGolden : public testing::TestWithParam<fs::path> {};

TEST_P(SlidingGolden, RtdDecodeReproducesEveryArray) {
    const fs::path& dir = GetParam();
    const json manifest = read_json(dir / "manifest.json");
    const fs::path artifact = repo_root / manifest.at("artifact").at("path").get<std::string>();
    const auto shots = manifest.at("count").get<std::size_t>();
    const auto windows = manifest.at("num_positions").get<std::size_t>();
    const fs::path root = test::scratch_dir(std::format("rtd_sliding_golden_{}_{}",
                                                        dir.parent_path().filename().string(),
                                                        dir.filename().string()));

    // The golden arrays this test knows; anything else the reference wrote is a failure.
    const std::set<std::string> compared{
        "win_iterations",        "win_legs",        "win_attempts",
        "win_converged",         "win_weight",      "win_committed_weight",
        "win_unexplained",       "win_flagged",     "win_virtual",
        "predicted_observables", "logical_failure", "flagged"};
    const std::set<std::string> inputs{"detectors", "observables"};
    std::set<std::string> seen;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".npy") {
            const std::string name = entry.path().stem().string();
            seen.insert(name);
            EXPECT_TRUE(compared.contains(name) || inputs.contains(name) || name == "commit_ptr" ||
                        name == "commit_faults")
                << "golden array " << name << " is not compared";
        }
    }
    for (const std::string& name : compared) {
        EXPECT_TRUE(seen.contains(name)) << "golden has no " << name << ".npy";
    }

    std::size_t decoded = 0;
    for (const auto& [first, count] : golden_ranges(dir, shots)) {
        const fs::path out = root / std::format("shots_{}_{}", first, count);
        std::vector<std::string> args = decode_args(artifact, dir, dir / "spec.json", out);
        args.insert(args.end(), {"--first", std::to_string(first), "--count", std::to_string(count),
                                 "--save-commits", "--workers", count > 1 ? "2" : "1"});
        const CliRun run = run_cli(args);
        ASSERT_EQ(run.code, 0) << dir << ": " << first_error(run).dump();
        for (const std::string& name : compared) {
            expect_rows_equal(dir / (name + ".npy"), out / (name + ".npy"), first, count);
        }
        expect_commits_equal(dir, out, first, count, windows);
        // Every decoded window that did not converge, or converged only after a deferral, is
        // one event; the totals do not depend on the rate limit.
        EXPECT_EQ(event_totals(run), expected_events(dir, manifest.at("window"), first, count));
        const json record = read_json(out / "run.json");
        EXPECT_EQ(record.at("plan").at("positions"), manifest.at("num_positions"));
        EXPECT_EQ(record.at("plan").at("shapes").size(), manifest.at("num_shapes"));
        EXPECT_EQ(record.at("window"), manifest.at("window"));
        decoded += count;
    }
    EXPECT_GT(decoded, 0U);
    fs::remove_all(root);
}

std::string golden_name(const testing::TestParamInfo<fs::path>& info) {
    std::string name =
        info.param.parent_path().filename().string() + "_" + info.param.filename().string();
    std::ranges::replace_if(
        name, [](char c) { return std::isalnum(static_cast<unsigned char>(c)) == 0; }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Fixture, SlidingGolden, testing::ValuesIn(window_goldens()), golden_name);

TEST(SlidingGoldenFixtures, AllFifteenArePresent) { EXPECT_EQ(window_goldens().size(), 15U); }

// ---- Identity: W ≥ Rt is the whole-shot decode ------------------------------------------------

// The version-2 spec that decodes like a whole-shot golden, with the given window object.
json golden_spec(const io::Golden& golden, json window) {
    json alpha;
    if (const auto* constant = std::get_if<ConstantAlpha>(&golden.min_sum.alpha)) {
        alpha = {{"rule", "constant"}, {"value", constant->value}};
    } else {
        alpha = {{"rule", "adaptive"},
                 {"scaling", std::get<AdaptiveAlpha>(golden.min_sum.alpha).scaling}};
    }
    json stopping;
    if (std::holds_alternative<AfterLeg0>(golden.relay.stopping)) {
        stopping = {{"rule", "after_leg0"}};
    } else if (const auto* n = std::get_if<AfterNConverged>(&golden.relay.stopping)) {
        stopping = {{"rule", "after_n_converged"}, {"count", n->count}};
    } else {
        stopping = {{"rule", "all_legs"}};
    }
    const json gammas = golden.gammas ? json{{"type", "explicit"},
                                             {"path", (golden.directory / "gammas.npy").string()}}
                                      : json{{"type", "none"}};
    return {{"version", 2},
            {"policy", golden.float_type},
            {"backend", "cpu"},
            {"layout", "row_major"},
            {"column_order", "wavefront"},
            {"block_rows", 64},
            {"executor", {{"type", "serial"}}},
            {"alpha", alpha},
            {"gamma0", golden.min_sum.gamma0 ? json(*golden.min_sum.gamma0) : json(nullptr)},
            {"pre_iter", golden.relay.pre_iter},
            {"set_max_iter", golden.relay.set_max_iter},
            {"num_sets", golden.relay.num_sets},
            {"stopping", stopping},
            {"gamma_source", gammas},
            {"window", std::move(window)}};
}

std::vector<std::uint64_t> bits_of(const std::vector<double>& values) {
    std::vector<std::uint64_t> bits;
    bits.reserve(values.size());
    for (const double v : values) {
        bits.push_back(std::bit_cast<std::uint64_t>(v));
    }
    return bits;
}

// Per shot, the columns of a dense [S, n] correction.
std::vector<std::vector<std::uint32_t>> supports_of(const std::vector<std::uint8_t>& dense,
                                                    std::size_t shots, std::size_t n) {
    std::vector<std::vector<std::uint32_t>> supports(shots);
    for (std::size_t s = 0; s < shots; ++s) {
        for (std::size_t j = 0; j < n; ++j) {
            if (dense[(s * n) + j] != 0) {
                supports[s].push_back(static_cast<std::uint32_t>(j));
            }
        }
    }
    return supports;
}

// A sliding run with one window against the whole-shot run of the same spec: the same decode.
void expect_same_decode(const fs::path& whole, const fs::path& sliding, std::size_t shots,
                        std::size_t n, const std::string& policy) {
    for (const char* name :
         {"success.npy", "predicted_observables.npy", "logical_failure.npy", "decodings.npy"}) {
        EXPECT_EQ(read_all<std::uint8_t>(whole / name), read_all<std::uint8_t>(sliding / name))
            << name;
    }
    for (const char* name : {"iterations.npy", "legs.npy"}) {
        EXPECT_EQ(read_all<std::uint32_t>(whole / name), read_all<std::uint32_t>(sliding / name))
            << name;
    }
    const auto success = read_all<std::uint8_t>(whole / "success.npy");
    EXPECT_EQ(read_all<std::uint8_t>(sliding / "win_converged.npy"), success);
    // The inner decoder's weight is the whole-shot weight, +∞ included.
    EXPECT_EQ(bits_of(read_all<double>(sliding / "win_weight.npy")),
              bits_of(read_all<double>(whole / "weight.npy")));
    // The per-shot weight is Σ λ over what the windows committed: with one window, its
    // committed weight.
    EXPECT_EQ(bits_of(read_all<double>(sliding / "weight.npy")),
              bits_of(read_all<double>(sliding / "win_committed_weight.npy")));
    EXPECT_EQ(read_all<std::int32_t>(sliding / "best_leg.npy"),
              std::vector<std::int32_t>(shots, -1));
    std::vector<std::uint8_t> flags(shots, 0);
    for (std::size_t s = 0; s < shots; ++s) {
        flags[s] = success[s] == 0 && policy != "commit_anyway" ? 1 : 0;
    }
    EXPECT_EQ(read_all<std::uint8_t>(sliding / "flagged.npy"), flags);
    // The single window commits the whole correction.
    const auto ptr = read_all<std::uint64_t>(sliding / "commit_ptr.npy");
    const auto faults = read_all<std::uint32_t>(sliding / "commit_faults.npy");
    const auto supports = supports_of(read_all<std::uint8_t>(whole / "decodings.npy"), shots, n);
    ASSERT_EQ(ptr.size(), shots + 1);
    for (std::size_t s = 0; s < shots; ++s) {
        EXPECT_TRUE(
            std::ranges::equal(std::span(faults).subspan(ptr[s], ptr[s + 1] - ptr[s]), supports[s]))
            << "shot " << s;
    }
}

class SlidingIdentity : public testing::TestWithParam<std::string> {};

TEST_P(SlidingIdentity, OneWindowReproducesTheWholeShotGolden) {
    const fs::path artifact_dir = test::fixture_root / "bb18_choi" / "artifact";
    auto artifact = io::load_artifact(artifact_dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto golden = io::load_golden(test::fixture_root / "bb18_choi" / GetParam(),
                                  artifact->num_detectors(), artifact->num_columns());
    ASSERT_TRUE(golden) << io::describe(golden.error());
    const fs::path root = test::scratch_dir("rtd_sliding_identity_" + GetParam());
    const fs::path shots = test::write_shots(*artifact, *golden, root / "shots");
    const std::size_t count = shots_for(golden->count(), 12);
    const std::size_t n = artifact->num_columns();

    const auto decode = [&](const std::string& name, const json& window) {
        const fs::path out = root / name;
        std::vector<std::string> args =
            decode_args(artifact_dir, shots,
                        write_spec(root, name + ".json", golden_spec(*golden, window)), out);
        // The shots directory written from the golden carries no checksums.
        args.insert(args.end(),
                    {"--count", std::to_string(count), "--save-decodings", "--no-verify"});
        if (window.at("mode") == "sliding") {
            args.emplace_back("--save-commits");
        }
        const CliRun run = run_cli(args);
        EXPECT_EQ(run.code, 0) << name << ": " << first_error(run).dump();
        return out;
    };
    const fs::path whole = decode("whole", {{"mode", "whole_shot"}});

    // The whole-shot run against the golden.
    const auto decodings = read_all<std::uint8_t>(whole / "decodings.npy");
    ASSERT_EQ(decodings.size(), count * n);
    EXPECT_TRUE(std::ranges::equal(std::span(decodings), golden->decoding.span().first(count * n)));
    EXPECT_TRUE(std::ranges::equal(read_all<std::uint8_t>(whole / "success.npy"),
                                   golden->success.span().first(count)));
    EXPECT_EQ(bits_of(read_all<double>(whole / "weight.npy")),
              bits_of({golden->weight.span().begin(),
                       golden->weight.span().begin() + static_cast<std::ptrdiff_t>(count)}));

    // Rt = 4. W ≥ Rt makes one final window, whatever C, C′ and the policy.
    const std::vector<std::pair<std::string, json>> windows{
        {"w4_c1", sliding_window(4, 1, 4, "exact", "commit_anyway", 0)},
        {"w10_c3", sliding_window(10, 3, 5, "exact", "flag", 0)},
        {"w4_c2_defer", sliding_window(4, 2, 3, "exact", "defer", 2)},
    };
    for (const auto& [name, window] : windows) {
        SCOPED_TRACE(name);
        const fs::path sliding = decode(name, window);
        expect_same_decode(whole, sliding, count, n, window.at("on_failure"));
        const json record = read_json(sliding / "run.json");
        EXPECT_EQ(record.at("plan").at("positions"), 1);
        EXPECT_EQ(record.at("plan").at("shapes").size(), 1U);
        test::read_array<std::uint32_t>(sliding / "win_iterations.npy", {count, 1});
    }
    fs::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(Bb18, SlidingIdentity,
                         testing::Values("min_sum_f32", "min_sum_f64", "mem_bp_f32",
                                         "min_sum_adaptive_f32", "relay_f32", "relay_f64",
                                         "relay_all_f32", "relay_preiter_f32",
                                         "single_columns_f32"));

// Uniform γ draws: window 0, attempt 0 of shot s draws from stream s, as the whole-shot decode of
// shot s does, so W ≥ Rt matches whole-shot runs for any shot range, worker count and executor.
TEST(SlidingIdentityUniform, OneWindowMatchesWholeShotRuns) {
    const fs::path root = test::scratch_dir("rtd_sliding_identity_uniform");
    const std::size_t count = shots_for(40, 8);
    const fs::path artifact = r9 / "artifact";
    const auto decode = [&](const std::string& name, const json& spec, unsigned workers) {
        const fs::path out = root / name;
        std::vector<std::string> args =
            decode_args(artifact, r9 / "shots", write_spec(root, name + ".json", spec), out);
        args.insert(args.end(), {"--first", "7", "--count", std::to_string(count),
                                 "--save-decodings", "--workers", std::to_string(workers)});
        if (spec.at("window").at("mode") == "sliding") {
            args.emplace_back("--save-commits");
        }
        const CliRun run = run_cli(args);
        EXPECT_EQ(run.code, 0) << name << ": " << first_error(run).dump();
        return out;
    };
    const fs::path whole = decode("whole", relay_spec({{"mode", "whole_shot"}}), 1);
    // Rt = 10.
    const fs::path serial =
        decode("serial", relay_spec(sliding_window(10, 1, 10, "exact", "commit_anyway", 0)), 2);
    const fs::path team =
        decode("team", relay_spec(sliding_window(12, 4, 7, "exact", "flag", 0), 2), 1);
    auto artifact_loaded = io::load_artifact(artifact);
    ASSERT_TRUE(artifact_loaded);
    const std::size_t n = artifact_loaded->num_columns();
    expect_same_decode(whole, serial, count, n, "commit_anyway");
    expect_same_decode(whole, team, count, n, "flag");
    fs::remove_all(root);
}

// ---- Rejections at startup ---------------------------------------------------------------------

struct Rejection {
    int code;
    std::string context;
    std::string text; // part of the logged error
};

void expect_rejected(const std::vector<std::string>& args, const fs::path& out,
                     const Rejection& expected) {
    const CliRun run = run_cli(args);
    EXPECT_EQ(run.code, expected.code) << expected.text;
    const json error = first_error(run);
    ASSERT_FALSE(error.is_null()) << expected.text;
    EXPECT_EQ(error.at("context"), expected.context);
    const std::string text = error.at("error");
    EXPECT_NE(text.find(expected.text), std::string::npos) << text;
    EXPECT_FALSE(logged(run, "decoding started"));
    EXPECT_FALSE(fs::exists(out)) << "an output directory was created";
}

TEST(SlidingRejections, APlanThatCannotBeBuiltStopsTheRun) {
    const fs::path root = test::scratch_dir("rtd_sliding_reject_plan");
    const fs::path out = root / "out";
    // Uniform boundary on Rt = 10 with W = 8, C = 2: window 1 spans rounds 3 … 10 and touches the
    // readout, so no bulk window exists to copy.
    const fs::path spec =
        write_spec(root, "spec.json", relay_spec(sliding_window(8, 2, 8, "uniform", "flag", 0)));
    const CliRun run = run_cli(decode_args(r9 / "artifact", r9 / "shots", spec, out));
    EXPECT_EQ(run.code, 2);
    const json error = first_error(run);
    ASSERT_FALSE(error.is_null());
    EXPECT_EQ(error.at("context"), "plan");
    EXPECT_EQ(error.at("message"), "window plan rejected");
    EXPECT_EQ(error.at("code"), "no_bulk_window");
    EXPECT_FALSE(logged(run, "shots loaded"));
    EXPECT_FALSE(fs::exists(out));
    fs::remove_all(root);
}

TEST(SlidingRejections, GammaTablesMustFitThePlan) {
    const fs::path root = test::scratch_dir("rtd_sliding_reject_gamma");
    const fs::path out = root / "out";
    const fs::path artifact = r9 / "artifact";
    const auto args = [&](const fs::path& spec) {
        return decode_args(artifact, r9 / "shots", spec, out);
    };

    // One whole-problem table, but the plan has several shapes.
    json spec = relay_spec(sliding_window(4, 2, 4, "exact", "flag", 0));
    spec["gamma_source"] = {{"type", "explicit"}, {"path", "gammas.npy"}};
    expect_rejected(args(write_spec(root, "single.json", spec)), out,
                    {.code = 2, .context = "gamma", .text = "explicit_shapes"});

    // One table per shape, copied from a golden, then broken.
    const fs::path golden = r9 / "window_4_2_exact_flag";
    fs::copy(golden / "gammas", root / "gammas");
    fs::copy_file(golden / "spec.json", root / "shapes.json");
    fs::remove(root / "gammas" / "shape_1.npy");
    expect_rejected(args(root / "shapes.json"), out,
                    {.code = 3, .context = "gamma", .text = "shape_1.npy"});

    // A table of the wrong width.
    const std::vector<double> narrow(std::size_t{3} * 5, 0.1);
    const std::array<std::size_t, 2> narrow_shape{3, 5};
    ASSERT_TRUE(io::write_npy<double>(root / "gammas" / "shape_1.npy", narrow, narrow_shape));
    expect_rejected(args(root / "shapes.json"), out,
                    {.code = 3, .context = "gamma", .text = "shape_1.npy"});

    // One table with one window, but not n columns wide.
    spec = relay_spec(sliding_window(10, 1, 10, "exact", "flag", 0));
    spec["gamma_source"] = {{"type", "explicit"},
                            {"path", (root / "gammas" / "shape_1.npy").string()}};
    expect_rejected(args(write_spec(root, "narrow.json", spec)), out,
                    {.code = 3, .context = "gamma", .text = "shape_1.npy"});
    fs::remove_all(root);
}

// ---- Outputs, recording, commits ----------------------------------------------------------------

// Everything a sliding run can write, on 16 shots of bb18 R = 9 under the exact boundary.
struct RecordedRun {
    fs::path root;
    fs::path out;
    fs::path plain; // the same run without any optional output
    fs::path config;
    std::size_t shots = 0;
    std::size_t windows = 0;
    std::uint32_t slots = 3;
};

RecordedRun recorded_run(const std::string& name, const json& window) {
    RecordedRun r;
    r.root = test::scratch_dir(name);
    r.out = r.root / "out";
    r.plain = r.root / "plain";
    r.shots = shots_for(16, 6);
    r.config = write_spec(r.root, "spec.json", relay_spec(window));
    std::vector<std::string> args = decode_args(r9 / "artifact", r9 / "shots", r.config, r.out);
    args.insert(args.end(),
                {"--first", "20", "--count", std::to_string(r.shots), "--record-solutions",
                 std::to_string(r.slots), "--save-solution-supports", "--save-commits",
                 "--save-decodings", "--workers", "2"});
    const CliRun run = run_cli(args);
    EXPECT_EQ(run.code, 0) << first_error(run).dump();
    std::vector<std::string> plain = decode_args(r9 / "artifact", r9 / "shots", r.config, r.plain);
    plain.insert(plain.end(), {"--first", "20", "--count", std::to_string(r.shots)});
    const CliRun plain_run = run_cli(plain);
    EXPECT_EQ(plain_run.code, 0) << first_error(plain_run).dump();
    r.windows = read_json(r.out / "run.json").at("plan").at("positions").get<std::size_t>();
    return r;
}

TEST(SlidingOutputs, EveryArrayHasItsShapeAndDtype) {
    const RecordedRun r =
        recorded_run("rtd_sliding_outputs", sliding_window(4, 2, 4, "exact", "flag", 0));
    auto artifact = io::load_artifact(r9 / "artifact");
    ASSERT_TRUE(artifact);
    const std::size_t s = r.shots;
    const std::size_t k = r.windows;
    const std::size_t n = r.slots;
    EXPECT_GT(k, 1U);

    std::set<std::string> files;
    for (const auto& entry : fs::directory_iterator(r.out)) {
        files.insert(entry.path().filename().string());
    }
    const std::set<std::string> expected{"success.npy",
                                         "iterations.npy",
                                         "legs.npy",
                                         "best_leg.npy",
                                         "weight.npy",
                                         "decode_ns.npy",
                                         "predicted_observables.npy",
                                         "logical_failure.npy",
                                         "decodings.npy",
                                         "win_iterations.npy",
                                         "win_legs.npy",
                                         "win_attempts.npy",
                                         "win_converged.npy",
                                         "win_cap_hit.npy",
                                         "win_weight.npy",
                                         "win_committed_weight.npy",
                                         "win_unexplained.npy",
                                         "win_flagged.npy",
                                         "win_virtual.npy",
                                         "win_decode_ns.npy",
                                         "flagged.npy",
                                         "commit_ptr.npy",
                                         "commit_faults.npy",
                                         "sol_count.npy",
                                         "sol_leg.npy",
                                         "sol_iterations.npy",
                                         "sol_weight.npy",
                                         "sol_class.npy",
                                         "sol_hash.npy",
                                         "sol_size.npy",
                                         "returned_class.npy",
                                         "solsup_ptr.npy",
                                         "solsup_idx.npy",
                                         "run.json"};
    EXPECT_EQ(files, expected);

    const fs::path& o = r.out;
    read_array<std::uint8_t>(o / "success.npy", {s});
    read_array<std::uint32_t>(o / "iterations.npy", {s});
    read_array<std::uint32_t>(o / "legs.npy", {s});
    read_array<std::int32_t>(o / "best_leg.npy", {s});
    read_array<double>(o / "weight.npy", {s});
    read_array<std::uint64_t>(o / "decode_ns.npy", {s});
    read_array<std::uint8_t>(o / "predicted_observables.npy", {s, artifact->num_observables()});
    read_array<std::uint8_t>(o / "logical_failure.npy", {s});
    read_array<std::uint8_t>(o / "decodings.npy", {s, artifact->num_columns()});
    for (const char* name : {"win_iterations.npy", "win_legs.npy", "win_unexplained.npy",
                             "win_virtual.npy", "sol_count.npy"}) {
        read_array<std::uint32_t>(o / name, {s, k});
    }
    for (const char* name :
         {"win_attempts.npy", "win_converged.npy", "win_cap_hit.npy", "win_flagged.npy"}) {
        read_array<std::uint8_t>(o / name, {s, k});
    }
    read_array<double>(o / "win_weight.npy", {s, k});
    read_array<double>(o / "win_committed_weight.npy", {s, k});
    read_array<std::uint64_t>(o / "win_decode_ns.npy", {s, k});
    read_array<std::uint64_t>(o / "returned_class.npy", {s, k});
    read_array<std::uint8_t>(o / "flagged.npy", {s});
    const auto commit_ptr = read_array<std::uint64_t>(o / "commit_ptr.npy", {(s * k) + 1});
    ASSERT_FALSE(commit_ptr.empty());
    read_array<std::uint32_t>(o / "commit_faults.npy", {commit_ptr.back()});
    for (const char* name : {"sol_leg.npy", "sol_iterations.npy", "sol_size.npy"}) {
        read_array<std::uint32_t>(o / name, {s, k, n});
    }
    read_array<double>(o / "sol_weight.npy", {s, k, n});
    read_array<std::uint64_t>(o / "sol_class.npy", {s, k, n});
    read_array<std::uint64_t>(o / "sol_hash.npy", {s, k, n});
    const auto solsup_ptr = read_array<std::uint64_t>(o / "solsup_ptr.npy", {(s * k * n) + 1});
    ASSERT_FALSE(solsup_ptr.empty());
    read_array<std::uint32_t>(o / "solsup_idx.npy", {solsup_ptr.back()});

    // Nothing optional changes what is decoded.
    for (const char* name :
         {"success.npy", "predicted_observables.npy", "logical_failure.npy", "flagged.npy",
          "win_attempts.npy", "win_converged.npy", "win_flagged.npy", "win_cap_hit.npy"}) {
        EXPECT_EQ(read_all<std::uint8_t>(r.plain / name), read_all<std::uint8_t>(o / name)) << name;
    }
    for (const char* name : {"iterations.npy", "legs.npy", "win_iterations.npy", "win_legs.npy",
                             "win_unexplained.npy", "win_virtual.npy"}) {
        EXPECT_EQ(read_all<std::uint32_t>(r.plain / name), read_all<std::uint32_t>(o / name))
            << name;
    }
    for (const char* name : {"weight.npy", "win_weight.npy", "win_committed_weight.npy"}) {
        EXPECT_EQ(bits_of(read_all<double>(r.plain / name)), bits_of(read_all<double>(o / name)))
            << name;
    }
    EXPECT_FALSE(fs::exists(r.plain / "commit_ptr.npy"));
    EXPECT_FALSE(fs::exists(r.plain / "sol_count.npy"));
    EXPECT_FALSE(fs::exists(r.plain / "decodings.npy"));

    // run.json: the window, the plan, the windowed summary.
    const json record = read_json(o / "run.json");
    EXPECT_EQ(record.at("window"), read_json(r.config).at("window"));
    const json& plan = record.at("plan");
    EXPECT_EQ(plan.at("positions"), k);
    EXPECT_EQ(plan.at("rounds_total"), 10);
    EXPECT_EQ(plan.at("detectors_per_round"), 18);
    EXPECT_GE(plan.at("shapes").size(), 2U);
    EXPECT_EQ(plan.at("placements").size(), plan.at("schedule_length"));
    const json& windows = record.at("summary").at("windows");
    const auto converged = read_all<std::uint8_t>(o / "win_converged.npy");
    const auto attempts = read_all<std::uint8_t>(o / "win_attempts.npy");
    std::size_t decoded = 0;
    std::size_t converged_count = 0;
    for (std::size_t q = 0; q < converged.size(); ++q) {
        decoded += attempts[q] > 0 ? 1U : 0U;
        converged_count += attempts[q] > 0 && converged[q] != 0 ? 1U : 0U;
    }
    EXPECT_EQ(windows.at("positions"), k);
    EXPECT_EQ(windows.at("decoded"), decoded);
    EXPECT_EQ(windows.at("converged").at("count"), converged_count);
    EXPECT_EQ(windows.at("converged").at("of"), decoded);
    const auto flagged = read_all<std::uint8_t>(o / "flagged.npy");
    EXPECT_EQ(windows.at("flagged_shots").at("count"), std::ranges::count(flagged, 1));
    for (const char* key : {"iterations", "legs", "max_window_iterations_per_shot",
                            "decode_time_us", "cap_hit", "deferred", "unexplained_total"}) {
        EXPECT_TRUE(windows.contains(key)) << key;
    }
    fs::remove_all(r.root);
}

// Column j of A as a k-bit mask.
std::vector<std::uint64_t> observable_masks(const io::Artifact& artifact) {
    std::vector<std::uint64_t> masks(artifact.num_columns(), 0);
    for (index_t j = 0; j < artifact.num_columns(); ++j) {
        for (const index_t o : artifact.observables.column(j)) {
            masks[j] ^= std::uint64_t{1} << o;
        }
    }
    return masks;
}

// The recorded solutions of one window against the plan: supports in the committed attempt's
// shape, classes by its commit classes, hashes and sizes by definition, and the returned class
// equal to the class of the best solution when every solution was recorded.
void check_window_solutions(const window::Shape& shape, std::size_t cell, std::uint32_t slots,
                            const std::vector<std::uint32_t>& count,
                            const std::vector<std::uint64_t>& solsup_ptr,
                            const std::vector<std::uint32_t>& solsup_idx,
                            const std::map<std::string, std::vector<std::uint64_t>>& u64,
                            const std::vector<std::uint32_t>& size,
                            const std::vector<double>& weight, bool converged) {
    const std::span<const std::uint64_t> commit_class = shape.commit_class();
    const std::uint32_t stored = std::min(count[cell], slots);
    std::optional<std::size_t> best;
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        const std::size_t q = (cell * slots) + slot;
        const std::span<const std::uint32_t> support(solsup_idx.data() + solsup_ptr[q],
                                                     solsup_ptr[q + 1] - solsup_ptr[q]);
        if (slot >= stored) {
            EXPECT_TRUE(support.empty());
            EXPECT_EQ(weight[q], std::numeric_limits<double>::infinity());
            EXPECT_EQ(u64.at("class")[q], 0U);
            continue;
        }
        EXPECT_EQ(std::ranges::adjacent_find(support, std::ranges::greater_equal{}), support.end())
            << "support not strictly ascending";
        EXPECT_TRUE(
            std::ranges::all_of(support, [&](std::uint32_t l) { return l < shape.num_columns(); }));
        EXPECT_EQ(u64.at("class")[q], solution_class(commit_class, support));
        EXPECT_EQ(u64.at("hash")[q], solution_hash(support));
        EXPECT_EQ(size[q], support.size());
        if (!best || weight[q] < weight[(cell * slots) + *best]) {
            best = slot;
        }
    }
    if (converged && count[cell] <= slots) {
        ASSERT_TRUE(best);
        EXPECT_EQ(u64.at("returned")[cell], u64.at("class")[(cell * slots) + *best]);
    }
}

TEST(SlidingOutputs, RecordedSolutionsAndCommitsFollowThePlan) {
    const RecordedRun r =
        recorded_run("rtd_sliding_recording", sliding_window(4, 2, 3, "exact", "flag", 0));
    auto artifact = io::load_artifact(r9 / "artifact");
    ASSERT_TRUE(artifact);
    auto spec = harness::load_spec(r.config);
    ASSERT_TRUE(spec);
    const window::ArtifactProblem problem(*artifact);
    auto plan = window::WindowPlan::build(problem.problem(), spec->window.sliding, spec->graph);
    ASSERT_TRUE(plan) << window::describe(plan.error());
    ASSERT_EQ(plan->num_positions(), r.windows);
    auto shots = io::load_shots(r9 / "shots", *artifact, false);
    ASSERT_TRUE(shots);

    const fs::path& out = r.out;
    const auto count = read_all<std::uint32_t>(out / "sol_count.npy");
    const auto size = read_all<std::uint32_t>(out / "sol_size.npy");
    const auto weight = read_all<double>(out / "sol_weight.npy");
    const std::map<std::string, std::vector<std::uint64_t>> u64{
        {"class", read_all<std::uint64_t>(out / "sol_class.npy")},
        {"hash", read_all<std::uint64_t>(out / "sol_hash.npy")},
        {"returned", read_all<std::uint64_t>(out / "returned_class.npy")}};
    const auto solsup_ptr = read_all<std::uint64_t>(out / "solsup_ptr.npy");
    const auto solsup_idx = read_all<std::uint32_t>(out / "solsup_idx.npy");
    const auto attempts = read_all<std::uint8_t>(out / "win_attempts.npy");
    const auto converged = read_all<std::uint8_t>(out / "win_converged.npy");
    const auto win_flagged = read_all<std::uint8_t>(out / "win_flagged.npy");
    const auto commit_ptr = read_all<std::uint64_t>(out / "commit_ptr.npy");
    const auto commit_faults = read_all<std::uint32_t>(out / "commit_faults.npy");
    const auto decodings = read_all<std::uint8_t>(out / "decodings.npy");
    const auto predicted = read_all<std::uint8_t>(out / "predicted_observables.npy");
    const auto success = read_all<std::uint8_t>(out / "success.npy");
    const auto flagged = read_all<std::uint8_t>(out / "flagged.npy");
    const auto masks = observable_masks(*artifact);
    const std::size_t n = artifact->num_columns();
    const index_t k = artifact->num_observables();
    std::size_t recorded = 0;

    for (std::size_t s = 0; s < r.shots; ++s) {
        SCOPED_TRACE(std::format("shot {}", s));
        std::uint64_t frame = 0;
        std::vector<std::uint8_t> committed(n, 0);
        bool any_flag = false;
        for (std::size_t w = 0; w < r.windows; ++w) {
            const std::size_t cell = (s * r.windows) + w;
            const std::span<const std::uint32_t> faults(commit_faults.data() + commit_ptr[cell],
                                                        commit_ptr[cell + 1] - commit_ptr[cell]);
            std::uint64_t delta = 0;
            for (const std::uint32_t j : faults) {
                delta ^= masks[j];
                committed[j] ^= std::uint8_t{1};
            }
            // Exact windows commit single faults with A's classes: the returned class is the
            // window's change of the frame.
            EXPECT_EQ(u64.at("returned")[cell], delta);
            frame ^= delta;
            any_flag = any_flag || win_flagged[cell] != 0;
            EXPECT_EQ(win_flagged[cell], attempts[cell] > 0 && converged[cell] == 0 ? 1 : 0);
            if (attempts[cell] == 0) {
                EXPECT_EQ(count[cell], 0U);
                continue;
            }
            const window::Placement* placement =
                plan->placement(static_cast<std::uint32_t>(w), attempts[cell] - 1U);
            ASSERT_NE(placement, nullptr);
            check_window_solutions(plan->shape_of(*placement), cell, r.slots, count, solsup_ptr,
                                   solsup_idx, u64, size, weight, converged[cell] != 0);
            recorded += std::min(count[cell], r.slots);
        }
        EXPECT_EQ(flagged[s], any_flag ? 1 : 0);
        // The frame of the commits is the prediction (the fixture has no observables bias), and
        // the dense correction is exactly the committed faults.
        for (index_t o = 0; o < k; ++o) {
            EXPECT_EQ(predicted[(s * k) + o], (frame >> o) & 1U);
        }
        EXPECT_TRUE(std::ranges::equal(std::span(decodings).subspan(s * n, n), committed));
        if (success[s] != 0) {
            // Every window converged: H·c = σ over the whole shot.
            std::vector<std::uint8_t> syndrome(artifact->num_detectors(), 0);
            for (std::size_t j = 0; j < n; ++j) {
                if (committed[j] != 0) {
                    for (const index_t row : artifact->graph.column_rows(
                             artifact->graph.internal_column(static_cast<index_t>(j)))) {
                        syndrome[row] ^= std::uint8_t{1};
                    }
                }
            }
            const auto sigma = shots->syndrome(20 + s);
            EXPECT_TRUE(std::ranges::equal(syndrome, sigma));
        }
    }
    EXPECT_GT(recorded, 0U) << "no solution was recorded";
    fs::remove_all(r.root);
}

// Team executors (one team per window shape) decode exactly as serial ones.
TEST(SlidingTeam, MatchesSerial) {
    const fs::path root = test::scratch_dir("rtd_sliding_team");
    const std::size_t count = shots_for(12, 4);
    const json window = sliding_window(4, 2, 4, "exact", "defer", 1);
    const auto decode = [&](const std::string& name, unsigned team) {
        const fs::path out = root / name;
        std::vector<std::string> args =
            decode_args(r9 / "artifact", r9 / "shots",
                        write_spec(root, name + ".json", relay_spec(window, team)), out);
        args.insert(args.end(), {"--count", std::to_string(count), "--save-commits",
                                 "--record-solutions", "2"});
        const CliRun run = run_cli(args);
        EXPECT_EQ(run.code, 0) << name << ": " << first_error(run).dump();
        return out;
    };
    const fs::path serial = decode("serial", 0);
    const fs::path team = decode("team", 2);
    for (const auto& entry : fs::directory_iterator(serial)) {
        const std::string name = entry.path().filename().string();
        if (entry.path().extension() != ".npy" || name.contains("decode_ns")) {
            continue;
        }
        const auto a = read_raw(entry.path());
        const auto b = read_raw(team / name);
        ASSERT_TRUE(a && b) << name;
        EXPECT_EQ(a->header.shape, b->header.shape) << name;
        EXPECT_TRUE(std::ranges::equal(a->bytes, b->bytes)) << name;
    }
    fs::remove_all(root);
}

// ---- Window events ------------------------------------------------------------------------------

TEST(SlidingEvents, AreLoggedUpToTheLimitAndCountedInFull) {
    const fs::path root = test::scratch_dir("rtd_sliding_events");
    const fs::path out = root / "out";
    // A cap of one iteration: nearly every window stops unconverged with its budget spent.
    const fs::path spec = write_spec(
        root, "spec.json", relay_spec(sliding_window(4, 2, 4, "exact", "commit_anyway", 0, 1)));
    std::vector<std::string> args = decode_args(r9 / "artifact", r9 / "shots", spec, out);
    args.insert(args.end(), {"--count", std::to_string(shots_for(30, 30)), "--workers", "2"});
    const CliRun run = run_cli(args);
    ASSERT_EQ(run.code, 0) << first_error(run).dump();

    const auto cap_hit = read_all<std::uint8_t>(out / "win_cap_hit.npy");
    const auto converged = read_all<std::uint8_t>(out / "win_converged.npy");
    const auto attempts = read_all<std::uint8_t>(out / "win_attempts.npy");
    std::map<std::string, std::uint64_t> expected;
    for (std::size_t q = 0; q < cap_hit.size(); ++q) {
        expected["cap_hit"] += cap_hit[q];
        expected["not_converged"] += attempts[q] > 0 && converged[q] == 0 ? 1U : 0U;
    }
    ASSERT_GT(expected["cap_hit"], 20U) << "the cap should stop most windows";
    EXPECT_EQ(event_totals(run), expected);

    std::map<std::string, std::uint64_t> individual;
    for (const json& line : run.log) {
        if (line.at("context") == "window" && !line.contains("total")) {
            ++individual[line.at("event").get<std::string>()];
            EXPECT_TRUE(line.contains("shot") && line.contains("window") &&
                        line.contains("policy"));
        }
    }
    for (const auto& [event, total] : expected) {
        EXPECT_EQ(individual[event], std::min<std::uint64_t>(total, 20)) << event;
    }
    const json windows = read_json(out / "run.json").at("summary").at("windows");
    EXPECT_EQ(windows.at("cap_hit").at("count"), expected["cap_hit"]);
    fs::remove_all(root);
}

} // namespace
