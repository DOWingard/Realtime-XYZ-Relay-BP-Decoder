// Golden tests of the stream decoder with the relay decoder inside.
//
// Identity: a window at least as wide as the shot (W ≥ Rt) is one final window holding the whole
// problem in its own column order, so the windowed decode must reproduce every whole-shot golden
// bit for bit: the correction ê, convergence, iterations, the per-leg record and the weight.
//
// Differential: every window golden written by the Python windowing reference (relay_bp inside,
// explicit γ tables per shape) must be reproduced array by array: per-window iterations, legs,
// attempts, convergence, weights (bitwise), unexplained detectors, flags, virtual commits, the
// committed faults of every window, and per shot the predicted observables, logical failure and
// flag.

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
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/decoder.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/sha256.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"
#include "unit/window_inner_support.hpp"
#include "unit/window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
namespace fs = std::filesystem;
using Json = nlohmann::json;

const fs::path fixture_root = RTD_FIXTURE_DIR;
// Golden manifests name their artifact relative to the repository root.
const fs::path repo_root = fs::path(RTD_FIXTURE_DIR).parent_path().parent_path();

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// Optimised builds decode every shot of every golden. Unoptimised and sanitizer builds, which
// look for undefined behaviour and races and run the decoder 5 to 20 times slower, decode a
// subset that reaches the same code paths (see shots_to_decode). RTD_WINDOW_GOLDEN_SHOTS=N
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

// Shots of the identity goldens decoded under each spec.
std::size_t identity_shots(std::size_t count) {
    const auto limit = shot_limit();
    if (limit) {
        return *limit == 0 ? count : std::min(count, *limit);
    }
    return every_shot ? count : std::min<std::size_t>(count, 12);
}

// ---- Identity: W ≥ Rt reproduces the whole-shot goldens. -----------------------------------

// The reference's "unique best" flag of each leg (see golden_test.cpp).
std::vector<bool> unique_best_flags(const test::InnerCall& call) {
    std::vector<bool> flags(call.legs_executed, false);
    if (!call.best_leg) {
        return flags;
    }
    bool unique = true;
    for (std::uint32_t leg = *call.best_leg + 1; leg < call.legs_executed; ++leg) {
        if (call.legs[leg].converged && call.legs[leg].weight == call.weight) {
            unique = false;
        }
    }
    flags[*call.best_leg] = unique;
    return flags;
}

// One shot of an identity run against the golden: the single window's commit and its inner call.
bool identity_shot_matches(const io::Golden& golden, std::size_t s, const WindowPlan& plan,
                           const Commit& commit, const ShotSummary& summary,
                           const test::InnerCall& call,
                           std::span<const std::uint64_t> column_class) {
    const WindowSpec& spec = plan.spec();
    const std::uint32_t rounds_total = plan.rounds_total();
    bool ok = true;
    const auto expected_hard = golden.decoding.row(s);
    std::vector<index_t> support;
    std::uint64_t frame = 0;
    for (index_t j = 0; j < expected_hard.size(); ++j) {
        if (expected_hard[j] != 0) {
            support.push_back(j);
            frame ^= column_class[j];
        }
    }
    const bool success = golden.success.data[s] != 0;
    const WindowRecord& r = commit.record;
    ok &= std::ranges::equal(commit.faults, support);
    ok &= std::ranges::equal(call.support, support);
    ok &= r.converged == success && summary.success == success;
    ok &= std::cmp_equal(r.iterations, golden.iterations.data[s]);
    ok &= same_bits(r.weight, golden.weight.data[s]);
    ok &= r.attempts == 1 && r.shape == 0 && commit.window == 0;
    ok &= r.flagged == (!success && spec.on_failure != OnFailure::commit_anyway);
    ok &= !success || r.unexplained == 0;
    ok &= summary.frame == frame;
    ok &= commit.rounds == rounds_total;
    if (golden.legs_ptr) {
        const auto begin = static_cast<std::size_t>(golden.legs_ptr->data[s]);
        const auto end = static_cast<std::size_t>(golden.legs_ptr->data[s + 1]);
        ok &= r.legs == end - begin && call.legs_executed == end - begin;
        if (call.legs_executed == end - begin) {
            const std::vector<bool> flags = unique_best_flags(call);
            for (std::size_t leg = 0; leg < end - begin; ++leg) {
                ok &= std::cmp_equal(call.legs[leg].iterations,
                                     golden.leg_iterations->data[begin + leg]);
                ok &= call.legs[leg].converged == (golden.leg_converged->data[begin + leg] != 0);
                ok &= flags[leg] == (golden.leg_unique_best->data[begin + leg] != 0);
            }
        }
    } else {
        ok &= r.legs == 1;
    }
    return ok;
}

template <class A, class Exec>
std::size_t identity_mismatches(const io::Artifact& artifact, const io::Golden& golden,
                                const WindowPlan& plan, const std::function<Exec()>& executor) {
    const std::size_t shots = identity_shots(golden.count());
    std::vector<test::InnerCall> calls;
    const GammaSource* gammas = golden.gammas ? &*golden.gammas : nullptr;
    using Relay = CpuRelayDecoder<A, Exec>;
    auto decoder = StreamDecoder<test::Spy<Relay>>::create(
        plan, [&](const Shape& shape) -> std::expected<test::Spy<Relay>, std::string> {
            auto backend = CpuBackend<A, Exec>::create(shape.graph(), shape.priors(), executor());
            if (!backend) {
                return std::unexpected(backend.error().detail);
            }
            auto relay =
                Relay::create(std::move(*backend), golden.min_sum, golden.relay, gammas);
            if (!relay) {
                return std::unexpected(relay.error().detail);
            }
            return test::Spy<Relay>(std::move(*relay), shape.index(), &calls);
        });
    EXPECT_TRUE(decoder) << (decoder ? "" : describe(decoder.error()));
    if (!decoder) {
        return golden.count();
    }
    const ArtifactProblem source(artifact);
    const auto classes = test::column_classes(source.problem());
    std::size_t mismatched = 0;
    for (std::size_t s = 0; s < shots; ++s) {
        calls.clear();
        std::vector<Commit> commits;
        auto summary = decode_shot(*decoder, golden.detectors.row(s), s,
                                   [&](const Commit& commit) { commits.push_back(commit); });
        EXPECT_TRUE(summary) << (summary ? "" : describe(summary.error()));
        if (!summary || commits.size() != 1 || calls.size() != 1) {
            ADD_FAILURE() << "shot " << s << ": expected one window and one decode";
            return golden.count();
        }
        // The commit's views stay valid: nothing was decoded after it.
        if (!identity_shot_matches(golden, s, plan, commits[0], *summary, calls[0], classes)) {
            ++mismatched;
            if (mismatched <= 5) {
                ADD_FAILURE() << std::format(
                    "{} shot {}: converged {} vs {}, iterations {} vs {}, weight {} vs {}, "
                    "legs {}",
                    golden.directory.filename().string(), s, commits[0].record.converged,
                    golden.success.data[s], commits[0].record.iterations,
                    golden.iterations.data[s], commits[0].record.weight, golden.weight.data[s],
                    commits[0].record.legs);
            }
        }
    }
    return mismatched;
}

template <class Exec>
std::size_t identity_dispatch(const io::Artifact& artifact, const io::Golden& golden,
                              const WindowPlan& plan, const std::function<Exec()>& executor) {
    if (golden.float_type == "f32") {
        return identity_mismatches<F32, Exec>(artifact, golden, plan, executor);
    }
    if (golden.float_type == "f64") {
        return identity_mismatches<F64, Exec>(artifact, golden, plan, executor);
    }
    ADD_FAILURE() << "unknown float type " << golden.float_type;
    return golden.count();
}

// W ≥ Rt under three specs: the identity must not depend on C, C′ or the policy.
std::vector<WindowSpec> identity_specs(std::uint32_t rounds_total) {
    const auto make = [](std::uint32_t w, std::uint32_t c, std::uint32_t converge, OnFailure policy,
                         std::uint32_t deferrals) {
        return WindowSpec{.width = w,
                          .commit = c,
                          .converge_rounds = converge,
                          .boundary = Boundary::exact,
                          .on_failure = policy,
                          .max_deferrals = deferrals,
                          .iteration_cap = std::nullopt};
    };
    return {make(rounds_total, 1, rounds_total, OnFailure::commit_anyway, 0),
            make(rounds_total + 6, 3, 5, OnFailure::flag, 0),
            make(rounds_total, 2, 3, OnFailure::defer, 2)};
}

class WindowIdentity : public testing::TestWithParam<std::string> {};

TEST_P(WindowIdentity, WholeShotWindowReproducesTheGolden) {
    auto artifact = io::load_artifact(fixture_root / "bb18_choi" / "artifact");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto golden = io::load_golden(fixture_root / "bb18_choi" / GetParam(),
                                  artifact->num_detectors(), artifact->num_columns());
    ASSERT_TRUE(golden) << io::describe(golden.error());
    const ArtifactProblem source(*artifact);
    const std::uint32_t rounds_total = 4;
    for (const WindowSpec& spec : identity_specs(rounds_total)) {
        auto plan = WindowPlan::build(source.problem(), spec);
        ASSERT_TRUE(plan) << describe(plan.error());
        ASSERT_EQ(plan->rounds_total(), rounds_total);
        ASSERT_EQ(plan->num_positions(), 1U);
        ASSERT_EQ(plan->shapes().size(), 1U);
        const std::function<Serial()> serial = [] { return Serial{}; };
        EXPECT_EQ(identity_dispatch(*artifact, *golden, *plan, serial), 0U)
            << std::format("({}, {}, {}) {}", spec.width, spec.commit, spec.converge_rounds,
                           to_string(spec.on_failure));
        if (spec.on_failure == OnFailure::commit_anyway) {
            const std::function<Team()> team = [] { return Team(2); };
            EXPECT_EQ(identity_dispatch(*artifact, *golden, *plan, team), 0U) << "team of 2";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Bb18, WindowIdentity,
                         testing::Values("min_sum_f32", "min_sum_f64", "mem_bp_f32",
                                         "min_sum_adaptive_f32", "relay_f32", "relay_f64",
                                         "relay_all_f32", "relay_preiter_f32",
                                         "single_columns_f32"));

// The gross relay goldens (R = 12), when RTD_GROSS_ARTIFACT and RTD_GROSS_GOLDEN name them.
TEST(WindowIdentityGross, WholeShotWindowReproducesTheRelayGoldens) {
    const char* artifact_dir = std::getenv("RTD_GROSS_ARTIFACT");
    const char* golden_root = std::getenv("RTD_GROSS_GOLDEN");
    if (artifact_dir == nullptr || golden_root == nullptr) {
        GTEST_SKIP() << "set RTD_GROSS_ARTIFACT and RTD_GROSS_GOLDEN to run";
    }
    auto artifact = io::load_artifact(artifact_dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const ArtifactProblem source(*artifact);
    std::size_t sets = 0;
    for (const char* name : {"relay5_f32", "relay5_f64"}) {
        const fs::path dir = fs::path(golden_root) / name;
        if (!fs::exists(dir / "manifest.json")) {
            continue;
        }
        auto golden = io::load_golden(dir, artifact->num_detectors(), artifact->num_columns());
        ASSERT_TRUE(golden) << io::describe(golden.error());
        const auto rounds_total =
            static_cast<std::uint32_t>(std::ranges::max(artifact->detector_round));
        auto plan = WindowPlan::build(source.problem(), identity_specs(rounds_total)[0]);
        ASSERT_TRUE(plan) << describe(plan.error());
        ASSERT_EQ(plan->num_positions(), 1U);
        const std::function<Serial()> serial = [] { return Serial{}; };
        const std::size_t mismatched = identity_dispatch(*artifact, *golden, *plan, serial);
        EXPECT_EQ(mismatched, 0U) << name;
        std::cout << std::format("[ identity ] gross {}: W = {}, {} shots, {} mismatched\n", name,
                                 rounds_total, golden->count(), mismatched);
        ++sets;
    }
    EXPECT_GT(sets, 0U) << "no relay5 golden under " << golden_root;
}

// ---- Differential: the Python reference's window goldens. ------------------------------------

struct Settings {
    std::string float_type;
    MinSumConfig min_sum;
    RelayConfig relay;
};

std::expected<Settings, std::string> decoder_settings(const Json& d) {
    Settings s;
    s.float_type = d.at("float").get<std::string>();
    if (!d.at("gamma0").is_null()) {
        s.min_sum.gamma0 = d.at("gamma0").get<double>();
    }
    s.relay.pre_iter = d.at("pre_iter").get<std::uint32_t>();
    s.relay.set_max_iter = d.at("set_max_iter").get<std::uint32_t>();
    s.relay.num_sets = d.at("num_sets").get<std::uint32_t>();
    const Json& stopping = d.at("stopping");
    const auto rule = stopping.at("rule").get<std::string>();
    if (rule == "after_n_converged") {
        s.relay.stopping = AfterNConverged{stopping.at("count").get<std::uint32_t>()};
    } else if (rule == "after_leg0") {
        s.relay.stopping = AfterLeg0{};
    } else if (rule == "all_legs") {
        s.relay.stopping = AllLegs{};
    } else {
        return std::unexpected("unknown stopping rule " + rule);
    }
    const Json& alpha = d.at("alpha");
    const auto alpha_rule = alpha.at("rule").get<std::string>();
    if (alpha_rule == "constant") {
        s.min_sum.alpha = ConstantAlpha{alpha.at("value").get<double>()};
    } else if (alpha_rule == "adaptive") {
        s.min_sum.alpha = AdaptiveAlpha{alpha.at("scaling").get<double>()};
    } else {
        return std::unexpected("unknown alpha rule " + alpha_rule);
    }
    return s;
}

std::expected<WindowSpec, std::string> window_spec(const Json& w) {
    if (w.at("mode").get<std::string>() != "sliding") {
        return std::unexpected("window mode is not sliding");
    }
    const auto boundary = parse_boundary(w.at("boundary").get<std::string>());
    const auto policy = parse_on_failure(w.at("on_failure").get<std::string>());
    if (!boundary || !policy) {
        return std::unexpected("unknown boundary or policy");
    }
    WindowSpec s{.width = w.at("width").get<std::uint32_t>(),
                 .commit = w.at("commit").get<std::uint32_t>(),
                 .converge_rounds = w.at("converge_rounds").get<std::uint32_t>(),
                 .boundary = *boundary,
                 .on_failure = *policy,
                 .max_deferrals = w.at("max_deferrals").get<std::uint32_t>(),
                 .iteration_cap = std::nullopt};
    if (!w.at("iteration_cap").is_null()) {
        s.iteration_cap = w.at("iteration_cap").get<std::uint32_t>();
    }
    return s;
}

// What the C++ stream decoder produced, per decoded shot, in the goldens' layouts.
struct Produced {
    std::size_t shots = 0;
    std::size_t windows = 0;
    std::size_t observables = 0;
    std::vector<std::size_t> decoded; // shot indices, ascending
    std::vector<std::uint32_t> win_iterations;
    std::vector<std::uint32_t> win_legs;
    std::vector<std::uint32_t> win_unexplained;
    std::vector<std::uint32_t> win_virtual;
    std::vector<std::uint8_t> win_attempts;
    std::vector<std::uint8_t> win_converged;
    std::vector<std::uint8_t> win_cap_hit;
    std::vector<std::uint8_t> win_flagged;
    std::vector<double> win_weight;
    std::vector<double> win_committed_weight;
    std::vector<std::vector<std::uint32_t>> commits; // [S·K]
    std::vector<std::uint8_t> predicted_observables;
    std::vector<std::uint8_t> logical_failure;
    std::vector<std::uint8_t> flagged;
    std::vector<std::uint8_t> success;
    std::vector<std::uint32_t> iterations;
    std::vector<std::uint32_t> legs;
    std::vector<double> weight;

    void size(std::size_t s, std::size_t k, std::size_t o) {
        shots = s;
        windows = k;
        observables = o;
        for (auto* v : {&win_iterations, &win_legs, &win_unexplained, &win_virtual}) {
            v->assign(s * k, 0);
        }
        for (auto* v : {&win_attempts, &win_converged, &win_cap_hit, &win_flagged}) {
            v->assign(s * k, 0);
        }
        win_weight.assign(s * k, 0.0);
        win_committed_weight.assign(s * k, 0.0);
        commits.assign(s * k, {});
        predicted_observables.assign(s * o, 0);
        for (auto* v : {&logical_failure, &flagged, &success}) {
            v->assign(s, 0);
        }
        iterations.assign(s, 0);
        legs.assign(s, 0);
        weight.assign(s, 0.0);
    }
};

template <class T>
bool equal_value(T a, T b) {
    if constexpr (std::same_as<T, double>) {
        return same_bits(a, b);
    } else {
        return a == b;
    }
}

// Compares the rows of the decoded shots of a golden array [S, width] with ours.
template <io::NpyElement T>
void compare_rows(const fs::path& file, const std::vector<T>& ours, std::size_t width,
                  const Produced& produced) {
    const std::string name = file.stem().string();
    auto golden = io::read_npy<T>(file);
    ASSERT_TRUE(golden) << io::describe(golden.error());
    ASSERT_EQ(golden->rows(), produced.shots) << name;
    ASSERT_EQ(golden->size(), produced.shots * width) << name;
    std::size_t mismatches = 0;
    for (const std::size_t s : produced.decoded) {
        for (std::size_t c = 0; c < width; ++c) {
            const T expected = golden->data[s * width + c];
            const T got = ours[s * width + c];
            if (!equal_value(expected, got)) {
                ++mismatches;
                if (mismatches <= 5) {
                    ADD_FAILURE() << std::format("{} shot {} column {}: golden {} ours {}", name,
                                                 s, c, expected, got);
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << name;
}

void compare_commits(const fs::path& dir, const Produced& produced) {
    auto ptr = io::read_npy<std::uint64_t>(dir / "commit_ptr.npy", 1);
    auto faults = io::read_npy<std::uint32_t>(dir / "commit_faults.npy", 1);
    ASSERT_TRUE(ptr) << io::describe(ptr.error());
    ASSERT_TRUE(faults) << io::describe(faults.error());
    const std::size_t k = produced.windows;
    ASSERT_EQ(ptr->size(), produced.shots * k + 1);
    ASSERT_EQ(ptr->data[produced.shots * k], faults->size());
    std::size_t mismatches = 0;
    for (const std::size_t s : produced.decoded) {
        for (std::size_t w = 0; w < k; ++w) {
            const std::size_t i = s * k + w;
            const std::span<const std::uint32_t> expected(
                faults->data.data() + ptr->data[i], ptr->data[i + 1] - ptr->data[i]);
            if (!std::ranges::equal(expected, produced.commits[i])) {
                ++mismatches;
                if (mismatches <= 5) {
                    ADD_FAILURE() << std::format(
                        "commit_faults shot {} window {}: golden {} faults, ours {}", s, w,
                        expected.size(), produced.commits[i].size());
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0U) << "commit_faults";
}

// Every array of the golden directory, by name; an array this test cannot produce is a failure,
// so nothing the reference wrote goes unchecked.
void compare_directory(const fs::path& dir, const Produced& p) {
    const std::set<std::string> required{
        "win_iterations", "win_legs",      "win_attempts",   "win_converged",
        "win_weight",     "win_committed_weight", "win_unexplained", "win_flagged",
        "win_virtual",    "commit_ptr",    "commit_faults",  "predicted_observables",
        "logical_failure", "flagged"};
    std::set<std::string> seen;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".npy") {
            continue;
        }
        const std::string name = entry.path().stem().string();
        seen.insert(name);
        // Inputs, wall times, and the faults (compared with their pointers).
        if (name == "detectors" || name == "observables" || name == "win_decode_ns" ||
            name == "decode_ns" || name == "commit_faults") {
            continue;
        }
        const std::size_t k = p.windows;
        const fs::path& f = entry.path();
        if (name == "win_iterations") {
            compare_rows(f, p.win_iterations, k, p);
        } else if (name == "win_legs") {
            compare_rows(f, p.win_legs, k, p);
        } else if (name == "win_attempts") {
            compare_rows(f, p.win_attempts, k, p);
        } else if (name == "win_converged") {
            compare_rows(f, p.win_converged, k, p);
        } else if (name == "win_cap_hit") {
            compare_rows(f, p.win_cap_hit, k, p);
        } else if (name == "win_weight") {
            compare_rows(f, p.win_weight, k, p);
        } else if (name == "win_committed_weight") {
            compare_rows(f, p.win_committed_weight, k, p);
        } else if (name == "win_unexplained") {
            compare_rows(f, p.win_unexplained, k, p);
        } else if (name == "win_flagged") {
            compare_rows(f, p.win_flagged, k, p);
        } else if (name == "win_virtual") {
            compare_rows(f, p.win_virtual, k, p);
        } else if (name == "commit_ptr") {
            compare_commits(dir, p);
        } else if (name == "predicted_observables") {
            compare_rows(f, p.predicted_observables, p.observables, p);
        } else if (name == "logical_failure") {
            compare_rows(f, p.logical_failure, 1, p);
        } else if (name == "flagged") {
            compare_rows(f, p.flagged, 1, p);
        } else if (name == "success") {
            compare_rows(f, p.success, 1, p);
        } else if (name == "iterations") {
            compare_rows(f, p.iterations, 1, p);
        } else if (name == "legs") {
            compare_rows(f, p.legs, 1, p);
        } else if (name == "weight") {
            compare_rows(f, p.weight, 1, p);
        } else {
            ADD_FAILURE() << "golden array " << name << " is not compared";
        }
    }
    for (const std::string& name : required) {
        EXPECT_TRUE(seen.contains(name)) << "golden has no " << name << ".npy";
    }
}

template <class A>
void decode_golden_shots(const fs::path& dir, const WindowPlan& plan, const Settings& settings,
                         std::span<const ExplicitGammaTable> tables,
                         const io::NpyArray<Bit>& detectors, const io::NpyArray<Bit>& truth,
                         std::uint64_t first, Produced& out) {
    using Relay = CpuRelayDecoder<A>;
    auto decoder = StreamDecoder<Relay>::create(
        plan, [&](const Shape& shape) -> std::expected<Relay, std::string> {
            auto backend = CpuBackend<A>::create(shape.graph(), shape.priors());
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
    for (const std::size_t s : out.decoded) {
        const auto summary = decode_shot(*decoder, detectors.row(s), first + s,
                                         [&](const Commit& c) {
                                             const std::size_t i = s * k + c.window;
                                             const WindowRecord& r = c.record;
                                             out.win_iterations[i] = r.iterations;
                                             out.win_legs[i] = r.legs;
                                             out.win_unexplained[i] = r.unexplained;
                                             out.win_virtual[i] = r.virtual_commits;
                                             out.win_attempts[i] =
                                                 static_cast<std::uint8_t>(r.attempts);
                                             out.win_converged[i] =
                                                 static_cast<std::uint8_t>(r.converged);
                                             out.win_cap_hit[i] =
                                                 static_cast<std::uint8_t>(r.cap_hit);
                                             out.win_flagged[i] =
                                                 static_cast<std::uint8_t>(r.flagged);
                                             out.win_weight[i] = r.weight;
                                             out.win_committed_weight[i] = r.committed_weight;
                                             out.commits[i].assign(c.faults.begin(),
                                                                   c.faults.end());
                                         });
        ASSERT_TRUE(summary) << dir.filename() << " shot " << s << ": "
                             << describe(summary.error());
        bool failure = false;
        for (std::size_t b = 0; b < o; ++b) {
            const Bit predicted = decoder->frame()[b];
            out.predicted_observables[s * o + b] = predicted;
            failure = failure || predicted != truth.row(s)[b];
        }
        out.logical_failure[s] = static_cast<std::uint8_t>(failure);
        out.flagged[s] = static_cast<std::uint8_t>(summary->flagged);
        out.success[s] = static_cast<std::uint8_t>(summary->success);
        out.iterations[s] = summary->iterations;
        out.legs[s] = summary->legs;
        out.weight[s] = summary->weight;
    }
}

// Reproduces one window golden directory (manifest.json, gammas/, arrays) with the C++ stream
// decoder. Returns the number of shots decoded.
// The shots of a window golden to decode: every one (optimised builds, or RTD_WINDOW_GOLDEN_SHOTS
// = 0), the first N (RTD_WINDOW_GOLDEN_SHOTS = N), or else the first two plus the first two shots
// with a window that did not converge, the first two with a deferred window and the first two
// with a window skipped after a final placement.
std::vector<std::size_t> shots_to_decode(const fs::path& dir, std::size_t shots) {
    std::vector<std::size_t> chosen;
    const auto limit = shot_limit();
    if (every_shot || (limit && *limit == 0) || (limit && *limit >= shots)) {
        chosen.resize(shots);
        std::ranges::iota(chosen, std::size_t{0});
        return chosen;
    }
    if (limit) {
        chosen.resize(*limit);
        std::ranges::iota(chosen, std::size_t{0});
        return chosen;
    }
    auto converged = io::read_npy<std::uint8_t>(dir / "win_converged.npy", 2);
    auto attempts = io::read_npy<std::uint8_t>(dir / "win_attempts.npy", 2);
    EXPECT_TRUE(converged && attempts);
    std::set<std::size_t> picked;
    for (std::size_t s = 0; s < std::min<std::size_t>(shots, 2); ++s) {
        picked.insert(s);
    }
    if (converged && attempts) {
        const std::array<std::function<bool(std::uint8_t, std::uint8_t)>, 3> kinds{
            [](std::uint8_t c, std::uint8_t /*a*/) { return c == 0; },
            [](std::uint8_t /*c*/, std::uint8_t a) { return a > 1; },
            [](std::uint8_t /*c*/, std::uint8_t a) { return a == 0; }};
        for (const auto& kind : kinds) {
            std::size_t found = 0;
            for (std::size_t s = 0; s < shots && found < 2; ++s) {
                for (std::size_t w = 0; w < converged->cols(); ++w) {
                    if (kind(converged->row(s)[w], attempts->row(s)[w])) {
                        picked.insert(s);
                        ++found;
                        break;
                    }
                }
            }
        }
    }
    chosen.assign(picked.begin(), picked.end());
    return chosen;
}

std::size_t check_window_golden(const fs::path& dir) {
    std::ifstream in(dir / "manifest.json");
    const Json manifest = Json::parse(in);
    const fs::path artifact_dir = repo_root / manifest.at("artifact").at("path").get<std::string>();
    auto artifact = io::load_artifact(artifact_dir);
    EXPECT_TRUE(artifact) << (artifact ? "" : io::describe(artifact.error()));
    if (!artifact) {
        return 0;
    }
    auto artifact_sha = io::sha256_file(artifact_dir / "manifest.json");
    EXPECT_TRUE(artifact_sha);
    EXPECT_EQ(artifact_sha.value_or(""),
              manifest.at("artifact").at("manifest_sha256").get<std::string>())
        << "the golden was made from another artifact";
    const ArtifactProblem source(*artifact);
    auto spec = window_spec(manifest.at("window"));
    EXPECT_TRUE(spec) << (spec ? "" : spec.error());
    auto settings = decoder_settings(manifest.at("decoder"));
    EXPECT_TRUE(settings) << (settings ? "" : settings.error());
    if (!spec || !settings) {
        return 0;
    }
    auto plan = WindowPlan::build(source.problem(), *spec);
    EXPECT_TRUE(plan) << (plan ? "" : describe(plan.error()));
    if (!plan) {
        return 0;
    }
    EXPECT_EQ(plan->shapes().size(), manifest.at("num_shapes").get<std::size_t>());
    EXPECT_EQ(plan->num_positions(), manifest.at("num_positions").get<std::size_t>());

    const auto rows = manifest.at("gamma").at("rows").get<std::size_t>();
    std::vector<ExplicitGammaTable> tables;
    tables.reserve(plan->shapes().size());
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
    EXPECT_EQ(detectors->cols(), artifact->num_detectors());
    const std::size_t shots = detectors->rows();
    Produced produced;
    produced.size(shots, plan->num_positions(), artifact->num_observables());
    produced.decoded = shots_to_decode(dir, shots);
    const std::size_t count = produced.decoded.size();
    const auto first = manifest.at("first").get<std::uint64_t>();
    if (settings->float_type == "f32") {
        decode_golden_shots<F32>(dir, *plan, *settings, tables, *detectors, *truth, first,
                                 produced);
    } else if (settings->float_type == "f64") {
        decode_golden_shots<F64>(dir, *plan, *settings, tables, *detectors, *truth, first,
                                 produced);
    } else {
        ADD_FAILURE() << "unknown float type " << settings->float_type;
        return 0;
    }
    if (testing::Test::HasFatalFailure()) {
        return 0;
    }
    compare_directory(dir, produced);
    return count;
}

std::vector<fs::path> fixture_window_goldens() {
    std::vector<fs::path> dirs;
    for (const char* fixture : {"bb18_choi", "bb18_choi_r9"}) {
        const fs::path root = fixture_root / fixture;
        if (!fs::exists(root)) {
            continue;
        }
        for (const auto& entry : fs::directory_iterator(root)) {
            if (entry.path().filename().string().starts_with("window_") &&
                fs::exists(entry.path() / "manifest.json")) {
                dirs.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(dirs);
    return dirs;
}

class WindowGolden : public testing::TestWithParam<fs::path> {};

TEST_P(WindowGolden, StreamDecoderReproducesTheReference) {
    const std::size_t decoded = check_window_golden(GetParam());
    EXPECT_GT(decoded, 0U);
}

std::string golden_name(const testing::TestParamInfo<fs::path>& info) {
    std::string name = info.param.parent_path().filename().string() + "_" +
                       info.param.filename().string();
    std::ranges::replace_if(
        name, [](char c) { return std::isalnum(static_cast<unsigned char>(c)) == 0; }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Fixture, WindowGolden, testing::ValuesIn(fixture_window_goldens()),
                         golden_name);

TEST(WindowGoldenFixtures, AllFifteenArePresent) {
    EXPECT_EQ(fixture_window_goldens().size(), 15U);
}

// Every window golden below RTD_WINDOW_GOLDEN_DIR (e.g. gross-code goldens under
// data/golden/window/).
TEST(WindowGoldenExternal, StreamDecoderReproducesEveryGolden) {
    const char* root = std::getenv("RTD_WINDOW_GOLDEN_DIR");
    if (root == nullptr) {
        GTEST_SKIP() << "set RTD_WINDOW_GOLDEN_DIR to run";
    }
    std::vector<fs::path> dirs;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_directory() && fs::exists(entry.path() / "manifest.json") &&
            fs::exists(entry.path() / "win_iterations.npy")) {
            dirs.push_back(entry.path());
        }
    }
    std::ranges::sort(dirs);
    ASSERT_FALSE(dirs.empty()) << "no window golden below " << root;
    for (const fs::path& dir : dirs) {
        SCOPED_TRACE(dir.string());
        const std::size_t decoded = check_window_golden(dir);
        std::cout << std::format("[ window   ] {}: {} shots\n", dir.string(), decoded);
        EXPECT_GT(decoded, 0U);
    }
}

} // namespace
