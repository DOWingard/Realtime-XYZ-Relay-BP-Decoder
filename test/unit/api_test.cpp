// The in-memory decoding API (rtd::api) against rtd_decode run in process on the same shots and
// spec: the whole-shot and sliding-window decoders must reproduce every array rtd_decode writes,
// bit for bit, recorded solutions, committed faults and (under a selection policy) every
// confidence and history value included, for any number format, any number of workers and any
// stream offset; a stream fed round by round must commit exactly what a batch commits; the window
// goldens and the fixed-point goldens of the bb18 fixtures are reproduced directly; biases are
// applied to the syndrome and to the prediction; and malformed problems, specs, batches and
// stream calls are refused with the error code that names the fault.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string_view>
#include <system_error>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"
#include "rtd/api/batch.hpp"
#include "rtd/api/decoder.hpp"
#include "rtd/api/error.hpp"
#include "rtd/api/problem.hpp"
#include "rtd/api/windowed.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/window/artifact_problem.hpp"

namespace {

using namespace rtd;
using json = nlohmann::json;
namespace fs = std::filesystem;
using test::read_all;
using test::read_json;
using test::run_cli;

const fs::path r9 = test::fixture_root / "bb18_choi_r9";
const fs::path repo_root = test::fixture_root.parent_path().parent_path();

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
// Unoptimised and sanitizer builds run the decoder 5 to 20 times slower and decode fewer shots.
constexpr std::size_t shots_to_decode = optimised && !sanitized ? 100 : 24;

// ---- Fixtures ---------------------------------------------------------------------------------

// The bb18 R = 9 artifact and its shots, loaded once. The artifact lives on the heap because
// `arrays` points into it.
// The arrays of a window::Problem as API input, with or without the detector rounds.
api::ProblemArrays arrays_of(const window::Problem& p, bool with_rounds) {
    return {.num_rows = p.num_rows,
            .num_columns = p.num_columns,
            .num_observables = p.num_observables,
            .h_row_ptr = p.h_row_ptr,
            .h_col_indices = p.h_col_indices,
            .priors = p.priors,
            .a_row_ptr = p.a_row_ptr,
            .a_col_indices = p.a_col_indices,
            .detector_round = with_rounds ? p.detector_round : std::span<const std::int32_t>{},
            .syndrome_bias = {},
            .observables_bias = {}};
}

struct Fixture {
    std::unique_ptr<io::Artifact> artifact;
    window::ArtifactProblem arrays;
    io::NpyArray<Bit> detectors;
    io::NpyArray<Bit> observables;

    [[nodiscard]] api::ProblemArrays problem_arrays(std::span<const Bit> syndrome_bias = {},
                                                    std::span<const Bit> observables_bias = {},
                                                    bool with_rounds = true) const {
        api::ProblemArrays out = arrays_of(arrays.problem(), with_rounds);
        out.syndrome_bias = syndrome_bias;
        out.observables_bias = observables_bias;
        return out;
    }
};

const Fixture& fixture() {
    static const Fixture loaded = [] {
        auto artifact = io::load_artifact(r9 / "artifact");
        if (!artifact) {
            throw std::runtime_error(io::describe(artifact.error()));
        }
        auto detectors = io::read_npy<Bit>(r9 / "shots" / "detectors.npy", 2);
        auto observables = io::read_npy<Bit>(r9 / "shots" / "observables.npy", 2);
        if (!detectors || !observables) {
            throw std::runtime_error("cannot read the bb18 R = 9 shots");
        }
        auto owned = std::make_unique<io::Artifact>(std::move(*artifact));
        const io::Artifact& stable = *owned;
        return Fixture{.artifact = std::move(owned),
                       .arrays = window::ArtifactProblem(stable),
                       .detectors = std::move(*detectors),
                       .observables = std::move(*observables)};
    }();
    return loaded;
}

json sliding_window(std::uint32_t width, std::uint32_t commit, const std::string& boundary,
                    const std::string& policy, std::uint32_t deferrals) {
    return {{"mode", "sliding"},          {"width", width},          {"commit", commit},
            {"converge_rounds", width},   {"boundary", boundary},    {"on_failure", policy},
            {"max_deferrals", deferrals}, {"iteration_cap", nullptr}};
}

// A small Relay-BP spec with uniform γ draws.
json relay_spec(json window, const std::string& policy = "f32") {
    return {{"version", 2},
            {"policy", policy},
            {"backend", "cpu"},
            {"layout", "row_major"},
            {"column_order", "wavefront"},
            {"block_rows", 64},
            {"executor", {{"type", "serial"}}},
            {"alpha", {{"rule", "constant"}, {"value", 1.0}}},
            {"gamma0", 0.125},
            {"pre_iter", 20},
            {"set_max_iter", 15},
            {"num_sets", 40},
            {"stopping", {{"rule", "after_n_converged"}, {"count", 3}}},
            {"gamma_source", {{"type", "uniform"}, {"seed", 7}, {"low", -0.24}, {"high", 0.66}}},
            {"window", std::move(window)}};
}

// The same spec as a version-3 spec with a selection policy.
json with_selection(json spec, json selection) {
    spec["version"] = 3;
    spec["selection"] = std::move(selection);
    return spec;
}

// The same spec in a fixed-point number format (min-sum scaling α = 1 is a shift form).
json with_arithmetic(json spec, const std::string& format) {
    spec.erase("policy");
    spec["arithmetic"] = format;
    return spec;
}

// Whole shot: the class-sum rule, stopped early once the gap reaches 1, and up to 4 legs more
// while the gap is below 3 or every solution falls into one class (3 + 4 ≤ capacity 10).
json whole_shot_selection() {
    return json::parse(R"({"rule": "class_sum",
                           "stop": {"rule": "gap", "threshold": 1.0},
                           "capacity": 10,
                           "confidence": {"signal": "gap", "threshold": 3.0,
                                          "single_class": "low", "extra_legs": 4,
                                          "on_low": "none"},
                           "history": null})");
}

// Windows: the largest-agreement rule, a window deferred (decoded again, wider) while its gap is
// below 2, and every history signal over the last 1, 2 and 3 windows.
json deferring_selection() {
    return json::parse(R"({"rule": "largest_agreement",
                           "stop": {"rule": "agree", "count": 2},
                           "capacity": 10,
                           "confidence": {"signal": "gap", "threshold": 2.0,
                                          "single_class": "high", "extra_legs": 3,
                                          "on_low": "defer"},
                           "history": {"lengths": [1, 2, 3],
                                       "signals": ["gap", "agreement", "weight", "first_legs",
                                                   "first_iterations", "q_supp", "density",
                                                   "commit_weight", "commit_q_supp"]}})");
}

// Windows: the default rule, a window flagged when its first solution took more than 30
// iterations, and two history signals.
json flagging_selection() {
    return json::parse(R"({"rule": "lowest_weight",
                           "stop": {"rule": "fixed"},
                           "capacity": 5,
                           "confidence": {"signal": "first_iterations", "threshold": 30.0,
                                          "single_class": "high", "extra_legs": 0,
                                          "on_low": "flag"},
                           "history": {"lengths": [1, 4], "signals": ["gap", "commit_weight"]}})");
}

harness::DecoderSpec parsed(const json& spec, const fs::path& base = fs::current_path()) {
    auto result = api::parse_spec(spec.dump(), base);
    if (!result) {
        throw std::runtime_error(api::describe(result.error()));
    }
    return std::move(*result);
}

std::shared_ptr<const api::Problem> make_problem(const harness::DecoderSpec& spec,
                                                 const api::ProblemArrays& arrays) {
    auto problem = api::Problem::create(arrays, spec.graph);
    if (!problem) {
        throw std::runtime_error(api::describe(problem.error()));
    }
    return std::move(*problem);
}

// Shots [first, first + count) of the fixture, unpacked.
api::BatchInput batch(std::size_t first, std::size_t count) {
    const io::NpyArray<Bit>& d = fixture().detectors;
    return {.detectors = d.span().subspan(first * d.cols(), count * d.cols()),
            .shots = count,
            .row_bytes = d.cols(),
            .bit_packed = false,
            .stream_offset = first};
}

// A token of this process, so that test runs from several build directories at once never share
// a scratch directory.
const std::string& run_token() {
    static const std::string token = [] {
        std::random_device device;
        return std::format("{:08x}{:08x}", device(), device());
    }();
    return token;
}

// Removes this process's scratch directories when the test binary ends.
class ScratchCleanup final : public testing::Environment {
public:
    void TearDown() override {
        const std::string prefix = std::format("rtd_api_{}_", run_token());
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(testing::TempDir(), ec)) {
            if (entry.path().filename().string().starts_with(prefix)) {
                fs::remove_all(entry.path(), ec);
            }
        }
    }
};

// gtest owns registered environments and tears them down after the last test.
// NOLINTBEGIN(cppcoreguidelines-owning-memory)
[[maybe_unused]] const testing::Environment* const scratch_cleanup =
    testing::AddGlobalTestEnvironment(new ScratchCleanup);
// NOLINTEND(cppcoreguidelines-owning-memory)

// rtd_decode over shots [first, first + count) of the fixture with `spec`; returns its output
// directory.
fs::path rtd_decode(const std::string& name, const json& spec, std::size_t first, std::size_t count,
                    const std::vector<std::string>& extra) {
    const fs::path root = test::scratch_dir(std::format("rtd_api_{}_{}", run_token(), name));
    const fs::path config = root / "spec.json";
    std::ofstream(config) << spec.dump(2);
    std::vector<std::string> arguments{"--artifact",  (r9 / "artifact").string(),
                                       "--shots",     (r9 / "shots").string(),
                                       "--config",    config.string(),
                                       "--out",       (root / "out").string(),
                                       "--first",     std::to_string(first),
                                       "--count",     std::to_string(count),
                                       "--log-level", "error"};
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    const test::CliRun run = run_cli(arguments);
    EXPECT_EQ(run.code, 0) << test::first_error(run).dump();
    return root / "out";
}

template <class T> std::vector<std::uint64_t> bits_of(const std::vector<T>& values) {
    std::vector<std::uint64_t> bits;
    bits.reserve(values.size());
    for (const T v : values) {
        if constexpr (std::same_as<T, double>) {
            bits.push_back(std::bit_cast<std::uint64_t>(v));
        } else {
            bits.push_back(static_cast<std::uint64_t>(v));
        }
    }
    return bits;
}

// Every array rtd_decode wrote that the API also records, compared by value (doubles bitwise).
void expect_same_arrays(const fs::path& out, const api::BatchResult& r, bool sliding,
                        bool solutions, bool commits) {
    EXPECT_EQ(read_all<Bit>(out / "predicted_observables.npy"), r.predicted);
    EXPECT_EQ(read_all<std::uint8_t>(out / "success.npy"), r.success);
    EXPECT_EQ(read_all<std::uint32_t>(out / "iterations.npy"), r.iterations);
    EXPECT_EQ(read_all<std::uint32_t>(out / "legs.npy"), r.legs);
    EXPECT_EQ(read_all<std::int32_t>(out / "best_leg.npy"), r.best_leg);
    EXPECT_EQ(bits_of(read_all<double>(out / "weight.npy")), bits_of(r.weight));
    EXPECT_EQ(read_all<Bit>(out / "decodings.npy"), r.decodings);
    if (sliding) {
        EXPECT_EQ(read_all<std::uint8_t>(out / "flagged.npy"), r.flagged);
        EXPECT_EQ(read_all<std::uint32_t>(out / "win_iterations.npy"), r.win_iterations);
        EXPECT_EQ(read_all<std::uint32_t>(out / "win_legs.npy"), r.win_legs);
        EXPECT_EQ(read_all<std::uint8_t>(out / "win_attempts.npy"), r.win_attempts);
        EXPECT_EQ(read_all<std::uint8_t>(out / "win_converged.npy"), r.win_converged);
        EXPECT_EQ(read_all<std::uint8_t>(out / "win_cap_hit.npy"), r.win_cap_hit);
        EXPECT_EQ(bits_of(read_all<double>(out / "win_weight.npy")), bits_of(r.win_weight));
        EXPECT_EQ(bits_of(read_all<double>(out / "win_committed_weight.npy")),
                  bits_of(r.win_committed_weight));
        EXPECT_EQ(read_all<std::uint32_t>(out / "win_unexplained.npy"), r.win_unexplained);
        EXPECT_EQ(read_all<std::uint8_t>(out / "win_flagged.npy"), r.win_flagged);
        EXPECT_EQ(read_all<std::uint32_t>(out / "win_virtual.npy"), r.win_virtual);
    }
    if (commits) {
        EXPECT_EQ(read_all<std::uint64_t>(out / "commit_ptr.npy"), r.commit_ptr);
        EXPECT_EQ(read_all<std::uint32_t>(out / "commit_faults.npy"), r.commit_faults);
    }
    if (solutions) {
        EXPECT_EQ(read_all<std::uint32_t>(out / "sol_count.npy"), r.sol_count);
        EXPECT_EQ(read_all<std::uint32_t>(out / "sol_leg.npy"), r.sol_leg);
        EXPECT_EQ(read_all<std::uint32_t>(out / "sol_iterations.npy"), r.sol_iterations);
        EXPECT_EQ(bits_of(read_all<double>(out / "sol_weight.npy")), bits_of(r.sol_weight));
        EXPECT_EQ(read_all<std::uint64_t>(out / "sol_class.npy"), r.sol_class);
        EXPECT_EQ(read_all<std::uint64_t>(out / "sol_hash.npy"), r.sol_hash);
        EXPECT_EQ(read_all<std::uint32_t>(out / "sol_size.npy"), r.sol_size);
        EXPECT_EQ(read_all<std::uint64_t>(out / "returned_class.npy"), r.returned_class);
    }
}

// The confidence and history arrays rtd_decode wrote under a selection policy, value by value
// (doubles bitwise, NaN included); without a policy, that neither side has any.
void expect_same_confidence(const fs::path& out, const api::BatchResult& r, bool selecting) {
    const harness::ConfidenceOutputs& c = r.confidence;
    ASSERT_EQ(c.enabled, selecting);
    ASSERT_EQ(fs::exists(out / "conf_found.npy"), selecting);
    if (!selecting) {
        return;
    }
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_found.npy"), c.found);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_seen.npy"), c.seen);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_distinct.npy"), c.distinct);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_classes.npy"), c.classes);
    EXPECT_EQ(read_all<std::uint64_t>(out / "conf_best_class.npy"), c.best_class);
    EXPECT_EQ(read_all<std::uint64_t>(out / "conf_second_class.npy"), c.second_class);
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_weight.npy")), bits_of(c.weight));
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_gap.npy")), bits_of(c.gap));
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_agreement.npy")), bits_of(c.agreement));
    EXPECT_EQ(read_all<std::uint8_t>(out / "conf_gap_state.npy"), c.gap_state);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_first_legs.npy"), c.first_legs);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_first_iterations.npy"), c.first_iterations);
    EXPECT_EQ(read_all<std::uint64_t>(out / "conf_class_sum_class.npy"), c.class_sum_class);
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_class_sum_top.npy")), bits_of(c.class_sum_top));
    EXPECT_EQ(read_all<std::uint64_t>(out / "conf_agreement_class.npy"), c.agreement_class);
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_q_supp.npy")), bits_of(c.q_supp));
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_q_sum_sq.npy")), bits_of(c.q_sum_sq));
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_q_total.npy")), bits_of(c.q_total));
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_components.npy"), c.components);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_syndrome_ones.npy"), c.syndrome_ones);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_syndrome_rows.npy"), c.syndrome_rows);
    EXPECT_EQ(read_all<std::uint64_t>(out / "conf_decided_class.npy"), c.decided_class);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_decided_leg.npy"), c.decided_leg);
    EXPECT_EQ(read_all<std::uint32_t>(out / "conf_extra_legs.npy"), c.extra_legs);
    EXPECT_EQ(bits_of(read_all<double>(out / "conf_score.npy")), bits_of(c.score));
    EXPECT_EQ(read_all<std::uint8_t>(out / "conf_low.npy"), c.low);
    EXPECT_EQ(read_all<std::uint8_t>(out / "conf_stopped_early.npy"), c.stopped_early);
    EXPECT_EQ(fs::exists(out / "conf_low_deferrals.npy"), c.sliding);
    if (c.sliding) {
        EXPECT_EQ(read_all<std::uint32_t>(out / "conf_low_deferrals.npy"), c.low_deferrals);
    }
    // The policy acted: some decode counted as low confidence.
    EXPECT_GT(std::ranges::count(c.low, std::uint8_t{1}), 0);
    const std::size_t signals = c.history_signals.size();
    const std::size_t lengths = c.history_lengths;
    EXPECT_EQ(fs::exists(out / "hist_state.npy"), signals > 0);
    if (signals == 0) {
        return;
    }
    EXPECT_EQ(read_all<std::uint8_t>(out / "hist_state.npy"), c.history_state);
    for (std::size_t g = 0; g < signals; ++g) {
        std::vector<double> ours;
        ours.reserve(c.cells * lengths);
        for (std::size_t cell = 0; cell < c.cells; ++cell) {
            for (std::size_t l = 0; l < lengths; ++l) {
                ours.push_back(c.history[(((cell * signals) + g) * lengths) + l]);
            }
        }
        const std::string name =
            std::format("hist_{}.npy", window::to_string(c.history_signals[g]));
        EXPECT_EQ(bits_of(read_all<double>(out / name)), bits_of(ours)) << name;
    }
}

std::vector<Bit> pack_rows(std::span<const Bit> bits, std::size_t rows, std::size_t width) {
    const std::size_t bytes = (width + 7) / 8;
    std::vector<Bit> packed(rows * bytes, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < width; ++i) {
            if (bits[(r * width) + i] != 0) {
                packed[(r * bytes) + (i / 8)] |= static_cast<Bit>(1U << (i % 8));
            }
        }
    }
    return packed;
}

// ---- Problems and specs -----------------------------------------------------------------------

TEST(ApiProblem, RejectsInconsistentArrays) {
    const harness::DecoderSpec spec = parsed(relay_spec({{"mode", "whole_shot"}}));
    const Fixture& f = fixture();
    const auto code_of = [&](const api::ProblemArrays& arrays) {
        auto problem = api::Problem::create(arrays, spec.graph);
        return problem ? std::optional<api::ApiError::Code>{} : problem.error().code;
    };
    EXPECT_FALSE(code_of(f.problem_arrays()).has_value());

    api::ProblemArrays short_priors = f.problem_arrays();
    short_priors.priors = short_priors.priors.first(short_priors.priors.size() - 1);
    EXPECT_EQ(code_of(short_priors), api::ApiError::Code::invalid_problem);

    api::ProblemArrays short_rounds = f.problem_arrays();
    short_rounds.detector_round = short_rounds.detector_round.first(3);
    EXPECT_EQ(code_of(short_rounds), api::ApiError::Code::invalid_problem);

    api::ProblemArrays bad_csr = f.problem_arrays();
    bad_csr.h_row_ptr = bad_csr.h_row_ptr.first(bad_csr.h_row_ptr.size() - 1);
    EXPECT_EQ(code_of(bad_csr), api::ApiError::Code::invalid_problem);

    const std::vector<Bit> wrong_length(3, 0);
    EXPECT_EQ(code_of(f.problem_arrays(wrong_length)), api::ApiError::Code::invalid_problem);
    std::vector<Bit> not_binary(f.artifact->num_detectors(), 0);
    not_binary.at(4) = 2;
    EXPECT_EQ(code_of(f.problem_arrays(not_binary)), api::ApiError::Code::invalid_problem);
    EXPECT_EQ(code_of(f.problem_arrays({}, wrong_length)), api::ApiError::Code::invalid_problem);

    std::vector<double> certain(f.artifact->priors.probabilities().begin(),
                                f.artifact->priors.probabilities().end());
    certain.at(7) = 1.0;
    api::ProblemArrays certain_fault = f.problem_arrays();
    certain_fault.priors = certain;
    EXPECT_EQ(code_of(certain_fault), api::ApiError::Code::invalid_problem);
}

TEST(ApiProblem, KeepsWhatItWasGiven) {
    const harness::DecoderSpec spec = parsed(relay_spec({{"mode", "whole_shot"}}));
    const Fixture& f = fixture();
    const auto problem = make_problem(spec, f.problem_arrays());
    EXPECT_EQ(problem->num_rows(), f.artifact->num_detectors());
    EXPECT_EQ(problem->num_columns(), f.artifact->num_columns());
    EXPECT_EQ(problem->num_observables(), f.artifact->num_observables());
    EXPECT_TRUE(problem->has_rounds());
    EXPECT_TRUE(problem->syndrome_bias().empty());
    ASSERT_EQ(problem->column_classes().size(), f.artifact->num_columns());
    for (index_t j = 0; j < f.artifact->num_columns(); ++j) {
        std::uint64_t mask = 0;
        for (const index_t o : f.artifact->observables.column(j)) {
            mask ^= std::uint64_t{1} << o;
        }
        ASSERT_EQ(problem->column_classes()[j], mask) << "column " << j;
    }
    const window::Problem w = problem->window_problem();
    EXPECT_TRUE(std::ranges::equal(w.detector_round, f.artifact->detector_round));
    EXPECT_TRUE(std::ranges::equal(w.priors, f.artifact->priors.probabilities()));
    EXPECT_FALSE(make_problem(spec, f.problem_arrays({}, {}, false))->has_rounds());
}

TEST(ApiSpec, ParsesWhatItCanDecodeAndRejectsTheRest) {
    const json whole = relay_spec({{"mode", "whole_shot"}});
    EXPECT_TRUE(api::parse_spec(whole.dump(), fs::current_path()));
    const auto sliding = api::parse_spec(
        relay_spec(sliding_window(4, 2, "exact", "flag", 0)).dump(), fs::current_path());
    ASSERT_TRUE(sliding);
    EXPECT_TRUE(sliding->window.is_sliding());

    const auto code_of = [](const std::string& text) {
        auto spec = api::parse_spec(text, fs::current_path());
        return spec ? std::optional<api::ApiError::Code>{} : spec.error().code;
    };
    EXPECT_EQ(code_of("{not json"), api::ApiError::Code::invalid_spec);
    json missing = whole;
    missing.erase("pre_iter");
    EXPECT_EQ(code_of(missing.dump()), api::ApiError::Code::invalid_spec);
    json version1 = whole;
    version1.erase("version");
    EXPECT_EQ(code_of(version1.dump()), api::ApiError::Code::invalid_spec);
    json cuda = whole;
    cuda["backend"] = "cuda";
    EXPECT_EQ(code_of(cuda.dump()), api::ApiError::Code::invalid_spec);
    // Fixed-point formats are named by "arithmetic"; "policy" names the IEEE formats only.
    json fixed = whole;
    fixed["policy"] = "int4.2.8";
    EXPECT_EQ(code_of(fixed.dump()), api::ApiError::Code::invalid_spec);
    for (const char* format : {"int4.2.8", "int5.2.8", "int6.2.8"}) {
        const auto spec = api::parse_spec(with_arithmetic(whole, format).dump(), fs::current_path());
        ASSERT_TRUE(spec) << format << ": " << api::describe(spec.error());
        EXPECT_EQ(harness::to_string(spec->policy), format);
    }
    // A fixed-point format needs a shift-form scaling α = 1 − 2^(−k).
    json scaled = with_arithmetic(whole, "int4.2.8");
    scaled["alpha"] = {{"rule", "constant"}, {"value", 0.8}};
    EXPECT_EQ(code_of(scaled.dump()), api::ApiError::Code::invalid_spec);
    // Version 3 without a selection policy is the version-2 decoder; with one it carries it.
    json version3 = whole;
    version3["version"] = 3;
    version3["selection"] = nullptr;
    const auto plain = api::parse_spec(version3.dump(), fs::current_path());
    ASSERT_TRUE(plain);
    EXPECT_FALSE(plain->selection);
    const auto selecting = api::parse_spec(with_selection(whole, whole_shot_selection()).dump(),
                                           fs::current_path());
    ASSERT_TRUE(selecting) << api::describe(selecting.error());
    ASSERT_TRUE(selecting->selection);
    EXPECT_EQ(selecting->selection->config.rule, SelectionRule::class_sum);
    // The selection object is checked against the relay schedule: 3 + 8 extra legs exceed the
    // capacity of 10.
    json too_many = whole_shot_selection();
    too_many["confidence"]["extra_legs"] = 8;
    const auto refused = api::parse_spec(with_selection(whole, too_many).dump(), fs::current_path());
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().code, api::ApiError::Code::invalid_spec);
    EXPECT_NE(refused.error().detail.find("extra_legs"), std::string::npos);
    // The message names the field.
    auto failed = api::parse_spec(missing.dump(), fs::current_path());
    ASSERT_FALSE(failed);
    EXPECT_NE(api::describe(failed.error()).find("pre_iter"), std::string::npos);
}

// ---- Whole shot ---------------------------------------------------------------------------------

// A spec rtd_decode and the API both decode, and the name of its scratch directory.
struct SpecCase {
    std::string name;
    json spec;
};

std::string case_name(const testing::TestParamInfo<SpecCase>& param_info) {
    return param_info.param.name;
}

class ApiWholeShot : public testing::TestWithParam<SpecCase> {};

TEST_P(ApiWholeShot, MatchesRtdDecodeArrayForArray) {
    const json& spec = GetParam().spec;
    const std::size_t first = 5;
    const std::size_t count = shots_to_decode - first;
    const fs::path out = rtd_decode("whole_" + GetParam().name, spec, first, count,
                                    {"--record-solutions", "3", "--save-decodings"});
    const harness::DecoderSpec parsed_spec = parsed(spec);
    const auto problem = make_problem(parsed_spec, fixture().problem_arrays());
    for (const unsigned workers : {1U, 3U}) {
        auto decoder = api::WholeShotDecoder::create(problem, parsed_spec,
                                                     {.workers = workers, .record_solutions = 3});
        ASSERT_TRUE(decoder) << api::describe(decoder.error());
        auto result = (*decoder)->decode_batch(batch(first, count),
                                               {.workers = workers, .save_decodings = true});
        ASSERT_TRUE(result) << api::describe(result.error());
        EXPECT_EQ(result->shots, count);
        EXPECT_EQ(result->predicted_row_bytes, fixture().artifact->num_observables());
        expect_same_arrays(out, *result, false, true, false);
        expect_same_confidence(out, *result, parsed_spec.selection.has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(
    Bb18, ApiWholeShot,
    testing::Values(
        SpecCase{"f32", relay_spec({{"mode", "whole_shot"}})},
        SpecCase{"selection", with_selection(relay_spec({{"mode", "whole_shot"}}),
                                             whole_shot_selection())},
        SpecCase{"int4", with_arithmetic(relay_spec({{"mode", "whole_shot"}}), "int4.2.8")},
        SpecCase{"int6_selection",
                 with_selection(with_arithmetic(relay_spec({{"mode", "whole_shot"}}), "int6.2.8"),
                                whole_shot_selection())}),
    case_name);

TEST(ApiWholeShotBatch, PackedInputAndOutputMatchUnpacked) {
    const harness::DecoderSpec spec = parsed(relay_spec({{"mode", "whole_shot"}}, "f64"));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder = api::WholeShotDecoder::create(problem, spec, {.workers = 2});
    ASSERT_TRUE(decoder);
    const std::size_t count = std::min<std::size_t>(shots_to_decode, 30);
    auto plain = (*decoder)->decode_batch(batch(0, count), {.workers = 2});
    ASSERT_TRUE(plain);
    const std::size_t m = fixture().artifact->num_detectors();
    const std::vector<Bit> packed = pack_rows(batch(0, count).detectors, count, m);
    const api::BatchInput packed_input{.detectors = packed,
                                       .shots = count,
                                       .row_bytes = (m + 7) / 8,
                                       .bit_packed = true,
                                       .stream_offset = 0};
    auto from_packed =
        (*decoder)->decode_batch(packed_input, {.workers = 1, .pack_predictions = true});
    ASSERT_TRUE(from_packed) << api::describe(from_packed.error());
    const std::size_t k = fixture().artifact->num_observables();
    EXPECT_EQ(from_packed->predicted_row_bytes, (k + 7) / 8);
    EXPECT_EQ(from_packed->predicted, pack_rows(plain->predicted, count, k));
    EXPECT_EQ(from_packed->iterations, plain->iterations);
    EXPECT_EQ(bits_of(from_packed->weight), bits_of(plain->weight));
}

TEST(ApiWholeShotBatch, ReproducesTheRelayGolden) {
    const fs::path golden = test::fixture_root / "bb18_choi" / "relay_f32";
    const json manifest = read_json(golden / "manifest.json");
    const json& config = manifest.at("config");
    json spec = relay_spec({{"mode", "whole_shot"}});
    spec["pre_iter"] = config.at("pre_iter");
    spec["set_max_iter"] = config.at("set_max_iter");
    spec["num_sets"] = config.at("num_sets");
    spec["stopping"] = {{"rule", "after_n_converged"}, {"count", config.at("stop_nconv")}};
    spec["gamma0"] = config.at("gamma0");
    spec["gamma_source"] = {{"type", "explicit"}, {"path", "gammas.npy"}};
    const harness::DecoderSpec decoder_spec = parsed(spec, golden);

    auto artifact = io::load_artifact(test::fixture_root / "bb18_choi" / "artifact");
    ASSERT_TRUE(artifact);
    const window::ArtifactProblem arrays(*artifact);
    const window::Problem p = arrays.problem();
    const auto problem = make_problem(decoder_spec, arrays_of(p, false));
    auto decoder = api::WholeShotDecoder::create(problem, decoder_spec, {.workers = 2});
    ASSERT_TRUE(decoder) << api::describe(decoder.error());
    auto detectors = io::read_npy<Bit>(golden / "detectors.npy", 2);
    ASSERT_TRUE(detectors);
    auto result = (*decoder)->decode_batch({.detectors = detectors->span(),
                                            .shots = detectors->rows(),
                                            .row_bytes = detectors->cols(),
                                            .bit_packed = false,
                                            .stream_offset = 0},
                                           {.workers = 2, .save_decodings = true});
    ASSERT_TRUE(result) << api::describe(result.error());
    EXPECT_EQ(read_all<std::uint8_t>(golden / "success.npy"), result->success);
    EXPECT_EQ(read_all<Bit>(golden / "decoding.npy"), result->decodings);
    const auto iterations = read_all<std::int64_t>(golden / "iterations.npy");
    EXPECT_TRUE(
        std::ranges::equal(iterations, result->iterations,
                           [](std::int64_t a, std::uint32_t b) { return std::cmp_equal(a, b); }));
    EXPECT_EQ(bits_of(read_all<double>(golden / "weight.npy")), bits_of(result->weight));
}

// ---- Sliding windows ----------------------------------------------------------------------------

class ApiWindowed : public testing::TestWithParam<SpecCase> {};

TEST_P(ApiWindowed, MatchesRtdDecodeArrayForArray) {
    const json& spec = GetParam().spec;
    const std::size_t first = 3;
    const std::size_t count = shots_to_decode - first;
    const fs::path out =
        rtd_decode("windowed_" + GetParam().name, spec, first, count,
                   {"--record-solutions", "3", "--save-commits", "--save-decodings"});
    const harness::DecoderSpec parsed_spec = parsed(spec);
    const auto problem = make_problem(parsed_spec, fixture().problem_arrays());
    for (const unsigned workers : {1U, 3U}) {
        auto decoder = api::WindowedDecoder::create(problem, parsed_spec,
                                                    {.workers = workers, .record_solutions = 3});
        ASSERT_TRUE(decoder) << api::describe(decoder.error());
        auto result = (*decoder)->decode_batch(
            batch(first, count),
            {.workers = workers, .save_decodings = true, .save_commits = true});
        ASSERT_TRUE(result) << api::describe(result.error());
        EXPECT_EQ(result->windows, (*decoder)->plan().num_positions());
        expect_same_arrays(out, *result, true, true, true);
        expect_same_confidence(out, *result, parsed_spec.selection.has_value());
    }
}

INSTANTIATE_TEST_SUITE_P(
    Bb18, ApiWindowed,
    testing::Values(
        SpecCase{"exact_commit", relay_spec(sliding_window(4, 2, "exact", "commit_anyway", 0))},
        SpecCase{"exact_defer", relay_spec(sliding_window(4, 2, "exact", "defer", 2))},
        SpecCase{"uniform_flag", relay_spec(sliding_window(5, 2, "uniform", "flag", 0))},
        SpecCase{"uniform_defer", relay_spec(sliding_window(4, 2, "uniform", "defer", 1))},
        SpecCase{"selection_defer",
                 with_selection(relay_spec(sliding_window(4, 2, "exact", "defer", 2)),
                                deferring_selection())},
        SpecCase{"selection_flag",
                 with_selection(relay_spec(sliding_window(5, 2, "uniform", "commit_anyway", 0)),
                                flagging_selection())},
        SpecCase{"int4_exact_commit",
                 with_arithmetic(relay_spec(sliding_window(4, 2, "exact", "commit_anyway", 0)),
                                 "int4.2.8")},
        SpecCase{"int5_selection_defer",
                 with_selection(
                     with_arithmetic(relay_spec(sliding_window(4, 2, "uniform", "defer", 1)),
                                     "int5.2.8"),
                     deferring_selection())}),
    case_name);

// The golden directories of the bb18 fixtures whose name starts with `prefix`.
std::vector<fs::path> goldens(std::string_view prefix) {
    std::vector<fs::path> dirs;
    for (const char* name : {"bb18_choi", "bb18_choi_r9"}) {
        for (const auto& entry : fs::directory_iterator(test::fixture_root / name)) {
            if (entry.path().filename().string().starts_with(prefix) &&
                fs::exists(entry.path() / "spec.json")) {
                dirs.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(dirs);
    return dirs;
}

std::vector<fs::path> window_goldens() { return goldens("window_"); }
std::vector<fs::path> fixed_goldens() { return goldens("fixed_"); }

class ApiWindowGolden : public testing::TestWithParam<fs::path> {};

// The window goldens were written by the Python windowing reference with relay_bp inside and
// explicit γ tables per shape, the fixed-point goldens by the integer emulator (rtd.fixed_ref)
// inside the same reference; the API must reproduce them without going through rtd_decode.
TEST_P(ApiWindowGolden, ReproducesEveryWindowRecord) {
    const fs::path& dir = GetParam();
    const json manifest = read_json(dir / "manifest.json");
    std::ifstream in(dir / "spec.json");
    std::stringstream text;
    text << in.rdbuf();
    auto spec = api::parse_spec(text.str(), dir);
    ASSERT_TRUE(spec) << api::describe(spec.error());
    auto artifact =
        io::load_artifact(repo_root / manifest.at("artifact").at("path").get<std::string>());
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    const window::ArtifactProblem arrays(*artifact);
    const window::Problem p = arrays.problem();
    const auto problem = make_problem(*spec, arrays_of(p, true));
    auto decoder = api::WindowedDecoder::create(problem, *spec, {.workers = 2});
    ASSERT_TRUE(decoder) << api::describe(decoder.error());
    auto detectors = io::read_npy<Bit>(dir / "detectors.npy", 2);
    ASSERT_TRUE(detectors);
    const std::size_t shots = std::min(detectors->rows(), shots_to_decode);
    auto result =
        (*decoder)->decode_batch({.detectors = detectors->span().first(shots * detectors->cols()),
                                  .shots = shots,
                                  .row_bytes = detectors->cols(),
                                  .bit_packed = false,
                                  .stream_offset = 0},
                                 {.workers = 2, .save_commits = true});
    ASSERT_TRUE(result) << api::describe(result.error());
    const std::size_t cells = shots * result->windows;
    const auto first_rows = [](auto values, std::size_t n) {
        values.resize(n);
        return values;
    };
    const std::size_t k = artifact->num_observables();
    EXPECT_EQ(first_rows(read_all<Bit>(dir / "predicted_observables.npy"), shots * k),
              result->predicted);
    EXPECT_EQ(first_rows(read_all<std::uint8_t>(dir / "flagged.npy"), shots), result->flagged);
    EXPECT_EQ(first_rows(read_all<std::uint32_t>(dir / "win_iterations.npy"), cells),
              result->win_iterations);
    EXPECT_EQ(first_rows(read_all<std::uint32_t>(dir / "win_legs.npy"), cells), result->win_legs);
    EXPECT_EQ(first_rows(read_all<std::uint8_t>(dir / "win_attempts.npy"), cells),
              result->win_attempts);
    EXPECT_EQ(first_rows(read_all<std::uint8_t>(dir / "win_converged.npy"), cells),
              result->win_converged);
    if (fs::exists(dir / "win_cap_hit.npy")) {
        EXPECT_EQ(first_rows(read_all<std::uint8_t>(dir / "win_cap_hit.npy"), cells),
                  result->win_cap_hit);
    }
    EXPECT_EQ(bits_of(first_rows(read_all<double>(dir / "win_weight.npy"), cells)),
              bits_of(result->win_weight));
    EXPECT_EQ(bits_of(first_rows(read_all<double>(dir / "win_committed_weight.npy"), cells)),
              bits_of(result->win_committed_weight));
    EXPECT_EQ(first_rows(read_all<std::uint32_t>(dir / "win_unexplained.npy"), cells),
              result->win_unexplained);
    EXPECT_EQ(first_rows(read_all<std::uint8_t>(dir / "win_flagged.npy"), cells),
              result->win_flagged);
    EXPECT_EQ(first_rows(read_all<std::uint32_t>(dir / "win_virtual.npy"), cells),
              result->win_virtual);
    const auto ptr = read_all<std::uint64_t>(dir / "commit_ptr.npy");
    const auto faults = read_all<std::uint32_t>(dir / "commit_faults.npy");
    ASSERT_GE(ptr.size(), cells + 1);
    EXPECT_EQ(first_rows(ptr, cells + 1), result->commit_ptr);
    EXPECT_EQ(first_rows(faults, ptr[cells]), result->commit_faults);
}

std::string golden_name(const testing::TestParamInfo<fs::path>& param_info) {
    std::string name = param_info.param.parent_path().filename().string() + "_" +
                       param_info.param.filename().string();
    std::ranges::replace(name, '.', '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Bb18, ApiWindowGolden, testing::ValuesIn(window_goldens()), golden_name);
INSTANTIATE_TEST_SUITE_P(Fixed, ApiWindowGolden, testing::ValuesIn(fixed_goldens()), golden_name);

TEST(ApiWindowGoldens, AllFifteenAreFound) { EXPECT_EQ(window_goldens().size(), 15U); }
TEST(ApiWindowGoldens, AllFiveFixedPointGoldensAreFound) {
    EXPECT_EQ(fixed_goldens().size(), 5U);
}

// ---- Streams ------------------------------------------------------------------------------------

TEST(ApiStream, RoundByRoundCommitsWhatTheBatchCommits) {
    const harness::DecoderSpec spec = parsed(relay_spec(sliding_window(4, 2, "exact", "defer", 2)));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder =
        api::WindowedDecoder::create(problem, spec, {.workers = 1, .record_solutions = 2});
    ASSERT_TRUE(decoder);
    const std::size_t count = std::min<std::size_t>(shots_to_decode, 30);
    auto expected = (*decoder)->decode_batch(batch(0, count), {.save_commits = true});
    ASSERT_TRUE(expected);
    auto stream = (*decoder)->open_stream();
    ASSERT_TRUE(stream) << api::describe(stream.error());
    const window::WindowPlan& plan = (*decoder)->plan();
    const std::size_t per_round = plan.detectors_per_round();
    const std::size_t rounds = plan.rounds_total();
    const std::size_t windows = plan.num_positions();
    const std::size_t k = problem->num_observables();
    for (std::size_t s = 0; s < count; ++s) {
        api::Stream& st = **stream;
        st.reset(s);
        const std::span<const Bit> shot = fixture().detectors.row(s);
        std::size_t decoded = 0;
        // Decode every window as soon as its rounds have arrived.
        for (std::size_t r = 0; r < rounds; ++r) {
            const auto bits = shot.subspan(r * per_round, per_round);
            ASSERT_TRUE(r + 1 < rounds ? st.push_round(bits) : st.push_final(bits));
            EXPECT_EQ(st.rounds_received(), r + 1);
            while (st.window_ready()) {
                auto commit = st.decode_next();
                ASSERT_TRUE(commit) << api::describe(commit.error());
                if (commit->deferred) {
                    continue;
                }
                const std::size_t cell = (s * windows) + commit->window;
                const auto begin = static_cast<std::ptrdiff_t>(expected->commit_ptr[cell]);
                const auto end = static_cast<std::ptrdiff_t>(expected->commit_ptr[cell + 1]);
                EXPECT_TRUE(std::ranges::equal(commit->faults,
                                               std::span(expected->commit_faults.begin() + begin,
                                                         expected->commit_faults.begin() + end)))
                    << "shot " << s << " window " << commit->window;
                EXPECT_EQ(commit->record.iterations, expected->win_iterations[cell]);
                EXPECT_EQ(commit->record.attempts, expected->win_attempts[cell]);
                EXPECT_EQ(commit->solutions_found, expected->sol_count[cell]);
                ++decoded;
            }
        }
        EXPECT_TRUE(st.finished());
        EXPECT_TRUE(st.closed());
        EXPECT_EQ(decoded, windows);
        EXPECT_TRUE(
            std::ranges::equal(st.predicted(), std::span(expected->predicted).subspan(s * k, k)));
        EXPECT_TRUE(std::ranges::equal(st.frame(), st.predicted()));
        EXPECT_EQ(st.flagged(), expected->flagged[s] != 0);
        EXPECT_EQ(st.summary().iterations, expected->iterations[s]);
        EXPECT_EQ(st.records().size(), windows);
    }
}

// Under a selection policy with a history, a stream reports what a batch records: each committed
// window's confidence, its low-confidence deferrals, and after each window the history values.
TEST(ApiStream, ConfidenceAndHistoryMatchTheBatch) {
    const harness::DecoderSpec spec = parsed(with_selection(
        relay_spec(sliding_window(4, 2, "exact", "defer", 2)), deferring_selection()));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder = api::WindowedDecoder::create(problem, spec, {.workers = 1});
    ASSERT_TRUE(decoder);
    const std::size_t count = std::min<std::size_t>(shots_to_decode, 20);
    auto expected = (*decoder)->decode_batch(batch(0, count), {});
    ASSERT_TRUE(expected);
    const harness::ConfidenceOutputs& c = expected->confidence;
    ASSERT_TRUE(c.enabled);
    auto stream = (*decoder)->open_stream();
    ASSERT_TRUE(stream);
    api::Stream& st = **stream;
    const window::WindowPlan& plan = (*decoder)->plan();
    const std::size_t per_round = plan.detectors_per_round();
    const std::size_t rounds = plan.rounds_total();
    const std::size_t windows = plan.num_positions();
    const std::size_t signals = c.history_signals.size();
    const std::size_t lengths = c.history_lengths;
    ASSERT_NE(st.history(), nullptr);
    std::size_t deferred = 0;
    for (std::size_t s = 0; s < count; ++s) {
        st.reset(s);
        const std::span<const Bit> shot = fixture().detectors.row(s);
        for (std::size_t r = 0; r < rounds; ++r) {
            const auto bits = shot.subspan(r * per_round, per_round);
            ASSERT_TRUE(r + 1 < rounds ? st.push_round(bits) : st.push_final(bits));
            while (st.window_ready()) {
                auto commit = st.decode_next();
                ASSERT_TRUE(commit) << api::describe(commit.error());
                if (commit->deferred) {
                    ++deferred;
                    continue;
                }
                const std::size_t cell = (s * windows) + commit->window;
                const window::WindowRecord& record = commit->record;
                EXPECT_EQ(record.confidence.has_value(), c.decoded[cell] != 0);
                if (record.confidence) {
                    EXPECT_EQ(record.confidence->low, c.low[cell] != 0);
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(record.confidence->gap),
                              std::bit_cast<std::uint64_t>(c.gap[cell]));
                }
                EXPECT_EQ(record.low_confidence_deferrals, c.low_deferrals[cell]);
                const window::SignalHistory* history = st.history();
                ASSERT_NE(history, nullptr);
                for (std::size_t g = 0; g < signals; ++g) {
                    for (std::size_t l = 0; l < lengths; ++l) {
                        EXPECT_EQ(std::bit_cast<std::uint64_t>(history->value(g, l)),
                                  std::bit_cast<std::uint64_t>(
                                      c.history[(((cell * signals) + g) * lengths) + l]))
                            << "shot " << s << " window " << commit->window;
                    }
                }
            }
        }
        EXPECT_TRUE(std::ranges::equal(st.predicted(),
                                       std::span(expected->predicted)
                                           .subspan(s * problem->num_observables(),
                                                    problem->num_observables())));
    }
    // A window deferred for low confidence returns while its wider attempt's rounds are missing.
    EXPECT_GT(deferred, 0U);
}

TEST(ApiStream, RefusesCallsOutOfOrder) {
    const harness::DecoderSpec spec =
        parsed(relay_spec(sliding_window(4, 2, "exact", "commit_anyway", 0)));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder = api::WindowedDecoder::create(problem, spec, {.workers = 1});
    ASSERT_TRUE(decoder);
    auto opened = (*decoder)->open_stream();
    ASSERT_TRUE(opened);
    api::Stream& stream = **opened;
    const window::WindowPlan& plan = (*decoder)->plan();
    const std::size_t per_round = plan.detectors_per_round();
    const std::vector<Bit> zeros(per_round, 0);
    stream.reset(0);
    EXPECT_FALSE(stream.window_ready());
    auto early = stream.decode_next();
    ASSERT_FALSE(early);
    EXPECT_EQ(early.error().code, api::ApiError::Code::stream_state);
    auto short_round = stream.push_round(std::span(zeros).first(per_round - 1));
    ASSERT_FALSE(short_round);
    EXPECT_EQ(short_round.error().code, api::ApiError::Code::invalid_input);
    auto early_final = stream.push_final(zeros);
    ASSERT_FALSE(early_final);
    EXPECT_EQ(early_final.error().code, api::ApiError::Code::stream_state);
    for (std::uint32_t r = 1; r < plan.rounds_total(); ++r) {
        ASSERT_TRUE(stream.push_round(zeros));
    }
    auto readout_as_round = stream.push_round(zeros);
    ASSERT_FALSE(readout_as_round);
    EXPECT_EQ(readout_as_round.error().code, api::ApiError::Code::stream_state);
    ASSERT_TRUE(stream.push_final(zeros));
    auto after_close = stream.push_round(zeros);
    ASSERT_FALSE(after_close);
    EXPECT_EQ(after_close.error().code, api::ApiError::Code::stream_state);
    while (!stream.finished()) {
        auto commit = stream.decode_next();
        ASSERT_TRUE(commit);
        EXPECT_TRUE(commit->faults.empty()); // σ = 0 has the solution ê = 0 in every window
    }
    EXPECT_FALSE(stream.window_ready());
    auto done = stream.decode_next();
    ASSERT_FALSE(done);
    EXPECT_EQ(done.error().code, api::ApiError::Code::stream_state);
    // A stream keeps what it needs: it outlives its decoder.
    const std::uint32_t rounds = plan.rounds_total();
    decoder->reset();
    stream.reset(1);
    for (std::uint32_t r = 1; r < rounds; ++r) {
        ASSERT_TRUE(stream.push_round(zeros));
    }
    ASSERT_TRUE(stream.push_final(zeros));
    while (!stream.finished()) {
        ASSERT_TRUE(stream.decode_next());
    }
}

// ---- Biases -------------------------------------------------------------------------------------

// A problem whose certain faults were removed decodes σ ⊕ b_σ and reports ℓ̂ ⊕ b_ℓ: with the bias
// applied by hand to the shots, it must give the unbiased problem's decode, flipped by b_ℓ.
TEST(ApiBias, IsAppliedToTheSyndromeAndThePrediction) {
    const Fixture& f = fixture();
    const std::size_t m = f.artifact->num_detectors();
    const std::size_t k = f.artifact->num_observables();
    std::vector<Bit> syndrome_bias(m, 0);
    for (std::size_t i = 0; i < m; i += 7) {
        syndrome_bias[i] = 1;
    }
    std::vector<Bit> observables_bias(k, 0);
    observables_bias.at(1) = 1;
    const std::size_t count = std::min<std::size_t>(shots_to_decode, 20);
    std::vector<Bit> shifted(f.detectors.span().begin(),
                             f.detectors.span().begin() + static_cast<std::ptrdiff_t>(count * m));
    for (std::size_t s = 0; s < count; ++s) {
        for (std::size_t i = 0; i < m; ++i) {
            shifted[(s * m) + i] ^= syndrome_bias[i];
        }
    }
    const api::BatchInput biased_input{
        .detectors = shifted, .shots = count, .row_bytes = m, .bit_packed = false};

    for (const json& window :
         {json{{"mode", "whole_shot"}}, sliding_window(4, 2, "exact", "commit_anyway", 0)}) {
        const harness::DecoderSpec spec = parsed(relay_spec(window));
        const auto plain = make_problem(spec, f.problem_arrays());
        const auto biased = make_problem(spec, f.problem_arrays(syndrome_bias, observables_bias));
        std::vector<Bit> expected;
        std::vector<Bit> got;
        if (spec.window.is_sliding()) {
            auto a = api::WindowedDecoder::create(plain, spec, {});
            auto b = api::WindowedDecoder::create(biased, spec, {});
            ASSERT_TRUE(a && b);
            expected = (*a)->decode_batch(batch(0, count), {})->predicted;
            got = (*b)->decode_batch(biased_input, {})->predicted;
            // The stream applies the same biases round by round.
            auto stream = (*b)->open_stream();
            ASSERT_TRUE(stream);
            const std::size_t per_round = (*b)->plan().detectors_per_round();
            const std::size_t rounds = (*b)->plan().rounds_total();
            for (std::size_t s = 0; s < count; ++s) {
                (*stream)->reset(s);
                for (std::size_t r = 0; r < rounds; ++r) {
                    const auto bits =
                        std::span<const Bit>(shifted).subspan((s * m) + (r * per_round), per_round);
                    ASSERT_TRUE(r + 1 < rounds ? (*stream)->push_round(bits)
                                               : (*stream)->push_final(bits));
                }
                while (!(*stream)->finished()) {
                    ASSERT_TRUE((*stream)->decode_next());
                }
                EXPECT_TRUE(
                    std::ranges::equal((*stream)->predicted(), std::span(got).subspan(s * k, k)));
            }
        } else {
            auto a = api::WholeShotDecoder::create(plain, spec, {});
            auto b = api::WholeShotDecoder::create(biased, spec, {});
            ASSERT_TRUE(a && b);
            expected = (*a)->decode_batch(batch(0, count), {})->predicted;
            got = (*b)->decode_batch(biased_input, {})->predicted;
        }
        ASSERT_EQ(got.size(), expected.size());
        for (std::size_t s = 0; s < count; ++s) {
            for (std::size_t o = 0; o < k; ++o) {
                EXPECT_EQ(got[(s * k) + o], expected[(s * k) + o] ^ observables_bias[o])
                    << "shot " << s << " observable " << o;
            }
        }
    }
}

// ---- Refusals and concurrency -------------------------------------------------------------------

TEST(ApiBatch, RejectsMisshapenBatchesAndOptions) {
    const harness::DecoderSpec spec = parsed(relay_spec({{"mode", "whole_shot"}}));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder = api::WholeShotDecoder::create(problem, spec, {.workers = 2});
    ASSERT_TRUE(decoder);
    const auto code_of = [&](const api::BatchInput& input, const api::BatchOptions& options) {
        auto result = (*decoder)->decode_batch(input, options);
        return result ? std::optional<api::ApiError::Code>{} : result.error().code;
    };
    const api::BatchInput good = batch(0, 4);
    EXPECT_FALSE(code_of(good, {}).has_value());
    api::BatchInput empty = good;
    empty.shots = 0;
    empty.detectors = {};
    EXPECT_EQ(code_of(empty, {}), api::ApiError::Code::invalid_input);
    api::BatchInput narrow = good;
    narrow.row_bytes -= 1;
    EXPECT_EQ(code_of(narrow, {}), api::ApiError::Code::invalid_input);
    api::BatchInput short_buffer = good;
    short_buffer.detectors = good.detectors.first(good.detectors.size() - 1);
    EXPECT_EQ(code_of(short_buffer, {}), api::ApiError::Code::invalid_input);
    api::BatchInput packed_width = good;
    packed_width.bit_packed = true;
    EXPECT_EQ(code_of(packed_width, {}), api::ApiError::Code::invalid_input);
    EXPECT_EQ(code_of(good, {.workers = 0}), api::ApiError::Code::invalid_input);
    EXPECT_EQ(code_of(good, {.workers = 3}), api::ApiError::Code::invalid_input);
    EXPECT_EQ(code_of(good, {.save_commits = true}), api::ApiError::Code::invalid_input);
}

TEST(ApiDecoders, RefuseTheWrongModeAndOptions) {
    const harness::DecoderSpec whole = parsed(relay_spec({{"mode", "whole_shot"}}));
    const harness::DecoderSpec sliding =
        parsed(relay_spec(sliding_window(4, 2, "exact", "commit_anyway", 0)));
    const auto problem = make_problem(whole, fixture().problem_arrays());
    const auto no_rounds = make_problem(whole, fixture().problem_arrays({}, {}, false));
    const auto code_of = [](const auto& made) {
        return made ? std::optional<api::ApiError::Code>{} : made.error().code;
    };
    EXPECT_EQ(code_of(api::WholeShotDecoder::create(problem, sliding, {})),
              api::ApiError::Code::invalid_spec);
    EXPECT_EQ(code_of(api::WindowedDecoder::create(problem, whole, {})),
              api::ApiError::Code::invalid_spec);
    EXPECT_EQ(code_of(api::WindowedDecoder::create(no_rounds, sliding, {})),
              api::ApiError::Code::invalid_problem);
    EXPECT_EQ(code_of(api::WholeShotDecoder::create(problem, whole, {.workers = 0})),
              api::ApiError::Code::invalid_input);
    EXPECT_EQ(code_of(api::WholeShotDecoder::create(problem, whole, {.record_solutions = 21})),
              api::ApiError::Code::invalid_input);
    EXPECT_EQ(code_of(api::WholeShotDecoder::create(nullptr, whole, {})),
              api::ApiError::Code::invalid_problem);
    // A window at least as wide as the shot under the uniform boundary has no bulk window.
    const harness::DecoderSpec too_wide =
        parsed(relay_spec(sliding_window(12, 2, "uniform", "commit_anyway", 0)));
    EXPECT_EQ(code_of(api::WindowedDecoder::create(problem, too_wide, {})),
              api::ApiError::Code::plan_rejected);
    // Explicit shape tables that do not exist.
    json missing_tables = relay_spec(sliding_window(4, 2, "exact", "commit_anyway", 0));
    missing_tables["gamma_source"] = {{"type", "explicit_shapes"}, {"directory", "no_such_dir"}};
    EXPECT_EQ(code_of(api::WindowedDecoder::create(problem, parsed(missing_tables), {})),
              api::ApiError::Code::gamma_source);
}

TEST(ApiConcurrency, OverlappingBatchesOnOneDecoderAgree) {
    const harness::DecoderSpec spec = parsed(relay_spec(sliding_window(5, 2, "exact", "flag", 0)));
    const auto problem = make_problem(spec, fixture().problem_arrays());
    auto decoder = api::WindowedDecoder::create(problem, spec, {.workers = 2});
    ASSERT_TRUE(decoder);
    constexpr std::size_t count = std::min<std::size_t>(shots_to_decode, 16);
    auto reference = (*decoder)->decode_batch(batch(0, count), {.workers = 2});
    ASSERT_TRUE(reference);
    std::vector<std::expected<api::BatchResult, api::ApiError>> results(3);
    {
        std::vector<std::jthread> callers;
        callers.reserve(results.size());
        for (auto& slot : results) {
            callers.emplace_back([&decoder, &slot] {
                slot = (*decoder)->decode_batch(batch(0, count), {.workers = 2});
            });
        }
    }
    for (const auto& result : results) {
        ASSERT_TRUE(result);
        EXPECT_EQ(result->predicted, reference->predicted);
        EXPECT_EQ(result->win_iterations, reference->win_iterations);
    }
}

TEST(ApiErrors, NameTheirCode) {
    EXPECT_EQ(api::to_string(api::ApiError::Code::stream_state), "stream_state");
    EXPECT_EQ(api::describe({.code = api::ApiError::Code::gamma_source, .detail = "x"}),
              "gamma_source: x");
    constexpr auto last = static_cast<std::uint8_t>(api::ApiError::Code::stream_state);
    for (auto code = std::uint8_t{0}; code <= last; ++code) {
        EXPECT_NE(api::to_string(static_cast<api::ApiError::Code>(code)), "unknown");
    }
}

} // namespace
