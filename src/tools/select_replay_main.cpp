// rtd_select_replay: replays the solutions an rtd_decode run recorded (--record-solutions N
// --save-solution-supports) through the selection policy's rules, decode by decode, as if the run
// had kept only the first S of them and had stopped by a given rule. It writes every confidence
// value and the stopped run's outcome per (shot, window), so that another implementation of the
// same rules can be compared with it value for value.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/selection.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"

namespace {

using namespace rtd;
using harness::JsonLogger;
using harness::Level;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

enum class ExitCode : std::uint8_t {
    ok = 0,
    internal_error = 1,
    usage_error = 2,
    input_error = 3,
    output_error = 4,
};
using enum ExitCode;

constexpr std::string_view usage =
    R"(usage: rtd_select_replay --artifact DIR --run DIR --slots S --out DIR [options]

required:
  --artifact DIR      the artifact the run decoded
  --run DIR           an rtd_decode output with sol_*.npy, solsup_*.npy and run.json
  --slots S           replay the first S recorded solutions of each decode (1 <= S <= N)
  --out DIR           output directory (created; must be empty unless --overwrite)

options:
  --stop RULE         fixed (default) | agree | agree_distinct | gap | gap_extend
  --count M           m of agree / agree_distinct, n0 of gap_extend
  --threshold T       t of gap / gap_extend
  --no-verify         skip SHA-256 verification of the artifact
  --overwrite         allow a non-empty --out directory
  --bench P           also time P passes of the decoder's own path over every decode (each
                      solution offered with its support, as a decode offers it, then the
                      confidence computed) and report ns per solution against the run's
                      decode time
  --log-level L       debug | info | warn | error (default info)
  --help              this text

writes rep_<value>.npy [shots, windows] for the values of the first S slots (seen, state,
weight, best_class, gap, second_class, distinct, agreement, first_legs, first_iterations,
class_sum_class, class_sum_top, agreement_class, q_supp, q_sum_sq, q_total, components) and the
stopped run's outcome (stop_solutions, stop_decision, stop_iterations, stop_legs, stop_by_rule),
plus replay.json. Windows that ran no decode keep empty values (state 0, weight +inf, NaN).
)";

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path run;
    std::filesystem::path out;
    std::uint32_t slots = 0;
    StopRule stop = StopRule::fixed;
    std::uint32_t count = 0;
    double threshold = 0.0;
    bool verify = true;
    bool overwrite = false;
    Level log_level = Level::info;
    bool help = false;
    std::uint32_t bench = 0;
};

std::expected<std::uint32_t, std::string> parse_count(std::string_view flag,
                                                      std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        return std::unexpected(
            std::format("{} expects a non-negative integer, got '{}'", flag, text));
    }
    return value;
}

std::expected<double, std::string> parse_number(std::string_view flag, std::string_view text) {
    double value = 0.0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        return std::unexpected(std::format("{} expects a number, got '{}'", flag, text));
    }
    return value;
}

std::expected<void, std::string> apply(Arguments& args, std::string_view flag,
                                       std::string_view text) {
    const auto set_count = [&](std::uint32_t& target) -> std::expected<void, std::string> {
        return parse_count(flag, text).transform([&](std::uint32_t v) { target = v; });
    };
    if (flag == "--artifact") {
        args.artifact = text;
    } else if (flag == "--run") {
        args.run = text;
    } else if (flag == "--out") {
        args.out = text;
    } else if (flag == "--slots") {
        return set_count(args.slots);
    } else if (flag == "--bench") {
        return set_count(args.bench);
    } else if (flag == "--count") {
        return set_count(args.count);
    } else if (flag == "--threshold") {
        return parse_number(flag, text).transform([&](double v) { args.threshold = v; });
    } else if (flag == "--stop") {
        const auto rule = parse_stop_rule(text);
        if (!rule) {
            return std::unexpected(std::format("unknown stop rule '{}'", text));
        }
        args.stop = *rule;
    } else if (flag == "--log-level") {
        const auto level = harness::parse_level(text);
        if (!level) {
            return std::unexpected(std::format("unknown log level '{}'", text));
        }
        args.log_level = *level;
    } else {
        return std::unexpected(std::format("unknown option {}", flag));
    }
    return {};
}

std::expected<Arguments, std::string> parse_arguments(std::span<char* const> argv) {
    Arguments args;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            args.help = true;
        } else if (flag == "--no-verify") {
            args.verify = false;
        } else if (flag == "--overwrite") {
            args.overwrite = true;
        } else if (i + 1 >= argv.size()) {
            return std::unexpected(std::format("{} needs a value", flag));
        } else if (auto applied = apply(args, flag, argv[++i]); !applied) {
            return std::unexpected(applied.error());
        }
    }
    if (args.help) {
        return args;
    }
    if (args.artifact.empty() || args.run.empty() || args.out.empty() || args.slots == 0) {
        return std::unexpected("--artifact, --run, --slots and --out are required");
    }
    return args;
}

// The recorded arrays of the run.
struct Recording {
    std::size_t shots = 0;
    std::size_t windows = 0;
    std::size_t slots = 0;
    io::NpyArray<std::uint32_t> count, leg, iterations, size;
    io::NpyArray<double> weight;
    io::NpyArray<std::uint64_t> klass, hash, returned;
    io::NpyArray<std::uint64_t> sup_ptr;
    io::NpyArray<std::uint32_t> sup_idx;
    std::optional<io::NpyArray<std::uint32_t>> win_iterations, win_legs;
    std::optional<io::NpyArray<std::uint8_t>> win_attempts;
    io::NpyArray<std::uint32_t> shot_iterations, shot_legs;
};

template <class T>
std::expected<io::NpyArray<T>, std::string> load(const std::filesystem::path& dir,
                                                 std::string_view name) {
    auto array = io::read_npy<T>(dir / std::format("{}.npy", name));
    if (!array) {
        return std::unexpected(io::describe(array.error()));
    }
    return std::move(*array);
}

std::expected<Recording, std::string> load_recording(const std::filesystem::path& dir,
                                                     bool sliding) {
    Recording r;
    std::string problem;
    const auto get = [&]<class T>(io::NpyArray<T>& into, std::string_view name) {
        if (!problem.empty()) {
            return;
        }
        auto array = load<T>(dir, name);
        if (!array) {
            problem = array.error();
            return;
        }
        into = std::move(*array);
    };
    get(r.count, "sol_count");
    get(r.leg, "sol_leg");
    get(r.iterations, "sol_iterations");
    get(r.size, "sol_size");
    get(r.weight, "sol_weight");
    get(r.klass, "sol_class");
    get(r.hash, "sol_hash");
    get(r.returned, "returned_class");
    get(r.sup_ptr, "solsup_ptr");
    get(r.sup_idx, "solsup_idx");
    get(r.shot_iterations, "iterations");
    get(r.shot_legs, "legs");
    if (sliding && problem.empty()) {
        io::NpyArray<std::uint32_t> it;
        io::NpyArray<std::uint32_t> lg;
        io::NpyArray<std::uint8_t> at;
        get(it, "win_iterations");
        get(lg, "win_legs");
        get(at, "win_attempts");
        r.win_iterations = std::move(it);
        r.win_legs = std::move(lg);
        r.win_attempts = std::move(at);
    }
    if (!problem.empty()) {
        return std::unexpected(problem);
    }
    if (r.leg.shape.size() != 3 || r.count.shape.size() != 2) {
        return std::unexpected("sol_leg must be [S, K, N] and sol_count [S, K]");
    }
    r.shots = r.leg.shape[0];
    r.windows = r.leg.shape[1];
    r.slots = r.leg.shape[2];
    if (r.sup_ptr.size() != (r.shots * r.windows * r.slots) + 1) {
        return std::unexpected("solsup_ptr does not have S·K·N + 1 entries");
    }
    return r;
}

// Everything the rules need for one decoding problem (the whole shot, or one window shape).
// shape_of value of a window position that ran no decode.
constexpr std::uint32_t no_shape = std::numeric_limits<std::uint32_t>::max();

struct ProblemView {
    std::span<const std::uint64_t> classes;
    std::span<const double> llr;
    const TannerGraph* graph = nullptr;
};

struct Outputs {
    std::vector<std::uint32_t> seen, distinct, first_legs, first_iterations, components;
    std::vector<std::uint8_t> state, by_rule;
    std::vector<double> weight, gap, agreement, class_sum_top, q_supp, q_sum_sq, q_total;
    std::vector<std::uint64_t> best_class, second_class, class_sum_class, agreement_class;
    std::vector<std::uint32_t> stop_solutions, stop_iterations, stop_legs;
    std::vector<std::uint64_t> stop_decision;

    explicit Outputs(std::size_t cells)
        : seen(cells), distinct(cells), first_legs(cells), first_iterations(cells),
          components(cells), state(cells), by_rule(cells), weight(cells), gap(cells),
          agreement(cells), class_sum_top(cells), q_supp(cells), q_sum_sq(cells), q_total(cells),
          best_class(cells), second_class(cells), class_sum_class(cells), agreement_class(cells),
          stop_solutions(cells), stop_iterations(cells), stop_legs(cells), stop_decision(cells) {}
};

// Replays one decode: the stopping rule on the first S slots in order (it fires after slot q or
// never), then every value over the first S slots.
void replay_cell(const Recording& rec, std::size_t cell, SelectionState& state,
                 std::uint32_t decode_iterations, std::uint32_t decode_legs, std::uint32_t s_max,
                 Outputs& out) {
    const std::size_t base = cell * rec.slots;
    const std::uint32_t found = rec.count.span()[cell];
    const auto stored = static_cast<std::uint32_t>(std::min<std::size_t>(found, rec.slots));
    const std::uint64_t returned = rec.returned.span()[cell];
    state.begin();
    std::optional<std::uint32_t> fired;
    std::uint64_t fired_class = 0;
    // The prefix's lowest-weight class (strict <, so the earliest of equal weights).
    double prefix_weight = std::numeric_limits<double>::infinity();
    std::uint64_t prefix_class = returned;
    for (std::uint32_t q = 0; q < stored; ++q) {
        const std::size_t slot = base + q;
        if (rec.weight.span()[slot] < prefix_weight) {
            prefix_weight = rec.weight.span()[slot];
            prefix_class = rec.klass.span()[slot];
        }
        const SolutionRecord record{.leg = rec.leg.span()[slot],
                                    .cumulative_iterations = rec.iterations.span()[slot],
                                    .weight = rec.weight.span()[slot],
                                    .logical_class = rec.klass.span()[slot],
                                    .hash = rec.hash.span()[slot],
                                    .size = rec.size.span()[slot]};
        const std::uint64_t begin = rec.sup_ptr.span()[slot];
        const std::uint64_t end = rec.sup_ptr.span()[slot + 1];
        state.add_record(record, rec.sup_idx.span().subspan(begin, end - begin));
        if (!fired && q < s_max && state.stop_requested()) {
            fired = q;
            fired_class = prefix_class;
        }
    }
    const Confidence c = state.finish(DecodeFacts{}).confidence;
    const bool none = c.gap_state == GapState::none;
    out.seen[cell] = c.seen;
    out.state[cell] = static_cast<std::uint8_t>(c.gap_state);
    out.weight[cell] = c.weight;
    out.best_class[cell] = none ? returned : c.best_class;
    out.gap[cell] = c.gap;
    out.second_class[cell] = c.second_class;
    out.distinct[cell] = c.distinct;
    out.agreement[cell] = c.agreement;
    out.first_legs[cell] = c.first_legs;
    out.first_iterations[cell] = c.first_iterations;
    out.class_sum_class[cell] = none ? returned : c.class_sum_class;
    out.class_sum_top[cell] = c.class_sum_top;
    out.agreement_class[cell] = none ? returned : c.agreement_class;
    out.q_supp[cell] = c.q_supp;
    out.q_sum_sq[cell] = c.q_sum_sq;
    out.q_total[cell] = c.q_total;
    out.components[cell] = c.components;
    if (fired) {
        const std::size_t slot = base + *fired;
        out.stop_solutions[cell] = *fired + 1;
        out.stop_decision[cell] = fired_class;
        out.stop_iterations[cell] = rec.iterations.span()[slot];
        out.stop_legs[cell] = rec.leg.span()[slot] + 1;
        out.by_rule[cell] = 1;
    } else if (found >= s_max) {
        // The run stops at its solution cap: the prefix of s_max slots.
        const std::size_t slot = base + s_max - 1;
        out.stop_solutions[cell] = s_max;
        out.stop_decision[cell] = c.best_class;
        out.stop_iterations[cell] = rec.iterations.span()[slot];
        out.stop_legs[cell] = rec.leg.span()[slot] + 1;
    } else {
        out.stop_solutions[cell] = found;
        out.stop_decision[cell] = none ? returned : c.best_class;
        out.stop_iterations[cell] = decode_iterations;
        out.stop_legs[cell] = decode_legs;
    }
}

std::expected<void, io::IoError> write_outputs(const std::filesystem::path& dir, const Outputs& out,
                                               std::size_t shots, std::size_t windows) {
    const std::array<std::size_t, 2> dims{shots, windows};
    std::expected<void, io::IoError> ok;
    const auto write = [&]<class T>(std::string_view name, const std::vector<T>& data) {
        if (ok) {
            ok = io::write_npy<T>(dir / std::format("rep_{}.npy", name), std::span<const T>(data),
                                  dims);
        }
    };
    write("seen", out.seen);
    write("state", out.state);
    write("weight", out.weight);
    write("best_class", out.best_class);
    write("gap", out.gap);
    write("second_class", out.second_class);
    write("distinct", out.distinct);
    write("agreement", out.agreement);
    write("first_legs", out.first_legs);
    write("first_iterations", out.first_iterations);
    write("class_sum_class", out.class_sum_class);
    write("class_sum_top", out.class_sum_top);
    write("agreement_class", out.agreement_class);
    write("q_supp", out.q_supp);
    write("q_sum_sq", out.q_sum_sq);
    write("q_total", out.q_total);
    write("components", out.components);
    write("stop_solutions", out.stop_solutions);
    write("stop_decision", out.stop_decision);
    write("stop_iterations", out.stop_iterations);
    write("stop_legs", out.stop_legs);
    write("stop_by_rule", out.by_rule);
    return ok;
}

// Times the online path of the selection rules on the recorded decodes: every solution offered as a
// SolutionEvent (class and hash computed from its support, as during a decode), then finish()
// (the confidence value, Q⁽²⁾ included). Decodes whose supports were not all saved are skipped.
struct BenchResult {
    std::uint64_t decodes = 0;
    std::uint64_t solutions = 0;
    double seconds = 0.0;
    double checksum = 0.0;
};

BenchResult bench_policy(const Recording& rec, std::vector<SelectionState>& states,
                         std::span<const std::uint32_t> shape_of, std::uint32_t slots,
                         std::uint32_t passes) {
    BenchResult r;
    std::vector<std::size_t> cells;
    for (std::size_t cell = 0; cell < rec.shots * rec.windows; ++cell) {
        if (shape_of[cell] == no_shape) {
            continue;
        }
        const std::size_t base = cell * rec.slots;
        const auto stored = static_cast<std::uint32_t>(
            std::min<std::size_t>({rec.count.span()[cell], rec.slots, slots}));
        bool complete = true;
        for (std::uint32_t q = 0; q < stored; ++q) {
            const std::size_t slot = base + q;
            complete = complete && rec.sup_ptr.span()[slot + 1] - rec.sup_ptr.span()[slot] ==
                                       rec.size.span()[slot];
        }
        if (complete) {
            cells.push_back(cell);
        }
    }
    const auto started = Clock::now();
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        for (const std::size_t cell : cells) {
            SelectionState& state = states[shape_of[cell]];
            const std::size_t base = cell * rec.slots;
            const auto stored = static_cast<std::uint32_t>(
                std::min<std::size_t>({rec.count.span()[cell], rec.slots, slots}));
            state.begin();
            for (std::uint32_t q = 0; q < stored; ++q) {
                const std::size_t slot = base + q;
                const std::uint64_t begin = rec.sup_ptr.span()[slot];
                const std::uint64_t end = rec.sup_ptr.span()[slot + 1];
                state.add(SolutionEvent{.leg = rec.leg.span()[slot],
                                        .cumulative_iterations = rec.iterations.span()[slot],
                                        .weight = rec.weight.span()[slot],
                                        .support = rec.sup_idx.span().subspan(begin, end - begin)});
                if (state.stop_requested()) {
                    r.checksum += 1.0;
                }
            }
            DecodeFacts facts;
            facts.success = stored > 0;
            const Confidence c = state.finish(facts).confidence;
            r.checksum +=
                static_cast<double>(c.distinct) + (std::isfinite(c.weight) ? c.weight : 0.0);
            r.solutions += stored;
            ++r.decodes;
        }
    }
    r.seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return r;
}

// The decoder spec the recording run used, from its run.json.
std::optional<harness::DecoderSpec> read_spec(const std::filesystem::path& run,
                                              JsonLogger& logger) {
    std::ifstream run_file(run / "run.json");
    if (!run_file) {
        logger.error("input", "cannot open run.json", "missing file", {{"run", run.string()}});
        return std::nullopt;
    }
    json run_record;
    try {
        run_record = json::parse(run_file);
    } catch (const json::exception& e) {
        logger.error("input", "run.json is not valid JSON", e.what(), {{"run", run.string()}});
        return std::nullopt;
    }
    if (!run_record.contains("decoder")) {
        logger.error("input", "run.json has no decoder spec", "missing field",
                     {{"run", run.string()}});
        return std::nullopt;
    }
    auto spec = harness::parse_spec(run_record.at("decoder"), run);
    if (!spec) {
        logger.error("input", "the run's decoder spec does not parse", spec.error().problem,
                     {{"field", spec.error().field}});
        return std::nullopt;
    }
    return std::move(*spec);
}

// The decoding problems of a run: the whole artifact, or every window shape of its plan. Views
// point into the other members, so an instance is filled in place and never moved.
struct Problems {
    std::vector<std::uint64_t> whole_classes;
    std::unique_ptr<window::ArtifactProblem> source;
    std::optional<window::WindowPlan> plan;
    std::vector<ProblemView> views;
};

ExitCode build_problems(const io::Artifact& artifact, const harness::DecoderSpec& spec,
                        const Recording& rec, Problems& out, JsonLogger& logger) {
    if (!spec.window.is_sliding()) {
        auto classes = harness::column_classes(artifact.observables);
        if (!classes) {
            logger.error("artifact", "cannot pack the observables", classes.error());
            return input_error;
        }
        out.whole_classes = std::move(*classes);
        out.views.push_back(ProblemView{
            .classes = out.whole_classes, .llr = artifact.priors.llr(), .graph = &artifact.graph});
        return ok;
    }
    out.source = std::make_unique<window::ArtifactProblem>(artifact);
    auto built = window::WindowPlan::build(out.source->problem(), spec.window.sliding, spec.graph);
    if (!built) {
        logger.error("plan", "cannot rebuild the run's window plan",
                     window::describe(built.error()));
        return input_error;
    }
    const window::WindowPlan& plan = out.plan.emplace(std::move(*built));
    for (const window::Shape& shape : plan.shapes()) {
        out.views.push_back(ProblemView{
            .classes = shape.commit_class(), .llr = shape.priors().llr(), .graph = &shape.graph()});
    }
    if (plan.num_positions() != rec.windows) {
        logger.error("input", "the recording's window count differs from the plan's",
                     "shape mismatch", {{"recorded", rec.windows}, {"plan", plan.num_positions()}});
        return input_error;
    }
    return ok;
}

// Empty values for a window position that ran no decode.
void clear_cell(std::size_t cell, Outputs& out) noexcept {
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    out.state[cell] = 0;
    out.weight[cell] = std::numeric_limits<double>::infinity();
    out.gap[cell] = nan;
    out.agreement[cell] = nan;
    out.class_sum_top[cell] = nan;
    out.q_supp[cell] = nan;
    out.q_sum_sq[cell] = nan;
    out.q_total[cell] = nan;
}

// Replays every recorded decode. shape_of[cell] is the problem each cell was decoded on, or
// no_shape for a window position that ran no decode (those are counted in `skipped`).
ExitCode replay_all(const Recording& rec, const Problems& problems,
                    std::vector<SelectionState>& states, std::uint32_t slots, Outputs& out,
                    std::vector<std::uint32_t>& shape_of, std::size_t& skipped,
                    JsonLogger& logger) {
    if (!problems.plan) {
        for (std::size_t cell = 0; cell < rec.shots * rec.windows; ++cell) {
            const std::size_t shot = cell / rec.windows;
            replay_cell(rec, cell, states[0], rec.shot_iterations.span()[shot],
                        rec.shot_legs.span()[shot], slots, out);
        }
        return ok;
    }
    if (!rec.win_attempts || !rec.win_iterations || !rec.win_legs) {
        logger.error("input", "a sliding run's recording lacks its window arrays",
                     "missing win_attempts, win_iterations or win_legs");
        return input_error;
    }
    const std::span<const std::uint8_t> attempts_of = rec.win_attempts->span();
    const std::span<const std::uint32_t> iterations_of = rec.win_iterations->span();
    const std::span<const std::uint32_t> legs_of = rec.win_legs->span();
    for (std::size_t cell = 0; cell < rec.shots * rec.windows; ++cell) {
        const std::uint8_t attempts = attempts_of[cell];
        if (attempts == 0) {
            ++skipped; // decided by an earlier final window: no decode to replay
            clear_cell(cell, out);
            shape_of[cell] = no_shape;
            continue;
        }
        // The committed attempt is the last one, whose inner decode was recorded.
        const auto k = static_cast<std::uint32_t>(cell % rec.windows);
        const window::Placement* placement = problems.plan->placement(k, attempts - 1U);
        if (placement == nullptr) {
            logger.error("input", "a window made more attempts than the plan allows",
                         "attempt out of range",
                         {{"shot", cell / rec.windows}, {"window", k}, {"attempts", attempts}});
            return input_error;
        }
        shape_of[cell] = placement->shape;
        replay_cell(rec, cell, states[placement->shape], iterations_of[cell], legs_of[cell], slots,
                    out);
    }
    return ok;
}

// Times the policy on the recording and relates it to the run's own decode times.
json bench_record(const Arguments& args, const Recording& rec, std::vector<SelectionState>& states,
                  const std::vector<std::uint32_t>& shape_of, bool sliding, JsonLogger& logger) {
    const BenchResult b = bench_policy(rec, states, shape_of, args.slots, args.bench);
    auto decode_ns = load<std::uint64_t>(args.run, sliding ? "win_decode_ns" : "decode_ns");
    double decode_mean_ns = std::numeric_limits<double>::quiet_NaN();
    if (decode_ns) {
        double sum = 0.0;
        std::size_t n = 0;
        for (std::size_t cell = 0; cell < decode_ns->size(); ++cell) {
            if (cell < shape_of.size() && shape_of[cell] != no_shape) {
                sum += static_cast<double>(decode_ns->span()[cell]);
                ++n;
            }
        }
        decode_mean_ns = n > 0 ? sum / static_cast<double>(n) : decode_mean_ns;
    } else {
        logger.warn("bench", "the run's decode times are unreadable; reported without them",
                    {{"error", decode_ns.error()}});
    }
    const double per_solution =
        b.solutions > 0 ? b.seconds * 1e9 / static_cast<double>(b.solutions) : 0.0;
    const double per_decode =
        b.decodes > 0 ? b.seconds * 1e9 / static_cast<double>(b.decodes) : 0.0;
    json bench = {{"passes", args.bench},
                  {"decodes", b.decodes},
                  {"solutions", b.solutions},
                  {"seconds", b.seconds},
                  {"ns_per_solution", per_solution},
                  {"ns_per_decode", per_decode},
                  {"decode_mean_ns", decode_mean_ns},
                  {"fraction_of_decode", per_decode / decode_mean_ns},
                  {"checksum", b.checksum}};
    logger.info("bench", "selection policy timed", bench);
    return bench;
}

ExitCode write_replay(const Arguments& args, const Outputs& out, const Recording& rec,
                      const json& record, JsonLogger& logger) {
    std::error_code ec;
    if (std::filesystem::exists(args.out) && !std::filesystem::is_empty(args.out) &&
        !args.overwrite) {
        logger.error("output", "output directory is not empty", "refusing to overwrite",
                     {{"out", args.out.string()}});
        return output_error;
    }
    std::filesystem::create_directories(args.out, ec);
    if (ec) {
        logger.error("output", "cannot create the output directory", ec.message(),
                     {{"out", args.out.string()}});
        return output_error;
    }
    if (auto written = write_outputs(args.out, out, rec.shots, rec.windows); !written) {
        logger.error("output", "cannot write the replay", io::describe(written.error()),
                     {{"out", args.out.string()}});
        return output_error;
    }
    std::ofstream replay_file(args.out / "replay.json");
    replay_file << record.dump(2) << '\n';
    if (!replay_file) {
        logger.error("output", "cannot write replay.json", "write failed",
                     {{"out", args.out.string()}});
        return output_error;
    }
    return ok;
}

ExitCode run(const Arguments& args, JsonLogger& logger) {
    const auto started = Clock::now();
    const std::optional<harness::DecoderSpec> spec = read_spec(args.run, logger);
    if (!spec) {
        return input_error;
    }
    const auto* stopping = std::get_if<AfterNConverged>(&spec->relay.stopping);
    SelectionConfig config;
    config.capacity = args.slots;
    config.stop = args.stop;
    config.stop_count = args.count;
    config.stop_gap = args.threshold;
    if (auto valid = validate(config); !valid) {
        logger.error("options", "invalid replay rule", valid.error().detail,
                     {{"slots", args.slots}, {"stop", to_string(args.stop)}});
        return usage_error;
    }

    auto artifact = io::load_artifact(
        args.artifact, io::ArtifactOptions{.graph = spec->graph, .verify_checksums = args.verify});
    if (!artifact) {
        logger.error("artifact", "cannot load the artifact", io::describe(artifact.error()),
                     {{"artifact", args.artifact.string()}});
        return input_error;
    }
    const bool sliding = spec->window.is_sliding();
    auto recording = load_recording(args.run, sliding);
    if (!recording) {
        logger.error("input", "cannot read the recorded solutions", recording.error(),
                     {{"run", args.run.string()}});
        return input_error;
    }
    const Recording& rec = *recording;
    if (args.slots > rec.slots) {
        logger.error("options", "more slots requested than recorded", "slots out of range",
                     {{"slots", args.slots}, {"recorded", rec.slots}});
        return usage_error;
    }
    // A run that stopped after `count` converged legs cannot tell what later legs would find.
    if (stopping != nullptr && args.slots > stopping->count) {
        logger.error("options", "the recording run stopped earlier than the replayed prefix",
                     "slots exceed the recording's stopping count",
                     {{"slots", args.slots}, {"stop_count", stopping->count}});
        return usage_error;
    }

    Problems problems;
    if (const ExitCode built = build_problems(*artifact, *spec, rec, problems, logger);
        built != ok) {
        return built;
    }
    std::vector<SelectionState> states;
    for (const ProblemView& p : problems.views) {
        auto state = SelectionState::create(config, p.classes, p.llr, p.graph);
        if (!state) {
            logger.error("selection", "cannot build the rules", state.error().detail);
            return internal_error;
        }
        states.push_back(std::move(*state));
    }

    Outputs out(rec.shots * rec.windows);
    std::vector<std::uint32_t> shape_of(rec.shots * rec.windows, 0);
    std::size_t skipped = 0;
    if (const ExitCode replayed =
            replay_all(rec, problems, states, args.slots, out, shape_of, skipped, logger);
        replayed != ok) {
        return replayed;
    }

    json record = {
        {"run", args.run.string()},
        {"artifact", args.artifact.string()},
        {"slots", args.slots},
        {"stop",
         {{"rule", to_string(args.stop)}, {"count", args.count}, {"threshold", args.threshold}}},
        {"shots", rec.shots},
        {"windows", rec.windows},
        {"skipped_windows", skipped}};
    if (args.bench > 0) {
        record["bench"] = bench_record(args, rec, states, shape_of, sliding, logger);
    }
    if (const ExitCode written = write_replay(args, out, rec, record, logger); written != ok) {
        return written;
    }
    logger.info("replay", "replay written",
                {{"cells", rec.shots * rec.windows},
                 {"skipped_windows", skipped},
                 {"seconds", std::chrono::duration<double>(Clock::now() - started).count()}});
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    JsonLogger logger(std::cerr, Level::info, harness::new_run_id());
    try {
        auto args = parse_arguments(std::span<char* const>(argv, static_cast<std::size_t>(argc)));
        if (!args) {
            logger.error("options", "invalid command line", args.error());
            std::cerr << usage;
            return static_cast<int>(usage_error);
        }
        if (args->help) {
            std::cout << usage;
            return static_cast<int>(ok);
        }
        JsonLogger run_logger(std::cerr, args->log_level, logger.run_id());
        run_logger.info("replay", "replay started",
                        {{"run", args->run.string()}, {"slots", args->slots}});
        return static_cast<int>(run(*args, run_logger));
    } catch (const std::exception& e) {
        logger.error("internal", "unexpected exception", e.what());
        return static_cast<int>(internal_error);
    }
}
