// rtd_window_plan: builds the sliding-window plan of an artifact for one window specification and
// writes it out (plan.json plus every shape's and placement's arrays as .npy), so that an
// independent implementation can be compared with it file by file.

#include <sys/resource.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/harness/logger.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/spec.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
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
    R"(usage: rtd_window_plan --artifact DIR --width W --commit C --converge C' --boundary B
                       --on-failure P --max-deferrals D --out DIR [options]

required:
  --artifact DIR        exported decoding problem (H, priors, observables, detector rounds)
  --width W             window width in rounds
  --commit C            rounds committed per window, 1 <= C < W
  --converge C'         rounds whose rows must satisfy H e = s, C <= C' <= W
  --boundary B          exact | uniform
  --on-failure P        commit_anyway | defer | flag
  --max-deferrals D     deferral attempts per window (0 unless P = defer)
  --out DIR             output directory (created; must be empty unless --overwrite)

options:
  --no-verify           skip SHA-256 verification of the artifact
  --overwrite           allow a non-empty --out directory
  --log-level L         debug | info | warn | error (default info)
  --help                this text

writes plan.json, shape_<i>_{H_indptr,H_indices,priors,commit,converge,class}.npy and
placement_<k>_<a>_columns.npy (+ _members_ptr.npy, _members.npy for the exact boundary).
)";

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path out;
    WindowSpec spec;
    bool verify = true;
    bool overwrite = false;
    Level log_level = Level::info;
    bool help = false;
};

// The window flags as given; every one is required.
struct SpecFlags {
    std::optional<std::uint32_t> width;
    std::optional<std::uint32_t> commit;
    std::optional<std::uint32_t> converge;
    std::optional<Boundary> boundary;
    std::optional<OnFailure> on_failure;
    std::optional<std::uint32_t> max_deferrals;
};

template <class T>
std::expected<T, std::string> required(std::string_view flag, const std::optional<T>& value) {
    if (!value) {
        return std::unexpected(std::format("{} is required", flag));
    }
    return *value;
}

std::expected<WindowSpec, std::string> spec_from(const SpecFlags& flags) {
    WindowSpec spec;
    std::expected<void, std::string> step =
        required("--width", flags.width).transform([&](std::uint32_t v) { spec.width = v; });
    step = step.and_then([&] {
        return required("--commit", flags.commit).transform([&](std::uint32_t v) {
            spec.commit = v;
        });
    });
    step = step.and_then([&] {
        return required("--converge", flags.converge).transform([&](std::uint32_t v) {
            spec.converge_rounds = v;
        });
    });
    step = step.and_then([&] {
        return required("--boundary", flags.boundary).transform([&](Boundary v) {
            spec.boundary = v;
        });
    });
    step = step.and_then([&] {
        return required("--on-failure", flags.on_failure).transform([&](OnFailure v) {
            spec.on_failure = v;
        });
    });
    step = step.and_then([&] {
        return required("--max-deferrals", flags.max_deferrals).transform([&](std::uint32_t v) {
            spec.max_deferrals = v;
        });
    });
    if (!step) {
        return std::unexpected(step.error());
    }
    return spec;
}

std::expected<std::uint32_t, std::string> parse_count(std::string_view flag, std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        return std::unexpected(std::format("{} expects a non-negative integer, got '{}'", flag, text));
    }
    return value;
}

std::expected<Arguments, std::string> parse_arguments(std::span<char* const> argv) {
    Arguments args;
    SpecFlags flags;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            args.help = true;
            continue;
        }
        if (flag == "--no-verify") {
            args.verify = false;
            continue;
        }
        if (flag == "--overwrite") {
            args.overwrite = true;
            continue;
        }
        if (i + 1 >= argv.size()) {
            return std::unexpected(std::format("{} needs a value", flag));
        }
        const std::string_view text = argv[++i];
        std::expected<void, std::string> step;
        const auto count_into = [&](std::optional<std::uint32_t>& target) {
            step = parse_count(flag, text).transform([&](std::uint32_t v) { target = v; });
        };
        if (flag == "--artifact") {
            args.artifact = text;
        } else if (flag == "--out") {
            args.out = text;
        } else if (flag == "--width") {
            count_into(flags.width);
        } else if (flag == "--commit") {
            count_into(flags.commit);
        } else if (flag == "--converge") {
            count_into(flags.converge);
        } else if (flag == "--max-deferrals") {
            count_into(flags.max_deferrals);
        } else if (flag == "--boundary") {
            flags.boundary = parse_boundary(text);
            if (!flags.boundary) {
                return std::unexpected(
                    std::format("--boundary expects exact or uniform, got '{}'", text));
            }
        } else if (flag == "--on-failure") {
            flags.on_failure = parse_on_failure(text);
            if (!flags.on_failure) {
                return std::unexpected(std::format(
                    "--on-failure expects commit_anyway, defer or flag, got '{}'", text));
            }
        } else if (flag == "--log-level") {
            const auto level = harness::parse_level(text);
            if (!level) {
                return std::unexpected(std::format("unknown log level '{}'", text));
            }
            args.log_level = *level;
        } else {
            return std::unexpected(std::format("unknown argument '{}'", flag));
        }
        if (!step) {
            return std::unexpected(step.error());
        }
    }
    if (args.help) {
        return args;
    }
    for (const auto& [flag, path] :
         {std::pair{"--artifact", &args.artifact}, std::pair{"--out", &args.out}}) {
        if (path->empty()) {
            return std::unexpected(std::format("{} is required", flag));
        }
    }
    auto spec = spec_from(flags);
    if (!spec) {
        return std::unexpected(spec.error());
    }
    args.spec = *spec;
    return args;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Peak resident set size of this process so far, in bytes (0 if unavailable).
std::size_t peak_rss_bytes() noexcept {
    rusage resources{};
    if (getrusage(RUSAGE_SELF, &resources) != 0) {
        return 0;
    }
    // glibc declares ru_maxrss inside an anonymous union.
    const long kib = resources.ru_maxrss; // NOLINT(cppcoreguidelines-pro-type-union-access)
    return static_cast<std::size_t>(kib) * 1024;
}

json io_error_fields(const io::IoError& error) {
    return {{"code", io::to_string(error.code)},
            {"path", error.path},
            {"expected", error.expected},
            {"found", error.found}};
}

// Creates the output directory, refusing to mix a plan into a previous one's files.
std::expected<void, std::string> prepare_output(const std::filesystem::path& out, bool overwrite) {
    std::error_code ec;
    if (std::filesystem::exists(out, ec)) {
        if (!std::filesystem::is_directory(out, ec)) {
            return std::unexpected(std::format("{} exists and is not a directory", out.string()));
        }
        if (!overwrite && !std::filesystem::is_empty(out, ec)) {
            return std::unexpected(
                std::format("{} is not empty (pass --overwrite to reuse it)", out.string()));
        }
        return {};
    }
    std::filesystem::create_directories(out, ec);
    if (ec) {
        return std::unexpected(std::format("cannot create {}: {}", out.string(), ec.message()));
    }
    return {};
}

json spec_json(const WindowSpec& spec) {
    return {{"width", spec.width},
            {"commit", spec.commit},
            {"converge_rounds", spec.converge_rounds},
            {"boundary", to_string(spec.boundary)},
            {"on_failure", to_string(spec.on_failure)},
            {"max_deferrals", spec.max_deferrals}};
}

json plan_json(const WindowPlan& plan) {
    json shapes = json::array();
    for (const Shape& shape : plan.shapes()) {
        shapes.push_back({{"index", shape.index()},
                          {"rows", shape.num_rows()},
                          {"columns", shape.num_columns()},
                          {"edges", shape.num_edges()},
                          {"merged_columns", shape.merged_columns()}});
    }
    json placements = json::array();
    for (const Placement& p : plan.schedule()) {
        placements.push_back({{"window", p.window},
                              {"attempt", p.attempt},
                              {"shape", p.shape},
                              {"first_round", p.first_round},
                              {"rounds", p.rounds},
                              {"commit_rounds", p.commit_rounds},
                              {"final", p.final}});
    }
    return {{"spec", spec_json(plan.spec())},
            {"rounds_total", plan.rounds_total()},
            {"detectors_per_round", plan.detectors_per_round()},
            {"num_positions", plan.num_positions()},
            {"shapes", std::move(shapes)},
            {"placements", std::move(placements)}};
}

template <class T>
std::expected<void, io::IoError> write_vector(const std::filesystem::path& path,
                                              std::span<const T> values) {
    const std::array<std::size_t, 1> shape{values.size()};
    return io::write_npy<T>(path, values, shape);
}

// Writes plan.json and every array; returns the number of files written.
std::expected<std::size_t, io::IoError> write_plan(const std::filesystem::path& out,
                                                   const WindowPlan& plan) {
    std::size_t files = 0;
    const auto count = [&](std::expected<void, io::IoError> written)
        -> std::expected<void, io::IoError> {
        if (written) {
            ++files;
        }
        return written;
    };
    for (const Shape& shape : plan.shapes()) {
        const std::string prefix = std::format("shape_{}_", shape.index());
        std::vector<std::uint8_t> commit(shape.commit().begin(), shape.commit().end());
        std::vector<std::uint8_t> converge(shape.converge().begin(), shape.converge().end());
        for (auto written :
             {count(write_vector<index_t>(out / (prefix + "H_indptr.npy"), shape.row_ptr())),
              count(write_vector<index_t>(out / (prefix + "H_indices.npy"), shape.col_indices())),
              count(write_vector<double>(out / (prefix + "priors.npy"),
                                         shape.priors().probabilities())),
              count(write_vector<std::uint8_t>(out / (prefix + "commit.npy"), commit)),
              count(write_vector<std::uint8_t>(out / (prefix + "converge.npy"), converge)),
              count(write_vector<std::uint64_t>(out / (prefix + "class.npy"),
                                                shape.commit_class()))}) {
            if (!written) {
                return std::unexpected(std::move(written.error()));
            }
        }
    }
    const bool exact = plan.spec().boundary == Boundary::exact;
    for (const Placement& p : plan.schedule()) {
        const std::string prefix = std::format("placement_{}_{}_", p.window, p.attempt);
        if (auto written = count(write_vector<index_t>(out / (prefix + "columns.npy"), p.columns));
            !written) {
            return std::unexpected(std::move(written.error()));
        }
        if (!exact) {
            continue;
        }
        for (auto written :
             {count(write_vector<index_t>(out / (prefix + "members_ptr.npy"), p.members_ptr)),
              count(write_vector<index_t>(out / (prefix + "members.npy"), p.members))}) {
            if (!written) {
                return std::unexpected(std::move(written.error()));
            }
        }
    }
    const std::filesystem::path json_path = out / "plan.json";
    std::ofstream stream(json_path);
    stream << plan_json(plan).dump(2) << '\n';
    stream.close();
    if (!stream) {
        return std::unexpected(io::IoError{.code = io::IoError::Code::write_failed,
                                           .path = json_path.string(),
                                           .expected = "plan.json written",
                                           .found = "stream error"});
    }
    return files + 1;
}

ExitCode run(const Arguments& args, JsonLogger& logger) {
    const auto started = Clock::now();
    const WindowSpec& spec = args.spec;
    if (auto valid = validate(spec); !valid) {
        logger.error("spec", "invalid window specification", describe(valid.error()),
                     {{"code", to_string(valid.error().code)}, {"spec", spec_json(spec)}});
        return usage_error;
    }
    logger.debug("spec", "window specification accepted", {{"spec", spec_json(spec)}});

    // Fail before the expensive steps if the plan could not be written.
    if (auto prepared = prepare_output(args.out, args.overwrite); !prepared) {
        logger.error("output", "cannot use the output directory", prepared.error(),
                     {{"out", args.out.string()}});
        return output_error;
    }

    auto step_start = Clock::now();
    auto loaded = io::load_artifact(args.artifact, io::ArtifactOptions{.graph = GraphOptions{},
                                                                      .verify_checksums =
                                                                          args.verify});
    if (!loaded) {
        logger.error("artifact", "cannot load the artifact", io::describe(loaded.error()),
                     io_error_fields(loaded.error()));
        return input_error;
    }
    const io::Artifact& artifact = *loaded;
    logger.info("artifact", "artifact loaded",
                {{"directory", args.artifact.string()},
                 {"detectors", artifact.num_detectors()},
                 {"columns", artifact.num_columns()},
                 {"edges", artifact.graph.num_edges()},
                 {"observables", artifact.num_observables()},
                 {"checksums_verified", args.verify},
                 {"seconds", seconds_since(step_start)}});

    step_start = Clock::now();
    const ArtifactProblem source(artifact);
    const Problem problem = source.problem();
    logger.debug("problem", "window problem assembled from the artifact",
                 {{"seconds", seconds_since(step_start)}});

    const std::size_t rss_before = peak_rss_bytes();
    step_start = Clock::now();
    auto plan = WindowPlan::build(problem, spec, GraphOptions{});
    const double build_seconds = seconds_since(step_start);
    if (!plan) {
        logger.error("plan", "window plan rejected", describe(plan.error()),
                     {{"code", to_string(plan.error().code)},
                      {"spec", spec_json(spec)},
                      {"artifact", args.artifact.string()}});
        return input_error;
    }
    const PlanStats stats = plan->stats();
    json shapes = json::array();
    for (const Shape& shape : plan->shapes()) {
        shapes.push_back({{"index", shape.index()},
                          {"rounds", shape.rounds()},
                          {"rows", shape.num_rows()},
                          {"columns", shape.num_columns()},
                          {"edges", shape.num_edges()},
                          {"merged_columns", shape.merged_columns()},
                          {"committed_columns", shape.committed_columns()}});
    }
    logger.info("plan", "window plan built",
                {{"rounds_total", plan->rounds_total()},
                 {"detectors_per_round", plan->detectors_per_round()},
                 {"positions", stats.positions},
                 {"placements", stats.placements},
                 {"shapes", std::move(shapes)},
                 {"merged_columns", stats.merged_columns},
                 {"virtual_committed", stats.virtual_committed},
                 {"memory_bytes_estimate", stats.memory_bytes},
                 {"peak_rss_bytes_before", rss_before},
                 {"peak_rss_bytes_after", peak_rss_bytes()},
                 {"seconds", build_seconds}});
    if (stats.virtual_committed > 0) {
        logger.warn("plan",
                    "some committed bulk columns have no counterpart at their position; their "
                    "commits are dropped (uniform boundary)",
                    {{"virtual_committed", stats.virtual_committed}});
    }

    step_start = Clock::now();
    auto written = write_plan(args.out, *plan);
    if (!written) {
        logger.error("output", "cannot write the plan", io::describe(written.error()),
                     io_error_fields(written.error()));
        return output_error;
    }
    logger.info("run", "plan written",
                {{"out", args.out.string()},
                 {"files", *written},
                 {"write_seconds", seconds_since(step_start)},
                 {"seconds", seconds_since(started)}});
    return ok;
}

ExitCode run_cli(std::span<char* const> argv) {
    auto args = parse_arguments(argv);
    JsonLogger logger(std::cerr, args ? args->log_level : Level::info, harness::new_run_id());
    if (!args) {
        logger.error("cli", "invalid arguments", args.error());
        std::cerr << usage;
        return usage_error;
    }
    if (args->help) {
        std::cout << usage;
        return ok;
    }
    logger.info("run", "run started",
                {{"artifact", args->artifact.string()},
                 {"out", args->out.string()},
                 {"spec", spec_json(args->spec)}});
    try {
        return run(*args, logger);
    } catch (const std::exception& e) {
        logger.error("run", "unexpected exception; aborting", e.what());
    } catch (...) {
        logger.error("run", "unexpected non-standard exception; aborting", "unknown");
    }
    return internal_error;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return static_cast<int>(
            run_cli(std::span<char* const>(argv, static_cast<std::size_t>(argc))));
    } catch (...) {
        // Thrown before the logger existed or by the logger itself, so report without it.
        std::fputs(R"({"level":"error","context":"run",)"
                   R"("message":"unexpected exception outside the run; aborting"})"
                   "\n",
                   stderr);
    }
    return static_cast<int>(internal_error);
}
