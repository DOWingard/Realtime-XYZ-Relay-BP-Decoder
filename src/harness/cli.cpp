// rtd_decode: decodes a range of sampled shots with one decoder configuration and writes
// per-shot results, summary statistics and a provenance record.

#include "rtd/harness/cli.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/gamma.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/harness/batch.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/sliding.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/harness/system.hpp"
#include "rtd/harness/writer.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/sha256.hpp"
#include "rtd/io/shots.hpp"

namespace rtd::harness {

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

enum class ExitCode : std::uint8_t {
    ok = 0,
    internal_error = 1,
    usage_error = 2,
    input_error = 3,
    output_error = 4,
    decode_error = 5,
};
using enum ExitCode;

constexpr std::string_view usage = R"(usage: rtd_decode --artifact DIR --shots DIR --config FILE --out DIR [options]

required:
  --artifact DIR     exported decoding problem (H, priors, observables)
  --shots DIR        sampled shots (detectors, observable flips) from the same circuit
  --config FILE      decoder spec (JSON, version 2, or version 3 with a "selection"
                     policy; every field required)
  --out DIR          result directory (created; must be empty unless --overwrite)

options:
  --first N          first shot to decode (default 0)
  --count N          number of shots (default: all from --first on)
  --workers N        concurrent decoders (default 1)
  --cpus LIST        pin threads to these CPUs, e.g. 0-2,6 (worker w takes the w-th group of
                     T CPUs, T = threads per decoder)
  --warmup N         untimed decodes per worker before timing (default 0)
  --save-decodings   also write every correction ê (shots × columns bytes); with sliding windows,
                     the faults all windows committed
  --record-solutions N
                     record the first N converged legs of every decode (0-20; 0 = off): leg,
                     cumulative iterations, weight, logical class, hash and size of each solution
                     (with sliding windows: of every window, classes by its commit classes)
  --save-solution-supports
                     also write the support of every recorded solution (needs N > 0; window-local
                     column indices with sliding windows)
  --save-commits     write the faults each window commits (window mode sliding only)
  --no-verify        skip SHA-256 verification of the inputs
  --overwrite        allow a non-empty --out directory
  --log-level L      debug | info | warn | error (default info)
  --help             this text
)";

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path shots;
    std::filesystem::path config;
    std::filesystem::path out;
    std::size_t first = 0;
    std::optional<std::size_t> count;
    unsigned workers = 1;
    std::vector<unsigned> cpus;
    std::string cpus_text;
    std::size_t warmup = 0;
    bool save_decodings = false;
    std::uint32_t record_solutions = 0;
    bool save_solution_supports = false;
    bool save_commits = false;
    bool verify = true;
    bool overwrite = false;
    Level log_level = Level::info;
    bool help = false;
};

template <class T>
std::expected<T, std::string> parse_number(std::string_view flag, std::string_view text) {
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || text.empty()) {
        return std::unexpected(std::format("{} expects a non-negative integer, got '{}'", flag, text));
    }
    return value;
}

std::expected<std::uint32_t, std::string> parse_solution_count(std::string_view flag,
                                                               std::string_view text) {
    return parse_number<std::uint32_t>(flag, text).and_then(
        [&](std::uint32_t n) -> std::expected<std::uint32_t, std::string> {
            if (n > RecordingSink::max_capacity) {
                return std::unexpected(std::format("{} must be in [0, {}], got {}", flag,
                                                   RecordingSink::max_capacity, n));
            }
            return n;
        });
}

// Applies a flag that takes no value; false when `flag` is not one.
bool apply_switch(Arguments& args, std::string_view flag) {
    if (flag == "--help" || flag == "-h") {
        args.help = true;
    } else if (flag == "--save-decodings") {
        args.save_decodings = true;
    } else if (flag == "--save-solution-supports") {
        args.save_solution_supports = true;
    } else if (flag == "--save-commits") {
        args.save_commits = true;
    } else if (flag == "--no-verify") {
        args.verify = false;
    } else if (flag == "--overwrite") {
        args.overwrite = true;
    } else {
        return false;
    }
    return true;
}

// Applies a flag that takes the value `text`.
std::expected<void, std::string> apply_option(Arguments& args, std::string_view flag,
                                              std::string_view text) {
    if (flag == "--artifact") {
        args.artifact = text;
    } else if (flag == "--shots") {
        args.shots = text;
    } else if (flag == "--config") {
        args.config = text;
    } else if (flag == "--out") {
        args.out = text;
    } else if (flag == "--first") {
        return parse_number<std::size_t>(flag, text).transform(
            [&](std::size_t v) { args.first = v; });
    } else if (flag == "--count") {
        return parse_number<std::size_t>(flag, text).transform(
            [&](std::size_t v) { args.count = v; });
    } else if (flag == "--workers") {
        return parse_number<unsigned>(flag, text).transform([&](unsigned v) { args.workers = v; });
    } else if (flag == "--warmup") {
        return parse_number<std::size_t>(flag, text).transform(
            [&](std::size_t v) { args.warmup = v; });
    } else if (flag == "--record-solutions") {
        return parse_solution_count(flag, text).transform(
            [&](std::uint32_t v) { args.record_solutions = v; });
    } else if (flag == "--cpus") {
        args.cpus_text = text;
        return parse_cpu_list(text).transform(
            [&](std::vector<unsigned> v) { args.cpus = std::move(v); });
    } else if (flag == "--log-level") {
        const auto level = parse_level(text);
        if (!level) {
            return std::unexpected(std::format("unknown log level '{}'", text));
        }
        args.log_level = *level;
    } else {
        return std::unexpected(std::format("unknown argument '{}'", flag));
    }
    return {};
}

std::expected<Arguments, std::string> parse_arguments(std::span<char* const> argv) {
    Arguments args;
    for (std::size_t i = 1; i < argv.size(); ++i) {
        const std::string_view flag = argv[i];
        if (apply_switch(args, flag)) {
            continue;
        }
        if (i + 1 >= argv.size()) {
            return std::unexpected(std::format("{} needs a value", flag));
        }
        if (auto applied = apply_option(args, flag, argv[++i]); !applied) {
            return std::unexpected(std::move(applied.error()));
        }
    }
    if (args.help) {
        return args;
    }
    for (const auto& [flag, path] : {std::pair{"--artifact", &args.artifact},
                                     std::pair{"--shots", &args.shots},
                                     std::pair{"--config", &args.config},
                                     std::pair{"--out", &args.out}}) {
        if (path->empty()) {
            return std::unexpected(std::format("{} is required", flag));
        }
    }
    if (args.workers == 0) {
        return std::unexpected(std::string("--workers must be at least 1"));
    }
    if (args.save_solution_supports && args.record_solutions == 0) {
        return std::unexpected(
            std::string("--save-solution-supports needs --record-solutions N with N > 0"));
    }
    return args;
}

json io_error_fields(const io::IoError& error) {
    return {{"code", io::to_string(error.code)},
            {"path", error.path},
            {"expected", error.expected},
            {"found", error.found}};
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::expected<std::unique_ptr<GammaSource>, std::string>
make_gamma_source(const GammaSpec& spec, index_t width, JsonLogger& logger, std::uint32_t num_sets) {
    switch (spec.kind) {
    case GammaSpec::Kind::none:
        return nullptr;
    case GammaSpec::Kind::uniform: {
        auto source = UniformGammaGenerator::create(spec.seed, spec.low, spec.high, width);
        if (!source) {
            return std::unexpected(source.error().detail);
        }
        return std::make_unique<UniformGammaGenerator>(std::move(*source));
    }
    case GammaSpec::Kind::explicit_table: {
        const std::array<std::optional<std::size_t>, 2> extents{std::nullopt, std::size_t{width}};
        auto table = io::read_npy<double>(spec.table, 2, extents);
        if (!table) {
            return std::unexpected(io::describe(table.error()));
        }
        const std::size_t rows = table->rows();
        const std::span<const double> values = table->span();
        auto source = ExplicitGammaTable::create({values.begin(), values.end()}, rows, width);
        if (!source) {
            return std::unexpected(source.error().detail);
        }
        if (source->reuses_rows(num_sets)) {
            logger.warn("gamma", "the gamma table has fewer rows than relay legs; rows are reused",
                        {{"rows", rows}, {"num_sets", num_sets}});
        }
        return std::make_unique<ExplicitGammaTable>(std::move(*source));
    }
    case GammaSpec::Kind::explicit_shapes:
        // The spec parser admits these only with sliding windows, whose decoders build one γ
        // source per window shape.
        return std::unexpected(
            std::string("explicit_shapes tables belong to sliding-window decoders"));
    }
    return std::unexpected(std::string("unknown gamma source"));
}

// Creates the output directory, refusing to mix results into a previous run's.
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

json file_record(const std::filesystem::path& file) {
    auto digest = io::sha256_file(file);
    return {{"path", std::filesystem::absolute(file).string()},
            {"sha256", digest ? json(*digest) : json(nullptr)}};
}

// What the run was asked to record beyond the per-shot arrays, for run.json.
json recording_record(const RunOptions& options) {
    return {{"record_solutions", options.record_solutions},
            {"save_solution_supports", options.save_solution_supports},
            {"save_commits", options.save_commits},
            {"save_decodings", options.save_decodings},
            {"class_bits", "bit o of sol_class and returned_class is observable o of A·ê "
                           "(sliding windows: A restricted to the columns the window commits); "
                           "the artifact's observables_bias is not applied"}};
}

ExitCode run(const Arguments& args, JsonLogger& logger) {
    const auto started = Clock::now();
    const auto started_wall = std::chrono::system_clock::now();

    auto spec = load_spec(args.config);
    if (!spec) {
        logger.error("config", "invalid decoder spec", spec.error().problem,
                     {{"file", args.config.string()}, {"field", spec.error().field}});
        return usage_error;
    }
    logger.info("config", "decoder spec loaded",
                {{"file", args.config.string()},
                 {"version", spec->version},
                 {"window", spec->raw.at("window")},
                 {"selection", spec->selection ? selection_json(*spec->selection) : json(nullptr)}});
    logger.debug("config", "decoder spec", {{"spec", spec->raw}});

    // The count is known only once the shots are loaded; everything else is checked now.
    RunOptions options{.first = args.first,
                       .count = 0,
                       .workers = args.workers,
                       .cpus = args.cpus,
                       .warmup = args.warmup,
                       .save_decodings = args.save_decodings,
                       .record_solutions = args.record_solutions,
                       .save_solution_supports = args.save_solution_supports,
                       .save_commits = args.save_commits};
    if (auto supported = check_supported(*spec, options); !supported) {
        logger.error(supported.error().context, "this build cannot run the requested decode",
                     supported.error().message, {{"file", args.config.string()}});
        return usage_error;
    }

    auto load_start = Clock::now();
    auto artifact = io::load_artifact(
        args.artifact, io::ArtifactOptions{.graph = spec->graph, .verify_checksums = args.verify});
    if (!artifact) {
        logger.error("artifact", "cannot load the artifact", io::describe(artifact.error()),
                     io_error_fields(artifact.error()));
        return input_error;
    }
    // Decoders keep pointers into the artifact; it stays in place from here on.
    const auto problem = std::make_unique<io::Artifact>(std::move(*artifact));
    logger.info("artifact", "artifact loaded",
                {{"directory", args.artifact.string()},
                 {"detectors", problem->num_detectors()},
                 {"columns", problem->num_columns()},
                 {"edges", problem->graph.num_edges()},
                 {"observables", problem->num_observables()},
                 {"checksums_verified", args.verify},
                 {"seconds", seconds_since(load_start)}});

    // Sliding windows: the plan and every window shape's γ source, built once and shared by all
    // workers, before anything else is loaded or written.
    std::unique_ptr<SlidingSetup> sliding;
    if (spec->window.is_sliding()) {
        auto setup = build_sliding(*problem, *spec, logger);
        if (!setup) {
            const SetupError& error = setup.error();
            logger.error(error.context,
                         error.context == "plan" ? "window plan rejected"
                                                 : "cannot build the window decoders' inputs",
                         error.message, error.fields);
            return error.input ? input_error : usage_error;
        }
        sliding = std::move(*setup);
    }

    load_start = Clock::now();
    auto shots = io::load_shots(args.shots, *problem, args.verify);
    if (!shots) {
        logger.error("shots", "cannot load the shots", io::describe(shots.error()),
                     io_error_fields(shots.error()));
        return input_error;
    }
    logger.info("shots", "shots loaded",
                {{"directory", args.shots.string()},
                 {"shots", shots->count()},
                 {"rounds", shots->rounds ? json(*shots->rounds) : json(nullptr)},
                 {"seconds", seconds_since(load_start)}});

    if (args.first > shots->count()) {
        logger.error("options", "--first is past the last shot", "out of range",
                     {{"first", args.first}, {"shots", shots->count()}});
        return usage_error;
    }
    const std::size_t count = args.count.value_or(shots->count() - args.first);
    options.count = count;

    // A sliding run has one γ source per window shape, in its setup.
    std::expected<std::unique_ptr<GammaSource>, std::string> gammas = nullptr;
    if (!sliding) {
        gammas =
            make_gamma_source(spec->gamma, problem->num_columns(), logger, spec->relay.num_sets);
    }
    if (!gammas) {
        logger.error("gamma", "cannot build the gamma source", gammas.error());
        return spec->gamma.kind == GammaSpec::Kind::explicit_table ? input_error : usage_error;
    }

    const unsigned threads = args.workers * std::max(spec->executor.team_threads, 1U);
    if (threads > hardware_threads()) {
        logger.warn("options", "more decoder threads than hardware threads; timings will be "
                               "distorted by oversubscription",
                    {{"threads", threads}, {"hardware_threads", hardware_threads()}});
    }
    if (args.cpus.empty() && spec->executor.team_threads > 1) {
        logger.warn("options", "a thread team runs unpinned; pass --cpus for stable latency");
    }

    if (auto prepared = prepare_output(args.out, args.overwrite); !prepared) {
        logger.error("output", "cannot use the output directory", prepared.error(),
                     {{"out", args.out.string()}});
        return output_error;
    }

    auto runner =
        BatchRunner::create(*problem, *shots, *spec, gammas->get(), options, logger, sliding.get());
    if (!runner) {
        logger.error(runner.error().context, "cannot set up the decoders", runner.error().message);
        return usage_error;
    }
    auto results = runner->run();
    if (!results) {
        logger.error(results.error().context, "decoding failed", results.error().message);
        return decode_error;
    }

    const std::optional<std::uint32_t> logical_qubits = [&]() -> std::optional<std::uint32_t> {
        auto k = shots->manifest.get<std::uint32_t>("/code/k");
        return k ? std::optional(*k) : std::nullopt;
    }();
    if (!shots->rounds) {
        logger.warn("summary", "the shots manifest lacks /rounds; no per-cycle rates");
    } else if (!logical_qubits) {
        logger.warn("summary", "the shots manifest lacks /code/k; no per-qubit rate");
    }
    json summary = summarize(*results, logical_qubits, shots->rounds);

    const BuildInfo build = build_info();
    json record = {
        {"run_id", logger.run_id()},
        {"started", std::format("{:%FT%TZ}", std::chrono::time_point_cast<std::chrono::seconds>(
                                                 started_wall))},
        {"elapsed_seconds", seconds_since(started)},
        {"decoder", spec->raw},
        {"window", spec->raw.at("window")},
        {"recording", recording_record(options)},
        {"inputs",
         {{"artifact", file_record(args.artifact / "manifest.json")},
          {"shots", file_record(args.shots / "manifest.json")},
          {"checksums_verified", args.verify},
          {"first", args.first},
          {"count", count}}},
        {"execution",
         {{"workers", args.workers},
          {"threads_per_decoder", std::max(spec->executor.team_threads, 1U)},
          {"cpus", args.cpus_text.empty() ? json(nullptr) : json(args.cpus_text)},
          {"warmup_per_worker", args.warmup}}},
        {"machine",
         {{"host", host_name()}, {"cpu", cpu_model()}, {"hardware_threads", hardware_threads()}}},
        {"build",
         {{"git_revision", build.git_revision},
          {"compiler", build.compiler},
          {"build_type", build.build_type},
          {"kernel_flags", build.flags}}},
        {"summary", summary},
    };
    if (sliding) {
        record["plan"] = plan_record(*sliding);
    }

    const auto write_start = Clock::now();
    if (auto written = write_results(args.out, *results, record); !written) {
        logger.error("output", "cannot write results", io::describe(written.error()),
                     io_error_fields(written.error()));
        return output_error;
    }
    logger.info("run", "run finished",
                {{"out", args.out.string()},
                 {"write_seconds", seconds_since(write_start)},
                 {"elapsed_seconds", seconds_since(started)},
                 {"summary", summary}});
    return ok;
}

} // namespace

int decode_main(std::span<char* const> argv, std::ostream& out, std::ostream& log) {
    const std::string run_id = new_run_id();
    auto args = parse_arguments(argv);
    JsonLogger logger(log, args ? args->log_level : Level::info, run_id);
    if (!args) {
        logger.error("cli", "invalid arguments", args.error());
        log << usage;
        return static_cast<int>(usage_error);
    }
    if (args->help) {
        out << usage;
        return static_cast<int>(ok);
    }
    logger.info("run", "run started",
                {{"artifact", args->artifact.string()},
                 {"shots", args->shots.string()},
                 {"config", args->config.string()},
                 {"out", args->out.string()},
                 {"first", args->first},
                 {"count", args->count ? json(*args->count) : json(nullptr)},
                 {"workers", args->workers},
                 {"cpus", args->cpus_text},
                 {"warmup", args->warmup},
                 {"record_solutions", args->record_solutions},
                 {"save_solution_supports", args->save_solution_supports},
                 {"save_commits", args->save_commits}});
    try {
        return static_cast<int>(run(*args, logger));
    } catch (const std::exception& e) {
        logger.error("run", "unexpected exception; aborting", e.what());
    } catch (...) {
        logger.error("run", "unexpected non-standard exception; aborting", "unknown");
    }
    return static_cast<int>(internal_error);
}

} // namespace rtd::harness
