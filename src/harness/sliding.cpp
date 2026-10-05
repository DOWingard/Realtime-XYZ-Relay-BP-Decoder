#include "rtd/harness/sliding.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include "rtd/core/decoder.hpp"
#include "rtd/core/decoder_fixed.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/worker.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/error.hpp"

namespace rtd::harness {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::uint64_t nanoseconds(Clock::duration elapsed) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
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

json window_spec_json(const window::WindowSpec& spec) {
    return {{"width", spec.width},
            {"commit", spec.commit},
            {"converge_rounds", spec.converge_rounds},
            {"boundary", window::to_string(spec.boundary)},
            {"on_failure", window::to_string(spec.on_failure)},
            {"max_deferrals", spec.max_deferrals},
            {"iteration_cap", spec.iteration_cap ? json(*spec.iteration_cap) : json(nullptr)}};
}

json shapes_json(const window::WindowPlan& plan) {
    json shapes = json::array();
    for (const window::Shape& shape : plan.shapes()) {
        shapes.push_back({{"index", shape.index()},
                          {"rounds", shape.rounds()},
                          {"rows", shape.num_rows()},
                          {"columns", shape.num_columns()},
                          {"edges", shape.num_edges()},
                          {"merged_columns", shape.merged_columns()},
                          {"committed_columns", shape.committed_columns()},
                          {"memory_bytes_estimate", shape.memory_bytes()}});
    }
    return shapes;
}

SetupError configuration_error(std::string context, std::string message, json fields = {}) {
    return SetupError{.input = false,
                      .context = std::move(context),
                      .message = std::move(message),
                      .fields = std::move(fields)};
}

SetupError input_error(std::string context, std::string message, json fields = {}) {
    return SetupError{.input = true,
                      .context = std::move(context),
                      .message = std::move(message),
                      .fields = std::move(fields)};
}

// Reads a [T, width] γ table; a missing file, another width or a non-finite value is an input
// error naming the file.
std::expected<ExplicitGammaTable, SetupError> read_gamma_table(const std::filesystem::path& file,
                                                               index_t width) {
    const std::array<std::optional<std::size_t>, 2> extents{std::nullopt, std::size_t{width}};
    auto table = io::read_npy<double>(file, 2, extents);
    if (!table) {
        return std::unexpected(input_error(
            "gamma", std::format("cannot read the gamma table: {}", io::describe(table.error())),
            {{"path", file.string()}, {"expected_columns", width}}));
    }
    const std::span<const double> values = table->span();
    auto source = ExplicitGammaTable::create({values.begin(), values.end()}, table->rows(), width);
    if (!source) {
        return std::unexpected(
            input_error("gamma", source.error().detail, {{"path", file.string()}}));
    }
    return std::move(*source);
}

std::expected<void, SetupError> build_gamma_sources(SlidingSetup& setup, const DecoderSpec& spec,
                                                    JsonLogger& logger) {
    const std::span<const window::Shape> shapes = setup.plan.shapes();
    const GammaSpec& gamma = spec.gamma;
    std::uint32_t reused = 0; // shapes whose table has fewer rows than relay legs
    for (const window::Shape& shape : shapes) {
        const index_t width = shape.num_columns();
        switch (gamma.kind) {
        case GammaSpec::Kind::none:
            setup.owned.push_back(nullptr);
            break;
        case GammaSpec::Kind::uniform: {
            auto source = UniformGammaGenerator::create(gamma.seed, gamma.low, gamma.high, width);
            if (!source) {
                return std::unexpected(configuration_error("gamma", source.error().detail));
            }
            setup.owned.push_back(std::make_unique<UniformGammaGenerator>(std::move(*source)));
            break;
        }
        case GammaSpec::Kind::explicit_table:
        case GammaSpec::Kind::explicit_shapes: {
            const bool single = gamma.kind == GammaSpec::Kind::explicit_table;
            if (single && shapes.size() != 1) {
                return std::unexpected(configuration_error(
                    "gamma",
                    std::format("an explicit gamma table has the width of the whole problem and "
                                "fits only a window plan with one shape (W >= Rt); this plan has "
                                "{} shapes: give one table per shape with "
                                "{{\"type\": \"explicit_shapes\", \"directory\": ...}}",
                                shapes.size()),
                    {{"path", gamma.table.string()}, {"shapes", shapes.size()}}));
            }
            const std::filesystem::path file =
                single ? gamma.table : gamma.directory / std::format("shape_{}.npy", shape.index());
            auto table = read_gamma_table(file, width);
            if (!table) {
                table.error().fields["shape"] = shape.index();
                return std::unexpected(std::move(table.error()));
            }
            reused += table->reuses_rows(spec.relay.num_sets) ? 1U : 0U;
            setup.owned.push_back(std::make_unique<ExplicitGammaTable>(std::move(*table)));
            break;
        }
        }
    }
    if (gamma.kind == GammaSpec::Kind::explicit_shapes) {
        // A table beyond the plan's shapes means the tables were made for another plan.
        const std::filesystem::path extra =
            gamma.directory / std::format("shape_{}.npy", shapes.size());
        std::error_code ec;
        if (std::filesystem::exists(extra, ec)) {
            logger.warn("gamma",
                        "the gamma directory holds more tables than the plan has shapes; were "
                        "they made for another window plan?",
                        {{"directory", gamma.directory.string()}, {"shapes", shapes.size()}});
        }
    }
    if (reused > 0) {
        logger.warn("gamma", "gamma tables have fewer rows than relay legs; rows are reused",
                    {{"shapes_affected", reused}, {"num_sets", spec.relay.num_sets}});
    }
    setup.gammas.reserve(setup.owned.size());
    for (const auto& source : setup.owned) {
        setup.gammas.push_back(source.get());
    }
    return {};
}

} // namespace

std::expected<std::unique_ptr<SlidingSetup>, SetupError>
build_sliding(const io::Artifact& artifact, const DecoderSpec& spec, JsonLogger& logger) {
    const window::WindowSpec& windows = spec.window.sliding;
    if (!spec.window.is_sliding()) {
        return std::unexpected(
            configuration_error("window", "a window plan is built for window mode sliding only"));
    }
    auto start = Clock::now();
    const window::ArtifactProblem source(artifact);
    logger.debug("plan", "window problem assembled from the artifact",
                 {{"seconds", seconds_since(start)}});

    const std::size_t rss_before = peak_rss_bytes();
    start = Clock::now();
    auto plan = window::WindowPlan::build(source.problem(), windows, spec.graph);
    const double build_seconds = seconds_since(start);
    if (!plan) {
        return std::unexpected(configuration_error("plan", window::describe(plan.error()),
                                                   {{"code", window::to_string(plan.error().code)},
                                                    {"window", window_spec_json(windows)},
                                                    {"artifact", artifact.directory.string()}}));
    }
    // win_attempts is stored as one byte per window.
    constexpr std::uint32_t max_attempts = std::numeric_limits<std::uint8_t>::max();
    for (std::uint32_t k = 0; k < plan->num_positions(); ++k) {
        if (plan->attempts(k) > max_attempts) {
            return std::unexpected(configuration_error(
                "plan",
                std::format("window position {} has {} attempts; at most {} are recorded", k,
                            plan->attempts(k), max_attempts),
                {{"max_deferrals", windows.max_deferrals}}));
        }
    }
    auto setup = std::make_unique<SlidingSetup>(std::move(*plan), build_seconds);
    if (auto sources = build_gamma_sources(*setup, spec, logger); !sources) {
        return std::unexpected(std::move(sources.error()));
    }

    const window::PlanStats stats = setup->plan.stats();
    logger.info("plan", "window plan built",
                {{"window", window_spec_json(windows)},
                 {"rounds_total", setup->plan.rounds_total()},
                 {"detectors_per_round", setup->plan.detectors_per_round()},
                 {"positions", stats.positions},
                 {"schedule_length", stats.placements},
                 {"shapes", shapes_json(setup->plan)},
                 {"merged_columns", stats.merged_columns},
                 {"virtual_committed", stats.virtual_committed},
                 {"memory_bytes_estimate", stats.memory_bytes},
                 {"peak_rss_bytes_before", rss_before},
                 {"peak_rss_bytes_after", peak_rss_bytes()},
                 {"seconds", build_seconds}});
    if (stats.virtual_committed > 0) {
        logger.warn("plan",
                    "some committed bulk columns have no counterpart at their window position "
                    "(before round 1 or past the readout); their commits change the residual "
                    "but no fault is recorded for them",
                    {{"virtual_committed", stats.virtual_committed}});
    }
    if (spec.executor.team_threads > 1 && setup->plan.shapes().size() > 1) {
        // Idle teams spin briefly after their last job and then sleep on an atomic wait, so the
        // teams of shapes not in use take no CPU time.
        logger.info("plan", "each window shape's decoder has its own thread team",
                    {{"shapes", setup->plan.shapes().size()},
                     {"team_threads", spec.executor.team_threads},
                     {"threads_per_worker",
                      1 + (setup->plan.shapes().size() * (spec.executor.team_threads - 1))}});
    }
    return setup;
}

json plan_record(const SlidingSetup& setup) {
    const window::WindowPlan& plan = setup.plan;
    const window::PlanStats stats = plan.stats();
    json placements = json::array();
    for (const window::Placement& p : plan.schedule()) {
        placements.push_back({{"window", p.window},
                              {"attempt", p.attempt},
                              {"shape", p.shape},
                              {"first_round", p.first_round},
                              {"rounds", p.rounds},
                              {"commit_rounds", p.commit_rounds},
                              {"final", p.final},
                              {"virtual_committed", p.virtual_committed}});
    }
    return {{"rounds_total", plan.rounds_total()},
            {"detectors_per_round", plan.detectors_per_round()},
            {"positions", stats.positions},
            {"schedule_length", stats.placements},
            {"shapes", shapes_json(plan)},
            {"placements", std::move(placements)},
            {"merged_columns", stats.merged_columns},
            {"virtual_committed", stats.virtual_committed},
            {"memory_bytes_estimate", stats.memory_bytes},
            {"build_seconds", setup.build_seconds}};
}

// ---- Window events ---------------------------------------------------------------------------

namespace {

struct EventText {
    std::string_view name;
    Level level;
    std::string_view message;
};

constexpr std::array<EventText, WindowEvents::kinds> event_text{{
    {.name = "not_converged",
     .level = Level::warn,
     .message = "window did not converge; policy applied"},
    {.name = "deferred",
     .level = Level::warn,
     .message = "window did not converge; decoded again with a wider window"},
    {.name = "deferral_limit", .level = Level::error, .message = "deferral limit reached"},
    {.name = "cap_hit", .level = Level::warn, .message = "window hit the iteration cap"},
}};

} // namespace

void WindowEvents::observe(std::size_t shot, const window::WindowRecord& record) {
    if (record.attempts == 0) {
        return; // decided by an earlier final window; nothing was decoded
    }
    const bool defer = spec_.on_failure == window::OnFailure::defer;
    if (!record.converged) {
        emit(defer ? Kind::deferral_limit : Kind::not_converged, shot, record);
    } else if (record.attempts > 1) {
        emit(Kind::deferred, shot, record);
    }
    if (record.cap_hit) {
        emit(Kind::cap_hit, shot, record);
    }
}

void WindowEvents::emit(Kind kind, std::size_t shot, const window::WindowRecord& record) {
    const auto index = static_cast<std::size_t>(kind);
    const std::uint64_t seen = counts_[index].fetch_add(1, std::memory_order_relaxed) + 1;
    if (seen > logged_per_kind) {
        return;
    }
    const EventText& text = event_text[index];
    if (!logger_->enabled(text.level)) {
        return;
    }
    json fields = {{"event", text.name},
                   {"count", seen},
                   {"shot", shot},
                   {"window", record.window},
                   {"attempts", record.attempts},
                   {"iterations", record.iterations},
                   {"legs", record.legs},
                   {"converged", record.converged},
                   {"unexplained", record.unexplained},
                   {"policy", window::to_string(spec_.on_failure)}};
    switch (kind) {
    case Kind::not_converged:
        fields["action"] = spec_.on_failure == window::OnFailure::flag
                               ? "committed leg 0's final correction and flagged the shot"
                               : "committed leg 0's final correction";
        break;
    case Kind::deferred:
        fields["action"] = std::format("converged at attempt {}", record.attempts);
        break;
    case Kind::deferral_limit:
        fields["max_deferrals"] = spec_.max_deferrals;
        fields["action"] =
            "committed leg 0's final correction of the widest attempt and flagged the shot";
        break;
    case Kind::cap_hit:
        fields["iteration_cap"] = spec_.iteration_cap ? json(*spec_.iteration_cap) : json(nullptr);
        break;
    }
    if (seen == logged_per_kind) {
        fields["further"] = "further events of this kind are counted, not logged; the total is "
                            "reported when decoding finishes";
    }
    // An event is an outcome of the decode, not a failure of this program: no stack trace.
    logger_->log(text.level, "window", text.message, std::move(fields),
                 text.level == Level::error ? std::optional(text.name) : std::nullopt);
}

json WindowEvents::counts() const {
    json counts = json::object();
    for (std::size_t i = 0; i < kinds; ++i) {
        counts[std::string(event_text[i].name)] = counts_[i].load(std::memory_order_relaxed);
    }
    return counts;
}

void WindowEvents::report() const {
    for (std::size_t i = 0; i < kinds; ++i) {
        const std::uint64_t total = counts_[i].load(std::memory_order_relaxed);
        if (total == 0) {
            continue;
        }
        const EventText& text = event_text[i];
        const json fields = {{"event", text.name},
                             {"total", total},
                             {"logged_individually", std::min(total, logged_per_kind)},
                             {"not_logged", total > logged_per_kind ? total - logged_per_kind : 0}};
        logger_->log(text.level, "window", std::format("{} (run total)", text.message), fields,
                     text.level == Level::error ? std::optional(text.name) : std::nullopt);
    }
}

// ---- The worker ------------------------------------------------------------------------------

namespace {

// Decodes each shot window by window: every round is pushed into the stream decoder, then every
// window position is decoded in order and its record, commits and solutions are written to the
// shot's row.
template <MessageArithmetic A, class Executor, class Sink>
class SlidingWorker final : public Worker {
public:
    using Inner = CpuRelayDecoder<A, Executor, Sink>;

    SlidingWorker(window::StreamDecoder<Inner> stream, const io::Artifact& artifact,
                  const RunOptions& options, WindowEvents& events,
                  std::optional<window::SignalHistory> history = std::nullopt)
        : stream_(std::move(stream)), artifact_(&artifact), events_(&events),
          save_decodings_(options.save_decodings), save_commits_(options.save_commits),
          history_(std::move(history)) {}

    [[nodiscard]] bool warm_up(std::span<const Bit> syndrome,
                               std::uint64_t stream) noexcept override {
        if (!feed(syndrome, stream)) {
            return false;
        }
        while (!stream_.finished()) {
            if (!stream_.decode_next()) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::expected<void, ShotError> decode_shot(const ShotTask& task,
                                                             RunOutputs& out) override {
        const auto start = Clock::now();
        if (auto fed = feed(task.syndrome, task.shot); !fed) {
            return std::unexpected(shot_error(fed.error()));
        }
        if (history_) {
            history_->reset();
        }
        ShotResults& results = out.results;
        const std::size_t row = task.row;
        const std::size_t first_cell = row * results.windows;
        std::uint64_t window_ns = 0;
        while (!stream_.finished()) {
            const auto window_start = Clock::now();
            auto commit = stream_.decode_next();
            window_ns += nanoseconds(Clock::now() - window_start);
            if (!commit) {
                return std::unexpected(shot_error(commit.error()));
            }
            // With every round pushed a failed attempt is retried at once; a deferral only
            // returns while rounds are missing. Its time counts towards the position.
            if (commit->deferred) {
                continue;
            }
            write_window(*commit, first_cell + commit->window, window_ns, row, out);
            window_ns = 0;
            events_->observe(task.shot, commit->record);
        }
        const window::ShotSummary& summary = stream_.summary();
        results.decode_ns[row] = nanoseconds(Clock::now() - start);
        results.success[row] = summary.success ? 1 : 0;
        results.iterations[row] = summary.iterations;
        results.legs[row] = summary.legs;
        results.best_leg[row] = -1;
        results.weight[row] = summary.weight;
        results.flagged[row] = summary.flagged ? 1 : 0;
        const index_t k = artifact_->num_observables();
        bool failed = false;
        for (index_t o = 0; o < k; ++o) {
            auto predicted = static_cast<Bit>((summary.frame >> o) & 1U);
            if (artifact_->observables_bias) {
                predicted ^= (*artifact_->observables_bias)[o];
            }
            results.predicted[(row * k) + o] = predicted;
            failed = failed || predicted != task.truth[o];
        }
        results.logical_failure[row] = failed ? 1 : 0;
        return {};
    }

private:
    // Resets the stream to `shot` and pushes every round of σ (Rt·M bits).
    std::expected<void, window::StreamError> feed(std::span<const Bit> syndrome,
                                                  std::uint64_t shot) noexcept {
        const window::WindowPlan& plan = stream_.plan();
        const std::size_t per_round = plan.detectors_per_round();
        const std::uint32_t rounds = plan.rounds_total();
        if (syndrome.size() != per_round * rounds) {
            window::StreamError error;
            error.code = window::StreamError::Code::wrong_syndrome_size;
            error.expected = per_round * rounds;
            error.found = syndrome.size();
            return std::unexpected(std::move(error));
        }
        stream_.reset(shot);
        for (std::uint32_t r = 0; r + 1 < rounds; ++r) {
            if (auto pushed = stream_.push_round(syndrome.subspan(r * per_round, per_round));
                !pushed) {
                return pushed;
            }
        }
        return stream_.push_final(syndrome.subspan((rounds - 1) * per_round, per_round));
    }

    static ShotError shot_error(const window::StreamError& error) noexcept {
        return ShotError{.code = window::to_string(error.code),
                         .window = error.window,
                         .expected = error.expected,
                         .found = error.found};
    }

    void write_window(const window::Commit& commit, std::size_t cell, std::uint64_t window_ns,
                      std::size_t row, RunOutputs& out) {
        ShotResults& results = out.results;
        const window::WindowRecord& record = commit.record;
        results.win_iterations[cell] = record.iterations;
        results.win_legs[cell] = record.legs;
        results.win_attempts[cell] = static_cast<std::uint8_t>(record.attempts);
        results.win_converged[cell] = record.converged ? 1 : 0;
        results.win_cap_hit[cell] = record.cap_hit ? 1 : 0;
        results.win_weight[cell] = record.weight;
        results.win_committed_weight[cell] = record.committed_weight;
        results.win_unexplained[cell] = record.unexplained;
        results.win_flagged[cell] = record.flagged ? 1 : 0;
        results.win_virtual[cell] = record.virtual_commits;
        results.win_decode_ns[cell] = window_ns;
        if (save_commits_) {
            out.commit_counts[cell] = static_cast<std::uint32_t>(commit.faults.size());
            std::vector<index_t>& faults = out.committed_faults[row];
            faults.insert(faults.end(), commit.faults.begin(), commit.faults.end());
        }
        if (save_decodings_) {
            // The committed correction c ⊕= ê restricted to the window's commit set; the commit
            // sets of the positions are disjoint, so each fault is written at most once.
            const std::size_t n = results.num_columns;
            Bit* dense = results.decodings.data() + (row * n);
            for (const index_t j : commit.faults) {
                dense[j] ^= Bit{1};
            }
        }
        if constexpr (Sink::enabled) {
            const SolutionRecorder* recorder =
                record.shape != window::no_shape ? recorder_of(stream_.inner(record.shape).sink())
                                                 : nullptr;
            if (recorder != nullptr) {
                // The committed attempt was the last decode of its shape's inner decoder, whose
                // sink therefore still holds its solutions. Its commit classes make the
                // returned class the window's change of the frame.
                record_solutions(*recorder, record.returned_class, row, commit.window, out);
            }
        }
        if (results.confidence.enabled) {
            if (record.confidence) {
                record_confidence(*record.confidence, cell, results.confidence);
            }
            results.confidence.low_deferrals[cell] = record.low_confidence_deferrals;
        }
        if (history_) {
            // The per-window signal over the last L windows, as a real-time rule would see it
            // after this window's commit.
            history_->push(record.confidence, record.shape != window::no_shape, commit.faults);
            record_history(*history_, cell, results.confidence);
        }
    }

    window::StreamDecoder<Inner> stream_;
    const io::Artifact* artifact_;
    WindowEvents* events_;
    bool save_decodings_;
    bool save_commits_;
    std::optional<window::SignalHistory> history_;
};

// One inner decoder per window shape: a CPU backend over the shape's graph and priors with its own
// executor, the spec's relay schedule, the shape's γ source and make_sink(shape)'s sink.
template <MessageArithmetic A, class Executor, class Sink, class MakeSink>
std::expected<std::unique_ptr<Worker>, HarnessError>
build_worker(const io::Artifact& artifact, const DecoderSpec& spec, const SlidingSetup& setup,
             const std::function<Executor()>& executor, const RunOptions& options,
             WindowEvents& events, MakeSink make_sink) {
    using Inner = CpuRelayDecoder<A, Executor, Sink>;
    auto stream = window::StreamDecoder<Inner>::create(
        setup.plan, [&](const window::Shape& shape) -> std::expected<Inner, std::string> {
            auto backend =
                CpuBackend<A, Executor>::create(shape.graph(), shape.priors(), executor());
            if (!backend) {
                return std::unexpected(
                    std::format("{}: {}", to_string(backend.error().code), backend.error().detail));
            }
            std::expected<Sink, std::string> sink = make_sink(shape);
            if (!sink) {
                return std::unexpected(std::move(sink.error()));
            }
            auto decoder = Inner::create(std::move(*backend), spec.min_sum, spec.relay,
                                         setup.gammas[shape.index()], std::move(*sink));
            if (!decoder) {
                return std::unexpected(
                    std::format("{}: {}", to_string(decoder.error().code), decoder.error().detail));
            }
            return std::move(*decoder);
        });
    if (!stream) {
        return std::unexpected(
            HarnessError{.context = "decoder", .message = window::describe(stream.error())});
    }
    std::optional<window::SignalHistory> history;
    if (spec.selection) {
        if (auto set = stream->set_on_low_confidence(spec.selection->on_low); !set) {
            return std::unexpected(
                HarnessError{.context = "selection", .message = window::describe(set.error())});
        }
        if (!spec.selection->history_lengths.empty()) {
            std::size_t max_commit = 0;
            for (const window::Shape& shape : setup.plan.shapes()) {
                max_commit = std::max<std::size_t>(max_commit, shape.committed_columns());
            }
            auto made = window::SignalHistory::create(
                spec.selection->history_lengths, spec.selection->history_signals, &artifact.graph,
                artifact.priors.llr(), max_commit);
            if (!made) {
                return std::unexpected(
                    HarnessError{.context = "selection", .message = made.error().detail});
            }
            history.emplace(std::move(*made));
        }
    }
    return std::make_unique<SlidingWorker<A, Executor, Sink>>(std::move(*stream), artifact, options,
                                                              events, std::move(history));
}

// A selection policy per window shape: the shape's commit classes classify its solutions (the
// class a solution assigns to the committed rounds), and Q_supp is measured on the shape's graph.
template <MessageArithmetic A, class Executor>
std::expected<std::unique_ptr<Worker>, HarnessError>
build_selecting_worker(const io::Artifact& artifact, const DecoderSpec& spec,
                       const SelectionConfig& config, const SlidingSetup& setup,
                       const std::function<Executor()>& executor, const RunOptions& options,
                       WindowEvents& events) {
    const auto state_of =
        [&](const window::Shape& shape) -> std::expected<SelectionState, std::string> {
        auto state = SelectionState::create(config, shape.commit_class(), shape.priors().llr(),
                                            &shape.graph());
        if (!state) {
            return std::unexpected(
                std::format("{}: {}", to_string(state.error().code), state.error().detail));
        }
        return std::move(*state);
    };
    if (options.record_solutions == 0) {
        return build_worker<A, Executor, SelectionSink<NoSink>>(
            artifact, spec, setup, executor, options, events,
            [&](const window::Shape& shape) -> std::expected<SelectionSink<NoSink>, std::string> {
                auto state = state_of(shape);
                if (!state) {
                    return std::unexpected(std::move(state.error()));
                }
                return SelectionSink<NoSink>(std::move(*state));
            });
    }
    return build_worker<A, Executor, SelectionSink<SolutionRecorder>>(
        artifact, spec, setup, executor, options, events,
        [&](const window::Shape& shape)
            -> std::expected<SelectionSink<SolutionRecorder>, std::string> {
            auto state = state_of(shape);
            if (!state) {
                return std::unexpected(std::move(state.error()));
            }
            auto records = RecordingSink::create(options.record_solutions, shape.commit_class());
            if (!records) {
                return std::unexpected(records.error().detail);
            }
            return SelectionSink<SolutionRecorder>(
                std::move(*state), SolutionRecorder(*records, options.save_solution_supports));
        });
}

} // namespace

template <MessageArithmetic A, class Executor>
std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker(const io::Artifact& artifact, const DecoderSpec& spec,
                    const SlidingSetup& setup, const std::function<Executor()>& executor,
                    const RunOptions& options, WindowEvents& events) {
    if (spec.selection) {
        return build_selecting_worker<A, Executor>(artifact, spec, spec.selection->config, setup,
                                                   executor, options, events);
    }
    if (options.record_solutions == 0) {
        return build_worker<A, Executor, NoSink>(
            artifact, spec, setup, executor, options, events,
            [](const window::Shape& /*shape*/) -> std::expected<NoSink, std::string> {
                return NoSink{};
            });
    }
    // Each shape's sink classifies a solution by the shape's commit classes: the XOR over its
    // support is the logical class it assigns to the committed rounds.
    return build_worker<A, Executor, SolutionRecorder>(
        artifact, spec, setup, executor, options, events,
        [&](const window::Shape& shape) -> std::expected<SolutionRecorder, std::string> {
            auto records = RecordingSink::create(options.record_solutions, shape.commit_class());
            if (!records) {
                return std::unexpected(records.error().detail);
            }
            return SolutionRecorder(*records, options.save_solution_supports);
        });
}

template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F32, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                 const std::function<Serial()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F64, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                 const std::function<Serial()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F32, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                               const std::function<Team()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F64, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                               const std::function<Team()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int4_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int5_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int6_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int4_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int5_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);
template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int6_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);

} // namespace rtd::harness
