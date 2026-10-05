#include "rtd/harness/batch.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <format>
#include <latch>
#include <limits>
#include <numeric>
#include <span>
#include <thread>
#include <utility>

#include "rtd/core/decoder.hpp"
#include "rtd/core/decoder_fixed.hpp"
#ifdef RTD_HAVE_CUDA_BACKEND
#include "rtd/core/backend_cuda.hpp"
#endif
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/sliding.hpp"
#include "rtd/harness/system.hpp"
#include "rtd/harness/worker.hpp"

namespace rtd::harness {

using Clock = std::chrono::steady_clock;

namespace {

// ℓ̂ = A·ê over the returned correction's support, then the frame change of pruned faults; the
// shot fails when ℓ̂ differs from the true flips anywhere.
void write_outcome(const io::Artifact& artifact, std::span<const index_t> support,
                   const ShotTask& task, std::span<Bit> predicted, ShotResults& results) {
    const index_t k = artifact.num_observables();
    artifact.observables.apply_support(support, predicted);
    if (artifact.observables_bias) {
        for (index_t o = 0; o < k; ++o) {
            predicted[o] ^= (*artifact.observables_bias)[o];
        }
    }
    bool failed = false;
    for (index_t o = 0; o < k; ++o) {
        results.predicted[(task.row * k) + o] = predicted[o];
        failed = failed || predicted[o] != task.truth[o];
    }
    results.logical_failure[task.row] = failed ? 1 : 0;
}

// Decodes each shot as one problem over all of its rounds: K = 1 window.
template <MessageArithmetic A, class Executor, class Sink>
class WholeShotWorker final : public Worker {
public:
    WholeShotWorker(CpuRelayDecoder<A, Executor, Sink> decoder, const io::Artifact& artifact,
                    bool save_decodings)
        : decoder_(std::move(decoder)), artifact_(&artifact),
          predicted_(artifact.num_observables()), save_decodings_(save_decodings) {}

    [[nodiscard]] bool warm_up(std::span<const Bit> syndrome,
                               std::uint64_t stream) noexcept override {
        return decoder_.decode(syndrome, stream).has_value();
    }

    [[nodiscard]] std::expected<void, ShotError> decode_shot(const ShotTask& task,
                                                             RunOutputs& out) override {
        const auto start = Clock::now();
        const auto result = decoder_.decode(task.syndrome, task.shot);
        const auto stop = Clock::now();
        if (!result) {
            return std::unexpected(ShotError{.code = to_string(result.error().code),
                                             .window = 0,
                                             .expected = result.error().expected,
                                             .found = result.error().found});
        }
        ShotResults& results = out.results;
        const std::size_t i = task.row;
        results.decode_ns[i] = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        results.success[i] = result->success ? 1 : 0;
        results.iterations[i] = result->iterations;
        results.legs[i] = result->legs_executed;
        results.best_leg[i] = result->best_leg ? static_cast<std::int32_t>(*result->best_leg) : -1;
        results.weight[i] = result->weight;
        write_outcome(*artifact_, result->support, task, predicted_, results);
        if (save_decodings_) {
            const std::size_t n = artifact_->num_columns();
            std::ranges::copy(result->hard,
                              results.decodings.begin() + static_cast<std::ptrdiff_t>(i * n));
        }
        if constexpr (Sink::enabled) {
            if (const SolutionRecorder* recorder = recorder_of(decoder_.sink())) {
                record_solutions(*recorder, recorder->class_of(result->support), i, 0, out);
            }
        }
        if (result->confidence) {
            record_confidence(*result->confidence, i, results.confidence);
        }
        return {};
    }

private:
    CpuRelayDecoder<A, Executor, Sink> decoder_;
    const io::Artifact* artifact_;
    std::vector<Bit> predicted_;
    bool save_decodings_;
};

HarnessError decoder_error(const ConfigError& error) {
    return HarnessError{.context = "decoder",
                        .message = std::format("{}: {}", to_string(error.code), error.detail)};
}

// Without recording the decoder carries NoSink, exactly the decoder of a run before recording
// existed; with it, a SolutionRecorder borrowing the run's column classes.
template <MessageArithmetic A, class Executor>
std::expected<std::unique_ptr<Worker>, HarnessError>
make_worker(const io::Artifact& artifact, const DecoderSpec& spec, const GammaSource* gammas,
            const std::function<Executor()>& executor, const RunOptions& options,
            std::span<const std::uint64_t> column_class) {
    auto backend = CpuBackend<A, Executor>::create(artifact.graph, artifact.priors, executor());
    if (!backend) {
        return std::unexpected(
            HarnessError{.context = "backend",
                         .message = std::format("{}: {}", to_string(backend.error().code),
                                                backend.error().detail)});
    }
    const auto build = [&]<class Sink>(Sink sink)
        -> std::expected<std::unique_ptr<Worker>, HarnessError> {
        auto decoder = CpuRelayDecoder<A, Executor, Sink>::create(
            std::move(*backend), spec.min_sum, spec.relay, gammas, std::move(sink));
        if (!decoder) {
            return std::unexpected(decoder_error(decoder.error()));
        }
        return std::make_unique<WholeShotWorker<A, Executor, Sink>>(std::move(*decoder), artifact,
                                                                    options.save_decodings);
    };
    std::optional<SolutionRecorder> recorder;
    if (options.record_solutions > 0) {
        auto records = RecordingSink::create(options.record_solutions, column_class);
        if (!records) {
            return std::unexpected(HarnessError{.context = "recording",
                                                .message = records.error().detail});
        }
        recorder.emplace(*records, options.save_solution_supports);
    }
    if (!spec.selection) {
        if (!recorder) {
            return build(NoSink{});
        }
        return build(std::move(*recorder));
    }
    // A selection policy: the decode's solutions go through it (and on to the recorder), and it
    // measures Q_supp on the whole problem's Tanner graph.
    auto state = SelectionState::create(spec.selection->config, column_class,
                                        artifact.priors.llr(), &artifact.graph);
    if (!state) {
        return std::unexpected(HarnessError{
            .context = "selection",
            .message = std::format("{}: {}", to_string(state.error().code), state.error().detail)});
    }
    if (!recorder) {
        return build(SelectionSink<NoSink>(std::move(*state)));
    }
    return build(SelectionSink<SolutionRecorder>(std::move(*state), std::move(*recorder)));
}

// What one worker's decoders are built from besides the spec.
struct WorkerInputs {
    const io::Artifact* artifact;
    const GammaSource* gammas;                   // whole shot
    const SlidingSetup* sliding;                 // sliding windows
    WindowEvents* events;                        // sliding windows
    std::span<const std::uint64_t> column_class; // whole shot, when recording
};

template <MessageArithmetic A, class Executor>
std::expected<std::unique_ptr<Worker>, HarnessError>
make_worker_with(const WorkerInputs& in, const DecoderSpec& spec,
                 const std::function<Executor()>& executor, const RunOptions& options) {
    if (in.sliding != nullptr) {
        return make_sliding_worker<A, Executor>(*in.artifact, spec, *in.sliding, executor, options,
                                                *in.events);
    }
    return make_worker<A, Executor>(*in.artifact, spec, in.gammas, executor, options,
                                    in.column_class);
}

// A sliding worker has one decoder per window shape, and each takes its own executor, so
// executors are handed out by a factory: a Serial, or a new Team pinned to the worker's CPUs.
template <MessageArithmetic A>
std::expected<std::unique_ptr<Worker>, HarnessError>
make_worker_for_executor(const WorkerInputs& in, const DecoderSpec& spec,
                         std::span<const unsigned> cpus, const RunOptions& options,
                         JsonLogger& logger) {
    if (spec.executor.team_threads == 0) {
        const std::function<Serial()> serial = [] { return Serial{}; };
        return make_worker_with<A, Serial>(in, spec, serial, options);
    }
    // Team worker 0 is the batch worker's own thread; the team pins the others as they start.
    std::vector<unsigned> team_cpus(cpus.begin(), cpus.end());
    auto pin = [team_cpus, &logger](unsigned index) {
        if (index < team_cpus.size()) {
            if (auto pinned = pin_current_thread(team_cpus[index]); !pinned) {
                logger.warn("batch", "could not pin a team thread; it runs unpinned",
                            {{"team_thread", index}, {"cpu", team_cpus[index]},
                             {"error", pinned.error()}});
            }
        }
    };
    std::function<void(unsigned)> on_thread_start;
    if (!cpus.empty()) {
        on_thread_start = std::move(pin);
    }
    const unsigned threads = spec.executor.team_threads;
    const std::function<Team()> team = [threads, on_thread_start] {
        return Team(threads, on_thread_start);
    };
    return make_worker_with<A, Team>(in, spec, team, options);
}

// The worker for the spec's number format.
std::expected<std::unique_ptr<Worker>, HarnessError>
make_worker_for_policy(const WorkerInputs& in, const DecoderSpec& spec,
                       std::span<const unsigned> cpus, const RunOptions& options,
                       JsonLogger& logger) {
    switch (spec.policy) {
    case Policy::f32:
        return make_worker_for_executor<F32>(in, spec, cpus, options, logger);
    case Policy::f64:
        return make_worker_for_executor<F64>(in, spec, cpus, options, logger);
    case Policy::int4_2_8:
        return make_worker_for_executor<Int4_2_8>(in, spec, cpus, options, logger);
    case Policy::int5_2_8:
        return make_worker_for_executor<Int5_2_8>(in, spec, cpus, options, logger);
    case Policy::int6_2_8:
        return make_worker_for_executor<Int6_2_8>(in, spec, cpus, options, logger);
    }
    return std::unexpected(HarnessError{.context = "config", .message = "unknown number format"});
}

} // namespace

void record_solutions(const SolutionRecorder& sink, std::uint64_t returned_class, std::size_t row,
                      std::size_t window, RunOutputs& out) {
    ShotResults& results = out.results;
    const std::size_t cell = (row * results.windows) + window;
    const std::size_t base = cell * results.solution_slots;
    results.sol_count[cell] = sink.found();
    results.returned_class[cell] = returned_class;
    const std::span<const SolutionRecord> records = sink.records();
    for (std::size_t s = 0; s < records.size(); ++s) {
        const SolutionRecord& record = records[s];
        results.sol_leg[base + s] = record.leg;
        results.sol_iterations[base + s] = record.cumulative_iterations;
        results.sol_weight[base + s] = record.weight;
        results.sol_class[base + s] = record.logical_class;
        results.sol_hash[base + s] = record.hash;
        results.sol_size[base + s] = record.size;
    }
    if (sink.keeps_supports()) {
        const std::span<const index_t> supports = sink.supports();
        out.solution_supports[row].insert(out.solution_supports[row].end(), supports.begin(),
                                          supports.end());
    }
}

std::expected<void, HarnessError> check_supported(const DecoderSpec& spec,
                                                  const RunOptions& options) {
    const auto fail = [](std::string context, std::string message) {
        return std::unexpected(
            HarnessError{.context = std::move(context), .message = std::move(message)});
    };
    if (options.save_commits && !spec.window.is_sliding()) {
        return fail("options",
                    "--save-commits records the faults each window commits and needs window mode "
                    "sliding; for a whole-shot decode use --save-decodings (every ê) or "
                    "--record-solutions N --save-solution-supports (the solutions' supports)");
    }
    if (options.record_solutions > RecordingSink::max_capacity) {
        return fail("options", std::format("--record-solutions must be in [0, {}], got {}",
                                           RecordingSink::max_capacity,
                                           options.record_solutions));
    }
    if (options.save_solution_supports && options.record_solutions == 0) {
        return fail("options", "--save-solution-supports needs --record-solutions N with N > 0");
    }
    return {};
}

BatchRunner::BatchRunner(const io::Artifact& artifact, const io::Shots& shots, RunOptions options,
                         unsigned threads_per_decoder, JsonLogger& logger)
    : artifact_(&artifact), shots_(&shots), options_(std::move(options)),
      threads_per_decoder_(threads_per_decoder), logger_(&logger) {}

BatchRunner::BatchRunner(BatchRunner&&) noexcept = default;
BatchRunner& BatchRunner::operator=(BatchRunner&&) noexcept = default;
BatchRunner::~BatchRunner() = default;

std::expected<BatchRunner, HarnessError>
BatchRunner::create(const io::Artifact& artifact, const io::Shots& shots, const DecoderSpec& spec,
                    const GammaSource* gammas, RunOptions options, JsonLogger& logger,
                    const SlidingSetup* sliding) {
    if (spec.backend == BackendKind::cuda) {
#ifdef RTD_HAVE_CUDA_BACKEND
        // Constructed only to surface the device backend's own verdict; a working device
        // backend will get a worker type here like the CPU one.
        const auto device = spec.policy == Policy::f32
                                ? CudaBackend<F32>::create(artifact.graph, artifact.priors).error()
                                : CudaBackend<F64>::create(artifact.graph, artifact.priors).error();
        return std::unexpected(HarnessError{
            .context = "backend",
            .message = std::format("cuda backend {}: {}", to_string(device.code), device.detail)});
#else
        return std::unexpected(HarnessError{
            .context = "backend",
            .message = "this binary was built without the cuda backend (RTD_ENABLE_CUDA=OFF)"});
#endif
    }
    if (auto supported = check_supported(spec, options); !supported) {
        return std::unexpected(std::move(supported.error()));
    }
    if (options.workers == 0) {
        return std::unexpected(
            HarnessError{.context = "options", .message = "at least one worker is needed"});
    }
    if (options.first > shots.count() || options.count > shots.count() - options.first) {
        return std::unexpected(HarnessError{
            .context = "options",
            .message = std::format("shots [{}, {}) are outside the {} available", options.first,
                                   options.first + options.count, shots.count())});
    }
    if (options.count == 0) {
        return std::unexpected(HarnessError{.context = "options", .message = "no shots to decode"});
    }
    if (spec.window.is_sliding() && sliding == nullptr) {
        return std::unexpected(HarnessError{
            .context = "window",
            .message = "a sliding-window spec needs its window plan and the gamma source of "
                       "every window shape (build_sliding)"});
    }
    if (!spec.window.is_sliding() && sliding != nullptr) {
        return std::unexpected(HarnessError{
            .context = "window", .message = "a window plan was given for a whole-shot spec"});
    }
    const unsigned per_worker = std::max(spec.executor.team_threads, 1U);
    const std::size_t cpus_needed = std::size_t{options.workers} * per_worker;
    if (!options.cpus.empty() && options.cpus.size() < cpus_needed) {
        return std::unexpected(HarnessError{
            .context = "options",
            .message = std::format("{} workers × {} threads need {} CPUs, {} were given",
                                   options.workers, per_worker, cpus_needed, options.cpus.size())});
    }

    BatchRunner runner(artifact, shots, std::move(options), per_worker, logger);
    const RunOptions& opts = runner.options_;
    if (opts.record_solutions > 0 || spec.selection) {
        auto classes = column_classes(artifact.observables);
        if (!classes) {
            return std::unexpected(HarnessError{
                .context = opts.record_solutions > 0 ? "recording" : "selection",
                .message = classes.error()});
        }
        runner.column_class_ = std::move(*classes);
    }
    if (spec.selection) {
        runner.selection_ = *spec.selection;
        logger.info("selection", "selection policy on", selection_json(*spec.selection));
    }
    if (opts.record_solutions > 0) {
        // The support store is address space reserved per worker; only the pages that supports
        // reach are ever touched.
        const std::size_t support_bytes =
            opts.save_solution_supports ? std::size_t{opts.record_solutions} *
                                              artifact.num_columns() * sizeof(index_t)
                                        : 0;
        logger.info("recording", "solution recording on",
                    {{"slots_per_decode", opts.record_solutions},
                     {"save_supports", opts.save_solution_supports},
                     {"observables", artifact.num_observables()},
                     {"support_store_bytes_per_worker", support_bytes}});
    }
    if (sliding != nullptr) {
        runner.sliding_ = sliding;
        runner.events_ = std::make_unique<WindowEvents>(logger, spec.window.sliding);
    }
    const WorkerInputs inputs{.artifact = &artifact,
                              .gammas = gammas,
                              .sliding = sliding,
                              .events = runner.events_.get(),
                              .column_class = runner.column_class_};
    runner.workers_.reserve(opts.workers);
    for (unsigned w = 0; w < opts.workers; ++w) {
        const std::span<const unsigned> cpus =
            opts.cpus.empty() ? std::span<const unsigned>{}
                              : std::span(opts.cpus).subspan(std::size_t{w} * per_worker,
                                                             per_worker);
        auto worker = make_worker_for_policy(inputs, spec, cpus, opts, logger);
        if (!worker) {
            return std::unexpected(std::move(worker.error()));
        }
        runner.workers_.push_back(std::move(*worker));
    }
    logger.debug("batch", "decoders built",
                 {{"workers", opts.workers}, {"threads_per_decoder", per_worker},
                  {"record_solutions", opts.record_solutions},
                  {"window_shapes_per_worker",
                   sliding != nullptr ? sliding->plan.shapes().size() : std::size_t{0}}});
    return runner;
}

// What the workers of one run() share. Each shot's result slots are written by exactly one
// worker, so only the counters are atomic.
struct BatchRunner::RunState {
    explicit RunState(std::ptrdiff_t workers) : warmed(workers + 1) {}

    RunOutputs outputs;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<std::size_t> decode_errors{0};
    std::atomic<bool> failed{false}; // some worker threw and stopped
    std::latch warmed;               // the workers and the main thread, after warmup
};

void BatchRunner::work(unsigned w, RunState& state) noexcept {
    bool warmed_up = false;
    try {
        decode_share(w, state, warmed_up);
    } catch (const std::exception& e) {
        abandon(w, state, warmed_up, e.what());
    } catch (...) {
        abandon(w, state, warmed_up, "a non-standard exception");
    }
}

void BatchRunner::abandon(unsigned w, RunState& state, bool warmed_up,
                          std::string_view what) noexcept {
    state.failed.store(true, std::memory_order_release);
    state.next.store(options_.count, std::memory_order_relaxed);
    if (!warmed_up) {
        state.warmed.count_down();
    }
    nlohmann::json fields;
    try {
        fields = {{"worker", w}};
    } catch (...) {
        fields = nullptr; // the line is still written, without the worker index
    }
    logger_->error("batch", "a worker failed; the run stops after the shots in flight", what,
                   std::move(fields));
}

void BatchRunner::decode_share(unsigned w, RunState& state, bool& warmed_up) {
    const RunOptions& opts = options_;
    const io::Artifact& artifact = *artifact_;
    const io::Shots& shots = *shots_;
    const std::size_t count = opts.count;
    Worker& worker = *workers_[w];

    if (!opts.cpus.empty()) {
        const unsigned cpu = opts.cpus[std::size_t{w} * threads_per_decoder_];
        if (auto pinned = pin_current_thread(cpu); !pinned) {
            logger_->warn("batch", "could not pin a worker; it runs unpinned",
                          {{"worker", w}, {"cpu", cpu}, {"error", pinned.error()}});
        }
    }
    std::vector<Bit> syndrome(artifact.num_detectors());
    const auto syndrome_of = [&](std::size_t shot) -> std::span<const Bit> {
        const std::span<const Bit> raw = shots.syndrome(shot);
        if (!artifact.syndrome_bias) {
            return raw;
        }
        std::ranges::transform(raw, *artifact.syndrome_bias, syndrome.begin(),
                               [](Bit a, Bit b) { return static_cast<Bit>(a ^ b); });
        return syndrome;
    };
    for (std::size_t i = 0; i < opts.warmup; ++i) {
        const std::size_t shot = opts.first + (i % count);
        if (!worker.warm_up(syndrome_of(shot), shot)) {
            logger_->warn("batch", "warmup decode rejected its input; the timed run will count it",
                          {{"worker", w}, {"shot", shot}});
        }
    }
    warmed_up = true;
    state.warmed.arrive_and_wait();

    for (;;) {
        const std::size_t i = state.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= count) {
            break;
        }
        const std::size_t shot = opts.first + i;
        const ShotTask task{
            .row = i, .shot = shot, .syndrome = syndrome_of(shot), .truth = shots.flips(shot)};
        if (auto decoded = worker.decode_shot(task, state.outputs); !decoded) {
            state.decode_errors.fetch_add(1, std::memory_order_relaxed);
            logger_->error("batch", "decode rejected its input", decoded.error().code,
                           {{"shot", shot},
                            {"window", decoded.error().window},
                            {"expected", decoded.error().expected},
                            {"found", decoded.error().found}});
        }
        state.done.fetch_add(1, std::memory_order_release);
    }
}

// Joins the per-row supports into one CSR array in row order. The slot pointers come from the
// recorded sizes, and each row must hold exactly the entries its slots claim.
std::expected<void, HarnessError> BatchRunner::gather_supports(RunState& state) const {
    ShotResults& results = state.outputs.results;
    std::vector<std::vector<index_t>>& rows = state.outputs.solution_supports;
    const std::size_t per_row = std::size_t{results.windows} * results.solution_slots;
    results.solsup_ptr.assign((results.count * per_row) + 1, 0);
    for (std::size_t q = 0; q < results.count * per_row; ++q) {
        results.solsup_ptr[q + 1] = results.solsup_ptr[q] + results.sol_size[q];
    }
    results.solsup_idx.resize(results.solsup_ptr.back());
    for (std::size_t row = 0; row < results.count; ++row) {
        const std::uint64_t begin = results.solsup_ptr[row * per_row];
        const std::uint64_t end = results.solsup_ptr[(row + 1) * per_row];
        if (rows[row].size() != end - begin) {
            logger_->error("batch", "recorded supports disagree with the recorded sizes",
                           "internal inconsistency",
                           {{"row", row}, {"stored", rows[row].size()}, {"sizes", end - begin}});
            return std::unexpected(HarnessError{
                .context = "recording",
                .message = std::format("row {} stores {} support entries but its sizes sum to {}",
                                       row, rows[row].size(), end - begin)});
        }
        std::ranges::copy(rows[row],
                          results.solsup_idx.begin() + static_cast<std::ptrdiff_t>(begin));
        std::vector<index_t>().swap(rows[row]);
    }
    return {};
}

// The per-window arrays [count, K] of a sliding-window run, and the per-row commit staging.
void BatchRunner::size_window_outputs(RunState& state) const {
    ShotResults& results = state.outputs.results;
    const std::size_t cells = results.count * results.windows;
    results.sliding = true;
    results.on_failure = sliding_->plan.spec().on_failure;
    results.win_iterations.assign(cells, 0);
    results.win_legs.assign(cells, 0);
    results.win_attempts.assign(cells, 0);
    results.win_converged.assign(cells, 0);
    results.win_cap_hit.assign(cells, 0);
    results.win_weight.assign(cells, std::numeric_limits<double>::infinity());
    results.win_committed_weight.assign(cells, 0.0);
    results.win_unexplained.assign(cells, 0);
    results.win_flagged.assign(cells, 0);
    results.win_virtual.assign(cells, 0);
    results.win_decode_ns.assign(cells, 0);
    results.flagged.assign(results.count, 0);
    if (options_.save_commits) {
        state.outputs.committed_faults.assign(results.count, {});
        state.outputs.commit_counts.assign(cells, 0);
    }
}

// Joins the per-row committed faults into one CSR array over the (row, window) cells, in row
// order; each row must hold exactly the faults its cells' counts claim.
std::expected<void, HarnessError> BatchRunner::gather_commits(RunState& state) const {
    ShotResults& results = state.outputs.results;
    std::vector<std::vector<index_t>>& rows = state.outputs.committed_faults;
    const std::vector<std::uint32_t>& counts = state.outputs.commit_counts;
    const std::size_t per_row = results.windows;
    results.commit_ptr.assign((results.count * per_row) + 1, 0);
    for (std::size_t q = 0; q < results.count * per_row; ++q) {
        results.commit_ptr[q + 1] = results.commit_ptr[q] + counts[q];
    }
    results.commit_faults.resize(results.commit_ptr.back());
    for (std::size_t row = 0; row < results.count; ++row) {
        const std::uint64_t begin = results.commit_ptr[row * per_row];
        const std::uint64_t end = results.commit_ptr[(row + 1) * per_row];
        if (rows[row].size() != end - begin) {
            logger_->error("batch", "committed faults disagree with the recorded counts",
                           "internal inconsistency",
                           {{"row", row}, {"stored", rows[row].size()}, {"counts", end - begin}});
            return std::unexpected(HarnessError{
                .context = "commits",
                .message = std::format("row {} stores {} committed faults but its counts sum to {}",
                                       row, rows[row].size(), end - begin)});
        }
        std::ranges::copy(rows[row],
                          results.commit_faults.begin() + static_cast<std::ptrdiff_t>(begin));
        std::vector<index_t>().swap(rows[row]);
    }
    return {};
}

std::expected<ShotResults, HarnessError> BatchRunner::run() {
    const RunOptions& opts = options_;
    const std::size_t count = opts.count;
    const index_t n = artifact_->num_columns();
    const index_t k = artifact_->num_observables();

    RunState state(static_cast<std::ptrdiff_t>(workers_.size()));
    ShotResults& results = state.outputs.results;
    results.first = opts.first;
    results.count = count;
    results.num_columns = n;
    results.num_observables = k;
    results.success.assign(count, 0);
    results.iterations.assign(count, 0);
    results.legs.assign(count, 0);
    results.best_leg.assign(count, -1);
    results.weight.assign(count, std::numeric_limits<double>::infinity());
    results.decode_ns.assign(count, 0);
    results.predicted.assign(count * k, 0);
    results.logical_failure.assign(count, 0);
    if (opts.save_decodings) {
        results.decodings.assign(count * n, 0);
    }
    results.windows = sliding_ != nullptr ? sliding_->plan.num_positions() : 1;
    results.solution_slots = opts.record_solutions;
    if (sliding_ != nullptr) {
        size_window_outputs(state);
    }
    if (selection_) {
        size_confidence_outputs(results.confidence, count * results.windows, *selection_,
                                sliding_ != nullptr);
    }
    if (opts.record_solutions > 0) {
        const std::size_t cells = count * results.windows;
        const std::size_t slots = cells * opts.record_solutions;
        results.sol_count.assign(cells, 0);
        results.returned_class.assign(cells, 0);
        results.sol_leg.assign(slots, 0);
        results.sol_iterations.assign(slots, 0);
        results.sol_weight.assign(slots, std::numeric_limits<double>::infinity());
        results.sol_class.assign(slots, 0);
        results.sol_hash.assign(slots, 0);
        results.sol_size.assign(slots, 0);
        if (opts.save_solution_supports) {
            state.outputs.solution_supports.assign(count, {});
        }
    }

    logger_->info("batch", "decoding started",
                  {{"shots", count},
                   {"first", opts.first},
                   {"workers", workers_.size()},
                   {"warmup_per_worker", opts.warmup},
                   {"mode", sliding_ != nullptr ? "sliding" : "whole_shot"},
                   {"windows_per_shot", results.windows}});
    std::vector<std::jthread> threads;
    threads.reserve(workers_.size());
    try {
        for (unsigned w = 0; w < workers_.size(); ++w) {
            threads.emplace_back([this, w, &state] { work(w, state); });
        }
    } catch (const std::system_error& e) {
        // Threads already started wait at the latch; release them with an empty queue.
        const std::size_t started = threads.size();
        state.next.store(count);
        for (std::size_t w = started; w < workers_.size(); ++w) {
            state.warmed.count_down();
        }
        state.warmed.arrive_and_wait();
        threads.clear();
        logger_->error("batch", "could not start worker threads", e.what(),
                       {{"started", started}, {"workers", workers_.size()}});
        return std::unexpected(
            HarnessError{.context = "batch",
                         .message = std::format("could not start worker threads: {}", e.what())});
    }
    state.warmed.arrive_and_wait();
    const auto start = Clock::now();
    logger_->debug("batch", "warmup finished; timed region started");

    // Progress, from the main thread, which otherwise only waits.
    auto last_report = start;
    constexpr auto report_every = std::chrono::seconds(10);
    const auto running = [&](std::size_t finished) {
        return finished < count && !state.failed.load(std::memory_order_acquire);
    };
    for (std::size_t finished = state.done.load(std::memory_order_acquire); running(finished);
         finished = state.done.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto now = Clock::now();
        if (now - last_report >= report_every) {
            last_report = now;
            const double seconds = std::chrono::duration<double>(now - start).count();
            nlohmann::json fields = {{"done", finished},
                                     {"shots", count},
                                     {"elapsed_s", seconds},
                                     {"shots_per_s", static_cast<double>(finished) / seconds}};
            if (events_) {
                fields["window_events"] = events_->counts();
            }
            logger_->info("batch", "progress", std::move(fields));
        }
    }
    threads.clear(); // joins
    results.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();

    if (state.failed.load()) {
        return std::unexpected(HarnessError{
            .context = "batch", .message = "a worker failed before finishing; see the log"});
    }
    if (const std::size_t errors = state.decode_errors.load(); errors != 0) {
        return std::unexpected(HarnessError{
            .context = "batch",
            .message = std::format("{} of {} decodes rejected their input", errors, count)});
    }
    if (opts.save_solution_supports) {
        if (auto gathered = gather_supports(state); !gathered) {
            return std::unexpected(std::move(gathered.error()));
        }
    }
    if (opts.save_commits) {
        if (auto gathered = gather_commits(state); !gathered) {
            return std::unexpected(std::move(gathered.error()));
        }
    }
    const auto converged = static_cast<std::size_t>(std::ranges::count(results.success, 1));
    nlohmann::json totals = {{"shots", count},
                             {"not_converged", count - converged},
                             {"wall_s", results.wall_seconds},
                             {"shots_per_s", static_cast<double>(count) / results.wall_seconds}};
    if (events_) {
        events_->report();
        totals["flagged"] = std::ranges::count(results.flagged, 1);
        totals["window_events"] = events_->counts();
    }
    logger_->info("batch", "decoding finished", std::move(totals));
    if (opts.record_solutions > 0) {
        const std::uint64_t found = std::accumulate(results.sol_count.begin(),
                                                    results.sol_count.end(), std::uint64_t{0});
        const auto beyond = static_cast<std::size_t>(std::ranges::count_if(
            results.sol_count, [&](std::uint32_t c) { return c > opts.record_solutions; }));
        logger_->info("recording", "solutions recorded",
                      {{"solutions_found", found},
                       {"decodes_with_unrecorded_solutions", beyond},
                       {"support_entries", results.solsup_idx.size()}});
    }
    if (selection_) {
        logger_->info("selection", "confidence recorded", confidence_summary(results.confidence));
    }
    return std::move(results);
}

} // namespace rtd::harness
