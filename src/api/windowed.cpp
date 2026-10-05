#include "rtd/api/windowed.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "dispatch.hpp"
#include "driver.hpp"
#include "sources.hpp"

namespace rtd::api {

// What every stream decoder of a windowed decoder shares: the problem, the spec, the window plan
// and the γ source of every shape. Relay decoders point into the plan's shapes and into the γ
// sources, so it is shared by pointer and outlives every decoder built from it.
struct WindowSetup {
    WindowSetup(std::shared_ptr<const Problem> p, harness::DecoderSpec s, window::WindowPlan built,
                std::uint32_t record)
        : problem(std::move(p)), spec(std::move(s)), plan(std::move(built)),
          record_solutions(record) {}

    std::shared_ptr<const Problem> problem;
    harness::DecoderSpec spec;
    window::WindowPlan plan;
    std::vector<std::unique_ptr<GammaSource>> gammas; // one per shape, null without relay legs
    std::uint32_t record_solutions = 0;
};

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t nanoseconds(Clock::duration elapsed) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

// A stream decoder whose inner decoder for each window shape is a CPU relay decoder over the
// shape's graph and priors, with the spec's schedule, the shape's γ source and a solution sink
// whose classes are the shape's commit classes (so a solution's class is the logical frame change
// it would commit). With a selection policy, each shape's policy measures Q_supp on the shape's
// graph, and the stream applies the policy's low-confidence action to every window.
template <MessageArithmetic A, class Executor, class Sink>
std::expected<window::StreamDecoder<CpuRelayDecoder<A, Executor, Sink>>, ApiError>
make_stream_decoder(const WindowSetup& setup) {
    using Inner = CpuRelayDecoder<A, Executor, Sink>;
    const harness::DecoderSpec& spec = setup.spec;
    auto stream = window::StreamDecoder<Inner>::create(
        setup.plan, [&](const window::Shape& shape) -> std::expected<Inner, std::string> {
            auto backend = CpuBackend<A, Executor>::create(
                shape.graph(), shape.priors(),
                detail::make_executor<Executor>(spec.executor.team_threads));
            if (!backend) {
                return std::unexpected(std::format(
                    "backend: {}: {}", to_string(backend.error().code), backend.error().detail));
            }
            auto sink = detail::make_sink<Sink>(
                {.record_solutions = setup.record_solutions,
                 .column_class = shape.commit_class(),
                 .selection = spec.selection ? &spec.selection->config : nullptr,
                 .llr = shape.priors().llr(),
                 .graph = &shape.graph()});
            if (!sink) {
                return std::unexpected(std::move(sink.error()));
            }
            auto decoder = Inner::create(std::move(*backend), spec.min_sum, spec.relay,
                                         setup.gammas[shape.index()].get(), std::move(*sink));
            if (!decoder) {
                return std::unexpected(std::format(
                    "decoder: {}: {}", to_string(decoder.error().code), decoder.error().detail));
            }
            return std::move(*decoder);
        });
    if (!stream) {
        return std::unexpected(ApiError{.code = ApiError::Code::construction_failed,
                                        .detail = window::describe(stream.error())});
    }
    if (spec.selection) {
        if (auto set = stream->set_on_low_confidence(spec.selection->on_low); !set) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::construction_failed,
                         .detail = std::format("selection.confidence.on_low: {}",
                                               window::describe(set.error()))});
        }
    }
    return std::move(*stream);
}

// The per-window signal over the last L windows that a spec's "history" asks for, or nothing.
// Its commit signals are measured on the global problem (the committed faults are global
// columns), and a window commits at most the largest commit set of any shape.
std::expected<std::optional<window::SignalHistory>, ApiError>
make_history(const WindowSetup& setup) {
    const harness::DecoderSpec& spec = setup.spec;
    if (!spec.selection || spec.selection->history_lengths.empty()) {
        return std::optional<window::SignalHistory>{};
    }
    std::size_t max_commit = 0;
    for (const window::Shape& shape : setup.plan.shapes()) {
        max_commit = std::max<std::size_t>(max_commit, shape.committed_columns());
    }
    auto made = window::SignalHistory::create(
        spec.selection->history_lengths, spec.selection->history_signals, &setup.problem->graph(),
        setup.problem->priors().llr(), max_commit);
    if (!made) {
        return std::unexpected(
            ApiError{.code = ApiError::Code::construction_failed,
                     .detail = std::format("selection.history: {}", made.error().detail)});
    }
    return std::optional<window::SignalHistory>(std::move(*made));
}

// Decodes each shot window by window, exactly as rtd_decode's sliding worker does: every round
// is pushed, then every window position is decoded in order and its record, commits and
// solutions are written to the shot's row.
template <MessageArithmetic A, class Executor, class Sink>
class SlidingWorker final : public detail::ShotWorker {
public:
    using Inner = CpuRelayDecoder<A, Executor, Sink>;

    SlidingWorker(window::StreamDecoder<Inner> stream, std::optional<window::SignalHistory> history)
        : stream_(std::move(stream)), history_(std::move(history)) {}

    [[nodiscard]] std::expected<void, ApiError> decode(std::span<const Bit> syndrome,
                                                       std::uint64_t stream, std::size_t row,
                                                       std::span<Bit> predicted, BatchResult& out,
                                                       detail::CommitStaging* commits) override {
        const auto start = Clock::now();
        if (auto fed = feed(syndrome, stream); !fed) {
            return std::unexpected(detail::stream_failure(fed.error()));
        }
        if (history_) {
            history_->reset();
        }
        std::uint64_t window_ns = 0;
        while (!stream_.finished()) {
            const auto window_start = Clock::now();
            auto commit = stream_.decode_next();
            window_ns += nanoseconds(Clock::now() - window_start);
            if (!commit) {
                return std::unexpected(detail::stream_failure(commit.error()));
            }
            // With every round pushed a failed attempt is retried at once, so a deferral does
            // not return here; its time would count towards the position.
            if (commit->deferred) {
                continue;
            }
            write_window(*commit, window_ns, row, out, commits);
            window_ns = 0;
        }
        const window::ShotSummary& summary = stream_.summary();
        out.decode_ns[row] = nanoseconds(Clock::now() - start);
        out.success[row] = summary.success ? 1 : 0;
        out.iterations[row] = summary.iterations;
        out.legs[row] = summary.legs;
        out.best_leg[row] = -1;
        out.weight[row] = summary.weight;
        out.flagged[row] = summary.flagged ? 1 : 0;
        for (std::size_t o = 0; o < predicted.size(); ++o) {
            predicted[o] = static_cast<Bit>((summary.frame >> o) & 1U);
        }
        return {};
    }

private:
    // Resets the stream to `shot` and pushes every round of σ (Rt·M bits).
    std::expected<void, window::StreamError> feed(std::span<const Bit> syndrome,
                                                  std::uint64_t shot) noexcept {
        const window::WindowPlan& plan = stream_.plan();
        const std::size_t per_round = plan.detectors_per_round();
        const std::uint32_t rounds = plan.rounds_total();
        stream_.reset(shot);
        for (std::uint32_t r = 0; r + 1 < rounds; ++r) {
            if (auto pushed = stream_.push_round(syndrome.subspan(r * per_round, per_round));
                !pushed) {
                return pushed;
            }
        }
        return stream_.push_final(syndrome.subspan((rounds - 1) * per_round, per_round));
    }

    // Records window position commit.window of shot `row`; `window_ns` is the wall time of the
    // position's decode_next calls.
    void write_window(const window::Commit& commit, std::uint64_t window_ns, std::size_t row,
                      BatchResult& out, detail::CommitStaging* commits) {
        const std::size_t cell = (row * out.windows) + commit.window;
        const window::WindowRecord& record = commit.record;
        out.win_iterations[cell] = record.iterations;
        out.win_legs[cell] = record.legs;
        out.win_attempts[cell] = static_cast<std::uint8_t>(record.attempts);
        out.win_converged[cell] = record.converged ? 1 : 0;
        out.win_cap_hit[cell] = record.cap_hit ? 1 : 0;
        out.win_weight[cell] = record.weight;
        out.win_committed_weight[cell] = record.committed_weight;
        out.win_unexplained[cell] = record.unexplained;
        out.win_flagged[cell] = record.flagged ? 1 : 0;
        out.win_virtual[cell] = record.virtual_commits;
        out.win_decode_ns[cell] = window_ns;
        if (commits != nullptr) {
            commits->counts[cell] = static_cast<std::uint32_t>(commit.faults.size());
            std::vector<index_t>& faults = commits->faults[row];
            faults.insert(faults.end(), commit.faults.begin(), commit.faults.end());
        }
        if (!out.decodings.empty()) {
            // The commit sets of the positions are disjoint, so each fault is set once.
            Bit* dense = out.decodings.data() + (row * out.num_columns);
            for (const index_t j : commit.faults) {
                dense[j] ^= Bit{1};
            }
        }
        if constexpr (Sink::enabled) {
            const harness::SolutionRecorder* recorder =
                record.shape != window::no_shape
                    ? harness::recorder_of(stream_.inner(record.shape).sink())
                    : nullptr;
            if (recorder != nullptr) {
                // The committed attempt was the last decode of its shape's inner decoder, whose
                // sink still holds its solutions.
                detail::write_solutions(*recorder, record.returned_class, cell, out);
            }
        }
        if (out.confidence.enabled) {
            if (record.confidence) {
                harness::record_confidence(*record.confidence, cell, out.confidence);
            }
            out.confidence.low_deferrals[cell] = record.low_confidence_deferrals;
        }
        if (history_) {
            // The signal over the last L windows as a real-time rule would see it after this
            // window's commit.
            history_->push(record.confidence, record.shape != window::no_shape, commit.faults);
            harness::record_history(*history_, cell, out.confidence);
        }
    }

    window::StreamDecoder<Inner> stream_;
    std::optional<window::SignalHistory> history_;
};

// A stream for a caller that pushes rounds itself; applies the problem's biases.
template <MessageArithmetic A, class Executor, class Sink> class StreamImpl final : public Stream {
public:
    using Inner = CpuRelayDecoder<A, Executor, Sink>;

    StreamImpl(std::shared_ptr<const WindowSetup> setup, window::StreamDecoder<Inner> decoder,
               std::optional<window::SignalHistory> history)
        : setup_(std::move(setup)), decoder_(std::move(decoder)), history_(std::move(history)),
          round_(setup_->plan.detectors_per_round()), predicted_(setup_->plan.num_observables()) {}

    void reset(std::uint64_t shot) noexcept override {
        decoder_.reset(shot);
        if (history_) {
            history_->reset();
        }
    }

    [[nodiscard]] std::expected<void, ApiError>
    push_round(std::span<const Bit> detectors) override {
        return lift(decoder_.push_round(biased(detectors)));
    }

    [[nodiscard]] std::expected<void, ApiError>
    push_final(std::span<const Bit> detectors) override {
        return lift(decoder_.push_final(biased(detectors)));
    }

    [[nodiscard]] bool window_ready() const noexcept override { return decoder_.window_ready(); }

    [[nodiscard]] std::expected<window::Commit, ApiError> decode_next() override {
        auto commit = decoder_.decode_next();
        if (!commit) {
            return std::unexpected(detail::stream_failure(commit.error()));
        }
        if (history_ && !commit->deferred) {
            history_->push(commit->record.confidence, commit->record.shape != window::no_shape,
                           commit->faults);
        }
        return *commit;
    }

    [[nodiscard]] const window::SignalHistory* history() const noexcept override {
        return history_ ? &*history_ : nullptr;
    }

    [[nodiscard]] bool finished() const noexcept override { return decoder_.finished(); }
    [[nodiscard]] bool closed() const noexcept override { return decoder_.closed(); }
    [[nodiscard]] bool flagged() const noexcept override { return decoder_.flagged(); }
    [[nodiscard]] std::uint32_t rounds_received() const noexcept override {
        return decoder_.core().rounds_received();
    }
    [[nodiscard]] std::span<const Bit> frame() const noexcept override { return decoder_.frame(); }
    [[nodiscard]] std::span<const Bit> predicted() noexcept override {
        const std::span<const Bit> frame = decoder_.frame();
        const std::span<const Bit> bias = setup_->problem->observables_bias();
        for (std::size_t o = 0; o < frame.size(); ++o) {
            predicted_[o] = bias.empty() ? frame[o] : static_cast<Bit>(frame[o] ^ bias[o]);
        }
        return predicted_;
    }
    [[nodiscard]] window::ShotSummary summary() const noexcept override {
        return decoder_.summary();
    }
    [[nodiscard]] std::span<const window::WindowRecord> records() const noexcept override {
        return decoder_.records();
    }

private:
    static std::expected<void, ApiError> lift(std::expected<void, window::StreamError> pushed) {
        if (!pushed) {
            return std::unexpected(detail::stream_failure(pushed.error()));
        }
        return {};
    }

    // The round with the syndrome bias of its rows applied. A round of the wrong size, or one
    // past the readout, is passed on unchanged for the stream decoder to refuse.
    [[nodiscard]] std::span<const Bit> biased(std::span<const Bit> detectors) noexcept {
        const std::span<const Bit> bias = setup_->problem->syndrome_bias();
        const std::size_t per_round = round_.size();
        const std::size_t first = std::size_t{decoder_.core().rounds_received()} * per_round;
        if (bias.empty() || detectors.size() != per_round || first + per_round > bias.size()) {
            return detectors;
        }
        for (std::size_t i = 0; i < per_round; ++i) {
            round_[i] = static_cast<Bit>((detectors[i] != 0 ? 1 : 0) ^ bias[first + i]);
        }
        return round_;
    }

    // Declared first, so the plan and γ sources outlive the decoders that point into them.
    std::shared_ptr<const WindowSetup> setup_;
    window::StreamDecoder<Inner> decoder_;
    std::optional<window::SignalHistory> history_;
    std::vector<Bit> round_;
    std::vector<Bit> predicted_;
};

} // namespace

struct WindowedDecoder::Impl {
    std::shared_ptr<const WindowSetup> setup;
    std::vector<std::unique_ptr<detail::ShotWorker>> workers;
    std::mutex batch;
};

WindowedDecoder::WindowedDecoder(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
WindowedDecoder::~WindowedDecoder() = default;

std::expected<std::unique_ptr<WindowedDecoder>, ApiError>
WindowedDecoder::create(std::shared_ptr<const Problem> problem, const harness::DecoderSpec& spec,
                        const DecoderOptions& options) {
    if (!problem) {
        return std::unexpected(
            ApiError{.code = ApiError::Code::invalid_problem, .detail = "no problem given"});
    }
    if (!spec.window.is_sliding()) {
        return std::unexpected(ApiError{
            .code = ApiError::Code::invalid_spec,
            .detail = "window.mode is whole_shot: a whole-shot spec needs a WholeShotDecoder"});
    }
    if (!problem->has_rounds()) {
        return std::unexpected(
            ApiError{.code = ApiError::Code::invalid_problem,
                     .detail = "sliding windows need the round of every detector"});
    }
    // Window shapes are classified by their commit classes, so k ≤ 64 is the plan's own
    // requirement; the plan builder reports it.
    if (auto valid =
            detail::check_decoder_setup(spec, options.workers, options.record_solutions, true);
        !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    auto plan =
        window::WindowPlan::build(problem->window_problem(), spec.window.sliding, spec.graph);
    if (!plan) {
        return std::unexpected(ApiError{.code = ApiError::Code::plan_rejected,
                                        .detail = window::describe(plan.error())});
    }
    // win_attempts is recorded in one byte per window, as rtd_decode records it.
    constexpr std::uint32_t max_attempts = std::numeric_limits<std::uint8_t>::max();
    for (std::uint32_t k = 0; k < plan->num_positions(); ++k) {
        if (plan->attempts(k) > max_attempts) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::plan_rejected,
                         .detail = std::format("window position {} has {} attempts; at most {} are "
                                               "recorded",
                                               k, plan->attempts(k), max_attempts)});
        }
    }
    auto setup = std::make_shared<WindowSetup>(std::move(problem), spec, std::move(*plan),
                                               options.record_solutions);
    auto gammas = detail::shape_gammas(spec.gamma, setup->plan);
    if (!gammas) {
        return std::unexpected(std::move(gammas.error()));
    }
    setup->gammas = std::move(*gammas);

    auto impl = std::make_unique<Impl>();
    impl->setup = setup;
    const WindowSetup& shared = *setup;
    const auto make_worker = [&]<class A, class Executor, class Sink>()
        -> std::expected<std::unique_ptr<detail::ShotWorker>, ApiError> {
        auto stream = make_stream_decoder<A, Executor, Sink>(shared);
        if (!stream) {
            return std::unexpected(std::move(stream.error()));
        }
        auto history = make_history(shared);
        if (!history) {
            return std::unexpected(std::move(history.error()));
        }
        return std::make_unique<SlidingWorker<A, Executor, Sink>>(std::move(*stream),
                                                                  std::move(*history));
    };
    impl->workers.reserve(options.workers);
    for (unsigned w = 0; w < options.workers; ++w) {
        auto worker = detail::with_decoder_types(spec, options.record_solutions, make_worker);
        if (!worker) {
            return std::unexpected(std::move(worker.error()));
        }
        impl->workers.push_back(std::move(*worker));
    }
    return std::unique_ptr<WindowedDecoder>(new WindowedDecoder(std::move(impl)));
}

const Problem& WindowedDecoder::problem() const noexcept { return *impl_->setup->problem; }
const harness::DecoderSpec& WindowedDecoder::spec() const noexcept { return impl_->setup->spec; }
const window::WindowPlan& WindowedDecoder::plan() const noexcept { return impl_->setup->plan; }
unsigned WindowedDecoder::workers() const noexcept {
    return static_cast<unsigned>(impl_->workers.size());
}
std::uint32_t WindowedDecoder::record_solutions() const noexcept {
    return impl_->setup->record_solutions;
}

std::expected<BatchResult, ApiError> WindowedDecoder::decode_batch(const BatchInput& input,
                                                                   const BatchOptions& options) {
    const std::scoped_lock lock(impl_->batch);
    const WindowSetup& setup = *impl_->setup;
    const harness::DecoderSpec& spec = setup.spec;
    return detail::run_batch(impl_->workers, *setup.problem,
                             detail::BatchShape{.sliding = true,
                                                .windows = setup.plan.num_positions(),
                                                .solution_slots = setup.record_solutions,
                                                .selection = spec.selection ? &*spec.selection
                                                                            : nullptr},
                             input, options);
}

std::expected<std::unique_ptr<Stream>, ApiError> WindowedDecoder::open_stream() const {
    const std::shared_ptr<const WindowSetup>& setup = impl_->setup;
    const auto make = [&]<class A, class Executor, class Sink>()
        -> std::expected<std::unique_ptr<Stream>, ApiError> {
        auto stream = make_stream_decoder<A, Executor, Sink>(*setup);
        if (!stream) {
            return std::unexpected(std::move(stream.error()));
        }
        auto history = make_history(*setup);
        if (!history) {
            return std::unexpected(std::move(history.error()));
        }
        return std::make_unique<StreamImpl<A, Executor, Sink>>(setup, std::move(*stream),
                                                               std::move(*history));
    };
    return detail::with_decoder_types(setup->spec, setup->record_solutions, make);
}

} // namespace rtd::api
