#include "rtd/api/decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <mutex>
#include <utility>
#include <vector>

#include "dispatch.hpp"
#include "driver.hpp"
#include "sources.hpp"

namespace rtd::api {

namespace {

using Clock = std::chrono::steady_clock;

// Decodes each shot as one problem over all its rounds, exactly as rtd_decode's whole-shot
// worker does: ℓ̂ = A·ê over the returned correction's support (with a selection policy, the
// support of the solution the policy decided), and the decode's confidence when there is one.
template <MessageArithmetic A, class Executor, class Sink>
class WholeShotWorker final : public detail::ShotWorker {
public:
    using Decoder = CpuRelayDecoder<A, Executor, Sink>;

    WholeShotWorker(Decoder decoder, const Problem& problem)
        : decoder_(std::move(decoder)), problem_(&problem) {}

    [[nodiscard]] std::expected<void, ApiError>
    decode(std::span<const Bit> syndrome, std::uint64_t stream, std::size_t row,
           std::span<Bit> predicted, BatchResult& out,
           detail::CommitStaging* /*commits*/) override {
        const auto start = Clock::now();
        const auto result = decoder_.decode(syndrome, stream);
        const auto elapsed = Clock::now() - start;
        if (!result) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::decode_failed,
                         .detail = std::format("{}: expected {} detector bits, found {}",
                                               to_string(result.error().code),
                                               result.error().expected, result.error().found)});
        }
        out.decode_ns[row] = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        out.success[row] = result->success ? 1 : 0;
        out.iterations[row] = result->iterations;
        out.legs[row] = result->legs_executed;
        out.best_leg[row] = result->best_leg ? static_cast<std::int32_t>(*result->best_leg) : -1;
        out.weight[row] = result->weight;
        problem_->observables().apply_support(result->support, predicted);
        if (!out.decodings.empty()) {
            const std::size_t n = out.num_columns;
            std::ranges::copy(result->hard,
                              out.decodings.begin() + static_cast<std::ptrdiff_t>(row * n));
        }
        if constexpr (Sink::enabled) {
            if (const harness::SolutionRecorder* recorder = harness::recorder_of(decoder_.sink())) {
                detail::write_solutions(*recorder, recorder->class_of(result->support), row, out);
            }
        }
        if (result->confidence) {
            harness::record_confidence(*result->confidence, row, out.confidence);
        }
        return {};
    }

private:
    Decoder decoder_;
    const Problem* problem_;
};

} // namespace

struct WholeShotDecoder::Impl {
    std::shared_ptr<const Problem> problem;
    harness::DecoderSpec spec;
    std::uint32_t record_solutions = 0;
    // Declared before the workers, whose decoders point into it.
    std::unique_ptr<GammaSource> gammas;
    std::vector<std::unique_ptr<detail::ShotWorker>> workers;
    std::mutex batch;
};

WholeShotDecoder::WholeShotDecoder(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
WholeShotDecoder::~WholeShotDecoder() = default;

std::expected<std::unique_ptr<WholeShotDecoder>, ApiError>
WholeShotDecoder::create(std::shared_ptr<const Problem> problem, const harness::DecoderSpec& spec,
                         const DecoderOptions& options) {
    if (!problem) {
        return std::unexpected(
            ApiError{.code = ApiError::Code::invalid_problem, .detail = "no problem given"});
    }
    if (spec.window.is_sliding()) {
        return std::unexpected(ApiError{
            .code = ApiError::Code::invalid_spec,
            .detail = "window.mode is sliding: a sliding-window spec needs a WindowedDecoder"});
    }
    if (auto valid = detail::check_decoder_setup(
            spec, options.workers, options.record_solutions,
            problem->column_classes().size() == problem->num_columns());
        !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    auto impl = std::make_unique<Impl>();
    impl->problem = std::move(problem);
    impl->spec = spec;
    impl->record_solutions = options.record_solutions;
    auto gammas = detail::whole_shot_gammas(spec.gamma, impl->problem->num_columns());
    if (!gammas) {
        return std::unexpected(std::move(gammas.error()));
    }
    impl->gammas = std::move(*gammas);

    const Problem& p = *impl->problem;
    const GammaSource* source = impl->gammas.get();
    const auto make_worker = [&]<class A, class Executor, class Sink>()
        -> std::expected<std::unique_ptr<detail::ShotWorker>, ApiError> {
        auto backend = CpuBackend<A, Executor>::create(
            p.graph(), p.priors(), detail::make_executor<Executor>(spec.executor.team_threads));
        if (!backend) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::construction_failed,
                         .detail = std::format("backend: {}: {}", to_string(backend.error().code),
                                               backend.error().detail)});
        }
        auto sink = detail::make_sink<Sink>(
            {.record_solutions = options.record_solutions,
             .column_class = p.column_classes(),
             .selection = spec.selection ? &spec.selection->config : nullptr,
             .llr = p.priors().llr(),
             .graph = &p.graph()});
        if (!sink) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::construction_failed, .detail = sink.error()});
        }
        auto decoder = CpuRelayDecoder<A, Executor, Sink>::create(
            std::move(*backend), spec.min_sum, spec.relay, source, std::move(*sink));
        if (!decoder) {
            return std::unexpected(
                ApiError{.code = ApiError::Code::construction_failed,
                         .detail = std::format("decoder: {}: {}", to_string(decoder.error().code),
                                               decoder.error().detail)});
        }
        return std::make_unique<WholeShotWorker<A, Executor, Sink>>(std::move(*decoder), p);
    };
    impl->workers.reserve(options.workers);
    for (unsigned w = 0; w < options.workers; ++w) {
        auto worker = detail::with_decoder_types(spec, options.record_solutions, make_worker);
        if (!worker) {
            return std::unexpected(std::move(worker.error()));
        }
        impl->workers.push_back(std::move(*worker));
    }
    return std::unique_ptr<WholeShotDecoder>(new WholeShotDecoder(std::move(impl)));
}

const Problem& WholeShotDecoder::problem() const noexcept { return *impl_->problem; }
const harness::DecoderSpec& WholeShotDecoder::spec() const noexcept { return impl_->spec; }
unsigned WholeShotDecoder::workers() const noexcept {
    return static_cast<unsigned>(impl_->workers.size());
}
std::uint32_t WholeShotDecoder::record_solutions() const noexcept {
    return impl_->record_solutions;
}

std::expected<BatchResult, ApiError> WholeShotDecoder::decode_batch(const BatchInput& input,
                                                                    const BatchOptions& options) {
    const std::scoped_lock lock(impl_->batch);
    const harness::DecoderSpec& spec = impl_->spec;
    return detail::run_batch(impl_->workers, *impl_->problem,
                             detail::BatchShape{.sliding = false,
                                                .windows = 1,
                                                .solution_slots = impl_->record_solutions,
                                                .selection = spec.selection ? &*spec.selection
                                                                            : nullptr},
                             input, options);
}

} // namespace rtd::api
