#include <algorithm>
#include <atomic>
#include <exception>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include "driver.hpp"
#include "rtd/api/batch.hpp"

namespace rtd::api {

namespace {

std::unexpected<ApiError> error(ApiError::Code code, std::string detail) {
    return std::unexpected(ApiError{.code = code, .detail = std::move(detail)});
}

} // namespace

std::expected<harness::DecoderSpec, ApiError> parse_spec(std::string_view json_text,
                                                         const std::filesystem::path& base) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        return error(ApiError::Code::invalid_spec, std::format("not valid JSON: {}", e.what()));
    }
    auto spec = harness::parse_spec(document, base);
    if (!spec) {
        const harness::SpecError& failure = spec.error();
        return error(ApiError::Code::invalid_spec,
                     failure.field.empty() ? failure.problem
                                           : std::format("{}: {}", failure.field, failure.problem));
    }
    if (spec->backend != harness::BackendKind::cpu) {
        return error(ApiError::Code::invalid_spec, "backend: this API runs the cpu backend only");
    }
    return std::move(*spec);
}

std::expected<void, ApiError> check_batch(const BatchInput& input, index_t num_rows) {
    const std::size_t expected_row = row_bytes_for(num_rows, input.bit_packed);
    if (input.shots == 0) {
        return error(ApiError::Code::invalid_input, "the batch holds no shots");
    }
    if (input.row_bytes != expected_row) {
        return error(ApiError::Code::invalid_input,
                     std::format("each shot needs {} bytes for {} detectors{}, found {}",
                                 expected_row, num_rows, input.bit_packed ? " (bit-packed)" : "",
                                 input.row_bytes));
    }
    const std::size_t bytes = input.detectors.size();
    const bool fits = expected_row == 0
                          ? bytes == 0
                          : bytes % expected_row == 0 && bytes / expected_row == input.shots;
    if (!fits) {
        return error(ApiError::Code::invalid_input,
                     std::format("{} shots of {} bytes need {} bytes, found {}", input.shots,
                                 expected_row, input.shots * expected_row, input.detectors.size()));
    }
    return {};
}

namespace detail {

namespace {

// Preallocates every array the batch records, with the value an undecoded entry keeps.
BatchResult allocate(const Problem& problem, const BatchShape& shape, std::size_t shots,
                     const BatchOptions& options) {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    BatchResult r;
    r.shots = shots;
    r.num_columns = problem.num_columns();
    r.num_observables = problem.num_observables();
    r.predicted_row_bytes = row_bytes_for(r.num_observables, options.pack_predictions);
    r.predicted.assign(shots * r.predicted_row_bytes, 0);
    r.success.assign(shots, 0);
    r.iterations.assign(shots, 0);
    r.legs.assign(shots, 0);
    r.best_leg.assign(shots, -1);
    r.weight.assign(shots, infinity);
    r.decode_ns.assign(shots, 0);
    if (options.save_decodings) {
        r.decodings.assign(shots * r.num_columns, 0);
    }
    r.windows = shape.windows;
    const std::size_t cells = shots * shape.windows;
    if (shape.sliding) {
        r.flagged.assign(shots, 0);
        r.win_iterations.assign(cells, 0);
        r.win_legs.assign(cells, 0);
        r.win_attempts.assign(cells, 0);
        r.win_converged.assign(cells, 0);
        r.win_cap_hit.assign(cells, 0);
        r.win_weight.assign(cells, infinity);
        r.win_committed_weight.assign(cells, 0.0);
        r.win_unexplained.assign(cells, 0);
        r.win_flagged.assign(cells, 0);
        r.win_virtual.assign(cells, 0);
        r.win_decode_ns.assign(cells, 0);
    }
    r.solution_slots = shape.solution_slots;
    if (shape.solution_slots > 0) {
        const std::size_t slots = cells * shape.solution_slots;
        r.sol_count.assign(cells, 0);
        r.returned_class.assign(cells, 0);
        r.sol_leg.assign(slots, 0);
        r.sol_iterations.assign(slots, 0);
        r.sol_weight.assign(slots, infinity);
        r.sol_class.assign(slots, 0);
        r.sol_hash.assign(slots, 0);
        r.sol_size.assign(slots, 0);
    }
    if (shape.selection != nullptr) {
        harness::size_confidence_outputs(r.confidence, cells, *shape.selection, shape.sliding);
    }
    return r;
}

// σ of one shot as m bytes of 0 or 1 with the syndrome bias applied: the row itself when it is
// unpacked and there is no bias, otherwise `scratch` filled from it.
std::span<const Bit> syndrome_of(std::span<const Bit> row, bool packed, std::span<const Bit> bias,
                                 std::span<Bit> scratch) noexcept {
    if (!packed && bias.empty()) {
        return row;
    }
    const std::size_t m = scratch.size();
    for (std::size_t i = 0; i < m; ++i) {
        const Bit bit = packed ? static_cast<Bit>((unsigned{row[i >> 3U]} >> (i & 7U)) & 1U)
                               : static_cast<Bit>(row[i] != 0);
        scratch[i] = bias.empty() ? bit : static_cast<Bit>(bit ^ bias[i]);
    }
    return scratch;
}

// Row `shot` of the predictions: ℓ̂ ⊕ b_ℓ, one byte per observable or bit-packed.
void write_prediction(std::span<const Bit> predicted, std::span<const Bit> bias, bool packed,
                      std::span<Bit> out) noexcept {
    std::ranges::fill(out, Bit{0});
    for (std::size_t o = 0; o < predicted.size(); ++o) {
        const auto bit = static_cast<Bit>((predicted[o] ^ (bias.empty() ? Bit{0} : bias[o])) & 1U);
        if (packed) {
            out[o >> 3U] = static_cast<Bit>(out[o >> 3U] | (bit << (o & 7U)));
        } else {
            out[o] = bit;
        }
    }
}

// Joins the per-shot committed faults into the CSR arrays over (shot, window) cells.
std::expected<void, ApiError> gather_commits(CommitStaging& staging, BatchResult& result) {
    const std::size_t cells = staging.counts.size();
    result.commit_ptr.assign(cells + 1, 0);
    for (std::size_t q = 0; q < cells; ++q) {
        result.commit_ptr[q + 1] = result.commit_ptr[q] + staging.counts[q];
    }
    result.commit_faults.resize(result.commit_ptr.back());
    const std::size_t per_row = result.windows;
    for (std::size_t row = 0; row < result.shots; ++row) {
        const std::uint64_t begin = result.commit_ptr[row * per_row];
        const std::uint64_t end = result.commit_ptr[(row + 1) * per_row];
        std::vector<index_t>& faults = staging.faults[row];
        if (faults.size() != end - begin) {
            return error(ApiError::Code::decode_failed,
                         std::format("shot {} stored {} committed faults but its windows count {}",
                                     row, faults.size(), end - begin));
        }
        std::ranges::copy(faults,
                          result.commit_faults.begin() + static_cast<std::ptrdiff_t>(begin));
        std::vector<index_t>().swap(faults);
    }
    return {};
}

// What the threads of one batch share. Each shot's row is written by exactly one worker.
struct BatchRun {
    std::span<const std::unique_ptr<ShotWorker>> workers;
    const Problem* problem = nullptr;
    const BatchInput* input = nullptr;
    const BatchOptions* options = nullptr;
    BatchResult* result = nullptr;
    CommitStaging* commits = nullptr;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex failure_mutex; // taken only when a worker fails
    std::optional<ApiError> failure;

    void fail(ApiError failed) noexcept {
        stop.store(true, std::memory_order_relaxed);
        const std::scoped_lock lock(failure_mutex);
        if (!failure) {
            failure = std::move(failed);
        }
    }

    // Worker w's share: shots claimed one at a time until none are left or a worker failed.
    // Exceptions (allocation) become the batch's failure.
    void work(unsigned w) {
        try {
            const index_t m = problem->num_rows();
            const index_t k = problem->num_observables();
            std::vector<Bit> syndrome(m);
            std::vector<Bit> predicted(k);
            const std::size_t out_bytes = result->predicted_row_bytes;
            for (;;) {
                const std::size_t s = next.fetch_add(1, std::memory_order_relaxed);
                if (s >= input->shots || stop.load(std::memory_order_relaxed)) {
                    break;
                }
                const std::span<const Bit> row =
                    input->detectors.subspan(s * input->row_bytes, input->row_bytes);
                const std::span<const Bit> sigma =
                    syndrome_of(row, input->bit_packed, problem->syndrome_bias(), syndrome);
                auto decoded = workers[w]->decode(sigma, input->stream_offset + s, s, predicted,
                                                  *result, commits);
                if (!decoded) {
                    decoded.error().detail = std::format("shot {}: {}", s, decoded.error().detail);
                    fail(std::move(decoded.error()));
                    break;
                }
                write_prediction(predicted, problem->observables_bias(), options->pack_predictions,
                                 std::span(result->predicted).subspan(s * out_bytes, out_bytes));
            }
        } catch (const std::exception& e) {
            fail(ApiError{.code = ApiError::Code::decode_failed,
                          .detail = std::format("worker {} stopped: {}", w, e.what())});
        } catch (...) {
            fail(ApiError{.code = ApiError::Code::decode_failed,
                          .detail = std::format("worker {} stopped: a non-standard exception", w)});
        }
    }
};

} // namespace

std::expected<BatchResult, ApiError> run_batch(std::span<const std::unique_ptr<ShotWorker>> workers,
                                               const Problem& problem, const BatchShape& shape,
                                               const BatchInput& input,
                                               const BatchOptions& options) {
    if (auto valid = check_batch(input, problem.num_rows()); !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    if (options.workers == 0 || options.workers > workers.size()) {
        return error(ApiError::Code::invalid_input,
                     std::format("workers must be in [1, {}] (the decoder was built with {}), "
                                 "got {}",
                                 workers.size(), workers.size(), options.workers));
    }
    if (options.save_commits && !shape.sliding) {
        return error(ApiError::Code::invalid_input,
                     "save_commits records the faults each window commits and needs a windowed "
                     "decoder; a whole-shot decoder returns ê with save_decodings");
    }
    BatchRun run;
    BatchResult result = allocate(problem, shape, input.shots, options);
    CommitStaging staging;
    if (options.save_commits) {
        staging.faults.resize(input.shots);
        staging.counts.assign(input.shots * shape.windows, 0);
        run.commits = &staging;
    }
    run.workers = workers;
    run.problem = &problem;
    run.input = &input;
    run.options = &options;
    run.result = &result;

    const auto threads = static_cast<unsigned>(std::min<std::size_t>(options.workers, input.shots));
    {
        std::vector<std::jthread> pool;
        try {
            pool.reserve(threads - 1);
            for (unsigned w = 1; w < threads; ++w) {
                pool.emplace_back([&run, w] { run.work(w); });
            }
        } catch (const std::system_error& e) {
            run.fail(
                ApiError{.code = ApiError::Code::decode_failed,
                         .detail = std::format("could not start worker threads: {}", e.what())});
        }
        run.work(0);
    } // joins
    if (run.failure) {
        return std::unexpected(std::move(*run.failure));
    }
    if (options.save_commits) {
        if (auto gathered = gather_commits(staging, result); !gathered) {
            return std::unexpected(std::move(gathered.error()));
        }
    }
    return result;
}

ApiError stream_failure(const window::StreamError& error) {
    using enum window::StreamError::Code;
    ApiError::Code code = ApiError::Code::decode_failed;
    if (error.code == wrong_round_size || error.code == wrong_syndrome_size) {
        code = ApiError::Code::invalid_input;
    } else if (error.code == too_many_rounds || error.code == missing_rounds ||
               error.code == stream_closed || error.code == not_ready) {
        code = ApiError::Code::stream_state;
    }
    return ApiError{.code = code, .detail = window::describe(error)};
}

} // namespace detail

} // namespace rtd::api
