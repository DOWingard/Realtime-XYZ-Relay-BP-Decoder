#pragma once

#include <expected>
#include <memory>

#include "rtd/api/batch.hpp"
#include "rtd/api/error.hpp"
#include "rtd/api/problem.hpp"
#include "rtd/harness/spec.hpp"

namespace rtd::api {

// Relay-BP (or plain min-sum, or Mem-BP: whatever the spec configures) over whole shots, from
// arrays in memory: the decoder rtd_decode runs for a whole-shot spec, with the same γ streams,
// so a batch reproduces an rtd_decode run over the same shots bit for bit.
//
// decode_batch runs one batch at a time (concurrent calls wait for each other); within a batch,
// shots are spread over the workers, each with its own decoder, and the results do not depend on
// the number of workers.
class WholeShotDecoder {
public:
    // Builds `options.workers` decoders for `spec` (whose window mode must be whole_shot) over
    // `problem`, whose graph must have been built with spec.graph.
    [[nodiscard]] static std::expected<std::unique_ptr<WholeShotDecoder>, ApiError>
    create(std::shared_ptr<const Problem> problem, const harness::DecoderSpec& spec,
           const DecoderOptions& options);

    WholeShotDecoder(const WholeShotDecoder&) = delete;
    WholeShotDecoder& operator=(const WholeShotDecoder&) = delete;
    WholeShotDecoder(WholeShotDecoder&&) = delete;
    WholeShotDecoder& operator=(WholeShotDecoder&&) = delete;
    ~WholeShotDecoder();

    [[nodiscard]] const Problem& problem() const noexcept;
    [[nodiscard]] const harness::DecoderSpec& spec() const noexcept;
    [[nodiscard]] unsigned workers() const noexcept;
    [[nodiscard]] std::uint32_t record_solutions() const noexcept;

    [[nodiscard]] std::expected<BatchResult, ApiError> decode_batch(const BatchInput& input,
                                                                    const BatchOptions& options);

private:
    struct Impl;
    explicit WholeShotDecoder(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace rtd::api
