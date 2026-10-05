#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include "rtd/api/batch.hpp"
#include "rtd/api/error.hpp"
#include "rtd/api/problem.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/window/history.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"

namespace rtd::api {

struct WindowSetup;

// One stream of syndrome rounds decoded window by window (window::StreamDecoder with a relay
// decoder per window shape), for a caller that receives rounds one at a time.
//
// Rounds are pushed raw: the problem's syndrome bias, if any, is applied here, and predicted()
// applies the observable bias to the frame. With a selection policy every committed window's
// record carries its confidence, and the policy's low-confidence action (defer or flag) applies. A stream is single-threaded: calls must not overlap.
// It keeps the plan and γ sources alive, so it may outlive the WindowedDecoder that opened it.
// Pushing and decoding never allocate; only building an error message does.
class Stream {
public:
    Stream() = default;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) = delete;
    Stream& operator=(Stream&&) = delete;
    virtual ~Stream() = default;

    // Starts a new shot; `shot` keys its γ streams (window k, attempt a draw from
    // shot + (k << 32) + (a << 56)).
    virtual void reset(std::uint64_t shot) noexcept = 0;
    // The next noisy round (rounds 1 … Rt − 1), M detector bits of one byte each.
    [[nodiscard]] virtual std::expected<void, ApiError>
    push_round(std::span<const Bit> detectors) = 0;
    // The readout round Rt; closes the stream.
    [[nodiscard]] virtual std::expected<void, ApiError>
    push_final(std::span<const Bit> detectors) = 0;
    [[nodiscard]] virtual bool window_ready() const noexcept = 0;
    // Decodes the oldest ready window. The views in the Commit stay valid until the next call
    // that changes the stream.
    [[nodiscard]] virtual std::expected<window::Commit, ApiError> decode_next() = 0;

    [[nodiscard]] virtual bool finished() const noexcept = 0;
    [[nodiscard]] virtual bool closed() const noexcept = 0;
    [[nodiscard]] virtual bool flagged() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t rounds_received() const noexcept = 0;
    // ℓ̂ accumulated by the commits so far (one byte per observable), without the observable
    // bias; predicted() is the same with the bias applied.
    [[nodiscard]] virtual std::span<const Bit> frame() const noexcept = 0;
    [[nodiscard]] virtual std::span<const Bit> predicted() noexcept = 0;
    [[nodiscard]] virtual window::ShotSummary summary() const noexcept = 0;
    [[nodiscard]] virtual std::span<const window::WindowRecord> records() const noexcept = 0;
    // With a selection policy whose spec asks for "history": the per-window signal over the last
    // L windows, after the last committed window of this shot (lengths and signals in the spec's
    // order). Null otherwise.
    [[nodiscard]] virtual const window::SignalHistory* history() const noexcept = 0;
};

// Sliding-window Relay-BP from arrays in memory: the decoder rtd_decode runs for a sliding spec,
// with the same window plan, γ sources and streams, so a batch reproduces an rtd_decode run over
// the same shots bit for bit. Batches run one at a time, spread over the workers.
class WindowedDecoder {
public:
    // Builds the window plan of spec.window.sliding over `problem` (which needs detector rounds
    // and a graph built with spec.graph), the γ source of every window shape, and
    // `options.workers` stream decoders.
    [[nodiscard]] static std::expected<std::unique_ptr<WindowedDecoder>, ApiError>
    create(std::shared_ptr<const Problem> problem, const harness::DecoderSpec& spec,
           const DecoderOptions& options);

    WindowedDecoder(const WindowedDecoder&) = delete;
    WindowedDecoder& operator=(const WindowedDecoder&) = delete;
    WindowedDecoder(WindowedDecoder&&) = delete;
    WindowedDecoder& operator=(WindowedDecoder&&) = delete;
    ~WindowedDecoder();

    [[nodiscard]] const Problem& problem() const noexcept;
    [[nodiscard]] const harness::DecoderSpec& spec() const noexcept;
    [[nodiscard]] const window::WindowPlan& plan() const noexcept;
    [[nodiscard]] unsigned workers() const noexcept;
    [[nodiscard]] std::uint32_t record_solutions() const noexcept;

    [[nodiscard]] std::expected<BatchResult, ApiError> decode_batch(const BatchInput& input,
                                                                    const BatchOptions& options);

    // A new, independent stream with its own inner decoders.
    [[nodiscard]] std::expected<std::unique_ptr<Stream>, ApiError> open_stream() const;

private:
    struct Impl;
    explicit WindowedDecoder(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace rtd::api
