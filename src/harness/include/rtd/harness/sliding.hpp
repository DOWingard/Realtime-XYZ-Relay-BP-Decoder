#pragma once

// Sliding-window decoding in rtd_decode: the window plan and per-shape γ sources that every
// worker of a run shares, the rate-limited log of window events, and the worker that decodes a
// shot window by window with a StreamDecoder over one relay decoder per window shape.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/arith.hpp"
#include "rtd/core/fixed_arith.hpp"
#include "rtd/core/executor.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/harness/batch.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/spec.hpp"
#include "rtd/window/stream.hpp"

namespace rtd::harness {

class Worker;

// What every worker of a sliding-window run shares: the window plan, built once from the
// artifact, and the γ source of each window shape. Stream decoders keep pointers to the plan
// object itself and relay decoders to its shapes and to the γ sources, so a setup is created on
// the heap and stays where it is until the run is over.
struct SlidingSetup {
    SlidingSetup(window::WindowPlan built, double seconds)
        : plan(std::move(built)), build_seconds(seconds) {}
    SlidingSetup(const SlidingSetup&) = delete;
    SlidingSetup& operator=(const SlidingSetup&) = delete;
    SlidingSetup(SlidingSetup&&) = delete;
    SlidingSetup& operator=(SlidingSetup&&) = delete;
    ~SlidingSetup() = default;

    window::WindowPlan plan;
    // One per shape, in shape order; null when the spec has no relay legs. `owned` holds them.
    std::vector<const GammaSource*> gammas;
    std::vector<std::unique_ptr<GammaSource>> owned;
    double build_seconds = 0.0; // plan construction alone
};

// Why a sliding-window run cannot start. `input` marks a γ table that is missing, unreadable or
// of the wrong width (an input error); otherwise the specification does not work with this
// artifact (a configuration error).
struct SetupError {
    bool input = false;
    std::string context;
    std::string message;
    nlohmann::json fields;
};

// Builds the plan of spec.window.sliding over `artifact`, every shape's Tanner graph built with
// the spec's graph options, and the γ source of every shape:
//   uniform          one generator per shape, same seed and interval, width n_i (the stream
//                    decoder keys window k, attempt a of shot s on s + (k << 32) + (a << 56), so
//                    a plan with a single window reproduces a whole-shot run);
//   explicit_shapes  directory/shape_<i>.npy, a [T, n_i] table for every shape i;
//   explicit         one [T, n] table, accepted only when the plan has a single shape.
// Logs the plan's statistics ("window plan built").
[[nodiscard]] std::expected<std::unique_ptr<SlidingSetup>, SetupError>
build_sliding(const io::Artifact& artifact, const DecoderSpec& spec, JsonLogger& logger);

// The plan's statistics for run.json: rounds, positions, placements, and per shape its size,
// merged and committed columns.
[[nodiscard]] nlohmann::json plan_record(const SlidingSetup& setup);

// The window events of one run, counted across its workers and logged at a bounded rate: the
// first `logged_per_kind` events of each kind one line each, the rest only counted and reported
// by report() when decoding has finished, so a bad operating point cannot flood the log.
class WindowEvents {
public:
    enum class Kind : std::uint8_t {
        not_converged,  // warn: no leg converged; commit_anyway or flag committed leg 0's ê
        deferred,       // warn: an attempt failed and the window was decoded again, wider
        deferral_limit, // error: every deferral attempt failed; the widest one was committed
        cap_hit,        // warn: the committed attempt ran out of its iteration budget
    };
    static constexpr std::size_t kinds = 4;
    static constexpr std::uint64_t logged_per_kind = 20;

    WindowEvents(JsonLogger& logger, const window::WindowSpec& spec) noexcept
        : logger_(&logger), spec_(spec) {}

    // Accounts one window position of one shot, from a worker, after its decode.
    void observe(std::size_t shot, const window::WindowRecord& record);
    // Per kind: events seen so far.
    [[nodiscard]] nlohmann::json counts() const;
    // One line per kind that occurred: its total and how many were not logged individually.
    void report() const;

private:
    void emit(Kind kind, std::size_t shot, const window::WindowRecord& record);

    JsonLogger* logger_;
    window::WindowSpec spec_;
    std::array<std::atomic<std::uint64_t>, kinds> counts_{};
};

// One sliding-window worker: a StreamDecoder whose inner decoder for each window shape is a CPU
// relay decoder with an executor from `executor()` (one per shape) and, when solutions are
// recorded, a SolutionRecorder whose column classes are the shape's commit classes.
template <MessageArithmetic A, class Executor>
[[nodiscard]] std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker(const io::Artifact& artifact, const DecoderSpec& spec,
                    const SlidingSetup& setup, const std::function<Executor()>& executor,
                    const RunOptions& options, WindowEvents& events);

extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F32, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                 const std::function<Serial()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F64, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                 const std::function<Serial()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F32, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                               const std::function<Team()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<F64, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                               const std::function<Team()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int4_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int5_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int6_2_8, Serial>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                      const std::function<Serial()>&, const RunOptions&,
                                      WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int4_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int5_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);
extern template std::expected<std::unique_ptr<Worker>, HarnessError>
make_sliding_worker<Int6_2_8, Team>(const io::Artifact&, const DecoderSpec&, const SlidingSetup&,
                                    const std::function<Team()>&, const RunOptions&, WindowEvents&);

} // namespace rtd::harness
