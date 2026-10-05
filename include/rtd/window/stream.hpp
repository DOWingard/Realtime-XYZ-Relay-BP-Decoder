#pragma once

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "rtd/core/confidence.hpp"
#include "rtd/core/result.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/core/types.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/spec.hpp"

namespace rtd::window {

// What the window layer needs of an inner decoder: its row and column counts, a convergence mask
// (1 = the row must satisfy H·ê = σ for the decode to count as converged), and a decode of one
// window syndrome within an iteration budget. `stream` keys the decoder's random draws; the
// returned DecodeResult's `support` is ê as ascending local column indices, and its views stay
// valid until the decoder's next decode. RelayDecoder satisfies it for every backend and sink.
template <class D>
concept SyndromeDecoder = requires(D d, const D cd, std::span<const Bit> s, std::uint64_t stream,
                                   DecodeLimits limits, std::span<const Bit> mask) {
    { cd.num_rows() } -> std::same_as<index_t>;
    { cd.num_columns() } -> std::same_as<index_t>;
    { d.set_convergence_rows(mask) } -> std::same_as<void>;
    { d.decode(s, stream, limits) } -> std::same_as<std::expected<DecodeResult, DecodeError>>;
};

// Shape index of a window record that ran no decode (a position already decided by an earlier
// final placement).
inline constexpr std::uint32_t no_shape = std::numeric_limits<std::uint32_t>::max();

// What happens to a window whose decode converged but whose inner decoder reports a low
// confidence (Confidence::low; only decoders with a selection policy report one).
//
// none: nothing; the window is committed as usual.
// defer: treated like a failed decode under OnFailure::defer: nothing is committed and the same
//   start round is decoded again, wider, once C more rounds have arrived. When the deferral
//   attempts run out, the last attempt is committed and the window is flagged.
// flag: committed as usual, and the stream is marked as having a low-confidence window.
// A window that did not converge has no solution, the lowest confidence there is: with defer or
// flag it is flagged when committed, whatever the failure policy (commit_anyway included).
enum class OnLowConfidence : std::uint8_t { none, defer, flag };

[[nodiscard]] std::string_view to_string(OnLowConfidence action) noexcept;
[[nodiscard]] std::optional<OnLowConfidence> parse_on_low_confidence(std::string_view text) noexcept;

// What happened at one window position of one shot. Iterations, legs and decode time are summed
// over the deferral attempts; everything else describes the attempt whose solution was committed.
struct WindowRecord {
    std::uint32_t window = 0; // position k
    // Shape of the committed attempt, i.e. which inner decoder produced the solution (its solution
    // sink, if any, still holds that decode); no_shape when no decode ran.
    std::uint32_t shape = no_shape;
    // a + 1 for the committed attempt a; 0 for a position already decided by an earlier final
    // placement, whose record is otherwise that of an empty window.
    std::uint32_t attempts = 0;
    std::uint32_t iterations = 0;
    std::uint32_t legs = 0;
    bool converged = true; // the committed attempt satisfied its convergence rows
    bool cap_hit = false;  // the committed attempt ran out of its iteration budget
    // Set in two cases, which `converged` tells apart:
    //   not converged, and the failure policy flags it (flag, or defer out of attempts) or a
    //   low-confidence action is set (defer or flag: no solution is the lowest confidence, so
    //   this holds even under commit_anyway);
    //   converged with low confidence, and the action is flag, or defer out of attempts.
    bool flagged = false;
    // W(ê) of the committed attempt's solution as the inner decoder reported it; +∞ if it did not
    // converge.
    double weight = 0.0;
    // Σ λ_l over the committed local columns of that solution (finite λ only, ascending l).
    double committed_weight = 0.0;
    // Detectors of the committed rounds (all real rows of a final placement) that the commits
    // leave unexplained.
    std::uint32_t unexplained = 0;
    // Committed local columns with no global counterpart (uniform boundary only).
    std::uint32_t virtual_commits = 0;
    // XOR of commit_class over the returned solution, i.e. the logical class it assigns to the
    // committed rounds; equal to this window's change of the frame.
    std::uint64_t returned_class = 0;
    std::uint64_t decode_ns = 0; // wall time inside the inner decoder
    // The committed attempt's confidence, when its inner decoder has a selection policy.
    std::optional<Confidence> confidence = std::nullopt;
    // The committed attempt converged with a low confidence.
    bool low_confidence = false;
    // Attempts of this position that converged but were deferred for low confidence.
    std::uint32_t low_confidence_deferrals = 0;
};

// The result of one decode_next call. The spans point into the stream decoder and stay valid
// until its next call that changes state.
struct Commit {
    std::uint32_t window = 0;      // position k
    std::uint32_t first_round = 0; // t_k
    // Rounds whose faults were decided: C, or every round of a final placement; 0 when nothing was
    // decided (a deferred or already decided position).
    std::uint32_t rounds = 0;
    // The window's decode failed and, under OnFailure::defer, it waits for C more rounds before
    // trying again with a wider window. Nothing was committed; `record` holds the attempts so
    // far, and the same position is decoded again by a later decode_next once window_ready().
    // Only happens while streaming: when every round has arrived the next attempt runs at once.
    bool deferred = false;
    // Committed faults as global column indices, ascending; virtual columns omitted.
    std::span<const index_t> faults;
    // A·(committed faults): one bit per logical observable.
    std::span<const Bit> frame_delta;
    WindowRecord record;
    // The committed attempt's converged legs as its inner decoder's solution sink recorded them,
    // and how many it found (possibly more than it stores); empty and 0 unless the inner decoder
    // has a recording sink.
    std::span<const SolutionRecord> solutions;
    std::uint32_t solutions_found = 0;
};

// Per-shot totals over the window records, in window order.
struct ShotSummary {
    bool success = true;  // every window converged
    bool flagged = false; // some window was flagged
    std::uint32_t iterations = 0;
    std::uint32_t legs = 0;
    double weight = 0.0; // Σ committed_weight, added in window order
    // Predicted flips of the logical observables, bit o = observable o.
    std::uint64_t frame = 0;
    std::uint64_t decode_ns = 0;
};

// Why a stream decoder could not be built, a round was refused, or a window could not be decoded.
// The numeric fields locate the problem; `detail` is filled only by construction errors (the
// decode path never allocates).
struct StreamError {
    enum class Code : std::uint8_t {
        // Construction
        inner_count_mismatch,      // not exactly one inner decoder per shape
        inner_size_mismatch,       // an inner decoder's rows or columns differ from its shape's
        inner_construction_failed, // the inner decoder factory reported an error
        // Rounds
        wrong_round_size,    // a round with other than M detector bits
        too_many_rounds,     // push_round of round Rt: the readout round is pushed by push_final
        missing_rounds,      // push_final before rounds 1 … Rt − 1 have arrived
        stream_closed,       // a push after push_final
        wrong_syndrome_size, // decode_shot with other than Rt·M detector bits
        // Decoding
        not_ready,    // decode_next while no window is ready (rounds missing, or all decoded)
        inner_failed, // the inner decoder returned an error
        // Configuration
        invalid_low_confidence_action, // defer on low confidence without deferral attempts
    };
    Code code = Code::not_ready;
    std::uint32_t window = 0; // position k, where it applies
    std::uint32_t shape = 0;  // shape index, where it applies
    std::size_t expected = 0;
    std::size_t found = 0;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(StreamError::Code code) noexcept;

// One line: "<code>: <what was expected and found>".
[[nodiscard]] std::string describe(const StreamError& error);

// The window arithmetic of one stream, without the inner decoders: the pending rounds, the
// residual syndrome with every commit applied (so the carry into the next window is already in
// it), the logical frame, and the records of the windows decoded so far. All of it lives in flat
// arrays sized once from the plan, so a decode allocates nothing.
//
// StreamDecoder drives it; the attempt protocol (begin_attempt, end_attempt, commit / deferred /
// skip) is public so that other drivers (a batched or device decoder) can reuse the same
// arithmetic. The plan must outlive it.
class StreamCore {
public:
    explicit StreamCore(const WindowPlan& plan);

    [[nodiscard]] const WindowPlan& plan() const noexcept { return *plan_; }

    // What a converged but low-confidence window does (none until set). Defer needs the plan's
    // deferral attempts: on_failure = defer and max_deferrals ≥ 1.
    [[nodiscard]] std::expected<void, StreamError>
    set_on_low_confidence(OnLowConfidence action) noexcept;
    [[nodiscard]] OnLowConfidence on_low_confidence() const noexcept { return low_policy_; }

    // Starts a new shot; `shot` is the base of every window's γ stream.
    void reset(std::uint64_t shot) noexcept;

    // Appends the next noisy round (rounds 1 … Rt − 1, M bits each). Never decodes.
    [[nodiscard]] std::expected<void, StreamError>
    push_round(std::span<const Bit> detectors) noexcept;
    // Appends the readout round Rt and closes the stream.
    [[nodiscard]] std::expected<void, StreamError>
    push_final(std::span<const Bit> detectors) noexcept;

    // Every row of the next attempt has arrived (rows past the readout count as arrived once the
    // stream is closed), or the next position was already decided and only needs its record.
    [[nodiscard]] bool window_ready() const noexcept;
    // Every window position has produced its commit.
    [[nodiscard]] bool finished() const noexcept { return position_ >= positions_; }
    // The readout round has arrived.
    [[nodiscard]] bool closed() const noexcept { return closed_; }
    [[nodiscard]] bool flagged() const noexcept { return summary_.flagged; }

    [[nodiscard]] std::uint64_t shot() const noexcept { return shot_; }
    [[nodiscard]] std::uint32_t rounds_received() const noexcept { return received_; }
    // The next window position to decode (K once finished).
    [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
    // ℓ̂ so far, one bit per observable, and as a mask.
    [[nodiscard]] std::span<const Bit> frame() const noexcept { return frame_bits_; }
    [[nodiscard]] std::uint64_t frame_mask() const noexcept { return summary_.frame; }
    // The syndrome of the rounds received so far with every commit applied: σ ⊕ H·c.
    [[nodiscard]] std::span<const Bit> residual() const noexcept {
        return {residual_.data(), std::size_t{received_} * per_round_};
    }
    // Records of the positions decoded so far in this shot, in position order.
    [[nodiscard]] std::span<const WindowRecord> records() const noexcept {
        return {records_.data(), position_};
    }
    [[nodiscard]] const ShotSummary& summary() const noexcept { return summary_; }

    // ---- The attempt protocol, in the order a driver calls it. -----------------------------

    // The next position was decided by an earlier final placement: skip() emits its empty record.
    [[nodiscard]] bool skipping() const noexcept { return complete_ && position_ < positions_; }
    [[nodiscard]] Commit skip() noexcept;

    // One decode to run: the inner decoder of `shape` on `syndrome`.
    struct Attempt {
        std::uint32_t shape = 0;
        std::span<const Bit> syndrome;
        std::uint64_t gamma_stream = 0; // shot + (k << 32) + (a << 56)
        DecodeLimits limits;
    };
    // Assembles the window syndrome of the current attempt from the residual, zero past the
    // readout round. Precondition: window_ready() and not skipping().
    [[nodiscard]] Attempt begin_attempt() noexcept;

    enum class Next : std::uint8_t {
        commit, // call commit() with the same result
        retry,  // deferred, and the wider attempt is ready: begin_attempt() again
        wait,   // deferred, and the wider attempt needs rounds that have not arrived: deferred()
    };
    // Accounts the attempt's iterations, legs and time and applies the non-convergence policy.
    [[nodiscard]] Next end_attempt(const DecodeResult& result, std::uint64_t decode_ns) noexcept;
    // Commits the result's columns that the window decides: updates the residual (and with it the
    // carry), the frame and the record, and moves to the next position.
    [[nodiscard]] Commit commit(const DecodeResult& result) noexcept;
    // The state after end_attempt returned wait.
    [[nodiscard]] Commit deferred() noexcept;

    [[nodiscard]] StreamError not_ready_error() const noexcept;
    [[nodiscard]] StreamError inner_error(const DecodeError& error) const noexcept;

private:
    [[nodiscard]] const Placement& current() const noexcept {
        return plan_->schedule()[begin_[position_] + attempt_];
    }
    [[nodiscard]] bool attempt_ready(const Placement& placement) const noexcept {
        return closed_ || placement.first_round + placement.rounds - 1 <= received_;
    }
    [[nodiscard]] std::expected<void, StreamError> accept(std::span<const Bit> detectors) noexcept;
    void clear_delta() noexcept;

    const WindowPlan* plan_;
    index_t per_round_;          // M
    index_t real_rows_;          // Rt·M
    std::uint32_t rounds_total_; // Rt
    std::uint32_t positions_;    // K
    index_t observables_;        // k
    OnFailure policy_;
    OnLowConfidence low_policy_ = OnLowConfidence::none;
    DecodeLimits limits_;
    std::vector<std::uint32_t> begin_;     // schedule index of attempt 0 of each position, [K]
    std::vector<std::uint32_t> attempts_;  // attempts available at each position, [K]
    std::vector<Bit> residual_;            // [Rt·M]
    std::vector<Bit> syndrome_;            // the current window syndrome, [max shape rows]
    std::vector<index_t> faults_;          // committed faults, [max committed columns]
    std::vector<Bit> frame_bits_;          // [k]
    std::vector<Bit> delta_bits_;          // [k]
    std::vector<WindowRecord> records_;    // [K]

    std::uint64_t shot_ = 0;
    std::uint32_t received_ = 0;
    bool closed_ = false;
    // A final placement was committed: every later position is already decided.
    bool complete_ = false;
    std::uint32_t position_ = 0;
    std::uint32_t attempt_ = 0;
    WindowRecord pending_;
    ShotSummary summary_;
};

// Sliding-window decoding of a stream of syndrome rounds.
//
// Rounds arrive with push_round / push_final and are only copied; decode_next decodes the oldest
// window whose rounds have all arrived with the inner decoder of its shape, commits the faults
// it decides and returns them with the window's record. The caller chooses when to decode: back
// to back for stored shots (decode_shot), or from a separate decoding loop for a real-time
// stream, where the gap between push and decode is the queue.
//
// Owns one inner decoder per shape of the plan, each given its shape's convergence rows once at
// construction. After construction nothing allocates, throws (unless the inner decoder does),
// logs or locks. One stream decoder per worker; the plan is shared, read-only, and must outlive
// it (the inner decoders typically point into its shapes as well).
template <SyndromeDecoder Inner>
    requires std::move_constructible<Inner>
class StreamDecoder {
    static constexpr bool nothrow_decode = noexcept(std::declval<Inner&>().decode(
        std::declval<std::span<const Bit>>(), std::uint64_t{}, DecodeLimits{}));
    static constexpr bool records_solutions = requires(const Inner& inner) {
        { inner.sink().records() } -> std::convertible_to<std::span<const SolutionRecord>>;
        { inner.sink().found() } -> std::convertible_to<std::uint32_t>;
    };

public:
    // One inner decoder per shape, in shape order; each must have its shape's row and column
    // counts.
    [[nodiscard]] static std::expected<StreamDecoder, StreamError>
    create(const WindowPlan& plan, std::vector<Inner> inners) {
        const auto shapes = plan.shapes();
        if (inners.size() != shapes.size()) {
            StreamError error;
            error.code = StreamError::Code::inner_count_mismatch;
            error.expected = shapes.size();
            error.found = inners.size();
            return std::unexpected(std::move(error));
        }
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            const Shape& shape = shapes[i];
            const Inner& inner = inners[i];
            if (inner.num_rows() != shape.num_rows() ||
                inner.num_columns() != shape.num_columns()) {
                StreamError error;
                error.code = StreamError::Code::inner_size_mismatch;
                error.shape = static_cast<std::uint32_t>(i);
                error.expected = shape.num_columns();
                error.found = inner.num_columns();
                error.detail = std::format("shape {} is {} x {}, its inner decoder {} x {}", i,
                                           shape.num_rows(), shape.num_columns(),
                                           inner.num_rows(), inner.num_columns());
                return std::unexpected(std::move(error));
            }
        }
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            inners[i].set_convergence_rows(shapes[i].converge());
        }
        return StreamDecoder(plan, std::move(inners));
    }

    // Builds the inner decoder of every shape with make_inner(shape), which returns an Inner or a
    // std::expected<Inner, E>; an E becomes inner_construction_failed with E's detail (or its
    // string form) and the shape index.
    template <class Factory>
        requires std::invocable<Factory&, const Shape&>
    [[nodiscard]] static std::expected<StreamDecoder, StreamError>
    create(const WindowPlan& plan, Factory make_inner) {
        std::vector<Inner> inners;
        inners.reserve(plan.shapes().size());
        for (const Shape& shape : plan.shapes()) {
            auto made = make_inner(shape);
            if constexpr (std::same_as<std::remove_cvref_t<decltype(made)>, Inner>) {
                inners.push_back(std::move(made));
            } else {
                if (!made) {
                    StreamError error;
                    error.code = StreamError::Code::inner_construction_failed;
                    error.shape = shape.index();
                    error.detail = detail_of(made.error());
                    return std::unexpected(std::move(error));
                }
                inners.push_back(std::move(*made));
            }
        }
        return create(plan, std::move(inners));
    }

    [[nodiscard]] const WindowPlan& plan() const noexcept { return core_.plan(); }

    // See StreamCore::set_on_low_confidence.
    [[nodiscard]] std::expected<void, StreamError>
    set_on_low_confidence(OnLowConfidence action) noexcept {
        return core_.set_on_low_confidence(action);
    }

    void reset(std::uint64_t shot) noexcept { core_.reset(shot); }
    [[nodiscard]] std::expected<void, StreamError>
    push_round(std::span<const Bit> detectors) noexcept {
        return core_.push_round(detectors);
    }
    [[nodiscard]] std::expected<void, StreamError>
    push_final(std::span<const Bit> detectors) noexcept {
        return core_.push_final(detectors);
    }
    [[nodiscard]] bool window_ready() const noexcept { return core_.window_ready(); }

    // Decodes the oldest ready window position, trying its deferral attempts as the policy and
    // the arrived rounds allow. Views in the Commit are valid until the next call.
    [[nodiscard]] std::expected<Commit, StreamError> decode_next() noexcept(nothrow_decode) {
        if (!core_.window_ready()) {
            return std::unexpected(core_.not_ready_error());
        }
        if (core_.skipping()) {
            return core_.skip();
        }
        for (;;) {
            const StreamCore::Attempt attempt = core_.begin_attempt();
            const auto start = std::chrono::steady_clock::now();
            auto result = inners_[attempt.shape].decode(attempt.syndrome, attempt.gamma_stream,
                                                        attempt.limits);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            if (!result) {
                return std::unexpected(core_.inner_error(result.error()));
            }
            const auto ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            switch (core_.end_attempt(*result, ns)) {
            case StreamCore::Next::retry:
                continue;
            case StreamCore::Next::wait:
                return core_.deferred();
            case StreamCore::Next::commit:
                break;
            }
            Commit commit = core_.commit(*result);
            if constexpr (records_solutions) {
                const auto& sink = inners_[attempt.shape].sink();
                commit.solutions = sink.records();
                commit.solutions_found = sink.found();
            }
            return commit;
        }
    }

    [[nodiscard]] bool finished() const noexcept { return core_.finished(); }
    [[nodiscard]] bool closed() const noexcept { return core_.closed(); }
    [[nodiscard]] bool flagged() const noexcept { return core_.flagged(); }
    [[nodiscard]] std::span<const Bit> frame() const noexcept { return core_.frame(); }
    [[nodiscard]] std::uint64_t frame_mask() const noexcept { return core_.frame_mask(); }
    [[nodiscard]] std::span<const Bit> residual() const noexcept { return core_.residual(); }
    [[nodiscard]] std::span<const WindowRecord> records() const noexcept { return core_.records(); }
    [[nodiscard]] const ShotSummary& summary() const noexcept { return core_.summary(); }
    [[nodiscard]] const StreamCore& core() const noexcept { return core_; }

    // The inner decoder of a shape (record.shape names the one that produced a commit, whose
    // solution sink still holds that decode until the shape is used again). Changing its
    // convergence rows changes what the window accepts.
    [[nodiscard]] Inner& inner(std::uint32_t shape) noexcept { return inners_[shape]; }
    [[nodiscard]] const Inner& inner(std::uint32_t shape) const noexcept { return inners_[shape]; }
    [[nodiscard]] std::span<Inner> inners() noexcept { return inners_; }

private:
    StreamDecoder(const WindowPlan& plan, std::vector<Inner> inners)
        : core_(plan), inners_(std::move(inners)) {}

    template <class E>
    static std::string detail_of(const E& error) {
        if constexpr (requires { std::string(error.detail); }) {
            return std::string(error.detail);
        } else if constexpr (std::convertible_to<const E&, std::string>) {
            return std::string(error);
        } else {
            return "the inner decoder factory reported an error";
        }
    }

    StreamCore core_;
    std::vector<Inner> inners_;
};

// Decodes one stored shot: resets the stream decoder to `shot`, feeds every round of `syndrome`
// (Rt·M bits), then decodes every window position in order, handing each Commit to `on_commit`
// before the next decode overwrites its views. Returns the shot's totals; the frame is the
// predicted flip of each observable.
template <class Inner, class OnCommit>
    requires std::invocable<OnCommit&, const Commit&>
[[nodiscard]] std::expected<ShotSummary, StreamError>
decode_shot(StreamDecoder<Inner>& decoder, std::span<const Bit> syndrome, std::uint64_t shot,
            OnCommit on_commit) {
    const WindowPlan& plan = decoder.plan();
    const std::size_t per_round = plan.detectors_per_round();
    const std::uint32_t rounds = plan.rounds_total();
    if (syndrome.size() != per_round * rounds) {
        StreamError error;
        error.code = StreamError::Code::wrong_syndrome_size;
        error.expected = per_round * rounds;
        error.found = syndrome.size();
        return std::unexpected(std::move(error));
    }
    decoder.reset(shot);
    for (std::uint32_t r = 0; r + 1 < rounds; ++r) {
        if (auto pushed = decoder.push_round(syndrome.subspan(r * per_round, per_round)); !pushed) {
            return std::unexpected(std::move(pushed.error()));
        }
    }
    if (auto pushed = decoder.push_final(syndrome.subspan((rounds - 1) * per_round, per_round));
        !pushed) {
        return std::unexpected(std::move(pushed.error()));
    }
    while (!decoder.finished()) {
        auto commit = decoder.decode_next();
        if (!commit) {
            return std::unexpected(std::move(commit.error()));
        }
        if (!commit->deferred) {
            on_commit(*commit);
        }
    }
    return decoder.summary();
}

template <class Inner>
[[nodiscard]] std::expected<ShotSummary, StreamError>
decode_shot(StreamDecoder<Inner>& decoder, std::span<const Bit> syndrome, std::uint64_t shot) {
    return decode_shot(decoder, syndrome, shot, [](const Commit& /*commit*/) noexcept {});
}

} // namespace rtd::window
