#include "rtd/window/stream.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

namespace rtd::window {

namespace {

StreamError stream_error(StreamError::Code code, std::uint32_t window, std::uint32_t shape,
                         std::size_t expected, std::size_t found) noexcept {
    StreamError error;
    error.code = code;
    error.window = window;
    error.shape = shape;
    error.expected = expected;
    error.found = found;
    return error;
}

} // namespace

std::string_view to_string(StreamError::Code code) noexcept {
    using Code = StreamError::Code;
    switch (code) {
    case Code::inner_count_mismatch:
        return "inner_count_mismatch";
    case Code::inner_size_mismatch:
        return "inner_size_mismatch";
    case Code::inner_construction_failed:
        return "inner_construction_failed";
    case Code::wrong_round_size:
        return "wrong_round_size";
    case Code::too_many_rounds:
        return "too_many_rounds";
    case Code::missing_rounds:
        return "missing_rounds";
    case Code::stream_closed:
        return "stream_closed";
    case Code::wrong_syndrome_size:
        return "wrong_syndrome_size";
    case Code::not_ready:
        return "not_ready";
    case Code::inner_failed:
        return "inner_failed";
    case Code::invalid_low_confidence_action:
        return "invalid_low_confidence_action";
    }
    return "unknown";
}

std::string_view to_string(OnLowConfidence action) noexcept {
    switch (action) {
    case OnLowConfidence::none:
        return "none";
    case OnLowConfidence::defer:
        return "defer";
    case OnLowConfidence::flag:
        return "flag";
    }
    return "unknown";
}

std::optional<OnLowConfidence> parse_on_low_confidence(std::string_view text) noexcept {
    for (const OnLowConfidence action :
         {OnLowConfidence::none, OnLowConfidence::defer, OnLowConfidence::flag}) {
        if (text == to_string(action)) {
            return action;
        }
    }
    return std::nullopt;
}

std::string describe(const StreamError& error) {
    using Code = StreamError::Code;
    std::string text;
    switch (error.code) {
    case Code::inner_count_mismatch:
        text = std::format("the plan has {} shapes but {} inner decoders were given",
                           error.expected, error.found);
        break;
    case Code::inner_size_mismatch:
        text = std::format("inner decoder of shape {} does not fit it", error.shape);
        break;
    case Code::inner_construction_failed:
        text = std::format("building the inner decoder of shape {} failed", error.shape);
        break;
    case Code::wrong_round_size:
        text = std::format("a round must hold {} detector bits, got {}", error.expected,
                           error.found);
        break;
    case Code::too_many_rounds:
        text = std::format("round {} pushed as a noisy round, but the plan has {} noisy rounds "
                           "before the readout (push_final)",
                           error.found, error.expected);
        break;
    case Code::missing_rounds:
        text = std::format("readout pushed after {} noisy rounds, expected {}", error.found,
                           error.expected);
        break;
    case Code::stream_closed:
        text = "round pushed after the readout closed the stream";
        break;
    case Code::wrong_syndrome_size:
        text = std::format("a shot must hold {} detector bits, got {}", error.expected,
                           error.found);
        break;
    case Code::not_ready:
        text = error.expected == 0
                   ? std::string("every window of the shot has been decoded")
                   : std::format("window {} is not ready: it needs round {}, {} rounds received",
                                 error.window, error.expected, error.found);
        break;
    case Code::inner_failed:
        text = std::format("inner decoder of shape {} failed on window {}: expected {}, found {}",
                           error.shape, error.window, error.expected, error.found);
        break;
    case Code::invalid_low_confidence_action:
        text = "deferring low-confidence windows needs on_failure = defer and max_deferrals >= 1";
        break;
    }
    if (!error.detail.empty()) {
        text += std::format(" ({})", error.detail);
    }
    return std::format("{}: {}", to_string(error.code), text);
}

StreamCore::StreamCore(const WindowPlan& plan)
    : plan_(&plan), per_round_(plan.detectors_per_round()), real_rows_(plan.num_detectors()),
      rounds_total_(plan.rounds_total()), positions_(plan.num_positions()),
      observables_(plan.num_observables()), policy_(plan.spec().on_failure),
      limits_{.max_total_iterations = plan.spec().iteration_cap}, begin_(positions_),
      attempts_(positions_), residual_(real_rows_, Bit{0}), frame_bits_(observables_, Bit{0}),
      delta_bits_(observables_, Bit{0}), records_(positions_) {
    std::uint32_t offset = 0;
    for (std::uint32_t k = 0; k < positions_; ++k) {
        begin_[k] = offset;
        attempts_[k] = plan.attempts(k);
        offset += attempts_[k];
    }
    index_t rows = 0;
    index_t committed = 0;
    for (const Shape& shape : plan.shapes()) {
        rows = std::max(rows, shape.num_rows());
        committed = std::max(committed, shape.committed_columns());
    }
    syndrome_.assign(rows, Bit{0});
    faults_.assign(committed, 0);
    reset(0);
}

std::expected<void, StreamError>
StreamCore::set_on_low_confidence(OnLowConfidence action) noexcept {
    if (action == OnLowConfidence::defer &&
        (policy_ != OnFailure::defer || plan_->spec().max_deferrals == 0)) {
        return std::unexpected(stream_error(StreamError::Code::invalid_low_confidence_action, 0,
                                            0, 1, plan_->spec().max_deferrals));
    }
    low_policy_ = action;
    return {};
}

void StreamCore::reset(std::uint64_t shot) noexcept {
    shot_ = shot;
    received_ = 0;
    closed_ = false;
    complete_ = false;
    position_ = 0;
    attempt_ = 0;
    pending_ = WindowRecord{};
    summary_ = ShotSummary{};
    std::ranges::fill(frame_bits_, Bit{0});
    clear_delta();
}

std::expected<void, StreamError> StreamCore::accept(std::span<const Bit> detectors) noexcept {
    if (closed_) {
        return std::unexpected(stream_error(StreamError::Code::stream_closed, position_, 0,
                                            rounds_total_, received_));
    }
    if (detectors.size() != per_round_) {
        return std::unexpected(stream_error(StreamError::Code::wrong_round_size, position_, 0,
                                            per_round_, detectors.size()));
    }
    // Any nonzero byte is a fired detector; the residual keeps strict 0/1 so that commits can
    // flip bits with XOR and unexplained defects can be counted by summing.
    Bit* out = residual_.data() + std::size_t{received_} * per_round_;
    for (index_t i = 0; i < per_round_; ++i) {
        out[i] = detectors[i] != 0 ? Bit{1} : Bit{0};
    }
    ++received_;
    return {};
}

std::expected<void, StreamError> StreamCore::push_round(std::span<const Bit> detectors) noexcept {
    if (!closed_ && received_ + 1 >= rounds_total_) {
        return std::unexpected(stream_error(StreamError::Code::too_many_rounds, position_, 0,
                                            rounds_total_ - 1, std::size_t{received_} + 1));
    }
    return accept(detectors);
}

std::expected<void, StreamError> StreamCore::push_final(std::span<const Bit> detectors) noexcept {
    if (!closed_ && received_ + 1 != rounds_total_) {
        return std::unexpected(stream_error(StreamError::Code::missing_rounds, position_, 0,
                                            rounds_total_ - 1, received_));
    }
    auto accepted = accept(detectors);
    if (accepted) {
        closed_ = true;
    }
    return accepted;
}

bool StreamCore::window_ready() const noexcept {
    if (position_ >= positions_) {
        return false;
    }
    return complete_ || attempt_ready(current());
}

void StreamCore::clear_delta() noexcept { std::ranges::fill(delta_bits_, Bit{0}); }

Commit StreamCore::skip() noexcept {
    WindowRecord& record = records_[position_];
    record = WindowRecord{};
    record.window = position_;
    summary_.weight += record.committed_weight;
    clear_delta();
    Commit commit;
    commit.window = position_;
    commit.first_round = plan_->schedule()[begin_[position_]].first_round;
    commit.frame_delta = delta_bits_;
    commit.record = record;
    ++position_;
    return commit;
}

StreamCore::Attempt StreamCore::begin_attempt() noexcept {
    const Placement& placement = current();
    const index_t rows = plan_->shape_of(placement).num_rows();
    // Rows past the readout round exist only under the uniform boundary; they read as zero.
    const index_t real = std::min(rows, real_rows_ - placement.first_row);
    std::copy_n(residual_.data() + placement.first_row, real, syndrome_.data());
    std::fill(syndrome_.data() + real, syndrome_.data() + rows, Bit{0});
    const std::uint64_t stream = shot_ + (std::uint64_t{position_} << 32U) +
                                 (std::uint64_t{attempt_} << 56U);
    return Attempt{.shape = placement.shape,
                   .syndrome = std::span<const Bit>(syndrome_.data(), rows),
                   .gamma_stream = stream,
                   .limits = limits_};
}

StreamCore::Next StreamCore::end_attempt(const DecodeResult& result,
                                         std::uint64_t decode_ns) noexcept {
    pending_.iterations += result.iterations;
    pending_.legs += result.legs_executed;
    pending_.decode_ns += decode_ns;
    const bool can_defer = policy_ == OnFailure::defer &&
                           attempt_ < plan_->spec().max_deferrals &&
                           attempt_ + 1 < attempts_[position_];
    if (result.success) {
        const bool low = result.confidence.has_value() && result.confidence->low;
        if (!low || low_policy_ == OnLowConfidence::none) {
            return Next::commit;
        }
        if (low_policy_ == OnLowConfidence::defer && can_defer) {
            ++pending_.low_confidence_deferrals;
            ++attempt_;
            return attempt_ready(current()) ? Next::retry : Next::wait;
        }
        // Flag, or defer with no wider attempt left: commit and mark the stream.
        pending_.flagged = true;
        return Next::commit;
    }
    if (can_defer) {
        ++attempt_;
        return attempt_ready(current()) ? Next::retry : Next::wait;
    }
    // A window with no solution is also a low-confidence one, so a low-confidence action flags it
    // even under commit_anyway.
    pending_.flagged =
        policy_ != OnFailure::commit_anyway || low_policy_ != OnLowConfidence::none;
    return Next::commit;
}

Commit StreamCore::commit(const DecodeResult& result) noexcept {
    const Placement& placement = current();
    const Shape& shape = plan_->shape_of(placement);
    const index_t real = std::min(shape.num_rows(), real_rows_ - placement.first_row);
    const std::span<const Bit> decides = shape.commit();
    const std::span<const std::uint64_t> classes = shape.commit_class();
    const std::span<const double> llr = shape.priors().llr();
    Bit* residual = residual_.data() + placement.first_row;

    std::uint64_t delta = 0;
    double committed_weight = 0.0;
    std::size_t faults = 0;
    std::uint32_t virtual_commits = 0;
    bool ascending = true;
    // The support ascends, so λ is added in ascending local order, as the solution weight is.
    for (const index_t l : result.support) {
        if (decides[l] == 0) {
            continue;
        }
        // Committing a fault removes its detectors from the residual; the ones in the round after
        // the committed rounds become the next window's carry.
        for (const index_t i : shape.column_rows(l)) {
            if (i >= real) {
                break;
            }
            residual[i] ^= Bit{1};
        }
        delta ^= classes[l];
        if (std::isfinite(llr[l])) {
            committed_weight += llr[l];
        }
        const index_t global = placement.columns[l];
        if (global == virtual_column) {
            ++virtual_commits;
            continue;
        }
        ascending = ascending && (faults == 0 || faults_[faults - 1] < global);
        faults_[faults++] = global;
    }
    // Exact windows list their columns in global order; uniform windows map the bulk shape's
    // columns, whose shifted counterparts need not be in the same order.
    if (!ascending) {
        std::sort(faults_.begin(), faults_.begin() + static_cast<std::ptrdiff_t>(faults));
    }

    summary_.frame ^= delta;
    for (index_t o = 0; o < observables_; ++o) {
        delta_bits_[o] = static_cast<Bit>((delta >> o) & 1U);
        frame_bits_[o] = static_cast<Bit>((summary_.frame >> o) & 1U);
    }

    const index_t committed_rows = std::min(placement.commit_rounds * per_round_, real);
    std::uint32_t unexplained = 0;
    for (index_t i = 0; i < committed_rows; ++i) {
        unexplained += residual[i];
    }

    WindowRecord& record = records_[position_];
    record = pending_;
    record.window = position_;
    record.shape = placement.shape;
    record.attempts = attempt_ + 1;
    record.converged = result.success;
    record.cap_hit = result.cap_hit;
    record.weight = result.success ? result.weight : std::numeric_limits<double>::infinity();
    record.committed_weight = committed_weight;
    record.unexplained = unexplained;
    record.virtual_commits = virtual_commits;
    record.returned_class = delta;
    record.confidence = result.confidence;
    record.low_confidence = result.success && result.confidence.has_value() && result.confidence->low;

    summary_.success = summary_.success && record.converged;
    summary_.flagged = summary_.flagged || record.flagged;
    summary_.iterations += record.iterations;
    summary_.legs += record.legs;
    summary_.weight += record.committed_weight;
    summary_.decode_ns += record.decode_ns;

    Commit commit;
    commit.window = position_;
    commit.first_round = placement.first_round;
    commit.rounds = placement.commit_rounds;
    commit.faults = std::span<const index_t>(faults_.data(), faults);
    commit.frame_delta = delta_bits_;
    commit.record = record;

    // A final placement decided every column from its first round to the readout.
    complete_ = complete_ || placement.final;
    ++position_;
    attempt_ = 0;
    pending_ = WindowRecord{};
    return commit;
}

Commit StreamCore::deferred() noexcept {
    clear_delta();
    Commit commit;
    commit.window = position_;
    commit.first_round = current().first_round;
    commit.deferred = true;
    commit.frame_delta = delta_bits_;
    commit.record = pending_;
    commit.record.window = position_;
    commit.record.shape = plan_->schedule()[begin_[position_] + attempt_ - 1].shape;
    commit.record.attempts = attempt_;
    commit.record.converged = false;
    commit.record.weight = std::numeric_limits<double>::infinity();
    return commit;
}

StreamError StreamCore::not_ready_error() const noexcept {
    if (position_ >= positions_) {
        return stream_error(StreamError::Code::not_ready, position_, 0, 0, received_);
    }
    const Placement& placement = current();
    const std::uint32_t last = std::min(placement.first_round + placement.rounds - 1,
                                        rounds_total_);
    return stream_error(StreamError::Code::not_ready, position_, placement.shape, last, received_);
}

StreamError StreamCore::inner_error(const DecodeError& error) const noexcept {
    return stream_error(StreamError::Code::inner_failed, position_, current().shape,
                        error.expected, error.found);
}

} // namespace rtd::window
