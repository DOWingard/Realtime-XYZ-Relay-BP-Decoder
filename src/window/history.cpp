#include "rtd/window/history.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <utility>

namespace rtd::window {

namespace {

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

constexpr bool uses_commits(HistorySignal signal) noexcept {
    return signal == HistorySignal::commit_weight || signal == HistorySignal::commit_q_supp;
}

} // namespace

std::string_view to_string(HistorySignal signal) noexcept {
    switch (signal) {
    case HistorySignal::gap:
        return "gap";
    case HistorySignal::agreement:
        return "agreement";
    case HistorySignal::weight:
        return "weight";
    case HistorySignal::first_legs:
        return "first_legs";
    case HistorySignal::first_iterations:
        return "first_iterations";
    case HistorySignal::q_supp:
        return "q_supp";
    case HistorySignal::density:
        return "density";
    case HistorySignal::commit_weight:
        return "commit_weight";
    case HistorySignal::commit_q_supp:
        return "commit_q_supp";
    }
    return "unknown";
}

std::optional<HistorySignal> parse_history_signal(std::string_view text) noexcept {
    for (const HistorySignal signal :
         {HistorySignal::gap, HistorySignal::agreement, HistorySignal::weight,
          HistorySignal::first_legs, HistorySignal::first_iterations, HistorySignal::q_supp,
          HistorySignal::density, HistorySignal::commit_weight, HistorySignal::commit_q_supp}) {
        if (text == to_string(signal)) {
            return signal;
        }
    }
    return std::nullopt;
}

SignalHistory::SignalHistory(std::vector<std::uint32_t> lengths, std::vector<HistorySignal> signals,
                             std::optional<SupportClusters> clusters,
                             std::span<const double> global_llr, std::size_t max_commit)
    : lengths_(std::move(lengths)), signals_(std::move(signals)),
      longest_(*std::ranges::max_element(lengths_)),
      commits_needed_(std::ranges::any_of(signals_, uses_commits)), clusters_(std::move(clusters)),
      llr_(global_llr), max_commit_(commits_needed_ ? max_commit : 0), windows_(longest_),
      ring_(std::size_t{longest_} * max_commit_), merged_(std::size_t{longest_} * max_commit_),
      values_(signals_.size() * lengths_.size(), nan), states_(lengths_.size(), GapState::none) {
    for (std::uint32_t slot = 0; slot < longest_; ++slot) {
        windows_[slot].begin = static_cast<std::uint32_t>(slot * max_commit_);
    }
}

std::expected<SignalHistory, HistoryError>
SignalHistory::create(std::span<const std::uint32_t> lengths,
                      std::span<const HistorySignal> signals, const TannerGraph* global_graph,
                      std::span<const double> global_llr, std::size_t max_commit) {
    if (lengths.empty() || signals.empty()) {
        return std::unexpected(HistoryError{"a signal history needs at least one length and one "
                                            "signal"});
    }
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] < 1 || lengths[i] > max_length) {
            return std::unexpected(HistoryError{
                std::format("history length {} is outside [1, {}]", lengths[i], max_length)});
        }
        if (std::ranges::count(lengths, lengths[i]) > 1) {
            return std::unexpected(
                HistoryError{std::format("history length {} is listed twice", lengths[i])});
        }
    }
    for (const HistorySignal signal : signals) {
        if (std::ranges::count(signals, signal) > 1) {
            return std::unexpected(
                HistoryError{std::format("history signal {} is listed twice", to_string(signal))});
        }
    }
    std::optional<SupportClusters> clusters;
    if (std::ranges::any_of(signals, uses_commits)) {
        if (global_graph == nullptr || global_llr.size() != global_graph->num_columns()) {
            return std::unexpected(HistoryError{
                "the commit signals need the global parity-check matrix and one λ per column"});
        }
        auto made = SupportClusters::create(*global_graph, global_llr);
        if (!made) {
            return std::unexpected(HistoryError{made.error().detail});
        }
        clusters.emplace(std::move(*made));
    }
    return SignalHistory({lengths.begin(), lengths.end()}, {signals.begin(), signals.end()},
                         std::move(clusters), global_llr, max_commit);
}

void SignalHistory::reset() noexcept {
    pushed_ = 0;
    std::ranges::fill(values_, nan);
    std::ranges::fill(states_, GapState::none);
}

void SignalHistory::push(const std::optional<Confidence>& confidence, bool decoded,
                         std::span<const index_t> committed) noexcept {
    Window& window = windows_[pushed_ % longest_];
    window.decoded = decoded;
    window.confidence = confidence;
    window.count = 0;
    if (commits_needed_) {
        const std::size_t count = std::min(committed.size(), max_commit_);
        std::copy_n(committed.data(), count, ring_.data() + window.begin);
        window.count = static_cast<std::uint32_t>(count);
    }
    ++pushed_;
    for (std::size_t l = 0; l < lengths_.size(); ++l) {
        compute(l);
    }
}

void SignalHistory::compute(std::size_t l) noexcept {
    const Totals totals = combine(std::min<std::uint64_t>(lengths_[l], pushed_));
    const bool usable = totals.any_decoded && !totals.any_none;
    if (!usable) {
        states_[l] = GapState::none;
    } else if (totals.any_defined) {
        states_[l] = GapState::defined;
    } else {
        states_[l] = GapState::single_class;
    }
    const CommitValues commit = commits_needed_ ? commit_values(totals.merged) : CommitValues{};
    for (std::size_t s = 0; s < signals_.size(); ++s) {
        values_[(s * lengths_.size()) + l] = value_of(signals_[s], totals, usable, commit);
    }
}

SignalHistory::Totals SignalHistory::combine(std::uint64_t span) noexcept {
    Totals totals;
    // Windows oldest first: pushed_ − span … pushed_ − 1.
    for (std::uint64_t w = pushed_ - span; w < pushed_; ++w) {
        const Window& window = windows_[w % longest_];
        if (commits_needed_) {
            std::copy_n(ring_.data() + window.begin, window.count, merged_.data() + totals.merged);
            totals.merged += window.count;
        }
        if (!window.decoded) {
            continue;
        }
        totals.any_decoded = true;
        if (!window.confidence || window.confidence->gap_state == GapState::none) {
            totals.any_none = true;
            continue;
        }
        add(totals, *window.confidence);
    }
    return totals;
}

void SignalHistory::add(Totals& totals, const Confidence& c) noexcept {
    if (c.gap_state == GapState::defined) {
        totals.any_defined = true;
        totals.gap = std::min(totals.gap, c.gap);
    }
    totals.agreement = totals.first_agreement ? c.agreement : totals.agreement * c.agreement;
    totals.first_agreement = false;
    totals.weight = totals.weight + c.weight;
    totals.first_legs = totals.first_legs + static_cast<double>(c.first_legs);
    totals.first_iterations = totals.first_iterations + static_cast<double>(c.first_iterations);
    if (std::isnan(c.q_sum_sq) || std::isnan(c.q_total)) {
        totals.q_known = false;
    } else {
        totals.sum_sq = totals.sum_sq + c.q_sum_sq;
        totals.total = totals.total + c.q_total;
    }
    totals.ones += c.syndrome_ones;
    totals.rows += c.syndrome_rows;
}

SignalHistory::CommitValues SignalHistory::commit_values(std::size_t merged) noexcept {
    // Commit sets of different windows are disjoint; their union is sorted into ascending global
    // order so that every sum runs in column order.
    std::sort(merged_.data(), merged_.data() + merged);
    const std::span<const index_t> faults(merged_.data(), merged);
    CommitValues commit;
    commit.weight = 0.0;
    for (const index_t j : faults) {
        if (j < llr_.size() && std::isfinite(llr_[j])) {
            commit.weight = commit.weight + llr_[j];
        }
    }
    if (clusters_.has_value()) {
        commit.q = clusters_->q2(clusters_->terms(faults));
    }
    return commit;
}

double SignalHistory::value_of(HistorySignal signal, const Totals& totals, bool usable,
                               const CommitValues& commit) noexcept {
    switch (signal) {
    case HistorySignal::gap:
        return usable && totals.any_defined ? totals.gap : nan;
    case HistorySignal::agreement:
        return usable ? totals.agreement : nan;
    case HistorySignal::weight:
        return usable ? totals.weight : infinity;
    case HistorySignal::first_legs:
        return usable ? totals.first_legs : nan;
    case HistorySignal::first_iterations:
        return usable ? totals.first_iterations : nan;
    case HistorySignal::q_supp:
        return usable && totals.q_known && totals.total > 0.0
                   ? std::sqrt(totals.sum_sq) / totals.total
                   : nan;
    case HistorySignal::density:
        return usable && totals.rows > 0
                   ? static_cast<double>(totals.ones) / static_cast<double>(totals.rows)
                   : nan;
    case HistorySignal::commit_weight:
        return commit.weight;
    case HistorySignal::commit_q_supp:
        return commit.q;
    }
    return nan;
}

} // namespace rtd::window
