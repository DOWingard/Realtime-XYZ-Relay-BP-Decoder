#pragma once

// A per-window confidence signal over the last L windows of a stream, the form of Lee, English and
// Bartlett's real-time post-selection rule: after each window, a value computed from that window
// and the L − 1 before it, compared with a cutoff to decide whether the run is aborted. The stream
// decoder never aborts; this class only computes the values, so that the decision can be made by
// the caller (or later, at any cutoff, in analysis).

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rtd/core/buffer.hpp"
#include "rtd/core/confidence.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/selection.hpp"
#include "rtd/core/types.hpp"

namespace rtd::window {

// What is combined over the windows k − L + 1 … k. The first seven combine the windows' own
// decode confidences like independent sub-problems (the joint solutions of independent problems
// are all tuples of their solutions, with summed weights):
//   gap               min Δ over the windows whose Δ is defined (strict-< scan, oldest first)
//   agreement         product of the agreements, oldest first
//   weight            Σ W(ê*), oldest first from 0.0
//   first_legs        Σ legs to the first solution
//   first_iterations  Σ iterations to the first solution
//   q_supp            sqrt(Σ Σ_c L_c²) / Σ Λ, both sums oldest first from 0.0
//   density           Σ fired detectors / Σ detectors
// Every one of them is NaN (weight: +∞) when some window in the range had no solution. The
// last two use only what the windows committed:
//   commit_weight     Σ λ_j over the union of the committed faults, ascending global column
//   commit_q_supp     Q^(2) of that union on the global problem: sqrt(Σ_c L_c²) / Λ_global
// A position that ran no decode (decided by an earlier final window) contributes nothing.
enum class HistorySignal : std::uint8_t {
    gap,
    agreement,
    weight,
    first_legs,
    first_iterations,
    q_supp,
    density,
    commit_weight,
    commit_q_supp,
};

[[nodiscard]] std::string_view to_string(HistorySignal signal) noexcept;
[[nodiscard]] std::optional<HistorySignal> parse_history_signal(std::string_view text) noexcept;

struct HistoryError {
    std::string detail;
};

class SignalHistory {
public:
    static constexpr std::uint32_t max_length = 64;

    // `lengths` (each 1 … max_length, distinct) and `signals` (distinct) fix the output layout:
    // value(s, l) for signal s and length l. The commit signals need the global problem's graph
    // and λ (borrowed) and `max_commit` = the most faults one window can commit.
    [[nodiscard]] static std::expected<SignalHistory, HistoryError>
    create(std::span<const std::uint32_t> lengths, std::span<const HistorySignal> signals,
           const TannerGraph* global_graph, std::span<const double> global_llr,
           std::size_t max_commit);

    // Starts a new stream.
    void reset() noexcept;
    // Accounts window k (in order): its decode's confidence (absent when it ran no decode or its
    // decoder reports none), whether it ran a decode at all, and its committed global faults
    // (ascending). Then computes every value for the windows up to and including k.
    void push(const std::optional<Confidence>& confidence, bool decoded,
              std::span<const index_t> committed) noexcept;

    [[nodiscard]] std::span<const std::uint32_t> lengths() const noexcept { return lengths_; }
    [[nodiscard]] std::span<const HistorySignal> signals() const noexcept { return signals_; }
    // The value of signal index s over the last lengths()[l] windows, after the last push.
    [[nodiscard]] double value(std::size_t s, std::size_t l) const noexcept {
        return values_[(s * lengths_.size()) + l];
    }
    // The combined gap category over the last lengths()[l] windows: none if some window had no
    // solution (or no window decoded), else defined if some window's Δ is defined, else
    // single_class.
    [[nodiscard]] GapState state(std::size_t l) const noexcept { return states_[l]; }

private:
    struct Window {
        bool decoded = false;
        std::optional<Confidence> confidence;
        std::uint32_t begin = 0; // committed faults in ring_[slot · max_commit, + count)
        std::uint32_t count = 0;
    };

    SignalHistory(std::vector<std::uint32_t> lengths, std::vector<HistorySignal> signals,
                  std::optional<SupportClusters> clusters, std::span<const double> global_llr,
                  std::size_t max_commit);

    // The per-window values summed over the decoded windows of one span, oldest first.
    struct Totals {
        bool any_none = false; // a decoded window without a solution
        bool any_decoded = false;
        bool any_defined = false; // a window whose Δ is defined
        double gap = std::numeric_limits<double>::infinity();
        double agreement = 1.0;
        bool first_agreement = true;
        double weight = 0.0;
        double first_legs = 0.0;
        double first_iterations = 0.0;
        double sum_sq = 0.0;
        double total = 0.0;
        bool q_known = true;
        std::uint64_t ones = 0;
        std::uint64_t rows = 0;
        std::size_t merged = 0; // committed faults gathered into merged_
    };
    // Commit-set signals of the gathered faults: W of their union and its Q_supp^(2).
    struct CommitValues {
        double weight = std::numeric_limits<double>::quiet_NaN();
        double q = std::numeric_limits<double>::quiet_NaN();
    };

    void compute(std::size_t l) noexcept;
    [[nodiscard]] Totals combine(std::uint64_t span) noexcept;
    static void add(Totals& totals, const Confidence& c) noexcept;
    [[nodiscard]] CommitValues commit_values(std::size_t merged) noexcept;
    [[nodiscard]] static double value_of(HistorySignal signal, const Totals& totals, bool usable,
                                         const CommitValues& commit) noexcept;

    std::vector<std::uint32_t> lengths_;
    std::vector<HistorySignal> signals_;
    std::uint32_t longest_ = 0;
    bool commits_needed_ = false;
    std::optional<SupportClusters> clusters_;
    std::span<const double> llr_;
    std::size_t max_commit_ = 0;
    std::vector<Window> windows_;   // ring of the last `longest_` windows
    AlignedBuffer<index_t> ring_;   // [longest_ · max_commit]
    AlignedBuffer<index_t> merged_; // [longest_ · max_commit]: the union, sorted
    std::uint64_t pushed_ = 0;
    std::vector<double> values_;   // [signals · lengths]
    std::vector<GapState> states_; // [lengths]
};

} // namespace rtd::window
