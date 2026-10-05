#pragma once

// Selection policies for the relay controller: which of the solutions a decode found it returns,
// when it stops, and how confident it is in the answer.
//
// Relay-BP produces a sequence of corrections ê (one per converged leg), each with H·ê = σ. Under
// independent faults P(ê) ∝ e^{−W(ê)} with W(ê) = Σ_{ê_j=1} λ_j and λ_j = ln((1 − p_j)/p_j). What
// matters for the logical outcome is only the class ℓ(ê) = A·ê. The optimal decision picks the
// class L with the largest total probability Σ_{ê: ℓ(ê)=L, H·ê=σ} e^{−W(ê)}; the relay decoder
// sees a small sample of those corrections, and the rules here are decisions over that sample
// (exact definitions in confidence.hpp):
//   lowest_weight      the class of the most probable correction found (Relay-BP's own rule);
//   class_sum          argmax_L Σ e^{−W} over the distinct solutions found in class L;
//   largest_agreement  the class with the most distinct solutions found.
// Stopping rules end the decode at the first converged leg after which they hold, evaluated on
// the solutions so far (ê* = the lowest-weight one):
//   fixed              never: only the relay schedule's own rule (stop after S converged legs);
//   agree              m converged legs (repeats included) have ê*'s class;
//   agree_distinct     m distinct solutions have ê*'s class;
//   gap                Δ is defined and Δ ≥ t;
//   gap_extend         at least n₀ solutions, and either every one has ê*'s class or Δ ≥ t: with
//                      the relay rule set to S > n₀, decodes whose first n₀ solutions contain a
//                      competing class within t of ê* keep searching up to S.
// The decision a stopped decode returns is the configured rule's over the solutions it holds.

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "rtd/core/buffer.hpp"
#include "rtd/core/confidence.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

enum class SelectionRule : std::uint8_t { lowest_weight, class_sum, largest_agreement };
enum class StopRule : std::uint8_t { fixed, agree, agree_distinct, gap, gap_extend };

// Which value of Confidence decides whether a decode counts as low-confidence. Every signal is
// turned into an inverse confidence `score` (larger = less trustworthy):
//   gap               −Δ (NaN when every solution has one class; such a decode counts as low
//                     or high as configured)
//   agreement         1 − a
//   weight            W(ê*)
//   first_legs        legs until the first solution
//   first_iterations  iterations until the first solution
//   q_supp            Q_supp^(2) of ê*
// A decode is low-confidence when score > threshold, where `threshold` is given in the signal's
// own units (for gap: low when Δ < threshold; for agreement: low when a < threshold).
enum class ConfidenceSignal : std::uint8_t {
    none,
    gap,
    agreement,
    weight,
    first_legs,
    first_iterations,
    q_supp,
};

[[nodiscard]] std::string_view to_string(SelectionRule rule) noexcept;
[[nodiscard]] std::string_view to_string(StopRule rule) noexcept;
[[nodiscard]] std::string_view to_string(ConfidenceSignal signal) noexcept;
[[nodiscard]] std::string_view to_string(GapState state) noexcept;
[[nodiscard]] std::optional<SelectionRule> parse_selection_rule(std::string_view text) noexcept;
[[nodiscard]] std::optional<StopRule> parse_stop_rule(std::string_view text) noexcept;
[[nodiscard]] std::optional<ConfidenceSignal>
parse_confidence_signal(std::string_view text) noexcept;

struct SelectionConfig {
    SelectionRule rule = SelectionRule::lowest_weight;
    StopRule stop = StopRule::fixed;
    std::uint32_t stop_count = 0; // m of agree / agree_distinct, n₀ of gap_extend; ≥ 1
    double stop_gap = 0.0;        // t of gap / gap_extend, finite
    // Converged legs the rules look at: the first `capacity` of each decode (1 … 20), the same
    // legs a recording of that many solution slots keeps.
    std::uint32_t capacity = RecordingSink::max_capacity;
    ConfidenceSignal signal = ConfidenceSignal::none;
    double threshold = 0.0;
    // For signal gap: whether a decode whose solutions all share one class is low-confidence.
    bool single_class_is_low = true;
    // Legs to run past the relay rule while the confidence is low; 0 = none. With a relay rule
    // of S converged legs, S + extra_legs ≤ capacity keeps every solution visible to the rules;
    // a larger value is cut to the free slots when the extension starts.
    std::uint32_t extra_legs = 0;
};

struct SelectionError {
    enum class Code : std::uint8_t {
        invalid_capacity,
        invalid_stop,
        invalid_threshold,
        missing_graph,
        size_mismatch,
    };
    Code code;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(SelectionError::Code code) noexcept;

// Checks the capacity (1 … 20), the stop parameters (m or n₀ ≥ 1, n₀ ≤ capacity, a finite t), a
// finite signal threshold, and that extra legs come with a signal to decide when to run them.
[[nodiscard]] std::expected<void, SelectionError> validate(const SelectionConfig& config);

// Connected components of a set of faults on a Tanner graph, two faults being connected when
// they flip a common detector, and the terms of the cluster fraction
//   Q^(2) = sqrt(Σ_c L_c²) / Λ,   L_c = Σ_{j ∈ c} λ_j,   Λ = Σ_j λ_j over every column
// (λ_j = +∞ terms skipped everywhere). Floating-point order, which the Python analysis
// reproduces: each L_c adds its members' λ in ascending column order starting from 0.0;
// components are ordered by their smallest column and Σ_c L_c² accumulates L_c·L_c in that order
// from 0.0; Λ adds every finite λ in ascending column order; Q = sqrt(Σ L_c²) / Λ.
//
// Scratch arrays are sized once from the graph (m row owners, n entries per support), so terms()
// never allocates. The graph and λ are borrowed.
class SupportClusters {
public:
    struct Terms {
        double sum_sq = 0.0; // Σ_c L_c²
        std::uint32_t components = 0;
    };

    [[nodiscard]] static std::expected<SupportClusters, SelectionError>
    create(const TannerGraph& graph, std::span<const double> llr);

    // Components of a strictly ascending support of external columns (columns outside the graph
    // are ignored).
    [[nodiscard]] Terms terms(std::span<const index_t> support) noexcept;
    // Λ, the problem's total λ.
    [[nodiscard]] double total() const noexcept { return total_; }
    // sqrt(sum_sq) / Λ; NaN when Λ is not positive.
    [[nodiscard]] double q2(const Terms& terms) const noexcept;

private:
    SupportClusters(const TannerGraph& graph, std::span<const double> llr);

    [[nodiscard]] std::uint32_t find(std::uint32_t x) noexcept;

    const TannerGraph* graph_;
    std::span<const double> llr_;
    double total_ = 0.0;
    AlignedBuffer<std::uint32_t> row_owner_; // [m]: position in the support of a row's first fault
    AlignedBuffer<std::uint32_t> parent_;    // [n]: union-find over support positions
    AlignedBuffer<std::uint32_t> label_;     // [n]: component index of a root, by first member
    AlignedBuffer<double> sums_;             // [n]: L_c
};

// The selection rules and one decode's bookkeeping. Feed it the converged legs of a decode in leg
// order (add, or add_record for a replay of recorded solutions), ask it after each whether to
// stop, and call finish once at the end.
class SelectionState {
public:
    static constexpr std::uint32_t max_capacity = RecordingSink::max_capacity;

    // `column_class` (k-bit class mask per column) and `llr` (λ per column, as the backend's
    // weights use them) are borrowed and must outlive the state. `graph` (may be null) enables
    // Q_supp; it must have the same columns.
    [[nodiscard]] static std::expected<SelectionState, SelectionError>
    create(const SelectionConfig& config, std::span<const std::uint64_t> column_class,
           std::span<const double> llr, const TannerGraph* graph);

    void begin() noexcept;
    // One converged leg; class and hash are computed from the support in one pass.
    void add(const SolutionEvent& event) noexcept;
    // One converged leg whose class and hash are already known (a replay). The support may be
    // empty when it was not kept; the class-level rules then return no replacement solution and
    // Q_supp is NaN.
    void add_record(const SolutionRecord& record, std::span<const index_t> support) noexcept;

    [[nodiscard]] bool stop_requested() const noexcept { return stop_; }
    [[nodiscard]] bool low_confidence() const noexcept;
    // Legs to run past the relay rule while the confidence is low: the configured count, but no
    // more than the slots the rules still have free. A leg converges at most once, so every
    // solution an extension finds is one the rules see, and the confidence it ends on describes
    // the solution the controller returns.
    [[nodiscard]] std::uint32_t extra_legs() const noexcept {
        return seen_ >= config_.capacity ? 0U
                                         : std::min(config_.extra_legs, config_.capacity - seen_);
    }
    // The confidence and, for the class-level rules, the decided solution. `facts` describes the
    // decode as the controller saw it (a replay passes what the recording holds).
    [[nodiscard]] Selection finish(const DecodeFacts& facts) noexcept;

    [[nodiscard]] const SelectionConfig& config() const noexcept { return config_; }
    [[nodiscard]] index_t num_columns() const noexcept {
        return static_cast<index_t>(column_class_.size());
    }
    [[nodiscard]] std::span<const std::uint64_t> column_class() const noexcept {
        return column_class_;
    }
    [[nodiscard]] std::uint64_t class_of(std::span<const index_t> support) const noexcept {
        return solution_class(column_class_, support);
    }
    // Converged legs of the current decode (including any beyond the capacity).
    [[nodiscard]] std::uint32_t found() const noexcept { return found_; }

private:
    // One converged leg within the capacity.
    struct Entry {
        std::uint32_t leg = 0;
        std::uint32_t cumulative_iterations = 0;
        double weight = 0.0;
        std::uint64_t logical_class = 0;
        std::uint64_t hash = 0;
        bool distinct = false;   // the first leg with this support
        std::uint32_t klass = 0; // index into classes_
    };
    // One logical class among the slots, in order of first appearance.
    struct ClassInfo {
        std::uint64_t logical_class = 0;
        std::uint32_t legs = 0;  // slots, repeats included
        std::uint32_t count = 0; // new slots (distinct solutions)
        std::uint32_t best = 0;  // entry of its lowest-weight distinct solution (earliest on ties)
        std::uint32_t slot = 0;  // support slot holding that solution
        bool has_support = false;
    };

    SelectionState(const SelectionConfig& config, std::span<const std::uint64_t> column_class,
                   std::span<const double> llr, std::optional<SupportClusters> clusters);

    void insert(const Entry& candidate, std::span<const index_t> support) noexcept;
    void store_support(ClassInfo& info, std::span<const index_t> support) noexcept;
    // The class each rule decides over the slots so far (indices into classes_).
    [[nodiscard]] std::uint32_t decided(SelectionRule rule) const noexcept;
    [[nodiscard]] std::uint32_t class_sum_choice(double& top) const noexcept;
    [[nodiscard]] std::uint32_t agreement_choice() const noexcept;
    // Δ and the index of the class of the first slot reaching W₂ (none_class unless defined).
    [[nodiscard]] double gap(std::uint32_t& second) const noexcept;
    [[nodiscard]] GapState gap_state() const noexcept;
    // Q_supp terms of ê*, computed at most once per change of ê*.
    [[nodiscard]] const SupportClusters::Terms* best_terms() const noexcept;
    [[nodiscard]] double score(const Confidence& c) const noexcept;
    void update_stop() noexcept;

    SelectionConfig config_;
    std::span<const std::uint64_t> column_class_;
    std::span<const double> llr_;
    // Mutable: Q_supp is evaluated lazily from const queries (low_confidence) and cached.
    mutable std::optional<SupportClusters> clusters_;
    // Supports kept per class: one slot (ê*'s) under the lowest-weight rule, else one per class.
    std::uint32_t slots_ = 1;
    AlignedBuffer<index_t> supports_; // [slots · n]
    std::array<std::uint32_t, max_capacity> slot_size_{};
    AlignedBuffer<Bit> hard_; // [n]: the replacement solution as a dense ê

    std::array<Entry, max_capacity> entries_{};
    std::array<ClassInfo, max_capacity> classes_{};
    std::uint32_t found_ = 0;
    std::uint32_t seen_ = 0;
    std::uint32_t distinct_ = 0;
    std::uint32_t class_count_ = 0;
    std::uint32_t best_ = 0; // entry of ê*
    bool stop_ = false;
    mutable bool q_valid_ = false;
    mutable bool q_known_ = false;
    mutable SupportClusters::Terms q_cache_;
};

// A solution sink that runs a SelectionState inside a RelayDecoder and forwards every solution
// to `Recorder` (NoSink, or a recording sink whose records the harness writes).
template <SolutionSink Recorder = NoSink> class SelectionSink {
public:
    static constexpr bool enabled = true;

    explicit SelectionSink(SelectionState state, Recorder recorder = Recorder{})
        : state_(std::move(state)), recorder_(std::move(recorder)) {}

    void on_decode_begin() noexcept {
        state_.begin();
        if constexpr (Recorder::enabled) {
            recorder_.on_decode_begin();
        }
    }
    void on_solution(const SolutionEvent& event) noexcept {
        state_.add(event);
        if constexpr (Recorder::enabled) {
            recorder_.on_solution(event);
        }
    }

    [[nodiscard]] bool stop_requested() const noexcept { return state_.stop_requested(); }
    [[nodiscard]] bool low_confidence() const noexcept { return state_.low_confidence(); }
    [[nodiscard]] std::uint32_t extra_legs() const noexcept { return state_.extra_legs(); }
    [[nodiscard]] Selection finish(const DecodeFacts& facts) noexcept {
        return state_.finish(facts);
    }

    [[nodiscard]] index_t num_columns() const noexcept { return state_.num_columns(); }
    [[nodiscard]] std::uint64_t class_of(std::span<const index_t> support) const noexcept {
        return state_.class_of(support);
    }
    [[nodiscard]] const SelectionState& state() const noexcept { return state_; }
    [[nodiscard]] Recorder& recorder() noexcept { return recorder_; }
    [[nodiscard]] const Recorder& recorder() const noexcept { return recorder_; }

    // A recording recorder's view, so that code reading `sink().records()` works unchanged.
    [[nodiscard]] auto records() const noexcept
        requires requires(const Recorder& r) { r.records(); }
    {
        return recorder_.records();
    }
    [[nodiscard]] std::uint32_t found() const noexcept
        requires requires(const Recorder& r) { r.found(); }
    {
        return recorder_.found();
    }

private:
    SelectionState state_;
    Recorder recorder_;
};

static_assert(SelectingSink<SelectionSink<NoSink>>);
static_assert(SelectingSink<SelectionSink<RecordingSink>>);

} // namespace rtd
