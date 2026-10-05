#include "rtd/core/selection.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

namespace rtd {

namespace {

constexpr std::uint32_t none_index = std::numeric_limits<std::uint32_t>::max();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

} // namespace

std::string_view to_string(SelectionRule rule) noexcept {
    switch (rule) {
    case SelectionRule::lowest_weight:
        return "lowest_weight";
    case SelectionRule::class_sum:
        return "class_sum";
    case SelectionRule::largest_agreement:
        return "largest_agreement";
    }
    return "unknown";
}

std::string_view to_string(StopRule rule) noexcept {
    switch (rule) {
    case StopRule::fixed:
        return "fixed";
    case StopRule::agree:
        return "agree";
    case StopRule::agree_distinct:
        return "agree_distinct";
    case StopRule::gap:
        return "gap";
    case StopRule::gap_extend:
        return "gap_extend";
    }
    return "unknown";
}

std::string_view to_string(ConfidenceSignal signal) noexcept {
    switch (signal) {
    case ConfidenceSignal::none:
        return "none";
    case ConfidenceSignal::gap:
        return "gap";
    case ConfidenceSignal::agreement:
        return "agreement";
    case ConfidenceSignal::weight:
        return "weight";
    case ConfidenceSignal::first_legs:
        return "first_legs";
    case ConfidenceSignal::first_iterations:
        return "first_iterations";
    case ConfidenceSignal::q_supp:
        return "q_supp";
    }
    return "unknown";
}

std::string_view to_string(GapState state) noexcept {
    switch (state) {
    case GapState::none:
        return "none";
    case GapState::single_class:
        return "single_class";
    case GapState::defined:
        return "defined";
    }
    return "unknown";
}

std::optional<SelectionRule> parse_selection_rule(std::string_view text) noexcept {
    for (const SelectionRule rule : {SelectionRule::lowest_weight, SelectionRule::class_sum,
                                     SelectionRule::largest_agreement}) {
        if (text == to_string(rule)) {
            return rule;
        }
    }
    return std::nullopt;
}

std::optional<StopRule> parse_stop_rule(std::string_view text) noexcept {
    for (const StopRule rule : {StopRule::fixed, StopRule::agree, StopRule::agree_distinct,
                                StopRule::gap, StopRule::gap_extend}) {
        if (text == to_string(rule)) {
            return rule;
        }
    }
    return std::nullopt;
}

std::optional<ConfidenceSignal> parse_confidence_signal(std::string_view text) noexcept {
    for (const ConfidenceSignal signal :
         {ConfidenceSignal::none, ConfidenceSignal::gap, ConfidenceSignal::agreement,
          ConfidenceSignal::weight, ConfidenceSignal::first_legs,
          ConfidenceSignal::first_iterations, ConfidenceSignal::q_supp}) {
        if (text == to_string(signal)) {
            return signal;
        }
    }
    return std::nullopt;
}

std::string_view to_string(SelectionError::Code code) noexcept {
    switch (code) {
    case SelectionError::Code::invalid_capacity:
        return "invalid_capacity";
    case SelectionError::Code::invalid_stop:
        return "invalid_stop";
    case SelectionError::Code::invalid_threshold:
        return "invalid_threshold";
    case SelectionError::Code::missing_graph:
        return "missing_graph";
    case SelectionError::Code::size_mismatch:
        return "size_mismatch";
    }
    return "unknown";
}

std::expected<void, SelectionError> validate(const SelectionConfig& config) {
    using Code = SelectionError::Code;
    if (config.capacity < 1 || config.capacity > SelectionState::max_capacity) {
        return std::unexpected(
            SelectionError{.code = Code::invalid_capacity,
                           .detail = std::format("capacity must be in [1, {}], got {}",
                                                 SelectionState::max_capacity, config.capacity)});
    }
    const bool counted = config.stop == StopRule::agree ||
                         config.stop == StopRule::agree_distinct ||
                         config.stop == StopRule::gap_extend;
    if (counted && config.stop_count == 0) {
        return std::unexpected(SelectionError{
            .code = Code::invalid_stop,
            .detail = std::format("the {} stop needs a count >= 1", to_string(config.stop))});
    }
    if (config.stop == StopRule::gap_extend && config.stop_count > config.capacity) {
        return std::unexpected(SelectionError{
            .code = Code::invalid_stop,
            .detail = std::format("gap_extend's n0 = {} exceeds the {} solutions the rules see",
                                  config.stop_count, config.capacity)});
    }
    const bool thresholded = config.stop == StopRule::gap || config.stop == StopRule::gap_extend;
    if (thresholded && !std::isfinite(config.stop_gap)) {
        return std::unexpected(
            SelectionError{.code = Code::invalid_stop,
                           .detail = std::format("the gap stop needs a finite threshold, got {}",
                                                 config.stop_gap)});
    }
    if (config.signal != ConfidenceSignal::none && !std::isfinite(config.threshold)) {
        return std::unexpected(
            SelectionError{.code = Code::invalid_threshold,
                           .detail = std::format("the confidence threshold must be finite, got {}",
                                                 config.threshold)});
    }
    if (config.extra_legs > 0 && config.signal == ConfidenceSignal::none) {
        return std::unexpected(SelectionError{
            .code = Code::invalid_threshold,
            .detail = "extra legs are run while the confidence is low, which needs a signal"});
    }
    return {};
}

// ---- SupportClusters ---------------------------------------------------------------------------

SupportClusters::SupportClusters(const TannerGraph& graph, std::span<const double> llr)
    : graph_(&graph), llr_(llr), row_owner_(graph.num_rows(), none_index),
      parent_(graph.num_columns()), label_(graph.num_columns()), sums_(graph.num_columns()) {
    for (const double lambda : llr) {
        if (std::isfinite(lambda)) {
            total_ += lambda;
        }
    }
}

std::expected<SupportClusters, SelectionError>
SupportClusters::create(const TannerGraph& graph, std::span<const double> llr) {
    if (llr.size() != graph.num_columns()) {
        return std::unexpected(SelectionError{
            .code = SelectionError::Code::size_mismatch,
            .detail = std::format("{} log-likelihood ratios for a graph of {} columns", llr.size(),
                                  graph.num_columns())});
    }
    return SupportClusters(graph, llr);
}

std::uint32_t SupportClusters::find(std::uint32_t x) noexcept {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]]; // path halving
        x = parent_[x];
    }
    return x;
}

SupportClusters::Terms SupportClusters::terms(std::span<const index_t> support) noexcept {
    Terms out;
    const index_t n = graph_->num_columns();
    const auto size = static_cast<std::uint32_t>(std::min<std::size_t>(support.size(), n));
    // Union-find over support positions: a row's first fault owns it, later faults on the same
    // row join the owner's set. Positions ascend with the columns, and the smaller root always
    // wins a union, so every root is its component's smallest column.
    for (std::uint32_t pos = 0; pos < size; ++pos) {
        parent_[pos] = pos;
    }
    for (std::uint32_t pos = 0; pos < size; ++pos) {
        const index_t j = support[pos];
        if (j >= n) {
            continue;
        }
        for (const index_t row : graph_->column_rows(graph_->internal_column(j))) {
            const std::uint32_t owner = row_owner_[row];
            if (owner == none_index) {
                row_owner_[row] = pos;
                continue;
            }
            const std::uint32_t a = find(pos);
            const std::uint32_t b = find(owner);
            if (a != b) {
                parent_[std::max(a, b)] = std::min(a, b);
            }
        }
    }
    // A component is labelled when its root, its first member, is reached; each L_c adds its
    // members' λ in ascending column order.
    for (std::uint32_t pos = 0; pos < size; ++pos) {
        const index_t j = support[pos];
        if (j >= n) {
            continue;
        }
        const std::uint32_t root = find(pos);
        if (root == pos) {
            label_[pos] = out.components;
            sums_[out.components] = 0.0;
            ++out.components;
        }
        if (std::isfinite(llr_[j])) {
            sums_[label_[root]] += llr_[j];
        }
    }
    for (std::uint32_t c = 0; c < out.components; ++c) {
        out.sum_sq += sums_[c] * sums_[c];
    }
    // Release the rows for the next call.
    for (std::uint32_t pos = 0; pos < size; ++pos) {
        const index_t j = support[pos];
        if (j >= n) {
            continue;
        }
        for (const index_t row : graph_->column_rows(graph_->internal_column(j))) {
            row_owner_[row] = none_index;
        }
    }
    return out;
}

double SupportClusters::q2(const Terms& terms) const noexcept {
    if (!(total_ > 0.0)) {
        return nan;
    }
    return std::sqrt(terms.sum_sq) / total_;
}

// ---- SelectionState ----------------------------------------------------------------------------

SelectionState::SelectionState(const SelectionConfig& config,
                               std::span<const std::uint64_t> column_class,
                               std::span<const double> llr, std::optional<SupportClusters> clusters)
    : config_(config), column_class_(column_class), llr_(llr), clusters_(std::move(clusters)),
      slots_(config.rule == SelectionRule::lowest_weight ? 1 : config.capacity),
      supports_(std::size_t{slots_} * column_class.size()), hard_(column_class.size(), Bit{0}) {}

std::expected<SelectionState, SelectionError>
SelectionState::create(const SelectionConfig& config, std::span<const std::uint64_t> column_class,
                       std::span<const double> llr, const TannerGraph* graph) {
    if (auto valid = validate(config); !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    if (llr.size() != column_class.size()) {
        return std::unexpected(
            SelectionError{.code = SelectionError::Code::size_mismatch,
                           .detail = std::format("{} log-likelihood ratios but {} column classes",
                                                 llr.size(), column_class.size())});
    }
    std::optional<SupportClusters> clusters;
    if (graph != nullptr) {
        auto made = SupportClusters::create(*graph, llr);
        if (!made) {
            return std::unexpected(std::move(made.error()));
        }
        clusters.emplace(std::move(*made));
    } else if (config.signal == ConfidenceSignal::q_supp) {
        return std::unexpected(SelectionError{
            .code = SelectionError::Code::missing_graph,
            .detail = "the q_supp signal needs the parity-check matrix of the problem"});
    }
    return SelectionState(config, column_class, llr, std::move(clusters));
}

void SelectionState::begin() noexcept {
    found_ = 0;
    seen_ = 0;
    distinct_ = 0;
    class_count_ = 0;
    best_ = 0;
    stop_ = false;
    q_valid_ = false;
}

void SelectionState::add(const SolutionEvent& event) noexcept {
    if (seen_ >= config_.capacity) {
        ++found_;
        return;
    }
    std::uint64_t mask = 0;
    std::uint64_t hash = 0;
    for (const index_t j : event.support) {
        mask ^= column_class_[j];
        hash = splitmix64_hash(hash ^ j);
    }
    insert(Entry{.leg = event.leg,
                 .cumulative_iterations = event.cumulative_iterations,
                 .weight = event.weight,
                 .logical_class = mask,
                 .hash = hash},
           event.support);
}

void SelectionState::add_record(const SolutionRecord& record,
                                std::span<const index_t> support) noexcept {
    if (seen_ >= config_.capacity) {
        ++found_;
        return;
    }
    // A record whose support was not kept is known only by its size.
    const bool known = support.size() == record.size;
    insert(Entry{.leg = record.leg,
                 .cumulative_iterations = record.cumulative_iterations,
                 .weight = record.weight,
                 .logical_class = record.logical_class,
                 .hash = record.hash},
           known ? support : std::span<const index_t>{});
    if (!known) {
        // The slot of this solution, if it took one, holds nothing usable.
        const Entry& entry = entries_[seen_ - 1];
        if (entry.distinct) {
            ClassInfo& info = classes_[entry.klass];
            const bool is_best = config_.rule == SelectionRule::lowest_weight
                                     ? best_ == seen_ - 1
                                     : info.best == seen_ - 1;
            if (is_best) {
                info.has_support = false;
                if (best_ == seen_ - 1) {
                    q_valid_ = false;
                }
            }
        }
    }
}

void SelectionState::store_support(ClassInfo& info, std::span<const index_t> support) noexcept {
    const std::size_t n = column_class_.size();
    const std::size_t count = std::min(support.size(), n);
    std::copy_n(support.data(), count, supports_.data() + (std::size_t{info.slot} * n));
    slot_size_[info.slot] = static_cast<std::uint32_t>(count);
    info.has_support = true;
}

void SelectionState::insert(const Entry& candidate, std::span<const index_t> support) noexcept {
    ++found_;
    Entry entry = candidate;
    const std::uint32_t index = seen_;
    std::uint32_t twin = none_index;
    for (std::uint32_t e = 0; e < seen_; ++e) {
        if (entries_[e].distinct && entries_[e].hash == entry.hash) {
            twin = e;
            break;
        }
    }
    if (twin != none_index) {
        // The same correction again: counted as a converged leg of its class, never as a
        // distinct solution.
        entry.distinct = false;
        entry.klass = entries_[twin].klass;
        ++classes_[entry.klass].legs;
        entries_[seen_++] = entry;
        update_stop();
        return;
    }
    entry.distinct = true;
    ++distinct_;
    std::uint32_t klass = none_index;
    for (std::uint32_t c = 0; c < class_count_; ++c) {
        if (classes_[c].logical_class == entry.logical_class) {
            klass = c;
            break;
        }
    }
    const bool new_class = klass == none_index;
    if (new_class) {
        klass = class_count_++;
        classes_[klass] = ClassInfo{.logical_class = entry.logical_class,
                                    .legs = 0,
                                    .count = 0,
                                    .best = index,
                                    .slot = slots_ == 1 ? 0 : klass,
                                    .has_support = false};
    }
    entry.klass = klass;
    entries_[seen_++] = entry;
    ClassInfo& info = classes_[klass];
    ++info.legs;
    ++info.count;
    const bool class_best = new_class || entry.weight < entries_[info.best].weight;
    if (class_best) {
        info.best = index;
    }
    const bool overall_best = distinct_ == 1 || entry.weight < entries_[best_].weight;
    if (overall_best) {
        best_ = index;
        q_valid_ = false;
    }
    // Under the lowest-weight rule one slot follows ê*; otherwise each class keeps its best.
    if (slots_ == 1 ? overall_best : class_best) {
        store_support(info, support);
        if (slots_ == 1) {
            // Slot 0 moved to this class's solution; the class that held it no longer has one.
            for (std::uint32_t c = 0; c < class_count_; ++c) {
                classes_[c].has_support = c == klass;
            }
        }
    }
    update_stop();
}

std::uint32_t SelectionState::class_sum_choice(double& top) const noexcept {
    // Z̃_L = Σ exp(−(W(ê) − W(ê*))) over the new slots of class L in slot order: Z_L scaled by
    // e^{W(ê*)}, so every term lies in (0, 1] and none underflows.
    std::array<double, max_capacity> z{};
    const double reference = entries_[best_].weight;
    for (std::uint32_t e = 0; e < seen_; ++e) {
        if (entries_[e].distinct) {
            const double d = entries_[e].weight - reference;
            z[entries_[e].klass] = z[entries_[e].klass] + std::exp(-d);
        }
    }
    // Strictly larger replaces, so ties go to the class that appeared first.
    std::uint32_t chosen = 0;
    for (std::uint32_t c = 1; c < class_count_; ++c) {
        if (z[c] > z[chosen]) {
            chosen = c;
        }
    }
    top = z[chosen];
    return chosen;
}

std::uint32_t SelectionState::agreement_choice() const noexcept {
    // More new slots wins; equal counts go to the class whose lightest new slot weighs less,
    // then to the class that appeared first.
    std::uint32_t chosen = 0;
    for (std::uint32_t c = 1; c < class_count_; ++c) {
        const double wc = entries_[classes_[c].best].weight;
        const double wb = entries_[classes_[chosen].best].weight;
        if (classes_[c].count > classes_[chosen].count ||
            (classes_[c].count == classes_[chosen].count && wc < wb)) {
            chosen = c;
        }
    }
    return chosen;
}

std::uint32_t SelectionState::decided(SelectionRule rule) const noexcept {
    if (class_count_ == 0) {
        return none_index;
    }
    switch (rule) {
    case SelectionRule::lowest_weight:
        return entries_[best_].klass;
    case SelectionRule::class_sum: {
        double top = 0.0;
        return class_sum_choice(top);
    }
    case SelectionRule::largest_agreement:
        return agreement_choice();
    }
    return entries_[best_].klass;
}

GapState SelectionState::gap_state() const noexcept {
    if (distinct_ == 0) {
        return GapState::none;
    }
    return class_count_ > 1 ? GapState::defined : GapState::single_class;
}

double SelectionState::gap(std::uint32_t& second) const noexcept {
    second = none_index;
    if (class_count_ < 2) {
        return nan;
    }
    // W₂ and the class of the first slot reaching it: each class's lightest new slot is its
    // earliest slot of that weight, so the first slot overall is the smallest such index.
    const std::uint32_t best_class = entries_[best_].klass;
    double other = infinity;
    std::uint32_t first = none_index;
    for (std::uint32_t c = 0; c < class_count_; ++c) {
        if (c == best_class) {
            continue;
        }
        const std::uint32_t slot = classes_[c].best;
        const double w = entries_[slot].weight;
        if (w < other || (w == other && slot < first)) {
            other = w;
            first = slot;
            second = c;
        }
    }
    return other - entries_[best_].weight;
}

const SupportClusters::Terms* SelectionState::best_terms() const noexcept {
    if (!clusters_ || distinct_ == 0) {
        return nullptr;
    }
    if (!q_valid_) {
        const ClassInfo& info = classes_[entries_[best_].klass];
        // With one slot the slot follows ê*; with one slot per class, ê* is its class's best.
        q_known_ = info.has_support;
        if (q_known_) {
            const std::size_t n = column_class_.size();
            q_cache_ = clusters_->terms(std::span<const index_t>(
                supports_.data() + (std::size_t{info.slot} * n), slot_size_[info.slot]));
        }
        q_valid_ = true;
    }
    return q_known_ ? &q_cache_ : nullptr;
}

double SelectionState::score(const Confidence& c) const noexcept {
    if (c.gap_state == GapState::none) {
        return infinity; // no solution: nothing to trust
    }
    switch (config_.signal) {
    case ConfidenceSignal::none:
        return nan;
    case ConfidenceSignal::gap:
        // One class only: Δ has no value, and no number stands in for it (least of all −∞,
        // which would read as certainty); gap_state carries the category and `low` follows
        // single_class_is_low.
        return c.gap_state == GapState::single_class ? nan : -c.gap;
    case ConfidenceSignal::agreement:
        return 1.0 - c.agreement;
    case ConfidenceSignal::weight:
        return c.weight;
    case ConfidenceSignal::first_legs:
        return static_cast<double>(c.first_legs);
    case ConfidenceSignal::first_iterations:
        return static_cast<double>(c.first_iterations);
    case ConfidenceSignal::q_supp:
        return c.q_supp;
    }
    return nan;
}

bool SelectionState::low_confidence() const noexcept {
    if (config_.signal == ConfidenceSignal::none) {
        return false;
    }
    const GapState state = gap_state();
    if (state == GapState::none) {
        return true;
    }
    const double t = config_.threshold;
    switch (config_.signal) {
    case ConfidenceSignal::none:
        return false;
    case ConfidenceSignal::gap: {
        std::uint32_t second = 0;
        return state == GapState::single_class ? config_.single_class_is_low : gap(second) < t;
    }
    case ConfidenceSignal::agreement:
        return static_cast<double>(classes_[entries_[best_].klass].count) /
                   static_cast<double>(distinct_) <
               t;
    case ConfidenceSignal::weight:
        return entries_[best_].weight > t;
    case ConfidenceSignal::first_legs:
        return static_cast<double>(entries_[0].leg) + 1.0 > t;
    case ConfidenceSignal::first_iterations:
        return static_cast<double>(entries_[0].cumulative_iterations) > t;
    case ConfidenceSignal::q_supp: {
        const SupportClusters::Terms* terms = best_terms();
        if (terms == nullptr || !clusters_.has_value()) {
            return false; // no value: the action is not taken
        }
        const double q = clusters_->q2(*terms);
        return !std::isnan(q) && q > t;
    }
    }
    return false;
}

void SelectionState::update_stop() noexcept {
    const GapState state = gap_state();
    std::uint32_t second = 0;
    switch (config_.stop) {
    case StopRule::fixed:
        stop_ = false;
        return;
    case StopRule::agree:
        stop_ =
            state != GapState::none && classes_[entries_[best_].klass].legs >= config_.stop_count;
        return;
    case StopRule::agree_distinct:
        stop_ =
            state != GapState::none && classes_[entries_[best_].klass].count >= config_.stop_count;
        return;
    case StopRule::gap:
        stop_ = state == GapState::defined && gap(second) >= config_.stop_gap;
        return;
    case StopRule::gap_extend:
        stop_ = seen_ >= config_.stop_count &&
                (state == GapState::single_class ||
                 (state == GapState::defined && gap(second) >= config_.stop_gap));
        return;
    }
}

Selection SelectionState::finish(const DecodeFacts& facts) noexcept {
    Selection selection;
    Confidence& c = selection.confidence;
    c.found = found_;
    c.seen = seen_;
    c.distinct = distinct_;
    c.classes = class_count_;
    c.stopped_early = facts.stopped_early;
    c.extra_legs = facts.extra_legs;
    c.syndrome_ones = facts.syndrome_ones;
    c.syndrome_rows = facts.syndrome_rows;
    if (facts.syndrome_rows > 0) {
        c.density =
            static_cast<double>(facts.syndrome_ones) / static_cast<double>(facts.syndrome_rows);
    }
    c.gap_state = gap_state();
    if (distinct_ == 0) {
        // No solution: every decision is the class of what the decoder returns (leg 0's final
        // estimate, or ê = 0 when no leg ran).
        const std::uint64_t returned = class_of(facts.best_support);
        c.best_class = returned;
        c.class_sum_class = returned;
        c.agreement_class = returned;
        c.decided_class = returned;
        c.score = config_.signal == ConfidenceSignal::none ? nan : infinity;
        c.low = config_.signal != ConfidenceSignal::none;
        return selection;
    }
    const Entry& best = entries_[best_];
    const ClassInfo& best_class = classes_[best.klass];
    c.best_class = best.logical_class;
    c.weight = best.weight;
    std::uint32_t second = none_index;
    c.gap = gap(second);
    if (second != none_index) {
        c.second_class = classes_[second].logical_class;
    }
    c.agreement = static_cast<double>(best_class.count) / static_cast<double>(distinct_);
    c.first_legs = entries_[0].leg + 1;
    c.first_iterations = entries_[0].cumulative_iterations;
    double top = 0.0;
    const std::uint32_t by_sum = class_sum_choice(top);
    c.class_sum_class = classes_[by_sum].logical_class;
    c.class_sum_top = top;
    c.agreement_class = classes_[agreement_choice()].logical_class;
    if (const SupportClusters::Terms* terms = best_terms();
        terms != nullptr && clusters_.has_value()) {
        c.q_total = clusters_->total();
        c.q_sum_sq = terms->sum_sq;
        c.components = terms->components;
        c.q_supp = clusters_->q2(*terms);
    }
    const std::uint32_t d = decided(config_.rule);
    const ClassInfo& chosen = classes_[d];
    const Entry& representative = entries_[chosen.best];
    c.decided_class = chosen.logical_class;
    c.decided_leg = representative.leg;
    c.score = score(c);
    c.low = low_confidence();

    if (config_.rule == SelectionRule::lowest_weight || !chosen.has_support) {
        return selection;
    }
    if (facts.success && representative.leg == facts.best_leg) {
        return selection; // the controller already returns this solution
    }
    const std::size_t n = column_class_.size();
    const std::span<const index_t> support(supports_.data() + (std::size_t{chosen.slot} * n),
                                           slot_size_[chosen.slot]);
    hard_.fill(Bit{0});
    for (const index_t j : support) {
        hard_[j] = Bit{1};
    }
    selection.replaces = true;
    selection.leg = representative.leg;
    selection.weight = representative.weight;
    selection.support = support;
    selection.hard = hard_.span();
    return selection;
}

} // namespace rtd
