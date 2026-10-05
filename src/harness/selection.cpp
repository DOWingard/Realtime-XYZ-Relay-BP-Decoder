#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <variant>

#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/selection_spec.hpp"
#include "rtd/io/npy.hpp"

namespace rtd {

template class RelayDecoder<CpuBackend<F32, Serial>, SelectionSink<NoSink>>;
template class RelayDecoder<CpuBackend<F64, Serial>, SelectionSink<NoSink>>;
template class RelayDecoder<CpuBackend<F32, Team>, SelectionSink<NoSink>>;
template class RelayDecoder<CpuBackend<F64, Team>, SelectionSink<NoSink>>;
template class RelayDecoder<CpuBackend<F32, Serial>, SelectionSink<harness::SolutionRecorder>>;
template class RelayDecoder<CpuBackend<F64, Serial>, SelectionSink<harness::SolutionRecorder>>;
template class RelayDecoder<CpuBackend<F32, Team>, SelectionSink<harness::SolutionRecorder>>;
template class RelayDecoder<CpuBackend<F64, Team>, SelectionSink<harness::SolutionRecorder>>;

} // namespace rtd

namespace rtd::harness {

namespace {

using json = nlohmann::json;

constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double infinity = std::numeric_limits<double>::infinity();

// The selection object is read field by field; a failure is thrown inside this file and turned
// into the error value at parse_selection's boundary, as the spec parser does.
struct Failure {
    SelectionSpecError error;
};

[[noreturn]] void fail(std::string field, std::string problem) {
    throw Failure{SelectionSpecError{.field = std::move(field), .problem = std::move(problem)}};
}

class Reader {
public:
    Reader(const json& value, std::string path) : value_(&value), path_(std::move(path)) {
        if (!value.is_object()) {
            fail(path_, std::format("expected an object, found {}", value.dump()));
        }
    }

    [[nodiscard]] std::string field(std::string_view key) const {
        return std::format("{}.{}", path_, key);
    }

    [[nodiscard]] const json& at(std::string_view key) const {
        const auto it = value_->find(key);
        if (it == value_->end()) {
            fail(field(key), "required field is missing");
        }
        return *it;
    }

    void only(std::initializer_list<std::string_view> allowed) const {
        for (const auto& item : value_->items()) {
            if (std::ranges::find(allowed, std::string_view(item.key())) == allowed.end()) {
                fail(field(item.key()), "unknown field");
            }
        }
    }

    [[nodiscard]] std::string string(std::string_view key) const {
        const json& value = at(key);
        if (!value.is_string()) {
            fail(field(key), std::format("expected a string, found {}", value.dump()));
        }
        return value.get<std::string>();
    }

    [[nodiscard]] double number(std::string_view key) const {
        const json& value = at(key);
        if (!value.is_number()) {
            fail(field(key), std::format("expected a number, found {}", value.dump()));
        }
        return value.get<double>();
    }

    [[nodiscard]] std::uint32_t count(std::string_view key) const {
        return count_of(at(key), field(key));
    }

    static std::uint32_t count_of(const json& value, const std::string& where) {
        const bool non_negative = value.is_number_unsigned() ||
                                  (value.is_number_integer() && value.get<std::int64_t>() >= 0);
        if (!non_negative) {
            fail(where, std::format("expected a non-negative integer, found {}", value.dump()));
        }
        const auto raw = value.get<std::uint64_t>();
        if (raw > std::numeric_limits<std::uint32_t>::max()) {
            fail(where, std::format("{} is too large", raw));
        }
        return static_cast<std::uint32_t>(raw);
    }

    template <class T, class Parse>
    [[nodiscard]] T choice(std::string_view key, Parse parse, std::string_view allowed) const {
        const std::string text = string(key);
        const std::optional<T> parsed = parse(text);
        if (!parsed) {
            fail(field(key), std::format("'{}' is not one of {}", text, allowed));
        }
        return *parsed;
    }

private:
    const json* value_;
    std::string path_;
};

void parse_stop(const Reader& stop, SelectionConfig& config) {
    config.stop = stop.choice<StopRule>("rule", parse_stop_rule,
                                        "fixed, agree, agree_distinct, gap, gap_extend");
    switch (config.stop) {
    case StopRule::fixed:
        stop.only({"rule"});
        break;
    case StopRule::agree:
    case StopRule::agree_distinct:
        stop.only({"rule", "count"});
        config.stop_count = stop.count("count");
        break;
    case StopRule::gap:
        stop.only({"rule", "threshold"});
        config.stop_gap = stop.number("threshold");
        break;
    case StopRule::gap_extend:
        stop.only({"rule", "count", "threshold"});
        config.stop_count = stop.count("count");
        config.stop_gap = stop.number("threshold");
        break;
    }
}

void parse_confidence(const Reader& node, SelectionSpec& out, bool sliding,
                      const window::WindowSpec& window) {
    node.only({"signal", "threshold", "single_class", "extra_legs", "on_low"});
    SelectionConfig& config = out.config;
    config.signal = node.choice<ConfidenceSignal>(
        "signal", parse_confidence_signal,
        "gap, agreement, weight, first_legs, first_iterations, q_supp");
    if (config.signal == ConfidenceSignal::none) {
        fail(node.field("signal"), "a confidence object needs a signal; use \"confidence\": null "
                                   "for none");
    }
    config.threshold = node.number("threshold");
    const std::string single = node.string("single_class");
    if (single != "low" && single != "high") {
        fail(node.field("single_class"), std::format("'{}' is not one of low, high", single));
    }
    config.single_class_is_low = single == "low";
    config.extra_legs = node.count("extra_legs");
    out.on_low = node.choice<window::OnLowConfidence>("on_low", window::parse_on_low_confidence,
                                                      "none, defer, flag");
    if (out.on_low != window::OnLowConfidence::none && !sliding) {
        fail(node.field("on_low"), "a low-confidence window action needs window mode sliding");
    }
    if (out.on_low == window::OnLowConfidence::defer &&
        (window.on_failure != window::OnFailure::defer || window.max_deferrals == 0)) {
        fail(node.field("on_low"), "deferring low-confidence windows needs window.on_failure = "
                                   "defer and window.max_deferrals >= 1");
    }
}

void parse_history(const Reader& node, SelectionSpec& out, bool sliding) {
    node.only({"lengths", "signals"});
    if (!sliding) {
        fail(node.field("lengths"), "a history over the last L windows needs window mode sliding");
    }
    const json& lengths = node.at("lengths");
    const json& signals = node.at("signals");
    if (!lengths.is_array() || lengths.empty()) {
        fail(node.field("lengths"), "expected a non-empty array of window counts");
    }
    if (!signals.is_array() || signals.empty()) {
        fail(node.field("signals"), "expected a non-empty array of signal names");
    }
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        out.history_lengths.push_back(
            Reader::count_of(lengths[i], std::format("{}[{}]", node.field("lengths"), i)));
    }
    for (std::size_t i = 0; i < signals.size(); ++i) {
        const std::string where = std::format("{}[{}]", node.field("signals"), i);
        if (!signals[i].is_string()) {
            fail(where, std::format("expected a string, found {}", signals[i].dump()));
        }
        const auto signal = window::parse_history_signal(signals[i].get<std::string>());
        if (!signal) {
            fail(where, std::format("'{}' is not one of gap, agreement, weight, first_legs, "
                                    "first_iterations, q_supp, density, commit_weight, "
                                    "commit_q_supp",
                                    signals[i].get<std::string>()));
        }
        out.history_signals.push_back(*signal);
    }
    // The same checks the history makes when it is built, reported against the spec's fields.
    for (const std::uint32_t length : out.history_lengths) {
        if (length < 1 || length > window::SignalHistory::max_length ||
            std::ranges::count(out.history_lengths, length) > 1) {
            fail(node.field("lengths"), std::format("lengths must be distinct and in [1, {}]",
                                                    window::SignalHistory::max_length));
        }
    }
    for (const window::HistorySignal signal : out.history_signals) {
        if (std::ranges::count(out.history_signals, signal) > 1) {
            fail(node.field("signals"), std::format("{} is listed twice", to_string(signal)));
        }
    }
}

SelectionSpec parse_object(const Reader& root, const RelayConfig& relay, bool sliding,
                           const window::WindowSpec& window) {
    root.only({"rule", "stop", "capacity", "confidence", "history"});
    SelectionSpec out;
    SelectionConfig& config = out.config;
    config.rule = root.choice<SelectionRule>("rule", parse_selection_rule,
                                             "lowest_weight, class_sum, largest_agreement");
    parse_stop(Reader(root.at("stop"), root.field("stop")), config);
    config.capacity = root.count("capacity");
    if (const json& confidence = root.at("confidence"); !confidence.is_null()) {
        parse_confidence(Reader(confidence, root.field("confidence")), out, sliding, window);
    }
    if (const json& history = root.at("history"); !history.is_null()) {
        parse_history(Reader(history, root.field("history")), out, sliding);
    }
    if (auto valid = validate(config); !valid) {
        std::string_view where = "confidence";
        if (valid.error().code == SelectionError::Code::invalid_capacity) {
            where = "capacity";
        } else if (valid.error().code == SelectionError::Code::invalid_stop) {
            where = "stop";
        }
        fail(root.field(where), valid.error().detail);
    }
    // A stopping rule, extra legs and a window action act on the solutions the relay schedule
    // produces; the rules must see all S of them, or a replay of a recording with S slots would
    // differ and a window's confidence could describe another solution than the one it commits.
    if (config.stop != StopRule::fixed || config.extra_legs > 0 ||
        out.on_low != window::OnLowConfidence::none) {
        const auto* count = std::get_if<AfterNConverged>(&relay.stopping);
        if (count == nullptr) {
            fail(root.field("stop"), "a stopping rule, extra legs or a low-confidence window action "
                                     "need the relay stopping rule after_n_converged");
        }
        if (count->count > config.capacity) {
            fail(root.field("capacity"),
                 std::format("the rules must see all {} solutions of the relay rule "
                             "after_n_converged, but capacity is {}",
                             count->count, config.capacity));
        }
        // Each extra leg may converge, so a low-confidence decode can end with S + extra_legs
        // solutions; beyond the capacity the rules would stop seeing them and could neither end
        // the extension nor describe the solution the decoder returns.
        if (config.extra_legs > config.capacity - count->count) {
            fail(std::format("{}.extra_legs", root.field("confidence")),
                 std::format("after_n_converged {} plus {} extra legs exceeds the {} solutions the "
                             "rules see (capacity); use at most {} extra legs",
                             count->count, config.extra_legs, config.capacity,
                             config.capacity - count->count));
        }
    }
    return out;
}

} // namespace

std::expected<std::optional<SelectionSpec>, SelectionSpecError>
parse_selection(const json& value, const RelayConfig& relay, bool sliding,
                const window::WindowSpec& window) {
    if (value.is_null()) {
        return std::optional<SelectionSpec>{};
    }
    try {
        return std::optional<SelectionSpec>(
            parse_object(Reader(value, "selection"), relay, sliding, window));
    } catch (const Failure& failure) {
        return std::unexpected(failure.error);
    } catch (const json::exception& e) {
        return std::unexpected(SelectionSpecError{.field = "selection", .problem = e.what()});
    }
}

json selection_json(const SelectionSpec& selection) {
    const SelectionConfig& c = selection.config;
    json stop = {{"rule", to_string(c.stop)}};
    if (c.stop == StopRule::agree || c.stop == StopRule::agree_distinct ||
        c.stop == StopRule::gap_extend) {
        stop["count"] = c.stop_count;
    }
    if (c.stop == StopRule::gap || c.stop == StopRule::gap_extend) {
        stop["threshold"] = c.stop_gap;
    }
    json confidence = nullptr;
    if (c.signal != ConfidenceSignal::none) {
        confidence = {{"signal", to_string(c.signal)},
                      {"threshold", c.threshold},
                      {"single_class", c.single_class_is_low ? "low" : "high"},
                      {"extra_legs", c.extra_legs},
                      {"on_low", window::to_string(selection.on_low)}};
    }
    json history = nullptr;
    if (!selection.history_lengths.empty()) {
        json signals = json::array();
        for (const window::HistorySignal signal : selection.history_signals) {
            signals.push_back(to_string(signal));
        }
        history = {{"lengths", selection.history_lengths}, {"signals", std::move(signals)}};
    }
    return {{"rule", to_string(c.rule)},
            {"stop", std::move(stop)},
            {"capacity", c.capacity},
            {"confidence", std::move(confidence)},
            {"history", std::move(history)}};
}

// ---- Outputs ------------------------------------------------------------------------------------

void size_confidence_outputs(ConfidenceOutputs& out, std::size_t cells,
                             const SelectionSpec& selection, bool sliding) {
    out.enabled = true;
    out.sliding = sliding;
    out.cells = cells;
    out.history_signals = selection.history_signals;
    out.history_lengths = selection.history_lengths.size();
    for (auto* v : {&out.found, &out.seen, &out.distinct, &out.classes, &out.first_legs,
                    &out.first_iterations, &out.components, &out.syndrome_ones, &out.syndrome_rows,
                    &out.decided_leg, &out.extra_legs, &out.low_deferrals}) {
        v->assign(cells, 0);
    }
    for (auto* v : {&out.best_class, &out.second_class, &out.class_sum_class, &out.agreement_class,
                    &out.decided_class}) {
        v->assign(cells, 0);
    }
    for (auto* v : {&out.gap, &out.agreement, &out.class_sum_top, &out.q_supp, &out.q_sum_sq,
                    &out.q_total, &out.score}) {
        v->assign(cells, nan);
    }
    out.weight.assign(cells, infinity);
    for (auto* v : {&out.gap_state, &out.low, &out.stopped_early, &out.decoded}) {
        v->assign(cells, 0);
    }
    out.history.assign(cells * out.history_signals.size() * out.history_lengths, nan);
    out.history_state.assign(cells * out.history_lengths, 0);
}

void record_confidence(const Confidence& c, std::size_t cell, ConfidenceOutputs& out) {
    out.decoded[cell] = 1;
    out.found[cell] = c.found;
    out.seen[cell] = c.seen;
    out.distinct[cell] = c.distinct;
    out.classes[cell] = c.classes;
    out.best_class[cell] = c.best_class;
    out.second_class[cell] = c.second_class;
    out.weight[cell] = c.weight;
    out.gap[cell] = c.gap;
    out.agreement[cell] = c.agreement;
    out.gap_state[cell] = static_cast<std::uint8_t>(c.gap_state);
    out.first_legs[cell] = c.first_legs;
    out.first_iterations[cell] = c.first_iterations;
    out.class_sum_class[cell] = c.class_sum_class;
    out.class_sum_top[cell] = c.class_sum_top;
    out.agreement_class[cell] = c.agreement_class;
    out.q_supp[cell] = c.q_supp;
    out.q_sum_sq[cell] = c.q_sum_sq;
    out.q_total[cell] = c.q_total;
    out.components[cell] = c.components;
    out.syndrome_ones[cell] = c.syndrome_ones;
    out.syndrome_rows[cell] = c.syndrome_rows;
    out.decided_class[cell] = c.decided_class;
    out.decided_leg[cell] = c.decided_leg;
    out.extra_legs[cell] = c.extra_legs;
    out.score[cell] = c.score;
    out.low[cell] = c.low ? 1 : 0;
    out.stopped_early[cell] = c.stopped_early ? 1 : 0;
}

void record_history(const window::SignalHistory& history, std::size_t cell,
                    ConfidenceOutputs& out) {
    const std::size_t signals = out.history_signals.size();
    const std::size_t lengths = out.history_lengths;
    for (std::size_t s = 0; s < signals; ++s) {
        for (std::size_t l = 0; l < lengths; ++l) {
            out.history[(((cell * signals) + s) * lengths) + l] = history.value(s, l);
        }
    }
    for (std::size_t l = 0; l < lengths; ++l) {
        out.history_state[(cell * lengths) + l] = static_cast<std::uint8_t>(history.state(l));
    }
}

std::expected<void, io::IoError> write_confidence(const std::filesystem::path& directory,
                                                  const ConfidenceOutputs& out, std::size_t shots,
                                                  std::size_t windows) {
    const std::array<std::size_t, 2> cells{shots, windows};
    std::expected<void, io::IoError> ok;
    const auto write = [&]<class T>(const char* name, const std::vector<T>& data,
                                    std::span<const std::size_t> dims) {
        if (ok) {
            ok = io::write_npy<T>(directory / name, std::span<const T>(data), dims);
        }
    };
    write("conf_found.npy", out.found, cells);
    write("conf_seen.npy", out.seen, cells);
    write("conf_distinct.npy", out.distinct, cells);
    write("conf_classes.npy", out.classes, cells);
    write("conf_best_class.npy", out.best_class, cells);
    write("conf_second_class.npy", out.second_class, cells);
    write("conf_weight.npy", out.weight, cells);
    write("conf_gap.npy", out.gap, cells);
    write("conf_agreement.npy", out.agreement, cells);
    write("conf_gap_state.npy", out.gap_state, cells);
    write("conf_first_legs.npy", out.first_legs, cells);
    write("conf_first_iterations.npy", out.first_iterations, cells);
    write("conf_class_sum_class.npy", out.class_sum_class, cells);
    write("conf_class_sum_top.npy", out.class_sum_top, cells);
    write("conf_agreement_class.npy", out.agreement_class, cells);
    write("conf_q_supp.npy", out.q_supp, cells);
    write("conf_q_sum_sq.npy", out.q_sum_sq, cells);
    write("conf_q_total.npy", out.q_total, cells);
    write("conf_components.npy", out.components, cells);
    write("conf_syndrome_ones.npy", out.syndrome_ones, cells);
    write("conf_syndrome_rows.npy", out.syndrome_rows, cells);
    write("conf_decided_class.npy", out.decided_class, cells);
    write("conf_decided_leg.npy", out.decided_leg, cells);
    write("conf_extra_legs.npy", out.extra_legs, cells);
    write("conf_score.npy", out.score, cells);
    write("conf_low.npy", out.low, cells);
    write("conf_stopped_early.npy", out.stopped_early, cells);
    if (out.sliding) {
        write("conf_low_deferrals.npy", out.low_deferrals, cells);
    }
    const std::size_t signals = out.history_signals.size();
    const std::size_t lengths = out.history_lengths;
    if (signals > 0 && lengths > 0) {
        const std::array<std::size_t, 3> dims{shots, windows, lengths};
        std::vector<double> one(shots * windows * lengths);
        for (std::size_t s = 0; s < signals; ++s) {
            for (std::size_t cell = 0; cell < shots * windows; ++cell) {
                for (std::size_t l = 0; l < lengths; ++l) {
                    one[(cell * lengths) + l] = out.history[(((cell * signals) + s) * lengths) + l];
                }
            }
            const std::string name = std::format("hist_{}.npy", to_string(out.history_signals[s]));
            write(name.c_str(), one, dims);
        }
        write("hist_state.npy", out.history_state, dims);
    }
    return ok;
}

json confidence_summary(const ConfidenceOutputs& out) {
    std::array<std::uint64_t, 3> states{};
    std::uint64_t low = 0;
    std::uint64_t early = 0;
    std::uint64_t extra = 0;
    std::uint64_t deferrals = 0;
    std::uint64_t other_decision = 0;
    std::uint64_t decoded = 0;
    for (std::size_t cell = 0; cell < out.cells; ++cell) {
        if (out.decoded[cell] == 0) {
            continue; // a window position that ran no decode
        }
        ++decoded;
        ++states.at(std::min<std::size_t>(out.gap_state[cell], 2));
        low += out.low[cell];
        early += out.stopped_early[cell];
        extra += out.extra_legs[cell];
        deferrals += out.low_deferrals[cell];
        other_decision += out.decided_class[cell] != out.best_class[cell] ? 1U : 0U;
    }
    return {
        {"decodes", decoded},
        {"gap_state", {{"none", states[0]}, {"single_class", states[1]}, {"defined", states[2]}}},
        {"low_confidence", low},
        {"stopped_early", early},
        {"extra_legs", extra},
        {"low_confidence_deferrals", deferrals},
        {"decision_differs_from_lowest_weight", other_decision}};
}

} // namespace rtd::harness
