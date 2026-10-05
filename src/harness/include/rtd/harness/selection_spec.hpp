#pragma once

// The "selection" object of a version-3 decoder spec: the selection policy every decode runs
// (which solution is returned, when a decode stops, how its confidence is measured) and, for
// sliding windows, what a low-confidence window does and the per-window signal over the last L
// windows that the run reports.
//
//   "selection": null                  (decodes exactly as a version-2 spec)
//   "selection": {
//     "rule": "lowest_weight" | "class_sum" | "largest_agreement",
//     "stop": {"rule": "fixed"}
//           | {"rule": "agree", "count": m} | {"rule": "agree_distinct", "count": m}
//           | {"rule": "gap", "threshold": t} | {"rule": "gap_extend", "count": n0, "threshold":
//           t},
//     "capacity": 20,
//     "confidence": null
//           | {"signal": "gap" | "agreement" | "weight" | "first_legs" | "first_iterations"
//                        | "q_supp",
//              "threshold": x, "single_class": "low" | "high", "extra_legs": 0,
//              "on_low": "none" | "defer" | "flag"},
//     "history": null
//           | {"lengths": [1, 2, 3, 5],
//              "signals": ["gap", "agreement", "weight", "first_legs", "first_iterations",
//                          "q_supp", "density", "commit_weight", "commit_q_supp"]}
//   }
//
// Every field is required. "on_low" other than "none" and "history" need window mode sliding;
// "on_low": "defer" needs on_failure = defer with max_deferrals ≥ 1. A stop rule other than fixed,
// and extra legs, need the relay rule after_n_converged with count S ≤ capacity, so that the rules
// see every solution the relay schedule would produce.

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/config.hpp"
#include "rtd/core/selection.hpp"
#include "rtd/window/history.hpp"
#include "rtd/window/spec.hpp"
#include "rtd/window/stream.hpp"

namespace rtd::harness {

struct SelectionSpec {
    SelectionConfig config{};
    window::OnLowConfidence on_low = window::OnLowConfidence::none;
    // The per-window signal over the last L windows: one value per (signal, length). Empty when
    // "history" is null.
    std::vector<std::uint32_t> history_lengths;
    std::vector<window::HistorySignal> history_signals;
};

struct SelectionSpecError {
    std::string field;
    std::string problem;
};

// Parses the "selection" value (null gives no policy). `relay` and `sliding` / `window` are the
// spec's already parsed relay schedule and window settings, for the cross-field checks.
[[nodiscard]] std::expected<std::optional<SelectionSpec>, SelectionSpecError>
parse_selection(const nlohmann::json& value, const RelayConfig& relay, bool sliding,
                const window::WindowSpec& window);

// The policy as run.json records it (the same fields as the spec, with every default spelled out).
[[nodiscard]] nlohmann::json selection_json(const SelectionSpec& selection);

} // namespace rtd::harness
