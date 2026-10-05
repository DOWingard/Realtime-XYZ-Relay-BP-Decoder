#pragma once

// The per-decode confidence values a run with a selection policy writes, and the relay decoders
// whose solution sink is a selection policy.
//
// Arrays (first axis = shot; K windows per shot, K = 1 for a whole-shot decode; L history lengths
// and G history signals):
//   conf_found, conf_seen, conf_distinct, conf_classes          u32 [S, K]
//   conf_best_class, conf_second_class                         u64 [S, K]
//   conf_weight, conf_gap, conf_agreement                      f64 [S, K]
//   conf_gap_state                                             u8  [S, K]  0 none, 1 single class
//                                                                          (Δ undefined), 2 defined
//   conf_first_legs, conf_first_iterations                     u32 [S, K]
//   conf_class_sum_class, conf_agreement_class                 u64 [S, K]
//   conf_class_sum_top                                         f64 [S, K]
//   conf_q_supp, conf_q_sum_sq, conf_q_total                   f64 [S, K]
//   conf_components, conf_syndrome_ones, conf_syndrome_rows    u32 [S, K]
//   conf_decided_class                                         u64 [S, K]
//   conf_decided_leg, conf_extra_legs                          u32 [S, K]
//   conf_score                                                 f64 [S, K]
//   conf_low, conf_stopped_early                               u8  [S, K]
//   conf_low_deferrals                                         u32 [S, K]  (sliding windows)
//   hist_<signal>                                              f64 [S, K, L]
//   hist_state                                                 u8  [S, K, L]
// Cells that ran no decode (a window position decided by an earlier final window) keep the empty
// values: counts 0, weight +∞, other doubles NaN, state 0.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/arith.hpp"
#include "rtd/core/confidence.hpp"
#include "rtd/core/decoder.hpp"
#include "rtd/core/executor.hpp"
#include "rtd/core/selection.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/selection_spec.hpp"
#include "rtd/io/error.hpp"
#include "rtd/window/history.hpp"

namespace rtd::harness {

struct ConfidenceOutputs {
    bool enabled = false;
    bool sliding = false;
    std::size_t cells = 0; // S · K
    std::vector<window::HistorySignal> history_signals;
    std::size_t history_lengths = 0;

    std::vector<std::uint32_t> found, seen, distinct, classes;
    std::vector<std::uint64_t> best_class, second_class;
    std::vector<double> weight, gap, agreement;
    std::vector<std::uint8_t> gap_state;
    std::vector<std::uint32_t> first_legs, first_iterations;
    std::vector<std::uint64_t> class_sum_class, agreement_class;
    std::vector<double> class_sum_top;
    std::vector<double> q_supp, q_sum_sq, q_total;
    std::vector<std::uint32_t> components, syndrome_ones, syndrome_rows;
    std::vector<std::uint64_t> decided_class;
    std::vector<std::uint32_t> decided_leg, extra_legs;
    std::vector<double> score;
    std::vector<std::uint8_t> low, stopped_early;
    std::vector<std::uint32_t> low_deferrals;
    std::vector<std::uint8_t> decoded; // the cell's confidence was recorded (not written)
    // [cells · G · L]: cell-major, then signal, then length.
    std::vector<double> history;
    std::vector<std::uint8_t> history_state; // [cells · L]
};

// Allocates every array for `cells` = S · K cells with the empty values.
void size_confidence_outputs(ConfidenceOutputs& out, std::size_t cells,
                             const SelectionSpec& selection, bool sliding);
// Writes one decode's confidence into cell `cell`.
void record_confidence(const Confidence& confidence, std::size_t cell, ConfidenceOutputs& out);
// Writes the history values after a window into cell `cell`.
void record_history(const window::SignalHistory& history, std::size_t cell, ConfidenceOutputs& out);
// The .npy files listed above, into `directory`.
[[nodiscard]] std::expected<void, io::IoError>
write_confidence(const std::filesystem::path& directory, const ConfidenceOutputs& out,
                 std::size_t shots, std::size_t windows);
// Counts for run.json: gap categories, low-confidence and early stops, extra legs, decisions that
// differ from the lowest-weight one.
[[nodiscard]] nlohmann::json confidence_summary(const ConfidenceOutputs& out);

// The solution recorder inside a sink, if it has one: the recorder itself, or the one a selection
// sink forwards to.
template <class Sink> [[nodiscard]] const SolutionRecorder* recorder_of(const Sink& sink) noexcept {
    if constexpr (std::same_as<Sink, SolutionRecorder>) {
        return &sink;
    } else if constexpr (std::same_as<Sink, SelectionSink<SolutionRecorder>>) {
        return &sink.recorder();
    } else {
        (void)sink;
        return nullptr;
    }
}

} // namespace rtd::harness

// Compiled once, in selection.cpp.
namespace rtd {
extern template class RelayDecoder<CpuBackend<F32, Serial>, SelectionSink<NoSink>>;
extern template class RelayDecoder<CpuBackend<F64, Serial>, SelectionSink<NoSink>>;
extern template class RelayDecoder<CpuBackend<F32, Team>, SelectionSink<NoSink>>;
extern template class RelayDecoder<CpuBackend<F64, Team>, SelectionSink<NoSink>>;
extern template class RelayDecoder<CpuBackend<F32, Serial>,
                                   SelectionSink<harness::SolutionRecorder>>;
extern template class RelayDecoder<CpuBackend<F64, Serial>,
                                   SelectionSink<harness::SolutionRecorder>>;
extern template class RelayDecoder<CpuBackend<F32, Team>, SelectionSink<harness::SolutionRecorder>>;
extern template class RelayDecoder<CpuBackend<F64, Team>, SelectionSink<harness::SolutionRecorder>>;
} // namespace rtd
