#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>

#include <nlohmann/json.hpp>

#include "rtd/harness/batch.hpp"
#include "rtd/io/error.hpp"

namespace rtd::harness {

// A binomial proportion with its Wilson score interval.
struct Proportion {
    std::size_t hits = 0;
    std::size_t trials = 0;
    double estimate = 0.0;
    double low = 0.0;
    double high = 0.0;
};

// Wilson score interval at the given two-sided normal quantile (1.96 for 95%).
[[nodiscard]] Proportion wilson(std::size_t hits, std::size_t trials, double z = 1.959963984540054);

// Logical error rate per syndrome cycle from the block error rate P (the probability that any
// logical qubit of the memory is wrong) over R cycles, assuming independent cycles:
//   p_L = 1 − (1 − P)^(1/R).
// This is the rate Bravyi et al. (2024) and Müller et al. (2025) report. Empty unless 0 ≤ P ≤ 1.
[[nodiscard]] std::optional<double> ler_per_cycle(double block_error_rate, std::uint32_t rounds);

// Logical error rate per logical qubit per syndrome cycle, additionally assuming the k qubits
// fail independently, each by a flip that composes like a binary symmetric channel:
//   p_L = [1 − (2(1 − P)^(1/k) − 1)^(1/R)] / 2.
// Empty when P ≥ 1 − 2^(−k), where the inversion has no solution.
[[nodiscard]] std::optional<double>
ler_per_qubit_per_cycle(double block_error_rate, std::uint32_t logical_qubits, std::uint32_t rounds);

// Nearest-rank quantile of an ascending sequence: the smallest value with at least a fraction q
// of the sample at or below it.
template <class T>
[[nodiscard]] T nearest_rank(std::span<const T> sorted, double q);

// Counts, rates with intervals, and distributions of iterations, legs and decode time.
// `rounds` enables the per-cycle rate; with `logical_qubits` too, the per-qubit rate.
[[nodiscard]] nlohmann::json summarize(const ShotResults& results,
                                       std::optional<std::uint32_t> logical_qubits,
                                       std::optional<std::uint32_t> rounds);

// Writes the per-shot arrays as .npy files and `record` as run.json into `directory`, which must
// exist. run.json is written last, so its presence marks a complete result set.
[[nodiscard]] std::expected<void, io::IoError> write_results(const std::filesystem::path& directory,
                                                             const ShotResults& results,
                                                             const nlohmann::json& record);

} // namespace rtd::harness
