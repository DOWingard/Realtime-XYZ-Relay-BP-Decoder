#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "rtd/core/types.hpp"
#include "rtd/window/error.hpp"
#include "rtd/window/problem.hpp"

namespace rtd::window {

// Which round every detector and every fault belongs to.
//
// Rounds are r = 1 … Rt, where Rt = R + 1 counts the R noisy rounds and the readout. Detectors
// must come in round order with the same number M in every round, so round r is rows
// [(r − 1)·M, r·M). The first round of fault j is s(j) = min{ρ(i) : H_ij = 1}. A detector compares
// two consecutive measurements, so a fault flips detectors in rounds s(j) and s(j) + 1 only;
// anything else means the rows are not what this layer assumes and the problem is rejected.
class TimeStructure {
public:
    // Validates the detector rounds and every column of H (H must be a valid CSR matrix with
    // num_rows + 1 row offsets; see validate(const Problem&)).
    [[nodiscard]] static std::expected<TimeStructure, PlanError> build(const Problem& problem);

    [[nodiscard]] std::uint32_t rounds_total() const noexcept { return rounds_total_; }
    [[nodiscard]] index_t detectors_per_round() const noexcept { return detectors_per_round_; }
    [[nodiscard]] index_t num_columns() const noexcept {
        return static_cast<index_t>(first_round_.size());
    }

    // s(j), in 1 … Rt.
    [[nodiscard]] std::uint32_t first_round(index_t column) const noexcept {
        return first_round_[column];
    }
    [[nodiscard]] std::span<const std::uint32_t> first_rounds() const noexcept {
        return first_round_;
    }

    [[nodiscard]] index_t first_row_of_round(std::uint32_t round) const noexcept {
        return (round - 1) * detectors_per_round_;
    }
    [[nodiscard]] std::uint32_t round_of_row(index_t row) const noexcept {
        return row / detectors_per_round_ + 1;
    }

private:
    TimeStructure() = default;

    std::uint32_t rounds_total_ = 0;
    index_t detectors_per_round_ = 0;
    std::vector<std::uint32_t> first_round_;
};

} // namespace rtd::window
