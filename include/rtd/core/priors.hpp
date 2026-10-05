#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rtd/core/types.hpp"

namespace rtd {

struct PriorsError {
    enum class Code : std::uint8_t { not_a_probability, certain_fault, too_large };
    Code code;
    index_t column; // first offending column
    std::string detail;
};

[[nodiscard]] std::string_view to_string(PriorsError::Code code) noexcept;

// Fault probabilities p_j and their log-odds λ_j = ln((1 − p_j)/p_j), both in double.
//
// λ_j > 0 means "probably no fault". p = 0 gives λ = +∞ (a fault that cannot happen) and is
// legal. p = 1 would give λ = −∞: such a column is certain to have fired and belongs in the
// syndrome, not in the decoding problem, so it is rejected.
//
// Kept in double regardless of the decoder's number format because the solution weight
// W(ê) = Σ_{ê_j=1} λ_j is always computed in double; each backend quantises λ itself.
class Priors {
public:
    [[nodiscard]] static std::expected<Priors, PriorsError>
    from_probabilities(std::span<const double> probabilities);

    [[nodiscard]] index_t size() const noexcept { return static_cast<index_t>(p_.size()); }
    [[nodiscard]] std::span<const double> probabilities() const noexcept { return p_; }
    [[nodiscard]] std::span<const double> llr() const noexcept { return llr_; }

private:
    std::vector<double> p_;
    std::vector<double> llr_;
};

} // namespace rtd
