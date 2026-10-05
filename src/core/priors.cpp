#include "rtd/core/priors.hpp"

#include <cmath>
#include <format>
#include <limits>

namespace rtd {

std::string_view to_string(PriorsError::Code code) noexcept {
    switch (code) {
    case PriorsError::Code::not_a_probability:
        return "not_a_probability";
    case PriorsError::Code::certain_fault:
        return "certain_fault";
    case PriorsError::Code::too_large:
        return "too_large";
    }
    return "unknown";
}

std::expected<Priors, PriorsError> Priors::from_probabilities(std::span<const double> probabilities) {
    if (probabilities.size() > std::numeric_limits<index_t>::max() / 2) {
        return std::unexpected(PriorsError{
            .code = PriorsError::Code::too_large,
            .column = 0,
            .detail = std::format("{} priors exceed 32-bit indexing", probabilities.size())});
    }
    Priors priors;
    priors.p_.assign(probabilities.begin(), probabilities.end());
    priors.llr_.resize(probabilities.size());
    for (std::size_t j = 0; j < probabilities.size(); ++j) {
        const double p = probabilities[j];
        // Written so that NaN fails the test.
        if (!(p >= 0.0 && p <= 1.0)) { // NOLINT(readability-simplify-boolean-expr)
            return std::unexpected(
                PriorsError{.code = PriorsError::Code::not_a_probability,
                            .column = static_cast<index_t>(j),
                            .detail = std::format("p[{}] = {} is not in [0, 1]", j, p)});
        }
        if (p == 1.0) {
            return std::unexpected(PriorsError{
                .code = PriorsError::Code::certain_fault,
                .column = static_cast<index_t>(j),
                .detail = std::format(
                    "p[{}] = 1: a certain fault belongs in the syndrome, not the problem", j)});
        }
        // One division then one log, the same two correctly rounded steps for every column.
        priors.llr_[j] = std::log((1.0 - p) / p);
    }
    return priors;
}

} // namespace rtd
