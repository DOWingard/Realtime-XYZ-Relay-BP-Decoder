#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "rtd/core/types.hpp"

namespace rtd {

// Observes a backend after every iteration. The default does nothing and compiles away: the
// backend converts the marginals to external order and double only when `enabled` is true.
struct NoTrace {
    static constexpr bool enabled = false;
    void on_iteration(std::uint32_t /*leg*/, std::uint32_t /*iteration*/,
                      std::span<const double> /*marginals*/) noexcept {}
};

// Records (leg, iteration, M) after every iteration, M in external column order, for comparing
// a decode against a reference trace step by step. Allocates; for debugging only.
class MarginalTracer {
public:
    static constexpr bool enabled = true;

    struct Step {
        std::uint32_t leg;
        std::uint32_t iteration;
        std::vector<double> marginals;
    };

    void on_iteration(std::uint32_t leg, std::uint32_t iteration,
                      std::span<const double> marginals) {
        steps_.push_back(Step{
            .leg = leg, .iteration = iteration, .marginals = {marginals.begin(), marginals.end()}});
    }

    [[nodiscard]] const std::vector<Step>& steps() const noexcept { return steps_; }
    void clear() noexcept { steps_.clear(); }

private:
    std::vector<Step> steps_;
};

} // namespace rtd
