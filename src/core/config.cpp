#include "rtd/core/config.hpp"

#include <cmath>
#include <format>
#include <utility>
#include <variant>

namespace rtd {

namespace {

std::unexpected<ConfigError> fail(ConfigError::Code code, std::string detail) {
    return std::unexpected(ConfigError{.code = code, .detail = std::move(detail)});
}

} // namespace

double alpha_at(const AlphaRule& rule, std::uint32_t iteration) noexcept {
    if (const auto* constant = std::get_if<ConstantAlpha>(&rule)) {
        return constant->value;
    }
    const auto* adaptive = std::get_if<AdaptiveAlpha>(&rule);
    if (adaptive == nullptr) {
        // Both alternatives are trivially copyable, so the variant is never valueless.
        std::unreachable();
    }
    const double exponent = -(static_cast<double>(iteration + 1) / adaptive->scaling);
    return 1.0 - std::exp2(exponent);
}

bool is_unit_alpha(const AlphaRule& rule) noexcept {
    const auto* constant = std::get_if<ConstantAlpha>(&rule);
    return constant != nullptr && constant->value == 1.0;
}

std::string_view to_string(ConfigError::Code code) noexcept {
    switch (code) {
    case ConfigError::Code::invalid_alpha:
        return "invalid_alpha";
    case ConfigError::Code::invalid_gamma0:
        return "invalid_gamma0";
    case ConfigError::Code::relay_without_memory:
        return "relay_without_memory";
    case ConfigError::Code::zero_pre_iter:
        return "zero_pre_iter";
    case ConfigError::Code::zero_set_max_iter:
        return "zero_set_max_iter";
    case ConfigError::Code::zero_stop_count:
        return "zero_stop_count";
    case ConfigError::Code::missing_gamma_source:
        return "missing_gamma_source";
    case ConfigError::Code::gamma_width_mismatch:
        return "gamma_width_mismatch";
    }
    return "unknown";
}

std::expected<void, ConfigError> validate(const MinSumConfig& min_sum, const RelayConfig& relay) {
    using Code = ConfigError::Code;
    if (const auto* a = std::get_if<ConstantAlpha>(&min_sum.alpha)) {
        if (!std::isfinite(a->value) || a->value <= 0.0) {
            return fail(Code::invalid_alpha,
                        std::format("constant alpha must be finite and > 0, got {}", a->value));
        }
    } else {
        const auto& adaptive = std::get<AdaptiveAlpha>(min_sum.alpha);
        if (!std::isfinite(adaptive.scaling) || adaptive.scaling <= 0.0) {
            return fail(Code::invalid_alpha,
                        std::format("adaptive alpha scaling must be finite and > 0, got {}",
                                    adaptive.scaling));
        }
    }
    if (min_sum.gamma0 && !std::isfinite(*min_sum.gamma0)) {
        return fail(Code::invalid_gamma0, std::format("gamma0 must be finite, got {}", *min_sum.gamma0));
    }
    if (relay.num_sets > 0 && !min_sum.gamma0) {
        return fail(Code::relay_without_memory,
                    std::format("num_sets = {} relay legs need gamma0: without a memory term every "
                                "leg restarts plain min-sum from the priors",
                                relay.num_sets));
    }
    if (relay.pre_iter == 0) {
        return fail(Code::zero_pre_iter, "pre_iter must be >= 1");
    }
    if (relay.num_sets > 0 && relay.set_max_iter == 0) {
        return fail(Code::zero_set_max_iter, "set_max_iter must be >= 1 when num_sets > 0");
    }
    if (const auto* rule = std::get_if<AfterNConverged>(&relay.stopping); rule && rule->count == 0) {
        return fail(Code::zero_stop_count, "AfterNConverged count must be >= 1");
    }
    return {};
}

} // namespace rtd
