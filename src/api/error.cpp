#include "rtd/api/error.hpp"

#include <format>

namespace rtd::api {

std::string_view to_string(ApiError::Code code) noexcept {
    using enum ApiError::Code;
    switch (code) {
    case invalid_spec:
        return "invalid_spec";
    case invalid_problem:
        return "invalid_problem";
    case plan_rejected:
        return "plan_rejected";
    case gamma_source:
        return "gamma_source";
    case construction_failed:
        return "construction_failed";
    case invalid_input:
        return "invalid_input";
    case decode_failed:
        return "decode_failed";
    case stream_state:
        return "stream_state";
    }
    return "unknown";
}

std::string describe(const ApiError& error) {
    return std::format("{}: {}", to_string(error.code), error.detail);
}

} // namespace rtd::api
