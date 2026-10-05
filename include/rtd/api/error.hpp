#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rtd::api {

// Why the in-memory decoding API refused a request. `detail` names the offending value, array or
// field, so a caller can log the error without re-deriving anything.
struct ApiError {
    enum class Code : std::uint8_t {
        invalid_spec,        // the decoder spec does not parse, or names something this API
                             // cannot run (a device backend, an unknown number format)
        invalid_problem,     // H, A, the priors, the detector rounds or a bias are inconsistent
        plan_rejected,       // no window plan exists for this problem and window spec
        gamma_source,        // a γ table is missing, unreadable or has the wrong width
        construction_failed, // a backend, relay decoder or stream decoder could not be built
        invalid_input,       // a batch, round or option has the wrong shape, size or value
        decode_failed,       // the decoder or the window layer refused an input mid-run
        stream_state,        // a stream call out of order: a push after the readout round,
                             // or a decode while no window is ready
    };
    Code code = Code::invalid_input;
    std::string detail;
};

[[nodiscard]] std::string_view to_string(ApiError::Code code) noexcept;

// One line: "<code>: <detail>".
[[nodiscard]] std::string describe(const ApiError& error);

} // namespace rtd::api
