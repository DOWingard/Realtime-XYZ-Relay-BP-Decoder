#include "rtd/io/error.hpp"

#include <format>

namespace rtd::io {

std::string_view to_string(IoError::Code code) noexcept {
    switch (code) {
    case IoError::Code::open_failed:
        return "open_failed";
    case IoError::Code::read_failed:
        return "read_failed";
    case IoError::Code::write_failed:
        return "write_failed";
    case IoError::Code::bad_format:
        return "bad_format";
    case IoError::Code::unsupported_dtype:
        return "unsupported_dtype";
    case IoError::Code::dtype_mismatch:
        return "dtype_mismatch";
    case IoError::Code::shape_mismatch:
        return "shape_mismatch";
    case IoError::Code::json_error:
        return "json_error";
    case IoError::Code::missing_field:
        return "missing_field";
    case IoError::Code::checksum_mismatch:
        return "checksum_mismatch";
    case IoError::Code::inconsistent:
        return "inconsistent";
    case IoError::Code::invalid_data:
        return "invalid_data";
    }
    return "unknown";
}

std::string describe(const IoError& error) {
    return std::format("{}: {}: expected {}, found {}", to_string(error.code), error.path,
                       error.expected, error.found);
}

} // namespace rtd::io
