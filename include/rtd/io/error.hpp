#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace rtd::io {

// A load or save failure, with enough context to log without re-deriving anything: the file,
// what the reader expected and what it found.
struct IoError {
    enum class Code : std::uint8_t {
        open_failed,
        read_failed,
        write_failed,
        bad_format,
        unsupported_dtype,
        dtype_mismatch,
        shape_mismatch,
        json_error,
        missing_field,
        checksum_mismatch,
        inconsistent,
        invalid_data,
    };
    Code code;
    std::string path;
    std::string expected;
    std::string found;
};

[[nodiscard]] std::string_view to_string(IoError::Code code) noexcept;

// One line: "<code>: <path>: expected <expected>, found <found>".
[[nodiscard]] std::string describe(const IoError& error);

} // namespace rtd::io
