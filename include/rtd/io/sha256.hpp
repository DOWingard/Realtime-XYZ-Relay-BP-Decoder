#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "rtd/io/error.hpp"

namespace rtd::io {

// Lowercase hex SHA-256 of a file's contents, streamed in 1 MiB blocks through OpenSSL (which
// uses the CPU's SHA instructions where present).
[[nodiscard]] std::expected<std::string, IoError> sha256_file(const std::filesystem::path& path);

} // namespace rtd::io
