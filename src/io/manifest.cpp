#include "rtd/io/manifest.hpp"

#include <cerrno>
#include <fstream>
#include <system_error>

#include "rtd/io/sha256.hpp"

namespace rtd::io {

std::expected<Manifest, IoError> Manifest::load(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) {
        return std::unexpected(
            IoError{.code = IoError::Code::open_failed,
                    .path = file.string(),
                    .expected = "a readable file",
                    .found = std::error_code(errno, std::generic_category()).message()});
    }
    try {
        return Manifest(nlohmann::json::parse(in), file);
    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(IoError{.code = IoError::Code::json_error,
                                       .path = file.string(),
                                       .expected = "valid JSON",
                                       .found = e.what()});
    }
}

std::optional<std::string> Manifest::sha256(std::string_view file, std::string_view section) const {
    const nlohmann::json::json_pointer ptr{std::string(section)};
    if (!raw_.contains(ptr)) {
        return std::nullopt;
    }
    const nlohmann::json& table = raw_.at(ptr);
    const auto entry = table.find(std::string(file));
    if (entry == table.end() || !entry->is_string()) {
        return std::nullopt;
    }
    return entry->get<std::string>();
}

std::expected<void, IoError> Manifest::verify(const std::filesystem::path& directory,
                                              std::string_view file) const {
    const auto recorded = sha256(file);
    if (!recorded) {
        return std::unexpected(IoError{.code = IoError::Code::missing_field,
                                       .path = path_.string(),
                                       .expected = std::format("a sha256 entry for {}", file),
                                       .found = "nothing"});
    }
    auto actual = sha256_file(directory / file);
    if (!actual) {
        return std::unexpected(std::move(actual.error()));
    }
    if (*actual != *recorded) {
        return std::unexpected(IoError{.code = IoError::Code::checksum_mismatch,
                                       .path = (directory / file).string(),
                                       .expected = *recorded,
                                       .found = *actual});
    }
    return {};
}

} // namespace rtd::io
