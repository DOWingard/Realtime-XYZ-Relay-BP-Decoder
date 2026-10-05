#pragma once

#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "rtd/io/error.hpp"

namespace rtd::io {

// A manifest.json: parsed, with typed access to required fields and to the checksums of the
// files it describes. The raw document is kept so a run record can embed it verbatim.
class Manifest {
public:
    [[nodiscard]] static std::expected<Manifest, IoError> load(const std::filesystem::path& file);

    [[nodiscard]] const nlohmann::json& raw() const noexcept { return raw_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    // The value at a JSON pointer such as "/num_detectors", converted to T.
    template <class T>
    [[nodiscard]] std::expected<T, IoError> get(std::string_view pointer) const {
        try {
            const nlohmann::json::json_pointer ptr{std::string(pointer)};
            if (!raw_.contains(ptr)) {
                return std::unexpected(IoError{.code = IoError::Code::missing_field,
                                               .path = path_.string(),
                                               .expected = std::format("a field at {}", pointer),
                                               .found = "nothing"});
            }
            return raw_.at(ptr).get<T>();
        } catch (const nlohmann::json::exception& e) {
            return std::unexpected(IoError{.code = IoError::Code::json_error,
                                           .path = path_.string(),
                                           .expected = std::format("a valid value at {}", pointer),
                                           .found = e.what()});
        }
    }

    // Checksum recorded for `file` under /sha256 (or under `section` when given, e.g. an
    // "/artifact/sha256" object), if any.
    [[nodiscard]] std::optional<std::string> sha256(std::string_view file,
                                                    std::string_view section = "/sha256") const;

    // Hashes directory/file and compares it with the recorded checksum; a missing record fails.
    [[nodiscard]] std::expected<void, IoError> verify(const std::filesystem::path& directory,
                                                      std::string_view file) const;

private:
    Manifest(nlohmann::json raw, std::filesystem::path path)
        : raw_(std::move(raw)), path_(std::move(path)) {}

    nlohmann::json raw_;
    std::filesystem::path path_;
};

} // namespace rtd::io
