#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace rtd::harness {

enum class Level : std::uint8_t { debug, info, warn, error };

[[nodiscard]] std::string_view to_string(Level level) noexcept;
[[nodiscard]] std::optional<Level> parse_level(std::string_view text) noexcept;

// Structured logging: one JSON object per line with timestamp (UTC, ISO 8601, milliseconds),
// level, context, message, the run id that ties every line of one run together, and any extra
// fields. Error lines carry "error" and, when given, "stack". Thread-safe; lines never interleave.
class JsonLogger {
public:
    JsonLogger(std::ostream& out, Level min_level, std::string run_id);

    [[nodiscard]] const std::string& run_id() const noexcept { return run_id_; }
    [[nodiscard]] bool enabled(Level level) const noexcept { return level >= min_level_; }

    // Never throws: a line that cannot be built or written is replaced by a fixed notice on
    // stderr, so reporting a failure cannot itself fail the caller.
    void log(Level level, std::string_view context, std::string_view message,
             nlohmann::json fields = nlohmann::json::object(),
             std::optional<std::string_view> error = std::nullopt,
             std::optional<std::string> stack = std::nullopt) noexcept;

    void debug(std::string_view context, std::string_view message,
               nlohmann::json fields = nlohmann::json::object()) noexcept {
        log(Level::debug, context, message, std::move(fields));
    }
    void info(std::string_view context, std::string_view message,
              nlohmann::json fields = nlohmann::json::object()) noexcept {
        log(Level::info, context, message, std::move(fields));
    }
    void warn(std::string_view context, std::string_view message,
              nlohmann::json fields = nlohmann::json::object()) noexcept {
        log(Level::warn, context, message, std::move(fields));
    }
    // An error with the call site's stack trace attached.
    void error(std::string_view context, std::string_view message, std::string_view error,
               nlohmann::json fields = nlohmann::json::object()) noexcept;

private:
    std::mutex mutex_;
    std::ostream* out_;
    Level min_level_;
    std::string run_id_;
};

// A random 16-hex-digit identifier for one run.
[[nodiscard]] std::string new_run_id();

} // namespace rtd::harness
