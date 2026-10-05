#include "rtd/harness/logger.hpp"

#include <chrono>
#include <cstdio>
#include <format>
#include <random>
#include <stacktrace>

namespace rtd::harness {

std::string_view to_string(Level level) noexcept {
    switch (level) {
    case Level::debug:
        return "debug";
    case Level::info:
        return "info";
    case Level::warn:
        return "warn";
    case Level::error:
        return "error";
    }
    return "info";
}

std::optional<Level> parse_level(std::string_view text) noexcept {
    for (const Level level : {Level::debug, Level::info, Level::warn, Level::error}) {
        if (text == to_string(level)) {
            return level;
        }
    }
    return std::nullopt;
}

JsonLogger::JsonLogger(std::ostream& out, Level min_level, std::string run_id)
    : out_(&out), min_level_(min_level), run_id_(std::move(run_id)) {}

void JsonLogger::log(Level level, std::string_view context, std::string_view message,
                     nlohmann::json fields, std::optional<std::string_view> error,
                     std::optional<std::string> stack) noexcept {
    if (!enabled(level)) {
        return;
    }
    try {
        const auto now = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now());
        nlohmann::json line = {
            {"timestamp", std::format("{:%FT%TZ}", now)},
            {"level", to_string(level)},
            {"context", context},
            {"message", message},
            {"run_id", run_id_},
        };
        if (error) {
            line["error"] = *error;
        }
        if (stack) {
            line["stack"] = *stack;
        }
        if (fields.is_object()) {
            for (const auto& [key, value] : fields.items()) {
                if (!line.contains(key)) {
                    line[key] = std::move(value);
                }
            }
        }
        // Replace invalid UTF-8 rather than throwing.
        const std::string text =
            line.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        const std::scoped_lock lock(mutex_);
        *out_ << text << '\n';
        out_->flush();
    } catch (...) {
        // Allocation, formatting or locking failed; fputs needs none of them.
        std::fputs(R"({"level":"error","context":"logger",)"
                   R"("message":"a log line could not be built or written"})"
                   "\n",
                   stderr);
    }
}

void JsonLogger::error(std::string_view context, std::string_view message, std::string_view error,
                       nlohmann::json fields) noexcept {
    std::optional<std::string> stack;
    try {
        stack = std::to_string(std::stacktrace::current(1));
    } catch (...) {
        stack.reset(); // the line is still worth writing without its trace
    }
    log(Level::error, context, message, std::move(fields), error, std::move(stack));
}

std::string new_run_id() {
    std::random_device device;
    const std::uint64_t id = (std::uint64_t{device()} << 32U) | device();
    return std::format("{:016x}", id);
}

} // namespace rtd::harness
