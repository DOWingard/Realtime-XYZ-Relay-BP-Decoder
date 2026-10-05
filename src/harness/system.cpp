#include "rtd/harness/system.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <system_error>
#include <thread>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#ifndef RTD_GIT_REVISION
#define RTD_GIT_REVISION "unknown"
#endif
#ifndef RTD_BUILD_TYPE
#define RTD_BUILD_TYPE "unknown"
#endif
#ifndef RTD_KERNEL_FLAGS
#define RTD_KERNEL_FLAGS "unknown"
#endif

namespace rtd::harness {

BuildInfo build_info() {
#ifdef __clang__
    std::string compiler = std::format("clang {}", __clang_version__);
#elifdef __GNUC__
    std::string compiler = std::format("gcc {}", __VERSION__);
#else
    std::string compiler = "unknown";
#endif
    return BuildInfo{.git_revision = RTD_GIT_REVISION,
                     .compiler = std::move(compiler),
                     .build_type = RTD_BUILD_TYPE,
                     .flags = RTD_KERNEL_FLAGS};
}

std::string cpu_model() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.starts_with("model name")) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                const auto start = line.find_first_not_of(' ', colon + 1);
                return start == std::string::npos ? std::string{} : line.substr(start);
            }
        }
    }
    return "unknown";
}

std::string host_name() {
    std::string name(256, '\0');
    if (gethostname(name.data(), name.size()) != 0) {
        return "unknown";
    }
    name.resize(name.find('\0'));
    return name;
}

unsigned hardware_threads() noexcept { return std::max(std::thread::hardware_concurrency(), 1U); }

std::expected<std::vector<unsigned>, std::string> parse_cpu_list(std::string_view text) {
    std::vector<unsigned> cpus;
    const auto number = [](std::string_view token) -> std::expected<unsigned, std::string> {
        unsigned value = 0;
        const auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec != std::errc{} || end != token.data() + token.size() || token.empty()) {
            return std::unexpected(std::format("'{}' is not a CPU number", token));
        }
        return value;
    };
    while (!text.empty()) {
        const auto comma = text.find(',');
        const std::string_view item = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        const auto dash = item.find('-');
        auto low = number(item.substr(0, dash));
        if (!low) {
            return std::unexpected(low.error());
        }
        unsigned high = *low;
        if (dash != std::string_view::npos) {
            auto parsed = number(item.substr(dash + 1));
            if (!parsed) {
                return std::unexpected(parsed.error());
            }
            high = *parsed;
            if (high < *low) {
                return std::unexpected(std::format("range '{}' is decreasing", item));
            }
        }
        for (unsigned cpu = *low; cpu <= high; ++cpu) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        return std::unexpected(std::string("empty CPU list"));
    }
    std::ranges::sort(cpus);
    const auto [first, last] = std::ranges::unique(cpus);
    cpus.erase(first, last);
    return cpus;
}

std::expected<void, std::string> pin_current_thread(unsigned cpu) {
    if (cpu >= CPU_SETSIZE) {
        return std::unexpected(std::format("CPU {} is beyond CPU_SETSIZE", cpu));
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set); rc != 0) {
        return std::unexpected(
            std::format("pthread_setaffinity_np(CPU {}): {}", cpu, std::generic_category().message(rc)));
    }
    return {};
}

} // namespace rtd::harness
