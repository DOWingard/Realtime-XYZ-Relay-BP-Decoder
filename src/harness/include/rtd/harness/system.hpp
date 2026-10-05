#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

// Facts about the machine and the build, for the run record, and thread placement.

namespace rtd::harness {

struct BuildInfo {
    std::string git_revision; // `git describe --always --dirty` at configure time
    std::string compiler;     // compiler identification and version
    std::string build_type;
    std::string flags; // compile flags of the decoder kernels
};

[[nodiscard]] BuildInfo build_info();

// "model name" from /proc/cpuinfo, or "unknown".
[[nodiscard]] std::string cpu_model();
[[nodiscard]] std::string host_name();
[[nodiscard]] unsigned hardware_threads() noexcept;

// Parses a CPU list such as "0-2,6,8-9" (the format of taskset and /sys) into ascending,
// de-duplicated CPU numbers.
[[nodiscard]] std::expected<std::vector<unsigned>, std::string> parse_cpu_list(std::string_view text);

// Restricts the calling thread to one CPU.
[[nodiscard]] std::expected<void, std::string> pin_current_thread(unsigned cpu);

} // namespace rtd::harness
