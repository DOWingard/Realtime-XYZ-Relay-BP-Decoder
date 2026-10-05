#pragma once

#include <ostream>
#include <span>

namespace rtd::harness {

// The rtd_decode command line: parses `argv` (argv[0] is the program name), decodes the selected
// shots, writes the results and returns the process exit code: 0 success, 1 internal error,
// 2 usage or configuration, 3 unreadable or inconsistent input, 4 output, 5 decode failure.
// --help goes to `out`; the JSON-lines log, and the usage text after an argument error, go to
// `log`.
[[nodiscard]] int decode_main(std::span<char* const> argv, std::ostream& out, std::ostream& log);

} // namespace rtd::harness
