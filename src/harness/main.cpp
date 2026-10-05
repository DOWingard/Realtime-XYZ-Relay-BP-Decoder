// rtd_decode: decodes a range of sampled shots with one decoder configuration and writes
// per-shot results, summary statistics and a provenance record (see rtd/harness/cli.hpp).

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <span>

#include "rtd/harness/cli.hpp"

int main(int argc, char** argv) {
    try {
        return rtd::harness::decode_main(
            std::span<char* const>(argv, static_cast<std::size_t>(argc)), std::cout, std::cerr);
    } catch (...) {
        // Thrown before the logger existed or by the logger itself, so report without it.
        std::fputs(R"({"level":"error","context":"run",)"
                   R"("message":"unexpected exception outside the run; aborting"})"
                   "\n",
                   stderr);
    }
    return 1;
}
