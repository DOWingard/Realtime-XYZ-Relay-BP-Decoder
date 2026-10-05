#pragma once

// Helpers shared by the harness tests: the committed bb18 fixture as a shots directory, small
// .npy readers, and rtd_decode's command line run in process.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/core/types.hpp"
#include "rtd/harness/cli.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/npy.hpp"

namespace rtd::test {

inline const std::filesystem::path fixture_root = RTD_FIXTURE_DIR;

// A shots directory holding the golden's syndromes with all-zero observable flips (the golden
// does not record the true flips; tests compare predictions, not failures).
inline std::filesystem::path write_shots(const io::Artifact& artifact, const io::Golden& golden,
                                         const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    const std::size_t shots = golden.count();
    const std::array<std::size_t, 2> det_shape{shots, artifact.num_detectors()};
    const std::array<std::size_t, 2> obs_shape{shots, artifact.num_observables()};
    const std::vector<Bit> flips(shots * artifact.num_observables(), 0);
    EXPECT_TRUE(io::write_npy<Bit>(dir / "detectors.npy", golden.detectors.span(), det_shape));
    EXPECT_TRUE(
        io::write_npy<Bit>(dir / "observables.npy", std::span<const Bit>(flips), obs_shape));
    const nlohmann::json manifest = {
        {"rounds", 3},
        {"code", {{"k", 8}}},
        {"sha256",
         {{"circuit.stim",
           artifact.manifest.get<std::string>("/source_circuit/sha256").value_or("")}}}};
    std::ofstream(dir / "manifest.json") << manifest.dump();
    return dir;
}

// Reads an .npy array written by the harness and checks its dtype (through T) and its shape.
template <io::NpyElement T>
std::vector<T> read_array(const std::filesystem::path& file,
                          const std::vector<std::size_t>& shape) {
    auto array = io::read_npy<T>(file);
    EXPECT_TRUE(array) << file << ": " << (array ? "" : io::describe(array.error()));
    if (!array) {
        return {};
    }
    EXPECT_EQ(array->shape, shape) << file;
    const std::span<const T> values = array->span();
    return {values.begin(), values.end()};
}

// A fresh directory under the test temporary directory.
inline std::filesystem::path scratch_dir(const std::string& name) {
    const std::filesystem::path dir = std::filesystem::path(testing::TempDir()) / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// Every value of an .npy array of element type T (its dtype must be T's).
template <io::NpyElement T> std::vector<T> read_all(const std::filesystem::path& file) {
    auto array = io::read_npy<T>(file);
    EXPECT_TRUE(array) << file;
    if (!array) {
        return {};
    }
    const std::span<const T> values = array->span();
    return {values.begin(), values.end()};
}

inline nlohmann::json read_json(const std::filesystem::path& file) {
    std::ifstream in(file);
    return nlohmann::json::parse(in);
}

struct CliRun {
    int code = -1;
    std::string out;
    // The JSON lines; the usage text after an argument error is skipped.
    std::vector<nlohmann::json> log;
};

// Runs rtd_decode's main with these arguments (the program name is added).
inline CliRun run_cli(const std::vector<std::string>& arguments) {
    std::vector<std::string> storage{"rtd_decode"};
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& argument : storage) {
        argv.push_back(argument.data());
    }
    std::ostringstream out;
    std::ostringstream log;
    CliRun run;
    run.code = harness::decode_main(argv, out, log);
    run.out = out.str();
    std::istringstream lines(log.str());
    std::string line;
    while (std::getline(lines, line)) {
        if (line.starts_with('{')) {
            run.log.push_back(nlohmann::json::parse(line));
        }
    }
    return run;
}

// The first error line of a run, or null.
inline nlohmann::json first_error(const CliRun& run) {
    for (const nlohmann::json& line : run.log) {
        if (line.at("level") == "error") {
            return line;
        }
    }
    return nullptr;
}

inline bool logged(const CliRun& run, const std::string& message) {
    return std::ranges::any_of(
        run.log, [&](const nlohmann::json& line) { return line.at("message") == message; });
}

} // namespace rtd::test
