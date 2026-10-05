#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "rtd/core/result.hpp"
#include "rtd/core/types.hpp"
#include "rtd/harness/batch.hpp"
#include "rtd/harness/recording.hpp"

namespace rtd::harness {

// One shot handed to a worker.
struct ShotTask {
    std::size_t row = 0;           // the shot's row in the run's outputs
    std::size_t shot = 0;          // the shot's index in the shots file; keys its relay γ draws
    std::span<const Bit> syndrome; // σ, with the artifact's syndrome bias applied
    std::span<const Bit> truth;    // the true observable flips
};

// Why a worker could not decode a shot: the decoder, or the window layer around it, refused its
// input. `code` names a static string; the numbers locate the problem.
struct ShotError {
    std::string_view code;
    std::size_t window = 0; // window position, where it applies
    std::size_t expected = 0;
    std::size_t found = 0;
};

// The outputs of a run while the workers fill them. Every row is written by exactly one worker,
// so the workers take no locks. Variable-length data are kept per row and joined into the CSR
// arrays of ShotResults once every shot is decoded.
struct RunOutputs {
    ShotResults results;
    // Per row, with RunOptions::save_solution_supports: the supports of the row's recorded
    // solutions, window after window and slot after slot (slot sizes are in results.sol_size).
    std::vector<std::vector<index_t>> solution_supports;
    // Per row, with RunOptions::save_commits: the faults each window committed, window after
    // window, and per (row, window) cell how many there are.
    std::vector<std::vector<index_t>> committed_faults;
    std::vector<std::uint32_t> commit_counts; // [count · K]
};

// One worker's decoder, with the number format, executor, solution sink and window mode chosen
// at run time. The batch loop hands it one shot at a time; the worker decodes the shot and
// writes that shot's row of every output, so whole-shot and windowed decoding share the threads,
// the shot queue and the writer. A virtual call per shot is nothing against a decode's cost.
class Worker {
public:
    Worker() = default;
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    Worker(Worker&&) = delete;
    Worker& operator=(Worker&&) = delete;
    virtual ~Worker() = default;

    // An untimed decode that writes nothing (warmup); false when the decoder rejected the input.
    [[nodiscard]] virtual bool warm_up(std::span<const Bit> syndrome,
                                       std::uint64_t stream) noexcept = 0;

    // Decodes one shot and writes row task.row of `out`, its decode time included. An error
    // means the decoder rejected the input; the row is then left as initialised, or partly
    // written.
    [[nodiscard]] virtual std::expected<void, ShotError> decode_shot(const ShotTask& task,
                                                                     RunOutputs& out) = 0;
};

// Copies what `sink` saw in one decode into window `window` of row `row`: the solution count,
// the class of the returned ê (`returned_class`), each stored solution's slot and, when the sink
// keeps them, the supports (appended to the row, so a row's windows must be recorded in order).
void record_solutions(const SolutionRecorder& sink, std::uint64_t returned_class, std::size_t row,
                      std::size_t window, RunOutputs& out);

} // namespace rtd::harness
