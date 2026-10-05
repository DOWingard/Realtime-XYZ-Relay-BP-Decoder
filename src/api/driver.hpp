#pragma once

// The batch loop shared by the whole-shot and the windowed decoder: shots are handed out to the
// workers from an atomic counter, each worker decodes its shots with its own decoder and writes
// their rows of the result, so the loop takes no lock and the result does not depend on how many
// workers ran or which one decoded which shot.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "rtd/api/batch.hpp"
#include "rtd/api/error.hpp"
#include "rtd/api/problem.hpp"
#include "rtd/core/types.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/selection_spec.hpp"
#include "rtd/window/stream.hpp"

namespace rtd::api::detail {

// Faults committed per (shot, window) cell, collected per shot while the workers run and joined
// into one CSR array afterwards.
struct CommitStaging {
    std::vector<std::vector<index_t>> faults; // [shots]: the shot's windows, in order
    std::vector<std::uint32_t> counts;        // [shots · K]
};

// One worker's decoder.
class ShotWorker {
public:
    ShotWorker() = default;
    ShotWorker(const ShotWorker&) = delete;
    ShotWorker& operator=(const ShotWorker&) = delete;
    ShotWorker(ShotWorker&&) = delete;
    ShotWorker& operator=(ShotWorker&&) = delete;
    virtual ~ShotWorker() = default;

    // Decodes the syndrome σ of one shot (m bytes of 0 or 1, syndrome bias applied) with γ
    // stream `stream`, writes the shot's records into row `row` of `out` (and its commits into
    // `commits` when that is not null), and the predicted flips A·ê, before the observable bias,
    // into `predicted` (k bytes).
    [[nodiscard]] virtual std::expected<void, ApiError>
    decode(std::span<const Bit> syndrome, std::uint64_t stream, std::size_t row,
           std::span<Bit> predicted, BatchResult& out, CommitStaging* commits) = 0;
};

// What a batch records besides the per-shot arrays.
struct BatchShape {
    bool sliding = false;
    std::uint32_t windows = 1;        // K
    std::uint32_t solution_slots = 0; // N
    // The spec's selection policy, when it has one: every decode's confidence is recorded.
    const harness::SelectionSpec* selection = nullptr;
};

// Runs a batch on the first options.workers of `workers`.
[[nodiscard]] std::expected<BatchResult, ApiError>
run_batch(std::span<const std::unique_ptr<ShotWorker>> workers, const Problem& problem,
          const BatchShape& shape, const BatchInput& input, const BatchOptions& options);

// Writes what a solution recorder saw in one decode into cell `cell` of `out` (N slots from
// cell · N), with `returned_class` the class of the returned ê.
inline void write_solutions(const harness::SolutionRecorder& sink, std::uint64_t returned_class,
                            std::size_t cell, BatchResult& out) {
    const std::size_t base = cell * out.solution_slots;
    out.sol_count[cell] = sink.found();
    out.returned_class[cell] = returned_class;
    const auto records = sink.records();
    for (std::size_t s = 0; s < records.size(); ++s) {
        out.sol_leg[base + s] = records[s].leg;
        out.sol_iterations[base + s] = records[s].cumulative_iterations;
        out.sol_weight[base + s] = records[s].weight;
        out.sol_class[base + s] = records[s].logical_class;
        out.sol_hash[base + s] = records[s].hash;
        out.sol_size[base + s] = records[s].size;
    }
}

// A failure the window layer reported, as an API error: a round or shot of the wrong size is
// invalid_input, a call out of order stream_state, anything else decode_failed.
[[nodiscard]] ApiError stream_failure(const window::StreamError& error);

} // namespace rtd::api::detail
