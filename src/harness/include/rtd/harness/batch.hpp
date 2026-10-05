#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rtd/core/gamma.hpp"
#include "rtd/core/types.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/shots.hpp"

namespace rtd::harness {

struct RunOptions {
    std::size_t first = 0; // first shot to decode
    std::size_t count = 0; // shots to decode
    unsigned workers = 1;  // decoders running concurrently, one thread (or team) each
    // CPUs to pin to, consumed in order: worker w's decoder threads take entries
    // [w·T, (w+1)·T) where T is the spec's team size (1 for the serial executor). Empty: no pinning.
    std::vector<unsigned> cpus;
    // Untimed decodes each worker performs before the timed run, to fault in its buffers and
    // warm its caches and branch predictors.
    std::size_t warmup = 0;
    // Every shot's correction as a dense row of n bytes: the returned ê of a whole-shot decode,
    // or the faults every window committed (virtual columns of the uniform boundary omitted).
    bool save_decodings = false;
    // Solutions recorded per decode: the first N converged legs, each as (leg, cumulative
    // iterations, weight, logical class, hash, size). 0 = off; at most RecordingSink::max_capacity.
    std::uint32_t record_solutions = 0;
    // Also keep the support of every recorded solution (needs record_solutions > 0).
    bool save_solution_supports = false;
    // The faults each window commits. Only sliding-window decoding commits by window; a
    // whole-shot run rejects it (the returned ê is `decodings`, or a recorded solution's support).
    bool save_commits = false;
};

// Everything recorded per decoded shot; entry i belongs to shot first + i.
//
// A sliding-window decode gives the per-shot arrays their windowed meaning: success = every
// window converged, iterations and legs = sums over the windows (deferral attempts included),
// best_leg = −1 (each window has its own), weight = Σ over windows of the committed columns' λ,
// decode_ns = wall time of the whole shot (every round pushed, every window decoded), predicted =
// the logical frame the commits accumulated, decodings = the committed correction.
struct ShotResults {
    std::size_t first = 0;
    std::size_t count = 0;
    index_t num_columns = 0;
    index_t num_observables = 0;

    std::vector<std::uint8_t> success;      // some relay leg converged
    std::vector<std::uint32_t> iterations;  // summed over the legs executed
    std::vector<std::uint32_t> legs;        // legs executed
    std::vector<std::int32_t> best_leg;     // leg of the returned solution; −1 if none converged
    std::vector<double> weight;             // W(ê); +∞ if none converged
    std::vector<std::uint64_t> decode_ns;   // wall time of the decode call alone
    std::vector<std::uint8_t> predicted;    // [count, k]: ℓ̂ = A·ê, frame-corrected
    std::vector<std::uint8_t> logical_failure; // ℓ̂ differs from the true flips somewhere
    std::vector<std::uint8_t> decodings;    // [count, n]: ê, only when saved

    // Sliding windows: the record of every window position k of every shot, [count, K] with
    // cell row·K + k. Iterations, legs and decode time are summed over the position's deferral
    // attempts; the other fields describe the attempt whose solution was committed. A position
    // that an earlier final window already decided has attempts 0 and an empty record.
    bool sliding = false;
    window::OnFailure on_failure = window::OnFailure::commit_anyway;
    std::vector<std::uint32_t> win_iterations;
    std::vector<std::uint32_t> win_legs;
    std::vector<std::uint8_t> win_attempts;
    std::vector<std::uint8_t> win_converged;
    std::vector<std::uint8_t> win_cap_hit;
    std::vector<double> win_weight;             // W(ê) reported by the inner decoder; +∞ if none
    std::vector<double> win_committed_weight;   // Σ λ over the committed local columns
    std::vector<std::uint32_t> win_unexplained; // detectors of the committed rounds left unmatched
    std::vector<std::uint8_t> win_flagged;
    std::vector<std::uint32_t> win_virtual;   // committed bulk columns with no counterpart
    std::vector<std::uint64_t> win_decode_ns; // wall time of the position's decode_next calls
    std::vector<std::uint8_t> flagged;        // [count]: some window was flagged
    // With RunOptions::save_commits, cell q committed the global faults
    // commit_faults[commit_ptr[q], commit_ptr[q + 1]), ascending.
    std::vector<std::uint64_t> commit_ptr; // [count·K + 1]
    std::vector<std::uint32_t> commit_faults;

    // Solution records, when RunOptions::record_solutions = N > 0: K windows per shot (K = 1 for
    // a whole-shot decode) with N slots each, holding the window's first N converged legs in leg
    // order. Slots at or past sol_count are empty: weight +∞, every other field 0. A class is A·ê
    // as a k-bit mask, bit o = observable o, without the artifact's observables_bias.
    std::uint32_t windows = 1;                  // K
    std::uint32_t solution_slots = 0;           // N; 0 = nothing recorded
    std::vector<std::uint32_t> sol_count;       // [count, K]: converged legs, may exceed N
    std::vector<std::uint32_t> sol_leg;         // [count, K, N]
    std::vector<std::uint32_t> sol_iterations;  // [count, K, N]: all legs up to and including it
    std::vector<double> sol_weight;             // [count, K, N]: W(ê)
    std::vector<std::uint64_t> sol_class;       // [count, K, N]: A·ê
    std::vector<std::uint64_t> sol_hash;        // [count, K, N]: solution_hash of the support
    std::vector<std::uint32_t> sol_size;        // [count, K, N]: |ê|
    std::vector<std::uint64_t> returned_class;  // [count, K]: A·ê of the returned ê, converged or not
    // With RunOptions::save_solution_supports, slot q = (row·K + window)·N + s holds the
    // ascending columns solsup_idx[solsup_ptr[q], solsup_ptr[q + 1]); empty slots are empty.
    std::vector<std::uint64_t> solsup_ptr;      // [count·K·N + 1]
    std::vector<std::uint32_t> solsup_idx;

    // With a selection policy (spec version 3): every decode's confidence, per (shot, window).
    ConfidenceOutputs confidence;

    double wall_seconds = 0.0; // timed region, all workers
};

struct HarnessError {
    std::string context;
    std::string message;
};

class Worker;
class WindowEvents;
struct SlidingSetup;

// Whether this build can run `spec` with the recording and output options of `options`, checked
// before any input is loaded (BatchRunner::create checks it again).
[[nodiscard]] std::expected<void, HarnessError> check_supported(const DecoderSpec& spec,
                                                                const RunOptions& options);

// Decodes a range of shots on a pool of workers. Each worker owns one decoder (one backend with
// its private buffers); all share the artifact and the γ source read-only. Workers claim shots
// from an atomic counter and write results into preallocated per-shot slots, so the decode path
// takes no locks, and the results do not depend on which worker decoded which shot (relay γ
// draws are keyed on the shot index).
class BatchRunner {
public:
    // Builds every worker's decoder up front, so configuration errors surface before the run.
    // A whole-shot spec decodes with `gammas`; a sliding spec needs `sliding` (build_sliding),
    // which holds the window plan and the γ source of each window shape. `artifact`, `shots`,
    // `gammas`, `sliding` and `logger` must outlive the runner.
    [[nodiscard]] static std::expected<BatchRunner, HarnessError>
    create(const io::Artifact& artifact, const io::Shots& shots, const DecoderSpec& spec,
           const GammaSource* gammas, RunOptions options, JsonLogger& logger,
           const SlidingSetup* sliding = nullptr);

    BatchRunner(BatchRunner&&) noexcept;
    BatchRunner& operator=(BatchRunner&&) noexcept;
    BatchRunner(const BatchRunner&) = delete;
    BatchRunner& operator=(const BatchRunner&) = delete;
    ~BatchRunner();

    [[nodiscard]] std::expected<ShotResults, HarnessError> run();

private:
    struct RunState;

    [[nodiscard]] std::expected<void, HarnessError> gather_supports(RunState& state) const;
    [[nodiscard]] std::expected<void, HarnessError> gather_commits(RunState& state) const;
    void size_window_outputs(RunState& state) const;

    BatchRunner(const io::Artifact& artifact, const io::Shots& shots, RunOptions options,
                unsigned threads_per_decoder, JsonLogger& logger);

    // Worker w's share of a run: pin, warm up, then decode shots until none are left. An
    // exception ends this worker only: it is logged, the queue is closed so the other workers
    // stop after their current shot, and run() reports the failure.
    void work(unsigned w, RunState& state) noexcept;
    void decode_share(unsigned w, RunState& state, bool& warmed_up);
    void abandon(unsigned w, RunState& state, bool warmed_up, std::string_view what) noexcept;

    const io::Artifact* artifact_;
    const io::Shots* shots_;
    RunOptions options_;
    unsigned threads_per_decoder_;
    JsonLogger* logger_;
    // A's columns as k-bit masks when solutions are recorded; the workers' sinks borrow it, so it
    // is declared before them and outlives them.
    std::vector<std::uint64_t> column_class_;
    // Sliding windows only: the shared plan, and the event log the workers report to (declared
    // before them, which point to it).
    const SlidingSetup* sliding_ = nullptr;
    // The spec's selection policy, when it has one (copied: the spec need not outlive the runner).
    std::optional<SelectionSpec> selection_;
    std::unique_ptr<WindowEvents> events_;
    std::vector<std::unique_ptr<Worker>> workers_;
};

} // namespace rtd::harness
