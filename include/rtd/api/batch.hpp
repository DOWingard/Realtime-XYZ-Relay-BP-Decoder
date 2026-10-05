#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

#include "rtd/api/error.hpp"
#include "rtd/core/types.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/spec.hpp"

namespace rtd::api {

// Parses a decoder spec (the JSON format of rtd_decode's --config, version 2 or 3) and checks
// that this API can run it: the CPU backend, with any number format (f32, f64 or a fixed-point
// intN.S.M) and, in version 3, any selection policy. Relative γ-table paths are resolved against
// `base`.
[[nodiscard]] std::expected<harness::DecoderSpec, ApiError>
parse_spec(std::string_view json_text, const std::filesystem::path& base);

// How a decoder is built. Every worker owns a complete decoder (buffers, and for sliding windows
// one inner decoder per window shape), so `workers` bounds the threads a batch can use.
struct DecoderOptions {
    unsigned workers = 1;
    // Converged legs recorded per decode (0 = off, at most 20): the first N, in leg order.
    std::uint32_t record_solutions = 0;
};

// A batch of shots: row s of `detectors` holds shot s's detection events, either one byte per
// detector (0 or 1) or bit-packed with detector i in bit i mod 8 of byte ⌊i/8⌋ (numpy's
// bitorder="little", stim's b8 format).
struct BatchInput {
    std::span<const Bit> detectors; // shots × row_bytes
    std::size_t shots = 0;
    std::size_t row_bytes = 0;
    bool bit_packed = false;
    // Shot s draws its relay γ from stream stream_offset + s, so a batch reproduces an
    // rtd_decode run over the same shots with --first equal to the offset of its first shot.
    std::uint64_t stream_offset = 0;
};

struct BatchOptions {
    unsigned workers = 1; // at most the decoder's
    // Predicted observable flips bit-packed like the input (⌈k/8⌉ bytes per shot), not one byte
    // per observable.
    bool pack_predictions = false;
    // Whole shot: every shot's returned ê as a dense row of n bytes.
    bool save_decodings = false;
    // Sliding windows: the global faults each window commits, as a CSR list over (shot, window).
    bool save_commits = false;
};

// Everything recorded for a batch, in the layout (and under the names) rtd_decode writes. Entry s
// belongs to shot s of the batch.
struct BatchResult {
    std::size_t shots = 0;
    index_t num_columns = 0;
    index_t num_observables = 0;         // k
    std::size_t predicted_row_bytes = 0; // k, or ⌈k/8⌉ when packed

    std::vector<Bit> predicted;            // [shots, predicted_row_bytes]: ℓ̂, bias applied
    std::vector<std::uint8_t> success;     // some leg (every window) converged
    std::vector<std::uint32_t> iterations; // summed over legs (and windows)
    std::vector<std::uint32_t> legs;
    std::vector<std::int32_t> best_leg;   // returned leg; −1 if none converged or windowed
    std::vector<double> weight;           // W(ê); windowed: Σ committed weight; +∞ if none
    std::vector<std::uint64_t> decode_ns; // wall time of the shot's decode
    std::vector<Bit> decodings;           // [shots, n], whole shot, when asked

    // Sliding windows: K positions per shot, cell s·K + k.
    std::uint32_t windows = 1;         // K
    std::vector<std::uint8_t> flagged; // [shots]
    std::vector<std::uint32_t> win_iterations;
    std::vector<std::uint32_t> win_legs;
    std::vector<std::uint8_t> win_attempts;
    std::vector<std::uint8_t> win_converged;
    std::vector<std::uint8_t> win_cap_hit;
    std::vector<double> win_weight;
    std::vector<double> win_committed_weight;
    std::vector<std::uint32_t> win_unexplained;
    std::vector<std::uint8_t> win_flagged;
    std::vector<std::uint32_t> win_virtual;
    std::vector<std::uint64_t> win_decode_ns;
    // Cell q committed the faults commit_faults[commit_ptr[q], commit_ptr[q + 1]).
    std::vector<std::uint64_t> commit_ptr;
    std::vector<std::uint32_t> commit_faults;

    // Solution records, N slots per decode (K decodes per shot when windowed). Empty slots have
    // weight +∞ and zeros elsewhere; sol_count says how many legs converged (it may exceed N).
    std::uint32_t solution_slots = 0;          // N
    std::vector<std::uint32_t> sol_count;      // [shots, K]
    std::vector<std::uint32_t> sol_leg;        // [shots, K, N]
    std::vector<std::uint32_t> sol_iterations; // [shots, K, N]
    std::vector<double> sol_weight;            // [shots, K, N]
    std::vector<std::uint64_t> sol_class;      // [shots, K, N]
    std::vector<std::uint64_t> sol_hash;       // [shots, K, N]
    std::vector<std::uint32_t> sol_size;       // [shots, K, N]
    std::vector<std::uint64_t> returned_class; // [shots, K]

    // With a selection policy (spec version 3): every decode's confidence per (shot, window) cell
    // and, for sliding windows with "history", the per-window signal over the last L windows. The
    // arrays, their layout and their empty values are rtd_decode's conf_* / hist_* outputs
    // (rtd/harness/confidence_outputs.hpp); `confidence.enabled` is false without a policy.
    harness::ConfidenceOutputs confidence;
};

// The bytes one shot of m detectors takes: m, or ⌈m/8⌉ packed.
[[nodiscard]] constexpr std::size_t row_bytes_for(std::size_t bits, bool packed) noexcept {
    return packed ? (bits + 7) / 8 : bits;
}

// Checks a batch against a problem of m detectors: at least one shot, rows of exactly
// row_bytes_for(m, bit_packed) bytes, and a buffer of shots × row_bytes. Byte values are not
// checked: an unpacked byte other than 0 counts as a detection event.
[[nodiscard]] std::expected<void, ApiError> check_batch(const BatchInput& input, index_t num_rows);

} // namespace rtd::api
