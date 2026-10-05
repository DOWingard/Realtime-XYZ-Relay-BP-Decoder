#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>

#include "rtd/core/config.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/core/types.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/manifest.hpp"
#include "rtd/io/npy.hpp"

namespace rtd::io {

// Golden decoder outputs on fixed inputs, written by the Python golden generator: the syndromes
// decoded, and for each the correction, convergence flag, iteration count and solution weight;
// optionally the final marginals of the first shots, and for relay decoders the explicit γ table
// and the per-leg record.
//
// The manifest's decoder configuration is translated into this decoder's configuration types, so
// a test decodes with exactly the settings the goldens were produced with.
struct Golden {
    std::filesystem::path directory;
    Manifest manifest;
    std::string decoder;    // "min_sum" or "relay"
    std::string float_type; // "f32" or "f64"
    MinSumConfig min_sum;
    RelayConfig relay;

    NpyArray<Bit> detectors;              // [S, m]
    NpyArray<Bit> decoding;               // [S, n]
    NpyArray<std::uint8_t> success;       // [S]
    NpyArray<std::int64_t> iterations;    // [S]
    NpyArray<double> weight;              // [S], +inf where success is 0
    std::optional<NpyArray<double>> posterior; // [K, n]

    std::optional<ExplicitGammaTable> gammas;
    // Legs of shot s are entries legs_ptr[s] .. legs_ptr[s+1] of the leg arrays.
    std::optional<NpyArray<std::int64_t>> legs_ptr;
    std::optional<NpyArray<std::int64_t>> leg_iterations;
    std::optional<NpyArray<std::uint8_t>> leg_converged;
    std::optional<NpyArray<std::uint8_t>> leg_unique_best;

    [[nodiscard]] std::size_t count() const noexcept { return detectors.rows(); }
};

// Loads a golden directory for a problem with m detectors and n columns.
[[nodiscard]] std::expected<Golden, IoError> load_golden(const std::filesystem::path& directory,
                                                         index_t num_detectors,
                                                         index_t num_columns);

} // namespace rtd::io
