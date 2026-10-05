#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>

#include "rtd/core/types.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/manifest.hpp"
#include "rtd/io/npy.hpp"

namespace rtd::io {

// Sampled shots: for each shot the detector outcomes σ (the syndrome) and the true flips of the
// logical observables.
struct Shots {
    std::filesystem::path directory;
    Manifest manifest;
    NpyArray<Bit> detectors;   // [S, m]
    NpyArray<Bit> observables; // [S, k]
    // Noisy syndrome cycles of the memory experiment, from the manifest; the unit of the logical
    // error rate per cycle.
    std::optional<std::uint32_t> rounds;

    [[nodiscard]] std::size_t count() const noexcept { return detectors.rows(); }
    [[nodiscard]] std::span<const Bit> syndrome(std::size_t shot) const noexcept {
        return detectors.row(shot);
    }
    [[nodiscard]] std::span<const Bit> flips(std::size_t shot) const noexcept {
        return observables.row(shot);
    }
};

// Loads a shots directory and checks it against the artifact it will be decoded with: m and k
// must match, and the circuit the shots were sampled from must be the one the artifact was
// exported from (compared by SHA-256).
[[nodiscard]] std::expected<Shots, IoError> load_shots(const std::filesystem::path& directory,
                                                       const Artifact& artifact,
                                                       bool verify_checksums = true);

} // namespace rtd::io
