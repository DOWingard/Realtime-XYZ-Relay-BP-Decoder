#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <vector>

#include "rtd/core/graph.hpp"
#include "rtd/core/observables.hpp"
#include "rtd/core/priors.hpp"
#include "rtd/core/types.hpp"
#include "rtd/io/error.hpp"
#include "rtd/io/manifest.hpp"

namespace rtd::io {

struct ArtifactOptions {
    GraphOptions graph{};
    // Hash every file and compare with the manifest before trusting it.
    bool verify_checksums = true;
};

// A decoding problem exported by the Python pipeline: H (rows = detectors, columns = faults),
// the priors p, the observable matrix A and per-detector metadata.
//
// Decoders keep pointers to `graph` and `priors`, so an Artifact must stay where it is while any
// decoder built from it is alive (hold it by unique_ptr or in a stable place).
struct Artifact {
    std::filesystem::path directory;
    Manifest manifest;
    TannerGraph graph;
    Priors priors;
    ObservableMatrix observables;
    std::vector<std::int32_t> detector_round;
    std::vector<std::uint8_t> detector_type;
    // Present only when the export pruned certain (p = 1) faults: σ ⊕ syndrome_bias is the
    // syndrome of the pruned problem, and ℓ̂ ⊕ observables_bias the frame change it implies.
    std::optional<std::vector<Bit>> syndrome_bias;
    std::optional<std::vector<Bit>> observables_bias;

    [[nodiscard]] index_t num_detectors() const noexcept { return graph.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return graph.num_columns(); }
    [[nodiscard]] index_t num_observables() const noexcept { return observables.num_rows(); }
};

// Loads and cross-checks an artifact directory: every array's length against the manifest's
// counts and the graph factory's validation, and (optionally) every file's SHA-256.
[[nodiscard]] std::expected<Artifact, IoError> load_artifact(const std::filesystem::path& directory,
                                                             const ArtifactOptions& options = {});

} // namespace rtd::io
