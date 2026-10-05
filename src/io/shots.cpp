#include "rtd/io/shots.hpp"

#include <array>
#include <format>
#include <string>

namespace rtd::io {

std::expected<Shots, IoError> load_shots(const std::filesystem::path& directory,
                                         const Artifact& artifact, bool verify_checksums) {
    auto manifest = Manifest::load(directory / "manifest.json");
    if (!manifest) {
        return std::unexpected(std::move(manifest.error()));
    }
    if (verify_checksums) {
        for (const char* file : {"detectors.npy", "observables.npy"}) {
            if (auto ok = manifest->verify(directory, file); !ok) {
                return std::unexpected(std::move(ok.error()));
            }
        }
    }
    const auto shots_circuit = manifest->sha256("circuit.stim");
    auto artifact_circuit = artifact.manifest.get<std::string>("/source_circuit/sha256");
    if (!shots_circuit || !artifact_circuit || *shots_circuit != *artifact_circuit) {
        return std::unexpected(IoError{
            .code = IoError::Code::inconsistent,
            .path = manifest->path().string(),
            .expected =
                std::format("the circuit the artifact was exported from (sha256 {})",
                            artifact_circuit ? *artifact_circuit : std::string("unrecorded")),
            .found = std::format("sha256 {}",
                                 shots_circuit ? *shots_circuit : std::string("unrecorded"))});
    }

    using Extent = std::optional<std::size_t>;
    const std::array<Extent, 2> detector_extents{std::nullopt, std::size_t{artifact.num_detectors()}};
    auto detectors = read_npy<Bit>(directory / "detectors.npy", 2, detector_extents);
    if (!detectors) {
        return std::unexpected(std::move(detectors.error()));
    }
    const std::array<Extent, 2> observable_extents{detectors->rows(),
                                                   std::size_t{artifact.num_observables()}};
    auto observables = read_npy<Bit>(directory / "observables.npy", 2, observable_extents);
    if (!observables) {
        return std::unexpected(std::move(observables.error()));
    }
    auto rounds = manifest->get<std::uint32_t>("/rounds");
    return Shots{.directory = directory,
                 .manifest = std::move(*manifest),
                 .detectors = std::move(*detectors),
                 .observables = std::move(*observables),
                 .rounds = rounds ? std::optional(*rounds) : std::nullopt};
}

} // namespace rtd::io
