#include "rtd/io/artifact.hpp"

#include <array>
#include <format>
#include <string>

#include "rtd/io/npy.hpp"

namespace rtd::io {

namespace {

template <class T>
std::unexpected<IoError> propagate(std::expected<T, IoError>& result) {
    return std::unexpected(std::move(result.error()));
}

std::unexpected<IoError> inconsistent(const std::filesystem::path& path, std::string expected,
                                      std::string found) {
    return std::unexpected(IoError{.code = IoError::Code::inconsistent,
                                   .path = path.string(),
                                   .expected = std::move(expected),
                                   .found = std::move(found)});
}

} // namespace

std::expected<Artifact, IoError> load_artifact(const std::filesystem::path& directory,
                                               const ArtifactOptions& options) {
    auto manifest = Manifest::load(directory / "manifest.json");
    if (!manifest) {
        return propagate(manifest);
    }
    auto format = manifest->get<int>("/format_version");
    auto m = manifest->get<index_t>("/num_detectors");
    auto n = manifest->get<index_t>("/num_columns");
    auto k = manifest->get<index_t>("/num_observables");
    auto nnz_h = manifest->get<std::size_t>("/nnz_H");
    auto nnz_a = manifest->get<std::size_t>("/nnz_A");
    for (auto* field : {&m, &n, &k}) {
        if (!*field) {
            return propagate(*field);
        }
    }
    if (!format) {
        return propagate(format);
    }
    if (!nnz_h) {
        return propagate(nnz_h);
    }
    if (!nnz_a) {
        return propagate(nnz_a);
    }
    if (*format != 1) {
        return std::unexpected(IoError{.code = IoError::Code::bad_format,
                                       .path = manifest->path().string(),
                                       .expected = "artifact format_version 1",
                                       .found = std::format("{}", *format)});
    }

    const bool has_syndrome_bias = std::filesystem::exists(directory / "syndrome_bias.npy");
    const bool has_observables_bias = std::filesystem::exists(directory / "observables_bias.npy");
    if (options.verify_checksums) {
        std::vector<std::string> files{"H_indptr.npy", "H_indices.npy", "A_indptr.npy",
                                       "A_indices.npy", "priors.npy",   "det_round.npy",
                                       "det_type.npy"};
        if (has_syndrome_bias) {
            files.emplace_back("syndrome_bias.npy");
        }
        if (has_observables_bias) {
            files.emplace_back("observables_bias.npy");
        }
        for (const std::string& file : files) {
            if (auto ok = manifest->verify(directory, file); !ok) {
                return propagate(ok);
            }
        }
    }

    using Extent = std::optional<std::size_t>;
    const std::array<Extent, 1> h_ptr_extent{std::size_t{*m} + 1};
    const std::array<Extent, 1> h_idx_extent{*nnz_h};
    const std::array<Extent, 1> a_ptr_extent{std::size_t{*k} + 1};
    const std::array<Extent, 1> a_idx_extent{*nnz_a};
    const std::array<Extent, 1> n_extent{std::size_t{*n}};
    const std::array<Extent, 1> m_extent{std::size_t{*m}};
    const std::array<Extent, 1> k_extent{std::size_t{*k}};

    auto h_ptr = read_npy<index_t>(directory / "H_indptr.npy", 1, h_ptr_extent);
    if (!h_ptr) {
        return propagate(h_ptr);
    }
    auto h_idx = read_npy<index_t>(directory / "H_indices.npy", 1, h_idx_extent);
    if (!h_idx) {
        return propagate(h_idx);
    }
    auto a_ptr = read_npy<index_t>(directory / "A_indptr.npy", 1, a_ptr_extent);
    if (!a_ptr) {
        return propagate(a_ptr);
    }
    auto a_idx = read_npy<index_t>(directory / "A_indices.npy", 1, a_idx_extent);
    if (!a_idx) {
        return propagate(a_idx);
    }
    auto p = read_npy<double>(directory / "priors.npy", 1, n_extent);
    if (!p) {
        return propagate(p);
    }
    auto rounds = read_npy<std::int32_t>(directory / "det_round.npy", 1, m_extent);
    if (!rounds) {
        return propagate(rounds);
    }
    auto types = read_npy<std::uint8_t>(directory / "det_type.npy", 1, m_extent);
    if (!types) {
        return propagate(types);
    }

    auto graph = TannerGraph::from_csr(*m, *n, h_ptr->span(), h_idx->span(), options.graph);
    if (!graph) {
        return inconsistent(directory / "H_indices.npy", "a valid CSR matrix",
                            std::format("{}: {}", to_string(graph.error().code), graph.error().detail));
    }
    auto observables = ObservableMatrix::from_csr(*k, *n, a_ptr->span(), a_idx->span());
    if (!observables) {
        return inconsistent(directory / "A_indices.npy", "a valid CSR matrix",
                            std::format("{}: {}", to_string(observables.error().code),
                                        observables.error().detail));
    }
    auto priors = Priors::from_probabilities(p->span());
    if (!priors) {
        return std::unexpected(IoError{.code = IoError::Code::invalid_data,
                                       .path = (directory / "priors.npy").string(),
                                       .expected = "probabilities in [0, 1)",
                                       .found = priors.error().detail});
    }

    Artifact artifact{.directory = directory,
                      .manifest = std::move(*manifest),
                      .graph = std::move(*graph),
                      .priors = std::move(*priors),
                      .observables = std::move(*observables),
                      .detector_round = {rounds->span().begin(), rounds->span().end()},
                      .detector_type = {types->span().begin(), types->span().end()},
                      .syndrome_bias = std::nullopt,
                      .observables_bias = std::nullopt};
    if (has_syndrome_bias) {
        auto bias = read_npy<std::uint8_t>(directory / "syndrome_bias.npy", 1, m_extent);
        if (!bias) {
            return propagate(bias);
        }
        artifact.syndrome_bias.emplace(bias->span().begin(), bias->span().end());
    }
    if (has_observables_bias) {
        auto bias = read_npy<std::uint8_t>(directory / "observables_bias.npy", 1, k_extent);
        if (!bias) {
            return propagate(bias);
        }
        artifact.observables_bias.emplace(bias->span().begin(), bias->span().end());
    }
    return artifact;
}

} // namespace rtd::io
