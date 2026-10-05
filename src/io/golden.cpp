#include "rtd/io/golden.hpp"

#include <array>
#include <format>
#include <vector>

namespace rtd::io {

namespace {

template <class T>
std::unexpected<IoError> propagate(std::expected<T, IoError>& result) {
    return std::unexpected(std::move(result.error()));
}

std::unexpected<IoError> bad_config(const Manifest& manifest, std::string expected,
                                    std::string found) {
    return std::unexpected(IoError{.code = IoError::Code::invalid_data,
                                   .path = manifest.path().string(),
                                   .expected = std::move(expected),
                                   .found = std::move(found)});
}

// The manifest's α convention: absent is 1, exactly 0 selects the adaptive rule
// 1 − 2^(−(t+1)/scaling), and a negative value also means 1.
std::expected<AlphaRule, IoError> alpha_rule(const Manifest& manifest) {
    const nlohmann::json& config = manifest.raw().at("config");
    auto scaling = manifest.get<double>("/config/alpha_iteration_scaling_factor");
    if (!scaling) {
        return propagate(scaling);
    }
    const nlohmann::json& alpha = config.at("alpha");
    if (alpha.is_null()) {
        return ConstantAlpha{1.0};
    }
    if (!alpha.is_number()) {
        return bad_config(manifest, "a number or null for config.alpha", alpha.dump());
    }
    const double value = alpha.get<double>();
    if (value == 0.0) {
        return AdaptiveAlpha{*scaling};
    }
    return ConstantAlpha{value < 0.0 ? 1.0 : value};
}

struct DecoderConfigs {
    MinSumConfig min_sum;
    RelayConfig relay;
};

std::expected<StoppingRule, IoError> stopping_rule(const Manifest& manifest) {
    auto criterion = manifest.get<std::string>("/config/stopping_criterion");
    auto stop_nconv = manifest.get<std::uint32_t>("/config/stop_nconv");
    if (!criterion) {
        return propagate(criterion);
    }
    if (!stop_nconv) {
        return propagate(stop_nconv);
    }
    if (*criterion == "pre_iter") {
        return AfterLeg0{};
    }
    if (*criterion == "nconv") {
        return AfterNConverged{*stop_nconv};
    }
    if (*criterion == "all") {
        return AllLegs{};
    }
    return bad_config(manifest, "stopping_criterion pre_iter, nconv or all", *criterion);
}

std::expected<RelayConfig, IoError> relay_config(const Manifest& manifest,
                                                 const std::string& decoder) {
    if (decoder == "min_sum") {
        auto max_iter = manifest.get<std::uint32_t>("/config/max_iter");
        if (!max_iter) {
            return propagate(max_iter);
        }
        return RelayConfig{
            .pre_iter = *max_iter, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}};
    }
    if (decoder != "relay") {
        return bad_config(manifest, "decoder min_sum or relay", decoder);
    }
    auto pre_iter = manifest.get<std::uint32_t>("/config/pre_iter");
    auto set_max_iter = manifest.get<std::uint32_t>("/config/set_max_iter");
    auto num_sets = manifest.get<std::uint32_t>("/config/num_sets");
    for (auto* field : {&pre_iter, &set_max_iter, &num_sets}) {
        if (!*field) {
            return propagate(*field);
        }
    }
    auto stopping = stopping_rule(manifest);
    if (!stopping) {
        return propagate(stopping);
    }
    return RelayConfig{.pre_iter = *pre_iter,
                       .set_max_iter = *set_max_iter,
                       .num_sets = *num_sets,
                       .stopping = *stopping};
}

std::expected<DecoderConfigs, IoError> decoder_configs(const Manifest& manifest,
                                                       const std::string& decoder) {
    if (!manifest.raw().contains("config") || !manifest.raw().at("config").contains("alpha")) {
        return std::unexpected(IoError{.code = IoError::Code::missing_field,
                                       .path = manifest.path().string(),
                                       .expected = "config.alpha",
                                       .found = "nothing"});
    }
    DecoderConfigs configs;
    auto alpha = alpha_rule(manifest);
    if (!alpha) {
        return propagate(alpha);
    }
    configs.min_sum.alpha = *alpha;
    const nlohmann::json gamma0 = manifest.raw().at("config").value("gamma0", nlohmann::json());
    if (!gamma0.is_null()) {
        configs.min_sum.gamma0 = gamma0.get<double>();
    }
    auto relay = relay_config(manifest, decoder);
    if (!relay) {
        return propagate(relay);
    }
    configs.relay = *relay;
    return configs;
}

using Extent = std::optional<std::size_t>;

// A relay golden's explicit γ table and per-leg record.
std::expected<void, IoError> load_relay_record(Golden& golden, std::size_t shots,
                                               index_t num_columns) {
    const std::filesystem::path& directory = golden.directory;
    const std::array<Extent, 2> gamma_extents{std::nullopt, std::size_t{num_columns}};
    auto table = read_npy<double>(directory / "gammas.npy", 2, gamma_extents);
    if (!table) {
        return propagate(table);
    }
    const std::size_t rows = table->rows();
    auto gammas = ExplicitGammaTable::create(
        std::vector<double>(table->span().begin(), table->span().end()), rows, num_columns);
    if (!gammas) {
        return std::unexpected(IoError{.code = IoError::Code::invalid_data,
                                       .path = (directory / "gammas.npy").string(),
                                       .expected = "a finite gamma table",
                                       .found = gammas.error().detail});
    }
    golden.gammas.emplace(std::move(*gammas));

    const std::array<Extent, 1> ptr_extent{shots + 1};
    auto legs_ptr = read_npy<std::int64_t>(directory / "legs_ptr.npy", 1, ptr_extent);
    if (!legs_ptr) {
        return propagate(legs_ptr);
    }
    const auto legs = static_cast<std::size_t>(legs_ptr->span().back());
    const std::array<Extent, 1> leg_extent{legs};
    auto leg_iterations = read_npy<std::int64_t>(directory / "leg_iterations.npy", 1, leg_extent);
    if (!leg_iterations) {
        return propagate(leg_iterations);
    }
    auto leg_converged = read_npy<std::uint8_t>(directory / "leg_converged.npy", 1, leg_extent);
    if (!leg_converged) {
        return propagate(leg_converged);
    }
    auto leg_best = read_npy<std::uint8_t>(directory / "leg_unique_best.npy", 1, leg_extent);
    if (!leg_best) {
        return propagate(leg_best);
    }
    golden.legs_ptr = std::move(*legs_ptr);
    golden.leg_iterations = std::move(*leg_iterations);
    golden.leg_converged = std::move(*leg_converged);
    golden.leg_unique_best = std::move(*leg_best);
    return {};
}

} // namespace

std::expected<Golden, IoError> load_golden(const std::filesystem::path& directory,
                                           index_t num_detectors, index_t num_columns) {
    auto manifest = Manifest::load(directory / "manifest.json");
    if (!manifest) {
        return propagate(manifest);
    }
    auto decoder = manifest->get<std::string>("/decoder");
    auto float_type = manifest->get<std::string>("/float");
    auto shots = manifest->get<std::size_t>("/num_shots");
    if (!decoder) {
        return propagate(decoder);
    }
    if (!float_type) {
        return propagate(float_type);
    }
    if (!shots) {
        return propagate(shots);
    }
    auto configs = decoder_configs(*manifest, *decoder);
    if (!configs) {
        return propagate(configs);
    }

    const std::size_t s = *shots;
    const std::array<Extent, 2> detector_extents{s, std::size_t{num_detectors}};
    const std::array<Extent, 2> decoding_extents{s, std::size_t{num_columns}};
    const std::array<Extent, 1> shot_extent{s};

    auto detectors = read_npy<Bit>(directory / "detectors.npy", 2, detector_extents);
    if (!detectors) {
        return propagate(detectors);
    }
    auto decoding = read_npy<Bit>(directory / "decoding.npy", 2, decoding_extents);
    if (!decoding) {
        return propagate(decoding);
    }
    auto success = read_npy<std::uint8_t>(directory / "success.npy", 1, shot_extent);
    if (!success) {
        return propagate(success);
    }
    auto iterations = read_npy<std::int64_t>(directory / "iterations.npy", 1, shot_extent);
    if (!iterations) {
        return propagate(iterations);
    }
    auto weight = read_npy<double>(directory / "weight.npy", 1, shot_extent);
    if (!weight) {
        return propagate(weight);
    }

    Golden golden{.directory = directory,
                  .manifest = std::move(*manifest),
                  .decoder = std::move(*decoder),
                  .float_type = std::move(*float_type),
                  .min_sum = configs->min_sum,
                  .relay = configs->relay,
                  .detectors = std::move(*detectors),
                  .decoding = std::move(*decoding),
                  .success = std::move(*success),
                  .iterations = std::move(*iterations),
                  .weight = std::move(*weight),
                  .posterior = std::nullopt,
                  .gammas = std::nullopt,
                  .legs_ptr = std::nullopt,
                  .leg_iterations = std::nullopt,
                  .leg_converged = std::nullopt,
                  .leg_unique_best = std::nullopt};

    if (std::filesystem::exists(directory / "posterior.npy")) {
        const std::array<Extent, 2> extents{std::nullopt, std::size_t{num_columns}};
        auto posterior = read_npy<double>(directory / "posterior.npy", 2, extents);
        if (!posterior) {
            return propagate(posterior);
        }
        golden.posterior = std::move(*posterior);
    }
    if (golden.decoder == "relay") {
        if (auto record = load_relay_record(golden, s, num_columns); !record) {
            return propagate(record);
        }
    }
    return golden;
}

} // namespace rtd::io
