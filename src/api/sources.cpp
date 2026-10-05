#include "sources.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "rtd/io/npy.hpp"

namespace rtd::api::detail {

namespace {

std::unexpected<ApiError> gamma_error(std::string detail) {
    return std::unexpected(
        ApiError{.code = ApiError::Code::gamma_source, .detail = std::move(detail)});
}

std::expected<std::unique_ptr<GammaSource>, ApiError> uniform(const harness::GammaSpec& spec,
                                                              index_t width) {
    auto source = UniformGammaGenerator::create(spec.seed, spec.low, spec.high, width);
    if (!source) {
        return gamma_error(source.error().detail);
    }
    return std::make_unique<UniformGammaGenerator>(std::move(*source));
}

// A [T, width] table from an .npy file.
std::expected<std::unique_ptr<GammaSource>, ApiError> table(const std::filesystem::path& file,
                                                            index_t width) {
    const std::array<std::optional<std::size_t>, 2> extents{std::nullopt, std::size_t{width}};
    auto array = io::read_npy<double>(file, 2, extents);
    if (!array) {
        return gamma_error(std::format("cannot read the gamma table {}: {}", file.string(),
                                       io::describe(array.error())));
    }
    const std::span<const double> values = array->span();
    auto source = ExplicitGammaTable::create({values.begin(), values.end()}, array->rows(), width);
    if (!source) {
        return gamma_error(std::format("{}: {}", file.string(), source.error().detail));
    }
    return std::make_unique<ExplicitGammaTable>(std::move(*source));
}

} // namespace

std::expected<std::unique_ptr<GammaSource>, ApiError>
whole_shot_gammas(const harness::GammaSpec& spec, index_t num_columns) {
    switch (spec.kind) {
    case harness::GammaSpec::Kind::none:
        return nullptr;
    case harness::GammaSpec::Kind::uniform:
        return uniform(spec, num_columns);
    case harness::GammaSpec::Kind::explicit_table:
        return table(spec.table, num_columns);
    case harness::GammaSpec::Kind::explicit_shapes:
        return gamma_error("explicit_shapes gamma tables belong to sliding-window decoders");
    }
    return gamma_error("unknown gamma source");
}

std::expected<std::vector<std::unique_ptr<GammaSource>>, ApiError>
shape_gammas(const harness::GammaSpec& spec, const window::WindowPlan& plan) {
    const std::span<const window::Shape> shapes = plan.shapes();
    std::vector<std::unique_ptr<GammaSource>> sources;
    sources.reserve(shapes.size());
    if (spec.kind == harness::GammaSpec::Kind::explicit_table && shapes.size() != 1) {
        return gamma_error(std::format(
            "an explicit gamma table has the width of the whole problem and fits only a window "
            "plan with one shape; this plan has {} shapes: give one table per shape with "
            "{{\"type\": \"explicit_shapes\", \"directory\": ...}}",
            shapes.size()));
    }
    for (const window::Shape& shape : shapes) {
        const index_t width = shape.num_columns();
        std::expected<std::unique_ptr<GammaSource>, ApiError> source = nullptr;
        switch (spec.kind) {
        case harness::GammaSpec::Kind::none:
            break;
        case harness::GammaSpec::Kind::uniform:
            source = uniform(spec, width);
            break;
        case harness::GammaSpec::Kind::explicit_table:
            source = table(spec.table, width);
            break;
        case harness::GammaSpec::Kind::explicit_shapes:
            source = table(spec.directory / std::format("shape_{}.npy", shape.index()), width);
            break;
        }
        if (!source) {
            return std::unexpected(std::move(source.error()));
        }
        sources.push_back(std::move(*source));
    }
    return sources;
}

} // namespace rtd::api::detail
