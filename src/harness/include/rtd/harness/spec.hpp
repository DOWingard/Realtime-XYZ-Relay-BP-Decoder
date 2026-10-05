#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "rtd/core/config.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/harness/selection_spec.hpp"
#include "rtd/window/spec.hpp"

namespace rtd::harness {

// The spec format this build reads. Version 2 added the required "window" object; a version-1
// spec becomes the identical version-2 decoder by adding "version": 2 and
// "window": {"mode": "whole_shot"}.
inline constexpr std::uint32_t spec_version = 2;
// Version 3 adds the required "selection" object (selection_spec.hpp); "selection": null gives
// the version-2 decoder. This build reads versions 2 and 3.
inline constexpr std::uint32_t selection_spec_version = 3;

// The number format: IEEE binary32 / binary64, or a fixed-point format intN.S.M (see
// rtd/core/fixed_arith.hpp).
enum class Policy : std::uint8_t { f32, f64, int4_2_8, int5_2_8, int6_2_8 };

// The spec spelling of a format ("f32", "int4.2.8", ...).
[[nodiscard]] std::string_view to_string(Policy policy) noexcept;
[[nodiscard]] constexpr bool is_fixed_point(Policy policy) noexcept {
    return policy != Policy::f32 && policy != Policy::f64;
}
enum class BackendKind : std::uint8_t { cpu, cuda };

struct ExecutorSpec {
    // 0 means the Serial executor: one thread per decoder. T ≥ 1 means a Team of T threads per
    // decoder. Which CPUs the threads run on is a property of the run, not of the decoder.
    unsigned team_threads = 0;
};

struct GammaSpec {
    enum class Kind : std::uint8_t { none, uniform, explicit_table, explicit_shapes };
    Kind kind = Kind::none;
    std::uint64_t seed = 0;
    double low = 0.0;
    double high = 0.0;
    std::filesystem::path table;     // explicit_table: an .npy file of shape [T, n]
    std::filesystem::path directory; // explicit_shapes: shape_<i>.npy of shape [T, n_i] per shape
};

// How a shot is decoded: as one problem over all its rounds, or as a sequence of overlapping
// windows that each decide the faults of their first rounds.
struct WindowSettings {
    enum class Mode : std::uint8_t { whole_shot, sliding };
    Mode mode = Mode::whole_shot;
    // The window schedule. Meaningful in sliding mode only, where it has passed
    // window::validate.
    window::WindowSpec sliding{};

    [[nodiscard]] bool is_sliding() const noexcept { return mode == Mode::sliding; }
};

// Everything that determines a decoder, parsed from a JSON file. Every field is required, so
// a run's configuration is always fully written down and never silently defaulted:
//
// {
//   "version": 2,
//   "policy": "f32" | "f64",       or instead
//   "arithmetic": "f32" | "f64" | "int4.2.8" | "int5.2.8" | "int6.2.8",
//   "backend": "cpu" | "cuda",
//   "layout": "row_major" | "column_blocked",
//   "column_order": "wavefront" | "degree_classes" | "natural",
//   "block_rows": 64,
//   "executor": {"type": "serial"} | {"type": "team", "threads": 3},
//   "alpha": {"rule": "constant", "value": 1.0} | {"rule": "adaptive", "scaling": 1.0},
//   "gamma0": 0.125 | null,
//   "pre_iter": 80, "set_max_iter": 60, "num_sets": 600,
//   "stopping": {"rule": "after_leg0"} | {"rule": "after_n_converged", "count": 5}
//             | {"rule": "all_legs"},
//   "gamma_source": {"type": "none"}
//                 | {"type": "uniform", "seed": 7, "low": -0.24, "high": 0.66}
//                 | {"type": "explicit", "path": "gammas.npy"}     (whole shot, or one shape)
//                 | {"type": "explicit_shapes", "directory": "gammas"},     (sliding only)
//   "window": {"mode": "whole_shot"}
//           | {"mode": "sliding", "width": 12, "commit": 8, "converge_rounds": 12,
//              "boundary": "exact" | "uniform",
//              "on_failure": "commit_anyway" | "defer" | "flag",
//              "max_deferrals": 0, "iteration_cap": 2000 | null}
// }
//
// The number format is named exactly once, by "policy" (the IEEE formats) or by "arithmetic"
// (every compiled format). A fixed-point format runs on the cpu backend only, and its min-sum
// scaling must be α = 1 or 1 − 2^(−k) on every iteration (constant, or adaptive with scaling 1).
// The window object holds exactly the fields of its mode. Relative gamma table paths and
// directories are resolved against the directory of the spec file. A single explicit table has
// the width of the whole problem, so a sliding decoder accepts it only when its window plan has
// one shape (W ≥ Rt); that is checked when the plan is built.
struct DecoderSpec {
    Policy policy = Policy::f32;
    BackendKind backend = BackendKind::cpu;
    GraphOptions graph{};
    ExecutorSpec executor{};
    MinSumConfig min_sum{};
    RelayConfig relay{};
    GammaSpec gamma{};
    WindowSettings window{};
    // The spec's "version" (2 or 3) and, for version 3, its selection policy (absent when null).
    std::uint32_t version = spec_version;
    std::optional<SelectionSpec> selection = std::nullopt;
    nlohmann::json raw;
};

struct SpecError {
    std::string field;
    std::string problem;
};

[[nodiscard]] std::expected<DecoderSpec, SpecError> parse_spec(const nlohmann::json& document,
                                                               const std::filesystem::path& base);
[[nodiscard]] std::expected<DecoderSpec, SpecError> load_spec(const std::filesystem::path& file);

} // namespace rtd::harness
