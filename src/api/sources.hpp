#pragma once

// The relay γ sources a spec asks for, built exactly as rtd_decode builds them, so that the same
// spec draws the same γ for the same shot.

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "rtd/api/error.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/core/types.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/window/plan.hpp"

namespace rtd::api::detail {

// The whole-shot source over n columns: null for "none", a generator for "uniform", the [T, n]
// table for "explicit". An "explicit_shapes" source belongs to sliding windows.
[[nodiscard]] std::expected<std::unique_ptr<GammaSource>, ApiError>
whole_shot_gammas(const harness::GammaSpec& spec, index_t num_columns);

// One source per window shape, in shape order: null for "none"; for "uniform" a generator per
// shape with the spec's seed and interval and the shape's width; for "explicit_shapes" the table
// directory/shape_<i>.npy of shape [T, n_i]; for "explicit" the single [T, n] table, which only a
// plan with one shape can use.
[[nodiscard]] std::expected<std::vector<std::unique_ptr<GammaSource>>, ApiError>
shape_gammas(const harness::GammaSpec& spec, const window::WindowPlan& plan);

} // namespace rtd::api::detail
