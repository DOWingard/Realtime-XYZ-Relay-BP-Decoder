#include "rtd/window/spec.hpp"

#include <format>
#include <string>

namespace rtd::window {

std::string_view to_string(Boundary boundary) noexcept {
    switch (boundary) {
    case Boundary::exact:
        return "exact";
    case Boundary::uniform:
        return "uniform";
    }
    return "unknown";
}

std::string_view to_string(OnFailure policy) noexcept {
    switch (policy) {
    case OnFailure::commit_anyway:
        return "commit_anyway";
    case OnFailure::defer:
        return "defer";
    case OnFailure::flag:
        return "flag";
    }
    return "unknown";
}

std::optional<Boundary> parse_boundary(std::string_view text) noexcept {
    for (const Boundary boundary : {Boundary::exact, Boundary::uniform}) {
        if (text == to_string(boundary)) {
            return boundary;
        }
    }
    return std::nullopt;
}

std::optional<OnFailure> parse_on_failure(std::string_view text) noexcept {
    for (const OnFailure policy : {OnFailure::commit_anyway, OnFailure::defer, OnFailure::flag}) {
        if (text == to_string(policy)) {
            return policy;
        }
    }
    return std::nullopt;
}

std::expected<void, PlanError> validate(const WindowSpec& spec) {
    using Code = PlanError::Code;
    const auto fail = [](Code code, std::string detail) {
        return std::unexpected(PlanError{.code = code, .detail = std::move(detail)});
    };
    // C < W leaves at least one buffer round, so the columns merged at the window's cut edge are
    // never among the committed ones.
    if (spec.commit == 0 || spec.commit >= spec.width) {
        return fail(Code::commit_out_of_range,
                    std::format("need 1 <= commit < width, got commit = {}, width = {}",
                                spec.commit, spec.width));
    }
    if (spec.converge_rounds < spec.commit || spec.converge_rounds > spec.width) {
        return fail(Code::converge_out_of_range,
                    std::format("need commit <= converge_rounds <= width, got converge_rounds = "
                                "{}, commit = {}, width = {}",
                                spec.converge_rounds, spec.commit, spec.width));
    }
    if (spec.max_deferrals != 0 && spec.on_failure != OnFailure::defer) {
        return fail(Code::deferrals_without_defer,
                    std::format("max_deferrals = {} needs on_failure = defer, got {}",
                                spec.max_deferrals, to_string(spec.on_failure)));
    }
    if (spec.iteration_cap && *spec.iteration_cap == 0) {
        return fail(Code::zero_iteration_cap, "iteration_cap = 0 allows no iteration at all");
    }
    return {};
}

} // namespace rtd::window
