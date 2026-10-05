#include "rtd/harness/spec.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include "rtd/core/fixed_arith.hpp"

namespace rtd::harness {

namespace {

using json = nlohmann::json;

// Parsing is a sequence of dozens of required-field reads, each of which can fail. Inside this
// file a failure is thrown and parse_spec converts it to the error value at its boundary, which
// keeps every read a single line.
struct SpecFailure {
    SpecError error;
};

[[noreturn]] void fail(std::string field, std::string problem) {
    throw SpecFailure{SpecError{.field = std::move(field), .problem = std::move(problem)}};
}

// A JSON object together with its path from the document root, for error messages.
class Node {
public:
    Node(const json& value, std::string path) : value_(&value), path_(std::move(path)) {}

    [[nodiscard]] const json& value() const noexcept { return *value_; }

    [[nodiscard]] std::string field(std::string_view key) const {
        return path_.empty() ? std::string(key) : std::format("{}.{}", path_, key);
    }

    [[nodiscard]] const json& at(std::string_view key) const {
        if (!value_->is_object()) {
            fail(path_, std::format("expected an object, found {}", value_->dump()));
        }
        const auto it = value_->find(key);
        if (it == value_->end()) {
            fail(field(key), "required field is missing");
        }
        return *it;
    }

    [[nodiscard]] Node child(std::string_view key) const { return {at(key), field(key)}; }

    // Rejects any key outside `allowed`, so a field meant for another mode or a misspelt one is
    // reported instead of ignored.
    void only(std::initializer_list<std::string_view> allowed, std::string_view context) const {
        for (const auto& item : value_->items()) {
            const std::string& key = item.key();
            if (std::ranges::find(allowed, std::string_view(key)) == allowed.end()) {
                fail(field(key), std::format("unknown field for {}", context));
            }
        }
    }

    [[nodiscard]] std::string string(std::string_view key) const {
        const json& value = at(key);
        if (!value.is_string()) {
            fail(field(key), std::format("expected a string, found {}", value.dump()));
        }
        return value.get<std::string>();
    }

    [[nodiscard]] double number(std::string_view key) const {
        const json& value = at(key);
        if (!value.is_number()) {
            fail(field(key), std::format("expected a number, found {}", value.dump()));
        }
        return value.get<double>();
    }

    template <std::unsigned_integral T>
    [[nodiscard]] T count(std::string_view key) const {
        const json& value = at(key);
        // Parsed non-negative integers are stored unsigned, but a document built in code may
        // hold them signed; floats and negative values are rejected either way.
        const bool non_negative = value.is_number_unsigned() ||
                                  (value.is_number_integer() && value.get<std::int64_t>() >= 0);
        if (!non_negative) {
            fail(field(key), std::format("expected a non-negative integer, found {}", value.dump()));
        }
        const auto raw = value.get<std::uint64_t>();
        if (raw > std::numeric_limits<T>::max()) {
            fail(field(key), std::format("{} exceeds the maximum {}", raw,
                                         std::numeric_limits<T>::max()));
        }
        return static_cast<T>(raw);
    }

    template <class Enum>
    [[nodiscard]] Enum choice(std::string_view key,
                              std::initializer_list<std::pair<std::string_view, Enum>> options) const {
        const std::string text = string(key);
        std::string allowed;
        for (const auto& [name, value] : options) {
            if (text == name) {
                return value;
            }
            allowed += std::format("{}{}", allowed.empty() ? "" : ", ", name);
        }
        fail(field(key), std::format("'{}' is not one of {}", text, allowed));
    }

private:
    const json* value_;
    std::string path_;
};

ExecutorSpec parse_executor(const Node& node) {
    enum class Type : std::uint8_t { serial, team };
    switch (node.choice<Type>("type", {{"serial", Type::serial}, {"team", Type::team}})) {
    case Type::serial:
        return ExecutorSpec{.team_threads = 0};
    case Type::team: {
        const auto threads = node.count<unsigned>("threads");
        if (threads == 0) {
            fail(node.field("threads"), "a team needs at least one thread");
        }
        return ExecutorSpec{.team_threads = threads};
    }
    }
    std::unreachable();
}

AlphaRule parse_alpha(const Node& node) {
    enum class Rule : std::uint8_t { constant, adaptive };
    switch (node.choice<Rule>("rule", {{"constant", Rule::constant}, {"adaptive", Rule::adaptive}})) {
    case Rule::constant:
        return ConstantAlpha{node.number("value")};
    case Rule::adaptive:
        return AdaptiveAlpha{node.number("scaling")};
    }
    std::unreachable();
}

StoppingRule parse_stopping(const Node& node) {
    enum class Rule : std::uint8_t { after_leg0, after_n_converged, all_legs };
    switch (node.choice<Rule>("rule", {{"after_leg0", Rule::after_leg0},
                                       {"after_n_converged", Rule::after_n_converged},
                                       {"all_legs", Rule::all_legs}})) {
    case Rule::after_leg0:
        return AfterLeg0{};
    case Rule::after_n_converged:
        return AfterNConverged{node.count<std::uint32_t>("count")};
    case Rule::all_legs:
        return AllLegs{};
    }
    std::unreachable();
}

std::filesystem::path resolve(const std::filesystem::path& path,
                              const std::filesystem::path& base) {
    return path.is_relative() ? base / path : path;
}

GammaSpec parse_gamma(const Node& node, const std::filesystem::path& base) {
    using Kind = GammaSpec::Kind;
    switch (node.choice<Kind>("type", {{"none", Kind::none},
                                       {"uniform", Kind::uniform},
                                       {"explicit", Kind::explicit_table},
                                       {"explicit_shapes", Kind::explicit_shapes}})) {
    case Kind::none:
        return GammaSpec{};
    case Kind::uniform:
        return GammaSpec{.kind = Kind::uniform,
                         .seed = node.count<std::uint64_t>("seed"),
                         .low = node.number("low"),
                         .high = node.number("high"),
                         .table = {},
                         .directory = {}};
    case Kind::explicit_table:
        return GammaSpec{.kind = Kind::explicit_table, .seed = 0, .low = 0, .high = 0,
                         .table = resolve(node.string("path"), base), .directory = {}};
    case Kind::explicit_shapes:
        return GammaSpec{.kind = Kind::explicit_shapes, .seed = 0, .low = 0, .high = 0,
                         .table = {}, .directory = resolve(node.string("directory"), base)};
    }
    std::unreachable();
}

// The number format, named by "policy" (the IEEE formats, as every spec before fixed point did)
// or by "arithmetic" (any compiled format), never both.
Policy parse_number_format(const Node& root) {
    const bool has_policy = root.value().contains("policy");
    const bool has_arithmetic = root.value().contains("arithmetic");
    if (has_policy && has_arithmetic) {
        fail("arithmetic", "the number format is given twice; keep \"arithmetic\" (f32, f64, "
                           "int4.2.8, int5.2.8, int6.2.8) or \"policy\" (f32, f64), not both");
    }
    if (!has_arithmetic) {
        if (!has_policy) {
            fail("policy", "required field is missing (or give \"arithmetic\": f32, f64, int4.2.8, "
                           "int5.2.8, int6.2.8)");
        }
        return root.choice<Policy>("policy", {{"f32", Policy::f32}, {"f64", Policy::f64}});
    }
    return root.choice<Policy>("arithmetic", {{"f32", Policy::f32},
                                              {"f64", Policy::f64},
                                              {"int4.2.8", Policy::int4_2_8},
                                              {"int5.2.8", Policy::int5_2_8},
                                              {"int6.2.8", Policy::int6_2_8}});
}

// Reads the "version" field first, so that a spec written for another version is reported as such
// and not as a list of missing or unknown fields. Returns the version (2 or 3).
std::uint32_t check_version(const Node& root) {
    const json& document = root.value();
    if (!document.contains("version")) {
        fail("version",
             std::format("required field is missing. This is a version-1 spec; version {0} adds "
                         "the required \"window\" object. To keep decoding whole shots exactly as "
                         "before, add \"version\": {0} and \"window\": {{\"mode\": "
                         "\"whole_shot\"}}",
                         spec_version));
    }
    const json& version = document.at("version");
    const bool integer = version.is_number_integer();
    if (integer && std::cmp_equal(version.get<std::int64_t>(), spec_version)) {
        if (document.contains("selection")) {
            fail("selection", std::format("the selection object needs \"version\": {}",
                                          selection_spec_version));
        }
        return spec_version;
    }
    if (integer && std::cmp_equal(version.get<std::int64_t>(), selection_spec_version)) {
        if (!document.contains("selection")) {
            fail("version",
                 std::format("unsupported spec version {0} without the \"selection\" object: "
                             "version {0} requires it (\"selection\": null keeps the version-{1} "
                             "decoder), or set \"version\": {1}",
                             selection_spec_version, spec_version));
        }
        return selection_spec_version;
    }
    fail("version",
         std::format("unsupported spec version {}; this rtd_decode reads versions {} and {} (a "
                     "version-1 spec is upgraded by setting \"version\": {} and adding "
                     "\"window\": {{\"mode\": \"whole_shot\"}})",
                     version.dump(), spec_version, selection_spec_version, spec_version));
}

// The field a window::validate failure is about.
std::string_view window_field(window::PlanError::Code code) noexcept {
    using Code = window::PlanError::Code;
    switch (code) {
    case Code::commit_out_of_range:
        return "window.commit";
    case Code::converge_out_of_range:
        return "window.converge_rounds";
    case Code::deferrals_without_defer:
        return "window.max_deferrals";
    case Code::zero_iteration_cap:
        return "window.iteration_cap";
    default:
        return "window";
    }
}

WindowSettings parse_window(const Node& node) {
    using Mode = WindowSettings::Mode;
    const Mode mode =
        node.choice<Mode>("mode", {{"whole_shot", Mode::whole_shot}, {"sliding", Mode::sliding}});
    if (mode == Mode::whole_shot) {
        node.only({"mode"}, "mode whole_shot (it has no other fields)");
        return WindowSettings{};
    }
    node.only({"mode", "width", "commit", "converge_rounds", "boundary", "on_failure",
               "max_deferrals", "iteration_cap"},
              "mode sliding");
    using window::Boundary;
    using window::OnFailure;
    window::WindowSpec spec{
        .width = node.count<std::uint32_t>("width"),
        .commit = node.count<std::uint32_t>("commit"),
        .converge_rounds = node.count<std::uint32_t>("converge_rounds"),
        .boundary = node.choice<Boundary>(
            "boundary", {{"exact", Boundary::exact}, {"uniform", Boundary::uniform}}),
        .on_failure = node.choice<OnFailure>("on_failure",
                                             {{"commit_anyway", OnFailure::commit_anyway},
                                              {"defer", OnFailure::defer},
                                              {"flag", OnFailure::flag}}),
        .max_deferrals = node.count<std::uint32_t>("max_deferrals"),
        .iteration_cap = std::nullopt,
    };
    if (!node.at("iteration_cap").is_null()) {
        spec.iteration_cap = node.count<std::uint32_t>("iteration_cap");
    }
    if (auto valid = window::validate(spec); !valid) {
        fail(std::string(window_field(valid.error().code)), window::describe(valid.error()));
    }
    return WindowSettings{.mode = Mode::sliding, .sliding = spec};
}

DecoderSpec parse(const json& document, const std::filesystem::path& base) {
    const Node root(document, "");
    const std::uint32_t version = check_version(root);
    DecoderSpec spec;
    spec.version = version;
    spec.raw = document;
    spec.policy = parse_number_format(root);
    spec.backend =
        root.choice<BackendKind>("backend", {{"cpu", BackendKind::cpu}, {"cuda", BackendKind::cuda}});
    spec.graph.layout = root.choice<EdgeLayout>(
        "layout", {{"row_major", EdgeLayout::row_major},
                   {"column_blocked", EdgeLayout::column_blocked}});
    spec.graph.column_order = root.choice<ColumnOrder>(
        "column_order", {{"wavefront", ColumnOrder::wavefront},
                         {"degree_classes", ColumnOrder::degree_classes},
                         {"natural", ColumnOrder::natural}});
    spec.graph.block_rows = root.count<index_t>("block_rows");
    spec.executor = parse_executor(root.child("executor"));

    spec.min_sum.alpha = parse_alpha(root.child("alpha"));
    if (const json& gamma0 = root.at("gamma0"); !gamma0.is_null()) {
        spec.min_sum.gamma0 = root.number("gamma0");
    }
    spec.relay.pre_iter = root.count<std::uint32_t>("pre_iter");
    spec.relay.set_max_iter = root.count<std::uint32_t>("set_max_iter");
    spec.relay.num_sets = root.count<std::uint32_t>("num_sets");
    spec.relay.stopping = parse_stopping(root.child("stopping"));
    spec.gamma = parse_gamma(root.child("gamma_source"), base);
    spec.window = parse_window(root.child("window"));

    if (spec.relay.num_sets > 0 && spec.gamma.kind == GammaSpec::Kind::none) {
        fail("gamma_source", std::format("num_sets = {} relay legs need a gamma source",
                                         spec.relay.num_sets));
    }
    // Every window shape has its own width. A single table fits a sliding decoder only when its
    // plan has one shape, which is known once the plan is built from the artifact.
    if (spec.gamma.kind == GammaSpec::Kind::explicit_shapes && !spec.window.is_sliding()) {
        fail("gamma_source.type", "explicit_shapes holds one table per window shape and needs "
                                  "window mode sliding; a whole-shot decoder takes "
                                  "{\"type\": \"explicit\", \"path\": ...}");
    }
    if (auto valid = validate(spec.min_sum, spec.relay); !valid) {
        fail(std::string(to_string(valid.error().code)), valid.error().detail);
    }
    if (is_fixed_point(spec.policy)) {
        if (spec.backend != BackendKind::cpu) {
            fail("backend", std::format("the fixed-point format {} runs on the cpu backend only",
                                        to_string(spec.policy)));
        }
        if (auto problem = fixed_alpha_problem(spec.min_sum.alpha)) {
            fail("alpha", *problem);
        }
    }
    if (version == selection_spec_version) {
        auto selection = parse_selection(document.at("selection"), spec.relay,
                                         spec.window.is_sliding(), spec.window.sliding);
        if (!selection) {
            fail(std::move(selection.error().field), std::move(selection.error().problem));
        }
        spec.selection = std::move(*selection);
    }
    return spec;
}

} // namespace

std::string_view to_string(Policy policy) noexcept {
    switch (policy) {
    case Policy::f32:
        return "f32";
    case Policy::f64:
        return "f64";
    case Policy::int4_2_8:
        return Int4_2_8::name;
    case Policy::int5_2_8:
        return Int5_2_8::name;
    case Policy::int6_2_8:
        return Int6_2_8::name;
    }
    return "unknown";
}

std::expected<DecoderSpec, SpecError> parse_spec(const json& document,
                                                 const std::filesystem::path& base) {
    try {
        if (!document.is_object()) {
            fail("", "a decoder spec must be a JSON object");
        }
        return parse(document, base);
    } catch (const SpecFailure& failure) {
        return std::unexpected(failure.error);
    } catch (const json::exception& e) {
        return std::unexpected(SpecError{.field = "", .problem = e.what()});
    }
}

std::expected<DecoderSpec, SpecError> load_spec(const std::filesystem::path& file) {
    std::ifstream in(file);
    if (!in) {
        return std::unexpected(
            SpecError{.field = "", .problem = std::format("cannot open {}", file.string())});
    }
    json document;
    try {
        document = json::parse(in);
    } catch (const json::exception& e) {
        return std::unexpected(SpecError{
            .field = "", .problem = std::format("{} is not valid JSON: {}", file.string(), e.what())});
    }
    return parse_spec(document, file.parent_path());
}

} // namespace rtd::harness
