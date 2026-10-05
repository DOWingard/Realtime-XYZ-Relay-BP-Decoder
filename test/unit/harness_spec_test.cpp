// Decoder spec version 2: the required version and window object, the sliding-window fields and
// their rules, the explicit_shapes gamma source, every rejection, and the specs shipped in
// configs/ and in the test fixtures.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/window/spec.hpp"

namespace {

using namespace rtd;
using namespace rtd::harness;
using json = nlohmann::json;
namespace fs = std::filesystem;

json whole_shot_spec() {
    return json::parse(R"({
        "version": 2,
        "policy": "f32",
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "serial"},
        "alpha": {"rule": "constant", "value": 1.0},
        "gamma0": 0.125,
        "pre_iter": 80,
        "set_max_iter": 60,
        "num_sets": 600,
        "stopping": {"rule": "after_n_converged", "count": 5},
        "gamma_source": {"type": "uniform", "seed": 1, "low": -0.24, "high": 0.66},
        "window": {"mode": "whole_shot"}
    })");
}

json sliding_window() {
    return json::parse(R"({"mode": "sliding", "width": 12, "commit": 8, "converge_rounds": 10,
                           "boundary": "uniform", "on_failure": "defer", "max_deferrals": 2,
                           "iteration_cap": 2000})");
}

json sliding_spec() {
    json doc = whole_shot_spec();
    doc["window"] = sliding_window();
    return doc;
}

// The rejection of `doc`, which must fail.
SpecError rejection(const json& doc) {
    auto spec = parse_spec(doc, "/specs");
    EXPECT_FALSE(spec) << "accepted " << doc.dump();
    return spec ? SpecError{} : spec.error();
}

bool mentions(const std::string& text, const std::string& part) { return text.contains(part); }

TEST(SpecV2, ParsesAWholeShotSpec) {
    auto spec = parse_spec(whole_shot_spec(), "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_EQ(spec->window.mode, WindowSettings::Mode::whole_shot);
    EXPECT_EQ(spec->gamma.kind, GammaSpec::Kind::uniform);
    EXPECT_EQ(spec->raw, whole_shot_spec());
}

TEST(SpecV2, ParsesEverySlidingField) {
    auto spec = parse_spec(sliding_spec(), "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    ASSERT_TRUE(spec->window.is_sliding());
    const window::WindowSpec& w = spec->window.sliding;
    EXPECT_EQ(w.width, 12U);
    EXPECT_EQ(w.commit, 8U);
    EXPECT_EQ(w.converge_rounds, 10U);
    EXPECT_EQ(w.boundary, window::Boundary::uniform);
    EXPECT_EQ(w.on_failure, window::OnFailure::defer);
    EXPECT_EQ(w.max_deferrals, 2U);
    EXPECT_EQ(w.iteration_cap, 2000U);

    json doc = sliding_spec();
    doc["window"]["iteration_cap"] = nullptr;
    doc["window"]["boundary"] = "exact";
    doc["window"]["on_failure"] = "flag";
    doc["window"]["max_deferrals"] = 0;
    spec = parse_spec(doc, "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_FALSE(spec->window.sliding.iteration_cap);
    EXPECT_EQ(spec->window.sliding.boundary, window::Boundary::exact);
    EXPECT_EQ(spec->window.sliding.on_failure, window::OnFailure::flag);

    doc["window"]["on_failure"] = "commit_anyway";
    spec = parse_spec(doc, "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_EQ(spec->window.sliding.on_failure, window::OnFailure::commit_anyway);
}

TEST(SpecV2, ExplicitShapesDirectoryIsResolvedAgainstTheSpec) {
    json doc = sliding_spec();
    doc["gamma_source"] = {{"type", "explicit_shapes"}, {"directory", "gammas"}};
    auto spec = parse_spec(doc, "/specs/run");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_EQ(spec->gamma.kind, GammaSpec::Kind::explicit_shapes);
    EXPECT_EQ(spec->gamma.directory, fs::path("/specs/run/gammas"));

    doc["gamma_source"]["directory"] = "/tables/gammas";
    spec = parse_spec(doc, "/specs/run");
    ASSERT_TRUE(spec);
    EXPECT_EQ(spec->gamma.directory, fs::path("/tables/gammas"));

    doc["gamma_source"].erase("directory");
    EXPECT_EQ(rejection(doc).field, "gamma_source.directory");
}

TEST(SpecV2, AVersionOneSpecIsRejectedWithTheUpgrade) {
    json doc = whole_shot_spec();
    doc.erase("version");
    doc.erase("window");
    const SpecError error = rejection(doc);
    EXPECT_EQ(error.field, "version");
    EXPECT_TRUE(mentions(error.problem, "version-1 spec")) << error.problem;
    EXPECT_TRUE(mentions(error.problem, R"("version": 2)")) << error.problem;
    EXPECT_TRUE(mentions(error.problem, R"("window": {"mode": "whole_shot"})")) << error.problem;

    // The upgrade the message describes is accepted.
    doc["version"] = 2;
    doc["window"] = {{"mode", "whole_shot"}};
    EXPECT_TRUE(parse_spec(doc, "/specs"));

    // A missing version is reported before anything else is missing.
    doc = whole_shot_spec();
    doc.erase("version");
    doc.erase("policy");
    EXPECT_EQ(rejection(doc).field, "version");
}

TEST(SpecV2, OtherVersionsAreRejected) {
    for (const json& version : {json(1), json(3), json(0), json(-2), json("2"), json(2.0),
                                json(2.5), json(nullptr), json::array({2})}) {
        json doc = whole_shot_spec();
        doc["version"] = version;
        const SpecError error = rejection(doc);
        EXPECT_EQ(error.field, "version") << version.dump();
        EXPECT_TRUE(mentions(error.problem, "unsupported spec version")) << error.problem;
    }
}

TEST(SpecV2, TheWindowObjectIsRequired) {
    json doc = whole_shot_spec();
    doc.erase("window");
    EXPECT_EQ(rejection(doc).field, "window");

    for (const json& bad : {json("whole_shot"), json::array(), json(nullptr), json::object()}) {
        doc["window"] = bad;
        const SpecError error = rejection(doc);
        EXPECT_TRUE(error.field == "window" || error.field == "window.mode")
            << bad.dump() << " -> " << error.field;
    }
    doc["window"] = {{"mode", "rolling"}};
    const SpecError error = rejection(doc);
    EXPECT_EQ(error.field, "window.mode");
    EXPECT_TRUE(mentions(error.problem, "whole_shot, sliding")) << error.problem;
}

TEST(SpecV2, EverySlidingFieldIsRequired) {
    for (const auto& item : sliding_window().items()) {
        json doc = sliding_spec();
        doc["window"].erase(item.key());
        const SpecError error = rejection(doc);
        EXPECT_EQ(error.field, "window." + item.key());
        if (item.key() != "mode") {
            EXPECT_EQ(error.problem, "required field is missing");
        }
    }
}

TEST(SpecV2, SlidingFieldsHaveTypesAndRanges) {
    const std::vector<std::pair<std::string, json>> bad{
        {"width", -1},           {"width", 1.5},           {"width", "12"},
        {"width", 1LL << 33},    {"commit", nullptr},      {"converge_rounds", true},
        {"boundary", "center"},  {"boundary", 0},          {"on_failure", "retry"},
        {"max_deferrals", -1},   {"iteration_cap", -5},    {"iteration_cap", 1.5},
        {"iteration_cap", "10"}, {"iteration_cap", false}, {"iteration_cap", 1LL << 32}};
    for (const auto& [key, value] : bad) {
        json doc = sliding_spec();
        doc["window"][key] = value;
        EXPECT_EQ(rejection(doc).field, "window." + key) << key << " = " << value.dump();
    }
}

TEST(SpecV2, WindowRulesNameTheirField) {
    struct Case {
        std::uint32_t width;
        std::uint32_t commit;
        std::uint32_t converge;
        std::string on_failure;
        std::uint32_t deferrals;
        json cap;
        std::string field; // empty: accepted
    };
    // (W, C, C′, policy, max_deferrals, cap, field that must be named; empty = accepted)
    const auto rule = [](std::uint32_t width, std::uint32_t commit, std::uint32_t converge,
                         std::string on_failure, std::uint32_t deferrals, json cap,
                         std::string field) {
        return Case{.width = width,
                    .commit = commit,
                    .converge = converge,
                    .on_failure = std::move(on_failure),
                    .deferrals = deferrals,
                    .cap = std::move(cap),
                    .field = std::move(field)};
    };
    const std::vector<Case> cases{
        rule(12, 12, 12, "flag", 0, nullptr, "window.commit"),
        rule(12, 13, 12, "flag", 0, nullptr, "window.commit"),
        rule(12, 0, 12, "flag", 0, nullptr, "window.commit"),
        rule(0, 0, 0, "flag", 0, nullptr, "window.commit"),
        rule(12, 8, 7, "flag", 0, nullptr, "window.converge_rounds"),
        rule(12, 8, 13, "flag", 0, nullptr, "window.converge_rounds"),
        rule(12, 8, 12, "flag", 1, nullptr, "window.max_deferrals"),
        rule(12, 8, 12, "commit_anyway", 3, nullptr, "window.max_deferrals"),
        rule(12, 8, 12, "flag", 0, 0, "window.iteration_cap"),
        rule(12, 8, 8, "flag", 0, 1, ""),
        rule(12, 8, 12, "defer", 0, nullptr, ""),
        rule(12, 8, 12, "defer", 4, 100000, ""),
        rule(2, 1, 1, "commit_anyway", 0, nullptr, ""),
    };
    for (const Case& c : cases) {
        json doc = sliding_spec();
        doc["window"]["width"] = c.width;
        doc["window"]["commit"] = c.commit;
        doc["window"]["converge_rounds"] = c.converge;
        doc["window"]["on_failure"] = c.on_failure;
        doc["window"]["max_deferrals"] = c.deferrals;
        doc["window"]["iteration_cap"] = c.cap;
        auto spec = parse_spec(doc, "/specs");
        if (c.field.empty()) {
            EXPECT_TRUE(spec) << doc["window"].dump() << ": "
                              << (spec ? "" : spec.error().field + " " + spec.error().problem);
        } else {
            ASSERT_FALSE(spec) << doc["window"].dump();
            EXPECT_EQ(spec.error().field, c.field) << doc["window"].dump();
        }
    }
}

TEST(SpecV2, TheWindowObjectHoldsOnlyItsModesFields) {
    json doc = whole_shot_spec();
    doc["window"]["width"] = 12;
    SpecError error = rejection(doc);
    EXPECT_EQ(error.field, "window.width");
    EXPECT_TRUE(mentions(error.problem, "whole_shot")) << error.problem;

    doc = sliding_spec();
    doc["window"]["commit_rounds"] = 8;
    error = rejection(doc);
    EXPECT_EQ(error.field, "window.commit_rounds");
    EXPECT_TRUE(mentions(error.problem, "sliding")) << error.problem;
}

TEST(SpecV2, TheGammaSourceMustSuitTheWindowMode) {
    json doc = whole_shot_spec();
    doc["gamma_source"] = {{"type", "explicit_shapes"}, {"directory", "gammas"}};
    SpecError error = rejection(doc);
    EXPECT_EQ(error.field, "gamma_source.type");
    EXPECT_TRUE(mentions(error.problem, "sliding")) << error.problem;

    // One whole-problem table parses with sliding windows; whether the plan has the single
    // shape it fits is checked when the plan is built.
    doc = sliding_spec();
    doc["gamma_source"] = {{"type", "explicit"}, {"path", "gammas.npy"}};
    auto single = parse_spec(doc, "/specs");
    ASSERT_TRUE(single) << single.error().field << ": " << single.error().problem;
    EXPECT_EQ(single->gamma.kind, GammaSpec::Kind::explicit_table);
    EXPECT_EQ(single->gamma.table, fs::path("/specs/gammas.npy"));

    // Plain min-sum windows need no gamma source at all.
    doc = sliding_spec();
    doc["gamma0"] = nullptr;
    doc["num_sets"] = 0;
    doc["stopping"] = {{"rule", "after_leg0"}};
    doc["gamma_source"] = {{"type", "none"}};
    auto spec = parse_spec(doc, "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_TRUE(spec->window.is_sliding());

    doc["num_sets"] = 3;
    doc["gamma0"] = 0.125;
    EXPECT_EQ(rejection(doc).field, "gamma_source");
}

// Every spec in configs/ is a whole-shot version-2 spec, and every spec of the windowed goldens is
// a sliding one whose gamma tables sit next to it.
TEST(SpecV2, ShippedSpecsParse) {
    const fs::path repository = test::fixture_root.parent_path().parent_path();
    std::size_t configs = 0;
    for (const auto& entry : fs::directory_iterator(repository / "configs")) {
        if (entry.path().extension() != ".json") {
            continue;
        }
        auto spec = load_spec(entry.path());
        ASSERT_TRUE(spec) << entry.path() << ": " << spec.error().field << ": "
                          << spec.error().problem;
        EXPECT_EQ(spec->raw.at("version"), 2) << entry.path();
        EXPECT_FALSE(spec->window.is_sliding()) << entry.path();
        ++configs;
    }
    EXPECT_GE(configs, 4U);

    std::size_t fixtures = 0;
    for (const auto& entry : fs::recursive_directory_iterator(test::fixture_root)) {
        if (entry.path().filename() != "spec.json") {
            continue;
        }
        auto spec = load_spec(entry.path());
        ASSERT_TRUE(spec) << entry.path() << ": " << spec.error().field << ": "
                          << spec.error().problem;
        EXPECT_TRUE(spec->window.is_sliding()) << entry.path();
        EXPECT_EQ(spec->gamma.kind, GammaSpec::Kind::explicit_shapes) << entry.path();
        EXPECT_EQ(spec->gamma.directory, entry.path().parent_path() / "gammas");
        EXPECT_TRUE(fs::is_directory(spec->gamma.directory)) << spec->gamma.directory;
        ++fixtures;
    }
    EXPECT_GE(fixtures, 1U);
}

} // namespace
