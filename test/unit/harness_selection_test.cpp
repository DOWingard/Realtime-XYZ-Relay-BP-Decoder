// Decoder spec version 3 and the selection policy in rtd_decode: the "selection" object and every
// rejection, the version rules, the confidence and history arrays of whole-shot and sliding runs,
// the default policy leaving every decode bit-identical, and the window actions on low confidence.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"
#include "rtd/core/confidence.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/selection_spec.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/io/npy.hpp"

namespace {

using namespace rtd;
using namespace rtd::harness;
using json = nlohmann::json;
namespace fs = std::filesystem;
using test::CliRun;
using test::first_error;
using test::read_all;
using test::read_array;
using test::read_json;
using test::run_cli;

const fs::path r9 = test::fixture_root / "bb18_choi_r9";

json sliding_window(const std::string& policy, std::uint32_t deferrals) {
    return {{"mode", "sliding"},
            {"width", 4},
            {"commit", 2},
            {"converge_rounds", 4},
            {"boundary", "exact"},
            {"on_failure", policy},
            {"max_deferrals", deferrals},
            {"iteration_cap", nullptr}};
}

json default_selection() {
    return json::parse(R"({"rule": "lowest_weight", "stop": {"rule": "fixed"}, "capacity": 5,
                           "confidence": null, "history": null})");
}

// A small Relay-BP spec (version 3 unless the selection is removed).
json spec_with(const json& window, const json& selection) {
    json doc = {
        {"version", 3},
        {"policy", "f32"},
        {"backend", "cpu"},
        {"layout", "row_major"},
        {"column_order", "wavefront"},
        {"block_rows", 64},
        {"executor", {{"type", "serial"}}},
        {"alpha", {{"rule", "constant"}, {"value", 1.0}}},
        {"gamma0", 0.125},
        {"pre_iter", 20},
        {"set_max_iter", 15},
        {"num_sets", 40},
        {"stopping", {{"rule", "after_n_converged"}, {"count", 3}}},
        {"gamma_source", {{"type", "uniform"}, {"seed", 7}, {"low", -0.24}, {"high", 0.66}}},
        {"window", window},
        {"selection", selection}};
    return doc;
}

json whole() { return {{"mode", "whole_shot"}}; }

SpecError rejection(const json& doc) {
    auto spec = parse_spec(doc, "/specs");
    EXPECT_FALSE(spec) << "accepted " << doc.dump();
    return spec ? SpecError{} : spec.error();
}

// ---- Spec -------------------------------------------------------------------------------------

TEST(SpecV3, ParsesEverySelectionField) {
    json sel = json::parse(R"({"rule": "class_sum",
                               "stop": {"rule": "gap_extend", "count": 2, "threshold": 1.5},
                               "capacity": 7,
                               "confidence": {"signal": "gap", "threshold": 2.5,
                                              "single_class": "low", "extra_legs": 4,
                                              "on_low": "defer"},
                               "history": {"lengths": [1, 3], "signals": ["gap", "commit_weight"]}})");
    auto spec = parse_spec(spec_with(sliding_window("defer", 1), sel), "/specs");
    ASSERT_TRUE(spec) << spec.error().field << ": " << spec.error().problem;
    EXPECT_EQ(spec->version, 3U);
    ASSERT_TRUE(spec->selection);
    const SelectionSpec& s = *spec->selection;
    EXPECT_EQ(s.config.rule, SelectionRule::class_sum);
    EXPECT_EQ(s.config.stop, StopRule::gap_extend);
    EXPECT_EQ(s.config.stop_count, 2U);
    EXPECT_EQ(s.config.stop_gap, 1.5);
    EXPECT_EQ(s.config.capacity, 7U);
    EXPECT_EQ(s.config.signal, ConfidenceSignal::gap);
    EXPECT_EQ(s.config.threshold, 2.5);
    EXPECT_TRUE(s.config.single_class_is_low);
    EXPECT_EQ(s.config.extra_legs, 4U);
    EXPECT_EQ(s.on_low, window::OnLowConfidence::defer);
    EXPECT_EQ(s.history_lengths, (std::vector<std::uint32_t>{1, 3}));
    EXPECT_EQ(s.history_signals,
              (std::vector<window::HistorySignal>{window::HistorySignal::gap,
                                                  window::HistorySignal::commit_weight}));
    // run.json's record parses back to the same policy.
    auto again = parse_selection(selection_json(s), spec->relay, true, spec->window.sliding);
    ASSERT_TRUE(again && *again);
    EXPECT_EQ(selection_json(**again), selection_json(s));
    EXPECT_EQ(selection_json(s), sel);
}

TEST(SpecV3, NullSelectionAndVersionRules) {
    auto spec = parse_spec(spec_with(whole(), nullptr), "/specs");
    ASSERT_TRUE(spec) << spec.error().problem;
    EXPECT_FALSE(spec->selection);
    EXPECT_EQ(spec->version, 3U);

    // Version 3 needs the object; version 2 must not have it.
    json doc = spec_with(whole(), nullptr);
    doc.erase("selection");
    SpecError error = rejection(doc);
    EXPECT_EQ(error.field, "version");
    EXPECT_TRUE(error.problem.contains("selection")) << error.problem;
    doc = spec_with(whole(), default_selection());
    doc["version"] = 2;
    EXPECT_EQ(rejection(doc).field, "selection");
    doc["version"] = 4;
    error = rejection(doc);
    EXPECT_EQ(error.field, "version");
    EXPECT_TRUE(error.problem.contains("versions 2 and 3")) << error.problem;
}

TEST(SpecV3, EverySelectionFieldIsRequiredAndChecked) {
    for (const auto& item : default_selection().items()) {
        json sel = default_selection();
        sel.erase(item.key());
        EXPECT_EQ(rejection(spec_with(whole(), sel)).field, "selection." + item.key());
    }
    struct Case {
        json window;
        json patch;
        std::string field;
    };
    const auto make_case = [](json window, json patch, std::string field) {
        return Case{
            .window = std::move(window), .patch = std::move(patch), .field = std::move(field)};
    };
    const json conf = json::parse(R"({"signal": "weight", "threshold": 10.0,
                                      "single_class": "high", "extra_legs": 0, "on_low": "none"})");
    const std::vector<Case> cases = {
        make_case(whole(), {{"rule", "median"}}, "selection.rule"),
        make_case(whole(), {{"extra", 1}}, "selection.extra"),
        make_case(whole(), {{"capacity", 0}}, "selection.capacity"),
        make_case(whole(), {{"capacity", 21}}, "selection.capacity"),
        make_case(whole(), {{"stop", {{"rule", "agree"}}}}, "selection.stop.count"),
        make_case(whole(), {{"stop", {{"rule", "agree"}, {"count", 0}}}}, "selection.stop"),
        make_case(whole(), {{"stop", {{"rule", "fixed"}, {"count", 2}}}}, "selection.stop.count"),
        make_case(whole(), {{"stop", {{"rule", "gap"}, {"threshold", "x"}}}},
                  "selection.stop.threshold"),
        // The rules must see the relay rule's 3 solutions.
        make_case(whole(), {{"capacity", 2}, {"stop", {{"rule", "agree"}, {"count", 2}}}},
                  "selection.capacity"),
        make_case(whole(),
                  {{"confidence",
                    [&] {
                        json c = conf;
                        c["signal"] = "none";
                        return c;
                    }()}},
                  "selection.confidence.signal"),
        make_case(whole(),
                  {{"confidence",
                    [&] {
                        json c = conf;
                        c["single_class"] = "maybe";
                        return c;
                    }()}},
                  "selection.confidence.single_class"),
        make_case(whole(),
                  {{"confidence",
                    [&] {
                        json c = conf;
                        c["on_low"] = "flag";
                        return c;
                    }()}},
                  "selection.confidence.on_low"),
        make_case(sliding_window("commit_anyway", 0),
                  {{"confidence",
                    [&] {
                        json c = conf;
                        c["on_low"] = "defer";
                        return c;
                    }()}},
                  "selection.confidence.on_low"),
        make_case(whole(), {{"history", {{"lengths", {1}}, {"signals", {"gap"}}}}},
                  "selection.history.lengths"),
        make_case(sliding_window("flag", 0),
                  {{"history", {{"lengths", {1, 1}}, {"signals", {"gap"}}}}},
                  "selection.history.lengths"),
        make_case(sliding_window("flag", 0),
                  {{"history", {{"lengths", {65}}, {"signals", {"gap"}}}}},
                  "selection.history.lengths"),
        make_case(sliding_window("flag", 0),
                  {{"history", {{"lengths", {2}}, {"signals", {"entropy"}}}}},
                  "selection.history.signals[0]"),
        make_case(sliding_window("flag", 0),
                  {{"history", {{"lengths", json::array()}, {"signals", {"gap"}}}}},
                  "selection.history.lengths"),
    };
    for (const Case& c : cases) {
        json sel = default_selection();
        sel.merge_patch(c.patch);
        const SpecError error = rejection(spec_with(c.window, sel));
        EXPECT_EQ(error.field, c.field) << c.patch.dump() << ": " << error.problem;
    }
    // Extra legs need a signal, and a stopping rule the relay's count.
    json sel = default_selection();
    sel["confidence"] = conf;
    sel["confidence"]["extra_legs"] = 2;
    json doc = spec_with(whole(), sel);
    doc["stopping"] = {{"rule", "all_legs"}};
    EXPECT_EQ(rejection(doc).field, "selection.stop");
    doc["stopping"] = {{"rule", "after_n_converged"}, {"count", 3}};
    EXPECT_TRUE(parse_spec(doc, "/specs"));
    // Each extra leg may converge: the relay's 3 solutions plus 3 extra legs could reach 6, more
    // than the 5 the rules see.
    doc["selection"]["confidence"]["extra_legs"] = 3;
    const SpecError over = rejection(doc);
    EXPECT_EQ(over.field, "selection.confidence.extra_legs");
    EXPECT_TRUE(over.problem.contains("at most 2 extra legs")) << over.problem;
    doc["selection"]["capacity"] = 6;
    EXPECT_TRUE(parse_spec(doc, "/specs"));
    // A window action on low confidence also needs the rules to see every solution.
    json flagged = default_selection();
    flagged["confidence"] = conf;
    flagged["confidence"]["on_low"] = "flag";
    json windowed = spec_with(sliding_window("flag", 0), flagged);
    EXPECT_TRUE(parse_spec(windowed, "/specs"));
    windowed["stopping"] = {{"rule", "all_legs"}};
    EXPECT_EQ(rejection(windowed).field, "selection.stop");
    windowed["stopping"] = {{"rule", "after_n_converged"}, {"count", 6}};
    EXPECT_EQ(rejection(windowed).field, "selection.capacity");
}

// ---- Runs -------------------------------------------------------------------------------------

std::vector<std::string> decode_args(const fs::path& config, const fs::path& out,
                                     std::size_t shots) {
    return {"--artifact", (r9 / "artifact").string(),
            "--shots",    (r9 / "shots").string(),
            "--config",   config.string(),
            "--out",      out.string(),
            "--first",    "10",
            "--count",    std::to_string(shots),
            "--workers",  "2"};
}

fs::path write_spec(const fs::path& dir, const std::string& name, const json& spec) {
    const fs::path file = dir / name;
    std::ofstream(file) << spec.dump(2);
    return file;
}

// Every array the two runs share has the same bytes.
void expect_same_arrays(const fs::path& a, const fs::path& b) {
    std::size_t compared = 0;
    for (const auto& entry : fs::directory_iterator(b)) {
        if (entry.path().extension() != ".npy" || entry.path().filename() == "decode_ns.npy" ||
            entry.path().filename() == "win_decode_ns.npy") {
            continue;
        }
        std::ifstream x(a / entry.path().filename(), std::ios::binary);
        std::ifstream y(entry.path(), std::ios::binary);
        const std::string bx((std::istreambuf_iterator<char>(x)), {});
        const std::string by((std::istreambuf_iterator<char>(y)), {});
        EXPECT_EQ(bx, by) << entry.path().filename();
        ++compared;
    }
    EXPECT_GT(compared, 5U);
}

const std::vector<std::string> conf_names = {
    "conf_found",           "conf_seen",          "conf_distinct",        "conf_classes",
    "conf_best_class",      "conf_second_class",  "conf_weight",          "conf_gap",
    "conf_agreement",       "conf_gap_state",     "conf_first_legs",      "conf_first_iterations",
    "conf_class_sum_class", "conf_class_sum_top", "conf_agreement_class", "conf_q_supp",
    "conf_q_sum_sq",        "conf_q_total",       "conf_components",      "conf_syndrome_ones",
    "conf_syndrome_rows",   "conf_decided_class", "conf_decided_leg",     "conf_extra_legs",
    "conf_score",           "conf_low",           "conf_stopped_early"};

TEST(SelectionRuns, TheDefaultPolicyChangesNoDecodeAndRecordsItsConfidence) {
    const fs::path root = test::scratch_dir("rtd_selection_whole");
    const std::size_t shots = 12;
    json v2 = spec_with(whole(), nullptr);
    v2["version"] = 2;
    v2.erase("selection");
    const fs::path plain = root / "plain";
    std::vector<std::string> args = decode_args(write_spec(root, "v2.json", v2), plain, shots);
    args.insert(args.end(), {"--record-solutions", "5", "--save-solution-supports"});
    CliRun run = run_cli(args);
    ASSERT_EQ(run.code, 0) << first_error(run).dump();

    const fs::path out = root / "sel";
    args = decode_args(write_spec(root, "v3.json", spec_with(whole(), default_selection())), out,
                       shots);
    args.insert(args.end(), {"--record-solutions", "5", "--save-solution-supports"});
    run = run_cli(args);
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    expect_same_arrays(out, plain);

    for (const std::string& name : conf_names) {
        EXPECT_TRUE(fs::exists(out / (name + ".npy"))) << name;
    }
    EXPECT_FALSE(fs::exists(out / "conf_low_deferrals.npy"));
    const auto found = read_array<std::uint32_t>(out / "conf_found.npy", {shots, 1});
    const auto count = read_all<std::uint32_t>(out / "sol_count.npy");
    EXPECT_EQ(found, count);
    const auto best = read_array<std::uint64_t>(out / "conf_best_class.npy", {shots, 1});
    const auto weight = read_array<double>(out / "conf_weight.npy", {shots, 1});
    const auto sol_w = read_all<double>(out / "sol_weight.npy");
    const auto sol_c = read_all<std::uint64_t>(out / "sol_class.npy");
    const auto state = read_array<std::uint8_t>(out / "conf_gap_state.npy", {shots, 1});
    const auto returned = read_all<std::uint64_t>(out / "returned_class.npy");
    const auto shot_weight = read_all<double>(out / "weight.npy");
    for (std::size_t s = 0; s < shots; ++s) {
        if (count[s] == 0) {
            EXPECT_EQ(state[s], 0U);
            EXPECT_EQ(best[s], returned[s]);
            continue;
        }
        const auto first = sol_w.begin() + static_cast<std::ptrdiff_t>(s * 5);
        const auto lightest = std::min_element(first, first + std::min<std::uint32_t>(count[s], 5));
        EXPECT_EQ(weight[s], *lightest);
        EXPECT_EQ(weight[s], shot_weight[s]);
        EXPECT_EQ(best[s], sol_c[static_cast<std::size_t>(lightest - sol_w.begin())]);
        EXPECT_EQ(best[s], returned[s]);
    }
    const json summary = read_json(out / "run.json").at("summary");
    ASSERT_TRUE(summary.contains("confidence"));
    EXPECT_EQ(summary["confidence"]["decodes"], shots);
    EXPECT_EQ(summary["confidence"]["decision_differs_from_lowest_weight"], 0);
    EXPECT_EQ(read_json(out / "run.json").at("decoder").at("version"), 3);
    fs::remove_all(root);
}

TEST(SelectionRuns, SlidingRunsRecordHistoryAndActOnLowConfidence) {
    const fs::path root = test::scratch_dir("rtd_selection_sliding");
    const std::size_t shots = 8;
    json sel = default_selection();
    sel["history"] = {{"lengths", {1, 2, 5}},
                      {"signals", {"gap", "weight", "density", "commit_weight", "commit_q_supp"}}};
    // Every decode is low when its weight exceeds -1.
    sel["confidence"] = {{"signal", "weight"},
                         {"threshold", -1.0},
                         {"single_class", "high"},
                         {"extra_legs", 0},
                         {"on_low", "none"}};
    const fs::path base = root / "base";
    CliRun run = run_cli(decode_args(
        write_spec(root, "base.json", spec_with(sliding_window("commit_anyway", 0), sel)), base,
        shots));
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    const std::size_t k =
        read_json(base / "run.json").at("plan").at("positions").get<std::size_t>();
    for (const char* name :
         {"hist_gap", "hist_weight", "hist_density", "hist_commit_weight", "hist_commit_q_supp"}) {
        read_array<double>(base / (std::string(name) + ".npy"), {shots, k, 3});
    }
    const auto hist_state = read_array<std::uint8_t>(base / "hist_state.npy", {shots, k, 3});
    const auto low = read_array<std::uint8_t>(base / "conf_low.npy", {shots, k});
    const auto deferrals = read_array<std::uint32_t>(base / "conf_low_deferrals.npy", {shots, k});
    const auto attempts = read_all<std::uint8_t>(base / "win_attempts.npy");
    const auto w = read_all<double>(base / "conf_weight.npy");
    const auto hw = read_all<double>(base / "hist_weight.npy");
    for (std::size_t c = 0; c < shots * k; ++c) {
        if (attempts[c] > 0) {
            EXPECT_EQ(low[c], 1U);
        }
        EXPECT_EQ(deferrals[c], 0U);
        // L = 1 is the window's own weight.
        if (attempts[c] > 0 && hist_state[c * 3] != 0) {
            EXPECT_EQ(hw[c * 3], w[c]);
        }
    }
    const auto flagged_base = read_all<std::uint8_t>(base / "win_flagged.npy");

    // flag: the same decodes, every decoded window flagged (one that did not converge too: it has
    // no solution, the lowest confidence).
    json flag_sel = sel;
    flag_sel["confidence"]["on_low"] = "flag";
    const fs::path flagged = root / "flag";
    run = run_cli(decode_args(
        write_spec(root, "flag.json", spec_with(sliding_window("commit_anyway", 0), flag_sel)),
        flagged, shots));
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    EXPECT_EQ(read_all<std::uint8_t>(flagged / "predicted_observables.npy"),
              read_all<std::uint8_t>(base / "predicted_observables.npy"));
    const auto win_flagged = read_all<std::uint8_t>(flagged / "win_flagged.npy");
    for (std::size_t c = 0; c < shots * k; ++c) {
        EXPECT_EQ(win_flagged[c], attempts[c] > 0 ? 1U : flagged_base[c]) << c;
    }

    // defer: every non-final window is deferred once, then committed and flagged.
    json defer_sel = sel;
    defer_sel["confidence"]["on_low"] = "defer";
    const fs::path deferred = root / "defer";
    run = run_cli(decode_args(
        write_spec(root, "defer.json", spec_with(sliding_window("defer", 1), defer_sel)), deferred,
        shots));
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    const auto low_deferrals = read_all<std::uint32_t>(deferred / "conf_low_deferrals.npy");
    EXPECT_GT(*std::ranges::max_element(low_deferrals), 0U);
    EXPECT_GT(read_json(deferred / "run.json")["summary"]["confidence"]["low_confidence_deferrals"]
                  .get<std::uint64_t>(),
              0U);
    fs::remove_all(root);
}

TEST(SelectionRuns, ClassRulesAndEarlyStopsRun) {
    const fs::path root = test::scratch_dir("rtd_selection_rules");
    const std::size_t shots = 10;
    json sel = default_selection();
    sel["rule"] = "largest_agreement";
    sel["stop"] = {{"rule", "agree"}, {"count", 2}};
    sel["confidence"] = {{"signal", "q_supp"},
                         {"threshold", 0.0},
                         {"single_class", "low"},
                         {"extra_legs", 2}, // the relay's 3 solutions + 2 = the capacity of 5
                         {"on_low", "none"}};
    const fs::path out = root / "out";
    std::vector<std::string> args =
        decode_args(write_spec(root, "rules.json", spec_with(whole(), sel)), out, shots);
    args.insert(args.end(), {"--record-solutions", "5"});
    const CliRun run = run_cli(args);
    ASSERT_EQ(run.code, 0) << first_error(run).dump();
    const auto early = read_array<std::uint8_t>(out / "conf_stopped_early.npy", {shots, 1});
    const auto decided = read_all<std::uint64_t>(out / "conf_decided_class.npy");
    const auto agreement = read_all<std::uint64_t>(out / "conf_agreement_class.npy");
    const auto returned = read_all<std::uint64_t>(out / "returned_class.npy");
    const auto q = read_all<double>(out / "conf_q_supp.npy");
    const auto state = read_all<std::uint8_t>(out / "conf_gap_state.npy");
    for (std::size_t s = 0; s < shots; ++s) {
        EXPECT_EQ(decided[s], agreement[s]);
        if (state[s] != 0) {
            EXPECT_EQ(decided[s], returned[s]) << "the returned correction is the rule's choice";
            EXPECT_GT(q[s], 0.0);
        }
    }
    EXPECT_GE(std::count(early.begin(), early.end(), 1), 0);
    fs::remove_all(root);
}

// ---- Output helpers ---------------------------------------------------------------------------

TEST(ConfidenceOutputs, EmptyCellsAndRecordedValues) {
    SelectionSpec spec;
    spec.history_lengths = {1, 2};
    spec.history_signals = {window::HistorySignal::weight};
    ConfidenceOutputs out;
    size_confidence_outputs(out, 6, spec, true);
    ASSERT_TRUE(out.enabled);
    EXPECT_EQ(out.weight.size(), 6U);
    EXPECT_TRUE(std::isinf(out.weight[0]));
    EXPECT_TRUE(std::isnan(out.gap[0]));
    EXPECT_EQ(out.history.size(), 6U * 1U * 2U);
    EXPECT_EQ(out.history_state.size(), 12U);

    Confidence c;
    c.found = 3;
    c.seen = 3;
    c.distinct = 2;
    c.weight = 4.5;
    c.gap_state = GapState::defined;
    c.gap = 1.25;
    c.low = true;
    record_confidence(c, 4, out);
    EXPECT_EQ(out.found[4], 3U);
    EXPECT_EQ(out.weight[4], 4.5);
    EXPECT_EQ(out.gap[4], 1.25);
    EXPECT_EQ(out.gap_state[4], 2U);
    EXPECT_EQ(out.low[4], 1U);

    const fs::path dir = test::scratch_dir("rtd_confidence_outputs");
    ASSERT_TRUE(write_confidence(dir, out, 3, 2));
    EXPECT_EQ(read_array<double>(dir / "conf_weight.npy", {3, 2})[4], 4.5);
    read_array<double>(dir / "hist_weight.npy", {3, 2, 2});
    read_array<std::uint8_t>(dir / "hist_state.npy", {3, 2, 2});
    read_array<std::uint32_t>(dir / "conf_low_deferrals.npy", {3, 2});
    const json summary = confidence_summary(out);
    EXPECT_EQ(summary["low_confidence"], 1);
    fs::remove_all(dir);
}

} // namespace
