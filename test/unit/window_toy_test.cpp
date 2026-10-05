// The hand-derived toy (test/fixtures/toy_rep3: a 3-bit repetition code, 6 noisy rounds and a
// noiseless readout, derivation in DERIVATION.md): with an exact minimum-weight inner decoder the
// stream decoder must reproduce every carry, window solution, commit and frame worked out on
// paper, in batch and in streaming use.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rtd/io/artifact.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"
#include "window_inner_support.hpp"
#include "window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
namespace fs = std::filesystem;
using Json = nlohmann::json;

const fs::path toy_dir = fs::path(RTD_FIXTURE_DIR) / "toy_rep3";

struct ExpectedWindow {
    std::uint32_t window = 0;
    std::uint32_t first_round = 0;
    std::vector<Bit> residual_first_round; // before the window's decode: σ of the round ⊕ carry
    std::vector<Bit> carry;
    std::vector<index_t> solution; // representatives of the unique minimum-weight solution
    std::vector<index_t> committed;
    std::uint64_t frame_after = 0;
};

struct ExpectedCase {
    std::string name;
    std::uint32_t width = 0;
    std::uint32_t commit = 0;
    std::vector<index_t> injected;
    std::vector<Bit> syndrome;
    std::uint64_t true_observables = 0;
    std::vector<ExpectedWindow> windows;
    std::uint64_t final_frame = 0;
};

std::vector<ExpectedCase> load_cases() {
    std::ifstream in(toy_dir / "expected.json");
    const Json doc = Json::parse(in);
    std::vector<ExpectedCase> cases;
    for (const Json& c : doc.at("cases")) {
        ExpectedCase e;
        e.name = c.at("name").get<std::string>();
        e.width = c.at("width").get<std::uint32_t>();
        e.commit = c.at("commit").get<std::uint32_t>();
        e.injected = c.at("injected_faults").get<std::vector<index_t>>();
        e.syndrome = c.at("syndrome").get<std::vector<Bit>>();
        e.true_observables = c.at("true_observables").get<std::uint64_t>();
        e.final_frame = c.at("final_frame").get<std::uint64_t>();
        for (const Json& w : c.at("windows")) {
            e.windows.push_back(
                {.window = w.at("window").get<std::uint32_t>(),
                 .first_round = w.at("first_round").get<std::uint32_t>(),
                 .residual_first_round =
                     w.at("residual_first_round_before_decode").get<std::vector<Bit>>(),
                 .carry = w.at("carry").get<std::vector<Bit>>(),
                 .solution = w.at("solution").get<std::vector<index_t>>(),
                 .committed = w.at("committed").get<std::vector<index_t>>(),
                 .frame_after = w.at("frame_after").get<std::uint64_t>()});
        }
        cases.push_back(std::move(e));
    }
    return cases;
}

struct ToyProblem {
    std::unique_ptr<io::Artifact> artifact;
    std::unique_ptr<ArtifactProblem> source;
};

// What one run of the stream decoder showed for each window.
struct Seen {
    std::vector<std::vector<index_t>> committed;
    std::vector<std::uint64_t> frame_after;
    std::uint64_t final_frame = 0;
};

void check_case(const ExpectedCase& e, const ToyProblem& toy, bool streaming) {
    SCOPED_TRACE(std::format("{} ({})", e.name, streaming ? "streaming" : "batch"));
    const Problem problem = toy.source->problem();
    const WindowSpec spec{.width = e.width,
                          .commit = e.commit,
                          .converge_rounds = e.width,
                          .boundary = Boundary::exact,
                          .on_failure = OnFailure::commit_anyway,
                          .max_deferrals = 0,
                          .iteration_cap = std::nullopt};
    auto plan = WindowPlan::build(problem, spec);
    ASSERT_TRUE(plan) << describe(plan.error());
    ASSERT_EQ(plan->num_positions(), e.windows.size());
    const index_t m_round = plan->detectors_per_round();

    // The fixture's syndrome is the one its injected faults produce, and so is its observable.
    const auto supports = test::column_supports(problem);
    const auto classes = test::column_classes(problem);
    std::vector<Bit> sigma(problem.num_rows, 0);
    std::uint64_t truth = 0;
    for (const index_t j : e.injected) {
        truth ^= classes[j];
        for (const index_t i : supports[j]) {
            sigma[i] ^= Bit{1};
        }
    }
    ASSERT_EQ(sigma, e.syndrome);
    ASSERT_EQ(truth, e.true_observables);

    std::vector<test::InnerCall> calls;
    auto decoder = StreamDecoder<test::Spy<test::BruteForceInner>>::create(
        *plan, [&calls](const Shape& shape) {
            return test::Spy<test::BruteForceInner>(test::BruteForceInner(shape), shape.index(),
                                                    &calls);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());

    Seen seen;
    const auto keep = [&](const Commit& commit) {
        seen.committed.emplace_back(commit.faults.begin(), commit.faults.end());
        seen.frame_after.push_back(decoder->frame_mask());
        EXPECT_FALSE(commit.deferred);
        EXPECT_TRUE(commit.record.converged);
        EXPECT_EQ(commit.record.attempts, 1U);
    };
    if (streaming) {
        decoder->reset(0);
        for (std::uint32_t r = 0; r < plan->rounds_total(); ++r) {
            const std::span<const Bit> bits(sigma.data() + std::size_t{r} * m_round, m_round);
            ASSERT_TRUE(r + 1 < plan->rounds_total() ? decoder->push_round(bits)
                                                     : decoder->push_final(bits));
            while (decoder->window_ready()) {
                auto commit = decoder->decode_next();
                ASSERT_TRUE(commit) << describe(commit.error());
                keep(*commit);
            }
        }
        ASSERT_TRUE(decoder->finished());
        seen.final_frame = decoder->frame_mask();
    } else {
        auto summary = decode_shot(*decoder, sigma, 0, keep);
        ASSERT_TRUE(summary) << describe(summary.error());
        EXPECT_TRUE(summary->success);
        seen.final_frame = summary->frame;
    }

    ASSERT_EQ(calls.size(), e.windows.size()) << "one decode per window";
    for (std::size_t k = 0; k < e.windows.size(); ++k) {
        const ExpectedWindow& w = e.windows[k];
        SCOPED_TRACE(std::format("window {}", k));
        const Placement* p = plan->placement(w.window, 0);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(p->first_round, w.first_round);
        const test::InnerCall& call = calls[k];
        EXPECT_EQ(test::split_stream(call.stream).window, w.window);
        const std::vector<Bit> first(call.syndrome.begin(), call.syndrome.begin() + m_round);
        EXPECT_EQ(first, w.residual_first_round);
        std::vector<Bit> carry(m_round);
        for (index_t i = 0; i < m_round; ++i) {
            carry[i] = static_cast<Bit>(first[i] ^ sigma[p->first_row + i]);
        }
        EXPECT_EQ(carry, w.carry);
        EXPECT_TRUE(call.success);
        EXPECT_EQ(call.optimal, 1U) << "the hand derivation assumes a unique minimum";
        std::vector<index_t> solution;
        solution.reserve(call.support.size());
        for (const index_t l : call.support) {
            solution.push_back(p->columns[l]);
        }
        EXPECT_EQ(solution, w.solution);
        EXPECT_EQ(seen.committed[k], w.committed);
        EXPECT_EQ(seen.frame_after[k], w.frame_after);
    }
    EXPECT_EQ(seen.final_frame, e.final_frame);
}

TEST(WindowToy, HandDerivedCarriesCommitsAndFrames) {
    ToyProblem toy;
    auto artifact = io::load_artifact(toy_dir);
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    toy.artifact = std::make_unique<io::Artifact>(std::move(*artifact));
    toy.source = std::make_unique<ArtifactProblem>(*toy.artifact);
    const std::vector<ExpectedCase> cases = load_cases();
    ASSERT_EQ(cases.size(), 3U);
    std::set<std::pair<std::uint32_t, std::uint32_t>> layouts;
    for (const ExpectedCase& e : cases) {
        layouts.insert({e.width, e.commit});
        for (const bool streaming : {false, true}) {
            check_case(e, toy, streaming);
        }
    }
    EXPECT_TRUE(layouts.contains({3, 1}));
    EXPECT_TRUE(layouts.contains({4, 2}));
}

} // namespace
