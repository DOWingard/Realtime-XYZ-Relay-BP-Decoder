// The stream decoder: construction and round bookkeeping, window readiness while streaming, the
// γ streams and budgets handed to the inner decoders, the three non-convergence policies with a
// scripted inner decoder, and properties on random time-structured problems with the relay
// decoder inside. Those are checked against a literal transcription of the stream-decoding rule,
// against the global problem (H·c ⊕ residual = σ, A·c = frame), and between batch and streaming
// use.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/shots.hpp"
#include "rtd/window/artifact_problem.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"
#include "unit/allocation_counter.hpp"
#include "window_inner_support.hpp"
#include "window_support.hpp"

namespace {

using namespace rtd;
using namespace rtd::window;
namespace fs = std::filesystem;
using Code = StreamError::Code;
using Fails = std::set<std::pair<std::uint32_t, std::uint32_t>>;

static_assert(SyndromeDecoder<CpuRelayDecoder<F32>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F64>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F32, Team>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F64, Team, RecordingSink>>);
static_assert(SyndromeDecoder<CpuRelayDecoder<F32, Serial, RecordingSink>>);

const fs::path fixture_root = RTD_FIXTURE_DIR;
constexpr double infinity = std::numeric_limits<double>::infinity();

WindowSpec spec(std::uint32_t width, std::uint32_t commit, std::uint32_t converge,
                Boundary boundary = Boundary::exact,
                OnFailure on_failure = OnFailure::commit_anyway, std::uint32_t deferrals = 0,
                std::optional<std::uint32_t> cap = std::nullopt) {
    return {.width = width,
            .commit = commit,
            .converge_rounds = converge,
            .boundary = boundary,
            .on_failure = on_failure,
            .max_deferrals = deferrals,
            .iteration_cap = cap};
}

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// The toy repetition-code artifact (Rt = 7 rounds of M = 2 detectors, 30 faults), held where the
// problem views can point at it.
struct Toy {
    std::unique_ptr<io::Artifact> artifact;
    std::unique_ptr<ArtifactProblem> source;
    [[nodiscard]] Problem problem() const { return source->problem(); }
};

Toy load_toy() {
    auto artifact = io::load_artifact(fixture_root / "toy_rep3");
    EXPECT_TRUE(artifact) << (artifact ? "" : io::describe(artifact.error()));
    Toy toy;
    if (artifact) {
        toy.artifact = std::make_unique<io::Artifact>(std::move(*artifact));
        toy.source = std::make_unique<ArtifactProblem>(*toy.artifact);
    }
    return toy;
}

// σ = H·e for the faults `columns` of a problem.
std::vector<Bit> syndrome_of(const Problem& problem, std::span<const index_t> columns) {
    const auto supports = test::column_supports(problem);
    std::vector<Bit> s(problem.num_rows, 0);
    for (const index_t j : columns) {
        for (const index_t i : supports[j]) {
            s[i] ^= Bit{1};
        }
    }
    return s;
}

std::expected<StreamDecoder<test::ScriptedInner>, StreamError>
scripted_stream(const WindowPlan& plan, const Fails& fails) {
    return StreamDecoder<test::ScriptedInner>::create(
        plan, [&fails](const Shape& shape) { return test::ScriptedInner(shape, &fails); });
}

// A shot's commits as decode_shot or a streaming loop hands them out, with the frame after each.
struct Collected {
    std::vector<WindowRecord> records;
    std::vector<std::vector<index_t>> faults;
    std::vector<std::vector<Bit>> frame_delta;
    std::vector<std::uint64_t> frame_after;
    std::vector<Bit> residual;
    ShotSummary summary;
    std::uint32_t deferred_returns = 0;
};

template <class Inner>
void collect(const StreamDecoder<Inner>& decoder, const Commit& commit, Collected& out) {
    EXPECT_EQ(commit.window, out.records.size());
    EXPECT_EQ(commit.record.window, commit.window);
    out.records.push_back(commit.record);
    out.faults.emplace_back(commit.faults.begin(), commit.faults.end());
    out.frame_delta.emplace_back(commit.frame_delta.begin(), commit.frame_delta.end());
    out.frame_after.push_back(decoder.frame_mask());
}

template <class Inner>
std::optional<Collected> batch(StreamDecoder<Inner>& decoder, std::span<const Bit> syndrome,
                               std::uint64_t shot) {
    Collected out;
    auto summary = decode_shot(decoder, syndrome, shot,
                               [&](const Commit& commit) { collect(decoder, commit, out); });
    EXPECT_TRUE(summary) << (summary ? "" : describe(summary.error()));
    if (!summary) {
        return std::nullopt;
    }
    out.summary = *summary;
    out.residual.assign(decoder.residual().begin(), decoder.residual().end());
    return out;
}

// Pushes one round at a time and decodes whenever a window is ready, as a real-time driver does.
template <class Inner>
std::optional<Collected> streamed(StreamDecoder<Inner>& decoder, std::span<const Bit> syndrome,
                                  std::uint64_t shot) {
    const WindowPlan& plan = decoder.plan();
    const index_t m_round = plan.detectors_per_round();
    const std::uint32_t rounds = plan.rounds_total();
    Collected out;
    decoder.reset(shot);
    for (std::uint32_t r = 0; r < rounds; ++r) {
        const auto bits = syndrome.subspan(std::size_t{r} * m_round, m_round);
        const auto pushed = r + 1 < rounds ? decoder.push_round(bits) : decoder.push_final(bits);
        EXPECT_TRUE(pushed) << (pushed ? "" : describe(pushed.error()));
        if (!pushed) {
            return std::nullopt;
        }
        while (decoder.window_ready()) {
            auto commit = decoder.decode_next();
            EXPECT_TRUE(commit) << (commit ? "" : describe(commit.error()));
            if (!commit) {
                return std::nullopt;
            }
            if (commit->deferred) {
                ++out.deferred_returns;
                EXPECT_TRUE(commit->faults.empty());
                EXPECT_FALSE(commit->record.converged);
                continue;
            }
            collect(decoder, *commit, out);
        }
    }
    EXPECT_TRUE(decoder.finished());
    out.summary = decoder.summary();
    out.residual.assign(decoder.residual().begin(), decoder.residual().end());
    return out;
}

void expect_same_record(const WindowRecord& a, const WindowRecord& b, const std::string& where) {
    EXPECT_EQ(a.window, b.window) << where;
    EXPECT_EQ(a.shape, b.shape) << where;
    EXPECT_EQ(a.attempts, b.attempts) << where;
    EXPECT_EQ(a.iterations, b.iterations) << where;
    EXPECT_EQ(a.legs, b.legs) << where;
    EXPECT_EQ(a.converged, b.converged) << where;
    EXPECT_EQ(a.cap_hit, b.cap_hit) << where;
    EXPECT_EQ(a.flagged, b.flagged) << where;
    EXPECT_TRUE(same_bits(a.weight, b.weight)) << where << ": " << a.weight << " vs " << b.weight;
    EXPECT_TRUE(same_bits(a.committed_weight, b.committed_weight))
        << where << ": " << a.committed_weight << " vs " << b.committed_weight;
    EXPECT_EQ(a.unexplained, b.unexplained) << where;
    EXPECT_EQ(a.virtual_commits, b.virtual_commits) << where;
    EXPECT_EQ(a.returned_class, b.returned_class) << where;
}

// Everything but the decode times.
void expect_same_shot(const Collected& a, const Collected& b, const std::string& where) {
    ASSERT_EQ(a.records.size(), b.records.size()) << where;
    for (std::size_t k = 0; k < a.records.size(); ++k) {
        const std::string at = std::format("{} window {}", where, k);
        expect_same_record(a.records[k], b.records[k], at);
        EXPECT_EQ(a.faults[k], b.faults[k]) << at;
        EXPECT_EQ(a.frame_delta[k], b.frame_delta[k]) << at;
        EXPECT_EQ(a.frame_after[k], b.frame_after[k]) << at;
    }
    EXPECT_EQ(a.residual, b.residual) << where;
    EXPECT_EQ(a.summary.success, b.summary.success) << where;
    EXPECT_EQ(a.summary.flagged, b.summary.flagged) << where;
    EXPECT_EQ(a.summary.iterations, b.summary.iterations) << where;
    EXPECT_EQ(a.summary.legs, b.summary.legs) << where;
    EXPECT_TRUE(same_bits(a.summary.weight, b.summary.weight)) << where;
    EXPECT_EQ(a.summary.frame, b.summary.frame) << where;
}

// ---- Construction, rounds and readiness. ---------------------------------------------------

TEST(WindowStream, ConstructionChecksTheInnerDecoders) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    auto plan = WindowPlan::build(toy.problem(), spec(3, 1, 3));
    ASSERT_TRUE(plan) << describe(plan.error());
    const auto shapes = static_cast<std::uint32_t>(plan->shapes().size());
    ASSERT_GE(shapes, 2U);
    const std::uint32_t last = shapes - 1;

    std::vector<test::BruteForceInner> too_few;
    too_few.emplace_back(plan->shapes()[0]);
    auto count = StreamDecoder<test::BruteForceInner>::create(*plan, std::move(too_few));
    ASSERT_FALSE(count);
    EXPECT_EQ(count.error().code, Code::inner_count_mismatch);
    EXPECT_EQ(count.error().expected, shapes);
    EXPECT_EQ(count.error().found, 1U);

    // The final window (rounds 5–7, faults of rounds 5 and 6) is smaller than the others: give it
    // the first window's decoder.
    std::vector<test::BruteForceInner> swapped;
    swapped.reserve(shapes);
    for (std::uint32_t s = 0; s < shapes; ++s) {
        swapped.emplace_back(plan->shapes()[s == last ? 0 : s]);
    }
    ASSERT_NE(plan->shapes()[last].num_columns(), plan->shapes()[0].num_columns());
    auto size = StreamDecoder<test::BruteForceInner>::create(*plan, std::move(swapped));
    ASSERT_FALSE(size);
    EXPECT_EQ(size.error().code, Code::inner_size_mismatch);
    EXPECT_EQ(size.error().shape, last);
    EXPECT_NE(describe(size.error()).find(std::format("shape {}", last)), std::string::npos)
        << describe(size.error());

    struct FactoryError {
        std::string detail;
    };
    auto failed = StreamDecoder<test::BruteForceInner>::create(
        *plan, [](const Shape& shape) -> std::expected<test::BruteForceInner, FactoryError> {
            if (shape.index() == 1) {
                return std::unexpected(FactoryError{"no backend for shape 1"});
            }
            return test::BruteForceInner(shape);
        });
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, Code::inner_construction_failed);
    EXPECT_EQ(failed.error().shape, 1U);
    EXPECT_EQ(failed.error().detail, "no backend for shape 1");

    auto from_string = StreamDecoder<test::BruteForceInner>::create(
        *plan, [](const Shape& /*shape*/) -> std::expected<test::BruteForceInner, std::string> {
            return std::unexpected(std::string("out of memory"));
        });
    ASSERT_FALSE(from_string);
    EXPECT_EQ(from_string.error().detail, "out of memory");

    // Each inner decoder gets its shape's convergence rows once.
    std::vector<test::InnerCall> log;
    auto spied = StreamDecoder<test::Spy<test::BruteForceInner>>::create(
        *plan, [&log](const Shape& shape) {
            return test::Spy<test::BruteForceInner>(test::BruteForceInner(shape), shape.index(),
                                                    &log);
        });
    ASSERT_TRUE(spied) << describe(spied.error());
    for (std::uint32_t s = 0; s < shapes; ++s) {
        EXPECT_TRUE(std::ranges::equal(spied->inner(s).convergence_rows(),
                                       plan->shapes()[s].converge()));
    }
}

TEST(WindowStream, ErrorsAreNamedAndDescribed) {
    for (const Code code :
         {Code::inner_count_mismatch, Code::inner_size_mismatch, Code::inner_construction_failed,
          Code::wrong_round_size, Code::too_many_rounds, Code::missing_rounds,
          Code::stream_closed, Code::wrong_syndrome_size, Code::not_ready, Code::inner_failed}) {
        StreamError error;
        error.code = code;
        error.expected = 3;
        error.found = 2;
        EXPECT_NE(to_string(code), "unknown");
        EXPECT_TRUE(describe(error).starts_with(std::string(to_string(code)) + ": "))
            << describe(error);
    }
}

TEST(WindowStream, RoundsAreCheckedAndTheStreamCloses) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    auto plan = WindowPlan::build(toy.problem(), spec(3, 1, 3));
    ASSERT_TRUE(plan) << describe(plan.error());
    const Fails none;
    auto decoder = scripted_stream(*plan, none);
    ASSERT_TRUE(decoder) << describe(decoder.error());
    const std::vector<Bit> round{0, 0};

    auto wrong = decoder->push_round(std::vector<Bit>{0, 0, 0});
    ASSERT_FALSE(wrong);
    EXPECT_EQ(wrong.error().code, Code::wrong_round_size);
    EXPECT_EQ(wrong.error().expected, 2U);
    EXPECT_EQ(wrong.error().found, 3U);
    auto early = decoder->push_final(round);
    ASSERT_FALSE(early);
    EXPECT_EQ(early.error().code, Code::missing_rounds);
    auto not_ready = decoder->decode_next();
    ASSERT_FALSE(not_ready);
    EXPECT_EQ(not_ready.error().code, Code::not_ready);
    EXPECT_EQ(not_ready.error().expected, 3U) << "window 0 needs rounds 1–3";

    for (int r = 0; r < 6; ++r) {
        ASSERT_TRUE(decoder->push_round(round));
    }
    auto seventh = decoder->push_round(round);
    ASSERT_FALSE(seventh);
    EXPECT_EQ(seventh.error().code, Code::too_many_rounds);
    EXPECT_FALSE(decoder->closed());
    ASSERT_TRUE(decoder->push_final(round));
    EXPECT_TRUE(decoder->closed());
    EXPECT_EQ(decoder->push_round(round).error().code, Code::stream_closed);
    EXPECT_EQ(decoder->push_final(round).error().code, Code::stream_closed);
    while (!decoder->finished()) {
        ASSERT_TRUE(decoder->window_ready());
        ASSERT_TRUE(decoder->decode_next());
    }
    EXPECT_FALSE(decoder->window_ready());
    auto after = decoder->decode_next();
    ASSERT_FALSE(after);
    EXPECT_EQ(after.error().code, Code::not_ready);
    EXPECT_EQ(describe(after.error()), "not_ready: every window of the shot has been decoded");

    // A new shot starts from nothing; any nonzero byte is a fired detector.
    decoder->reset(5);
    EXPECT_FALSE(decoder->closed());
    EXPECT_TRUE(decoder->residual().empty());
    ASSERT_TRUE(decoder->push_round(std::vector<Bit>{2, 255}));
    EXPECT_TRUE(std::ranges::equal(decoder->residual(), std::vector<Bit>{1, 1}));

    auto short_shot = decode_shot(*decoder, std::vector<Bit>(13, 0), 0);
    ASSERT_FALSE(short_shot);
    EXPECT_EQ(short_shot.error().code, Code::wrong_syndrome_size);
    EXPECT_EQ(short_shot.error().expected, 14U);
}

// Window k of an exact (3, 1) plan needs rounds up to k + 3; the final window (rounds 5–7) needs
// the readout. Under the uniform boundary the positions reaching the readout or beyond wait for
// the stream to close.
TEST(WindowStream, WindowsBecomeReadyAsTheirRoundsArrive) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const Fails none;
    const std::vector<Bit> round{0, 0};
    for (const Boundary boundary : {Boundary::exact, Boundary::uniform}) {
        auto plan = WindowPlan::build(toy.problem(), spec(3, 1, 3, boundary));
        ASSERT_TRUE(plan) << describe(plan.error());
        auto decoder = scripted_stream(*plan, none);
        ASSERT_TRUE(decoder) << describe(decoder.error());
        decoder->reset(0);
        std::vector<std::uint32_t> decoded_after; // rounds received when each window decoded
        for (std::uint32_t r = 1; r <= 7; ++r) {
            ASSERT_TRUE(r < 7 ? decoder->push_round(round) : decoder->push_final(round));
            while (decoder->window_ready()) {
                auto commit = decoder->decode_next();
                ASSERT_TRUE(commit) << describe(commit.error());
                ASSERT_FALSE(commit->deferred);
                decoded_after.push_back(r);
            }
        }
        EXPECT_TRUE(decoder->finished());
        const std::vector<std::uint32_t> expected =
            boundary == Boundary::exact ? std::vector<std::uint32_t>{3, 4, 5, 6, 7}
                                        : std::vector<std::uint32_t>{3, 4, 5, 6, 7, 7, 7};
        EXPECT_EQ(decoded_after, expected) << to_string(boundary);
    }
}

// Every decode of shot s at window k, attempt a uses γ stream s + (k << 32) + (a << 56) and the
// spec's iteration cap.
TEST(WindowStream, InnerDecodesGetTheirStreamsAndBudget) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    auto plan = WindowPlan::build(toy.problem(),
                                  spec(3, 1, 3, Boundary::exact, OnFailure::defer, 2, 17));
    ASSERT_TRUE(plan) << describe(plan.error());
    const Fails fails{{0, 0}, {0, 1}, {2, 0}};
    std::vector<test::InnerCall> log;
    auto decoder = StreamDecoder<test::Spy<test::ScriptedInner>>::create(
        *plan, [&](const Shape& shape) {
            return test::Spy<test::ScriptedInner>(test::ScriptedInner(shape, &fails),
                                                  shape.index(), &log);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());
    constexpr std::uint64_t shot = 0x1234'5678ULL;
    ASSERT_TRUE(decode_shot(*decoder, std::vector<Bit>(14, 0), shot));
    std::vector<test::StreamKey> keys;
    for (const test::InnerCall& call : log) {
        keys.push_back(test::split_stream(call.stream));
        EXPECT_EQ(call.stream, shot + (std::uint64_t{keys.back().window} << 32U) +
                                   (std::uint64_t{keys.back().attempt} << 56U));
        EXPECT_EQ(call.cap, std::optional<std::uint32_t>(17));
    }
    const auto key = [](std::uint32_t window, std::uint32_t attempt) {
        return test::StreamKey{.shot = shot, .window = window, .attempt = attempt};
    };
    const std::vector<test::StreamKey> expected{key(0, 0), key(0, 1), key(0, 2), key(1, 0),
                                                key(2, 0), key(2, 1), key(3, 0), key(4, 0)};
    EXPECT_EQ(keys, expected);
}

// With a recording sink inside, each commit carries the committed attempt's solutions (the sink's
// records, in place), and the lowest-weight one, when stored, is the returned ê: same weight, and
// its logical class over the commit classes is the window's frame change.
TEST(WindowStream, CommitsCarryTheRecordedSolutions) {
    auto artifact = io::load_artifact(fixture_root / "bb18_choi_r9" / "artifact");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto shots = io::load_shots(fixture_root / "bb18_choi_r9" / "shots", *artifact);
    ASSERT_TRUE(shots) << io::describe(shots.error());
    const ArtifactProblem source(*artifact);
    auto plan = WindowPlan::build(source.problem(),
                                  spec(4, 2, 4, Boundary::exact, OnFailure::defer, 2));
    ASSERT_TRUE(plan) << describe(plan.error());
    std::vector<UniformGammaGenerator> gammas;
    for (const Shape& shape : plan->shapes()) {
        auto g = UniformGammaGenerator::create(9, -0.24, 0.66, shape.num_columns());
        ASSERT_TRUE(g);
        gammas.push_back(*g);
    }
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig relay{
        .pre_iter = 80, .set_max_iter = 60, .num_sets = 50, .stopping = AfterNConverged{3}};
    using Recording = CpuRelayDecoder<F32, Serial, RecordingSink>;
    auto decoder = StreamDecoder<Recording>::create(
        *plan, [&](const Shape& shape) -> std::expected<Recording, std::string> {
            auto backend = CpuBackend<F32>::create(shape.graph(), shape.priors());
            if (!backend) {
                return std::unexpected(backend.error().detail);
            }
            auto made = Recording::create(std::move(*backend), min_sum, relay,
                                          &gammas[shape.index()],
                                          RecordingSink(2, shape.commit_class()));
            if (!made) {
                return std::unexpected(made.error().detail);
            }
            return std::move(*made);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());
    std::size_t with_solutions = 0;
    std::size_t overflowing = 0;
    for (std::size_t s = 0; s < 8; ++s) {
        const auto check = [&](const Commit& c) {
            if (c.record.attempts == 0) {
                EXPECT_TRUE(c.solutions.empty());
                EXPECT_EQ(c.solutions_found, 0U);
                return;
            }
            const RecordingSink& sink = decoder->inner(c.record.shape).sink();
            EXPECT_EQ(c.solutions.data(), sink.records().data());
            EXPECT_EQ(c.solutions.size(), sink.records().size());
            EXPECT_EQ(c.solutions_found, sink.found());
            EXPECT_EQ(c.record.converged, c.solutions_found > 0);
            overflowing += c.solutions_found > sink.capacity() ? 1U : 0U;
            if (c.solutions.empty()) {
                return;
            }
            ++with_solutions;
            const auto best = std::ranges::min_element(c.solutions, {}, &SolutionRecord::weight);
            if (c.solutions_found <= sink.capacity()) {
                EXPECT_TRUE(same_bits(best->weight, c.record.weight));
                EXPECT_EQ(best->logical_class, c.record.returned_class);
            }
        };
        ASSERT_TRUE(decode_shot(*decoder, shots->syndrome(s), s, check));
    }
    EXPECT_GT(with_solutions, 15U);
    EXPECT_GT(overflowing, 0U);
}

// After construction the stream decoder allocates nothing: decoding shots in batch and streaming,
// with deferrals, skipped positions, uniform placements, recorded solutions and a refused round,
// performs no heap allocation at all (the relay decoder inside included).
TEST(WindowStream, DecodingAllocatesNothing) {
    if constexpr (!test::allocations_counted) {
        GTEST_SKIP() << "sanitizer builds keep their own allocation functions";
    }
    auto artifact = io::load_artifact(fixture_root / "bb18_choi_r9" / "artifact");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto shots = io::load_shots(fixture_root / "bb18_choi_r9" / "shots", *artifact);
    ASSERT_TRUE(shots) << io::describe(shots.error());
    const ArtifactProblem source(*artifact);
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig relay{
        .pre_iter = 40, .set_max_iter = 30, .num_sets = 20, .stopping = AfterNConverged{3}};
    using Recording = CpuRelayDecoder<F32, Serial, RecordingSink>;
    std::size_t deferred = 0;
    for (const WindowSpec& s : {spec(4, 2, 4, Boundary::exact, OnFailure::defer, 2),
                                spec(4, 2, 3, Boundary::uniform, OnFailure::defer, 1, 150)}) {
        auto plan = WindowPlan::build(source.problem(), s);
        ASSERT_TRUE(plan) << describe(plan.error());
        std::vector<UniformGammaGenerator> gammas;
        for (const Shape& shape : plan->shapes()) {
            gammas.push_back(*UniformGammaGenerator::create(3, -0.24, 0.66, shape.num_columns()));
        }
        const std::uint64_t unbuilt = test::allocations();
        auto decoder = StreamDecoder<Recording>::create(*plan, [&](const Shape& shape) {
            auto backend = CpuBackend<F32>::create(shape.graph(), shape.priors());
            return std::move(*Recording::create(std::move(*backend), min_sum, relay,
                                                &gammas[shape.index()],
                                                RecordingSink(4, shape.commit_class())));
        });
        ASSERT_TRUE(decoder) << describe(decoder.error());
        EXPECT_GT(test::allocations(), unbuilt) << "the counter does not see allocations";
        const index_t m_round = plan->detectors_per_round();
        const std::uint32_t rounds = plan->rounds_total();
        std::size_t failures = 0;
        std::size_t commits = 0;
        const std::uint64_t before = test::allocations();
        for (std::size_t shot = 0; shot < 12; ++shot) {
            const auto sigma = shots->syndrome(shot);
            if (!decode_shot(*decoder, sigma, shot, [&commits](const Commit&) { ++commits; })) {
                ++failures;
            }
            decoder->reset(shot);
            for (std::uint32_t r = 0; r < rounds; ++r) {
                const auto bits = sigma.subspan(std::size_t{r} * m_round, m_round);
                if (r == 1 && decoder->push_round(bits.first(1))) {
                    ++failures; // a short round must be refused
                }
                if (!(r + 1 < rounds ? decoder->push_round(bits) : decoder->push_final(bits))) {
                    ++failures;
                }
                while (decoder->window_ready()) {
                    const auto commit = decoder->decode_next();
                    failures += commit ? 0U : 1U;
                    deferred += commit && commit->deferred ? 1U : 0U;
                }
            }
        }
        const std::uint64_t allocations = test::allocations() - before;
        EXPECT_EQ(allocations, 0U) << to_string(s.boundary);
        EXPECT_EQ(failures, 0U);
        EXPECT_EQ(commits, 12 * plan->num_positions());
    }
    EXPECT_GT(deferred, 0U) << "no streaming deferral was exercised";
}

// ---- Non-convergence policies (toy, exact (3, 1)). -----------------------------------------

struct PolicyRun {
    Collected shot;
    std::vector<test::InnerCall> calls;
};

// Case A of the toy fixture (faults 6, 15 and 27), decoded with the given failures scripted in.
std::optional<PolicyRun> run_policy(const Toy& toy, const WindowSpec& s, const Fails& fails,
                                    bool stream) {
    auto plan = WindowPlan::build(toy.problem(), s);
    EXPECT_TRUE(plan) << (plan ? "" : describe(plan.error()));
    if (!plan) {
        return std::nullopt;
    }
    PolicyRun run;
    auto decoder = StreamDecoder<test::Spy<test::ScriptedInner>>::create(
        *plan, [&](const Shape& shape) {
            return test::Spy<test::ScriptedInner>(test::ScriptedInner(shape, &fails),
                                                  shape.index(), &run.calls);
        });
    EXPECT_TRUE(decoder);
    if (!decoder) {
        return std::nullopt;
    }
    const std::vector<index_t> faults{6, 15, 27};
    const std::vector<Bit> sigma = syndrome_of(toy.problem(), faults);
    auto shot = stream ? streamed(*decoder, sigma, 3) : batch(*decoder, sigma, 3);
    if (!shot) {
        return std::nullopt;
    }
    run.shot = std::move(*shot);
    return run;
}

TEST(WindowStreamPolicy, CommitAnywayCommitsTheUnconvergedSolution) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const Fails fails{{1, 0}};
    const auto clean = run_policy(toy, spec(3, 1, 3), {}, false);
    const auto run = run_policy(toy, spec(3, 1, 3), fails, false);
    ASSERT_TRUE(clean && run);
    const WindowRecord& w1 = run->shot.records[1];
    EXPECT_FALSE(w1.converged);
    EXPECT_FALSE(w1.flagged);
    EXPECT_EQ(w1.attempts, 1U);
    EXPECT_EQ(w1.iterations, test::ScriptedInner::failed_iterations);
    EXPECT_EQ(w1.legs, test::ScriptedInner::failed_legs);
    EXPECT_EQ(w1.weight, infinity);
    // The unconverged decode returned the exact solution, so the same faults are committed.
    EXPECT_EQ(run->shot.faults[1], clean->shot.faults[1]);
    EXPECT_TRUE(same_bits(w1.committed_weight, clean->shot.records[1].committed_weight));
    EXPECT_FALSE(run->shot.summary.success);
    EXPECT_FALSE(run->shot.summary.flagged);
    EXPECT_EQ(run->shot.summary.frame, clean->shot.summary.frame);
    EXPECT_EQ(run->shot.summary.iterations,
              clean->shot.summary.iterations - 1 + test::ScriptedInner::failed_iterations);
}

TEST(WindowStreamPolicy, FlagMarksTheWindowAndTheShot) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const auto run =
        run_policy(toy, spec(3, 1, 3, Boundary::exact, OnFailure::flag), {{1, 0}}, false);
    ASSERT_TRUE(run);
    for (std::size_t k = 0; k < run->shot.records.size(); ++k) {
        EXPECT_EQ(run->shot.records[k].flagged, k == 1) << k;
    }
    EXPECT_TRUE(run->shot.summary.flagged);
    EXPECT_FALSE(run->shot.summary.success);
}

TEST(WindowStreamPolicy, DeferRetriesWithAWiderWindow) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const WindowSpec s = spec(3, 1, 3, Boundary::exact, OnFailure::defer, 2);
    const auto run = run_policy(toy, s, {{1, 0}}, false);
    ASSERT_TRUE(run);
    auto plan = WindowPlan::build(toy.problem(), s);
    ASSERT_TRUE(plan);
    const Placement* wider = plan->placement(1, 1);
    ASSERT_NE(wider, nullptr);
    const WindowRecord& w1 = run->shot.records[1];
    EXPECT_TRUE(w1.converged);
    EXPECT_FALSE(w1.flagged);
    EXPECT_EQ(w1.attempts, 2U);
    EXPECT_EQ(w1.shape, wider->shape);
    EXPECT_EQ(w1.iterations, test::ScriptedInner::failed_iterations + 1);
    EXPECT_EQ(w1.legs, test::ScriptedInner::failed_legs + 1);
    EXPECT_TRUE(run->shot.summary.success);
    EXPECT_EQ(run->shot.summary.frame, 1U) << "case A's true observable flip";
    // The wider attempt spans rounds 2–5 but still commits only faults with s = 2.
    EXPECT_EQ(run->shot.faults[1], (std::vector<index_t>{6}));
}

TEST(WindowStreamPolicy, DeferOutOfAttemptsIsFlagged) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const auto run = run_policy(toy, spec(3, 1, 3, Boundary::exact, OnFailure::defer, 2),
                                {{1, 0}, {1, 1}, {1, 2}}, false);
    ASSERT_TRUE(run);
    const WindowRecord& w1 = run->shot.records[1];
    EXPECT_FALSE(w1.converged);
    EXPECT_TRUE(w1.flagged);
    EXPECT_EQ(w1.attempts, 3U);
    EXPECT_EQ(w1.iterations, 3 * test::ScriptedInner::failed_iterations);
    EXPECT_TRUE(run->shot.summary.flagged);
}

// Window 3 (t = 4) fails; its first deferral attempt covers rounds 4–7, reaches the readout and
// commits every fault with s ≥ 4, so window 4 is not decoded and gets an empty record.
TEST(WindowStreamPolicy, AFinalDeferralAttemptCompletesTheShot) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const WindowSpec s = spec(3, 1, 3, Boundary::exact, OnFailure::defer, 2);
    for (const bool stream : {false, true}) {
        const auto run = run_policy(toy, s, {{3, 0}}, stream);
        ASSERT_TRUE(run);
        ASSERT_EQ(run->shot.records.size(), 5U);
        const WindowRecord& w3 = run->shot.records[3];
        EXPECT_EQ(w3.attempts, 2U);
        EXPECT_TRUE(w3.converged);
        EXPECT_EQ(run->shot.faults[3], (std::vector<index_t>{15, 27}));
        const WindowRecord& w4 = run->shot.records[4];
        EXPECT_EQ(w4.attempts, 0U);
        EXPECT_EQ(w4.shape, no_shape);
        EXPECT_TRUE(w4.converged);
        EXPECT_EQ(w4.iterations, 0U);
        EXPECT_EQ(w4.legs, 0U);
        EXPECT_EQ(w4.weight, 0.0);
        EXPECT_EQ(w4.committed_weight, 0.0);
        EXPECT_EQ(w4.unexplained, 0U);
        EXPECT_FALSE(w4.flagged);
        EXPECT_EQ(w4.decode_ns, 0U);
        EXPECT_TRUE(run->shot.faults[4].empty());
        EXPECT_EQ(run->shot.summary.frame, 1U);
        EXPECT_TRUE(run->shot.summary.success);
        EXPECT_TRUE(std::ranges::all_of(run->shot.residual, [](Bit b) { return b == 0; }));
        for (const test::InnerCall& call : run->calls) {
            EXPECT_NE(test::split_stream(call.stream).window, 4U) << "window 4 was decoded";
        }
    }
}

// While streaming, a failed window waits for the rounds of its wider attempt: decode_next reports
// the deferral and the window is decoded again once they arrive, with the batch result.
TEST(WindowStreamPolicy, StreamingDeferralWaitsForItsRounds) {
    const Toy toy = load_toy();
    ASSERT_TRUE(toy.artifact);
    const WindowSpec s = spec(3, 1, 3, Boundary::exact, OnFailure::defer, 2);
    const Fails fails{{1, 0}, {1, 1}, {3, 0}};
    const auto together = run_policy(toy, s, fails, false);
    const auto streaming = run_policy(toy, s, fails, true);
    ASSERT_TRUE(together && streaming);
    expect_same_shot(streaming->shot, together->shot, "streamed");
    // Window 1 waits twice (its wider attempts need rounds 5 and 6), window 3 once (its wider
    // attempt reaches the readout).
    EXPECT_EQ(streaming->shot.deferred_returns, 3U);
    EXPECT_EQ(together->shot.deferred_returns, 0U);

    auto plan = WindowPlan::build(toy.problem(), s);
    ASSERT_TRUE(plan);
    auto decoder = scripted_stream(*plan, fails);
    ASSERT_TRUE(decoder);
    const std::vector<Bit> round{0, 0};
    decoder->reset(0);
    for (int r = 0; r < 4; ++r) {
        ASSERT_TRUE(decoder->push_round(round));
    }
    ASSERT_TRUE(decoder->decode_next()); // window 0
    auto deferred = decoder->decode_next();
    ASSERT_TRUE(deferred);
    EXPECT_TRUE(deferred->deferred);
    EXPECT_EQ(deferred->window, 1U);
    EXPECT_EQ(deferred->rounds, 0U);
    EXPECT_EQ(deferred->record.attempts, 1U);
    EXPECT_EQ(deferred->record.iterations, test::ScriptedInner::failed_iterations);
    EXPECT_FALSE(decoder->window_ready()) << "attempt (1, 1) needs round 5";
    ASSERT_TRUE(decoder->push_round(round));
    ASSERT_TRUE(decoder->window_ready());
    auto again = decoder->decode_next();
    ASSERT_TRUE(again);
    EXPECT_TRUE(again->deferred);
    EXPECT_EQ(again->record.attempts, 2U);
    ASSERT_TRUE(decoder->push_round(round));
    auto done = decoder->decode_next();
    ASSERT_TRUE(done);
    EXPECT_FALSE(done->deferred);
    EXPECT_EQ(done->record.attempts, 3U);
    EXPECT_EQ(done->record.iterations, 2 * test::ScriptedInner::failed_iterations + 1);
}

// ---- Properties on random time-structured problems, relay decoder inside. --------------------

// A literal transcription of the stream-decoding rule over the global syndrome, with its own
// column → rows lists (rebuilt from each shape's CSR) and its own inner decoders.
struct ReferenceShot {
    std::vector<WindowRecord> records;
    std::vector<std::vector<index_t>> faults;
    std::vector<std::uint64_t> frame_after;
    std::vector<Bit> residual;
    std::uint64_t frame = 0;
};

using ShapeColumns = std::vector<std::vector<std::vector<index_t>>>;

ShapeColumns shape_columns(const WindowPlan& plan) {
    ShapeColumns out;
    for (const Shape& shape : plan.shapes()) {
        std::vector<std::vector<index_t>> cols(shape.num_columns());
        for (index_t i = 0; i < shape.num_rows(); ++i) {
            for (index_t e = shape.row_ptr()[i]; e < shape.row_ptr()[i + 1]; ++e) {
                cols[shape.col_indices()[e]].push_back(i);
            }
        }
        out.push_back(std::move(cols));
    }
    return out;
}

std::optional<std::size_t> find_placement(const WindowPlan& plan, std::uint32_t k,
                                          std::uint32_t a) {
    for (std::size_t i = 0; i < plan.schedule().size(); ++i) {
        if (plan.schedule()[i].window == k && plan.schedule()[i].attempt == a) {
            return i;
        }
    }
    return std::nullopt;
}

struct Decided {
    std::size_t placement = 0;
    bool success = false;
    bool cap_hit = false;
    double weight = 0.0;
    std::vector<index_t> support;
    std::uint32_t iterations = 0;
    std::uint32_t legs = 0;
    std::uint32_t attempts = 0;
    bool flagged = false;
};

template <class Inner>
std::optional<Decided> reference_window(const WindowPlan& plan, std::vector<Inner>& inners,
                                        std::span<const Bit> residual, std::uint64_t shot,
                                        std::uint32_t k) {
    const WindowSpec& s = plan.spec();
    Decided d;
    for (std::uint32_t a = 0;; ++a) {
        const auto index = find_placement(plan, k, a);
        if (!index) {
            ADD_FAILURE() << "no placement " << k << ", " << a;
            return std::nullopt;
        }
        const Placement& p = plan.schedule()[*index];
        const Shape& shape = plan.shape_of(p);
        std::vector<Bit> local(shape.num_rows(), 0);
        for (index_t i = 0; i < shape.num_rows(); ++i) {
            if (std::size_t{p.first_row} + i < residual.size()) {
                local[i] = residual[p.first_row + i];
            }
        }
        const std::uint64_t stream = shot + (std::uint64_t{k} << 32U) + (std::uint64_t{a} << 56U);
        auto res = inners[p.shape].decode(local, stream, DecodeLimits{s.iteration_cap});
        if (!res) {
            ADD_FAILURE() << "reference decode failed";
            return std::nullopt;
        }
        d.placement = *index;
        d.success = res->success;
        d.cap_hit = res->cap_hit;
        d.weight = res->weight;
        d.support.assign(res->support.begin(), res->support.end());
        d.iterations += res->iterations;
        d.legs += res->legs_executed;
        d.attempts = a + 1;
        if (res->success) {
            return d;
        }
        if (s.on_failure == OnFailure::defer && a < s.max_deferrals &&
            find_placement(plan, k, a + 1)) {
            continue;
        }
        d.flagged = s.on_failure != OnFailure::commit_anyway;
        return d;
    }
}

template <class Inner>
std::optional<ReferenceShot> reference_decode(const WindowPlan& plan, std::vector<Inner>& inners,
                                              const ShapeColumns& columns,
                                              std::span<const Bit> sigma, std::uint64_t shot) {
    const std::size_t m = sigma.size();
    const std::size_t m_round = plan.detectors_per_round();
    ReferenceShot out;
    out.residual.assign(sigma.begin(), sigma.end());
    bool complete = false;
    for (std::uint32_t k = 0; k < plan.num_positions(); ++k) {
        WindowRecord record;
        record.window = k;
        if (complete) {
            out.records.push_back(record);
            out.faults.emplace_back();
            out.frame_after.push_back(out.frame);
            continue;
        }
        const auto d = reference_window(plan, inners, out.residual, shot, k);
        if (!d) {
            return std::nullopt;
        }
        const Placement& p = plan.schedule()[d->placement];
        const Shape& shape = plan.shape_of(p);
        std::vector<index_t> faults;
        for (const index_t l : d->support) {
            if (shape.commit()[l] == 0) {
                continue;
            }
            for (const index_t i : columns[p.shape][l]) {
                if (std::size_t{p.first_row} + i < m) {
                    out.residual[p.first_row + i] ^= Bit{1};
                }
            }
            record.returned_class ^= shape.commit_class()[l];
            if (std::isfinite(shape.priors().llr()[l])) {
                record.committed_weight += shape.priors().llr()[l];
            }
            if (p.columns[l] == virtual_column) {
                ++record.virtual_commits;
            } else {
                faults.push_back(p.columns[l]);
            }
        }
        std::ranges::sort(faults);
        const std::size_t end = std::min(p.first_row + p.commit_rounds * m_round, m);
        for (std::size_t i = p.first_row; i < end; ++i) {
            record.unexplained += out.residual[i];
        }
        out.frame ^= record.returned_class;
        record.shape = p.shape;
        record.attempts = d->attempts;
        record.iterations = d->iterations;
        record.legs = d->legs;
        record.converged = d->success;
        record.cap_hit = d->cap_hit;
        record.flagged = d->flagged;
        record.weight = d->success ? d->weight : infinity;
        out.records.push_back(record);
        out.faults.push_back(std::move(faults));
        out.frame_after.push_back(out.frame);
        complete = p.final;
    }
    return out;
}

void expect_matches_reference(const Collected& got, const ReferenceShot& ref,
                              const std::string& where) {
    ASSERT_EQ(got.records.size(), ref.records.size()) << where;
    for (std::size_t k = 0; k < ref.records.size(); ++k) {
        const std::string at = std::format("{} window {}", where, k);
        expect_same_record(got.records[k], ref.records[k], at);
        EXPECT_EQ(got.faults[k], ref.faults[k]) << at;
        EXPECT_EQ(got.frame_after[k], ref.frame_after[k]) << at;
        std::uint64_t delta = 0;
        for (std::size_t o = 0; o < got.frame_delta[k].size(); ++o) {
            delta |= std::uint64_t{got.frame_delta[k][o]} << o;
        }
        EXPECT_EQ(delta, ref.records[k].returned_class) << at;
    }
    EXPECT_EQ(got.residual, ref.residual) << where;
    EXPECT_EQ(got.summary.frame, ref.frame) << where;
    double weight = 0.0;
    std::uint32_t iterations = 0;
    bool success = true;
    bool flagged = false;
    for (const WindowRecord& r : ref.records) {
        weight += r.committed_weight;
        iterations += r.iterations;
        success = success && r.converged;
        flagged = flagged || r.flagged;
    }
    EXPECT_TRUE(same_bits(got.summary.weight, weight)) << where;
    EXPECT_EQ(got.summary.iterations, iterations) << where;
    EXPECT_EQ(got.summary.success, success) << where;
    EXPECT_EQ(got.summary.flagged, flagged) << where;
}

// What the random cases exercised, so that a silently vacuous property shows up.
struct Coverage {
    std::size_t cases = 0;
    std::size_t exact = 0;
    std::size_t uniform = 0;
    std::size_t shots = 0;
    std::size_t windows = 0;
    std::size_t not_converged = 0;
    std::size_t deferred = 0;
    std::size_t streaming_deferrals = 0;
    std::size_t skipped = 0;
    std::size_t flagged = 0;
    std::size_t cap_hits = 0;
    std::size_t partial_convergence_checks = 0;
    std::size_t virtual_commits = 0;
    std::size_t all_converged_shots = 0;
};

// One window's record against the policy and the cap, and its faults against the commit set of
// the attempt that committed them; counts each column's commits.
void check_window(const WindowPlan& plan, const Collected& got, std::size_t k, bool& completed,
                  std::vector<int>& committed, Coverage& coverage) {
    const WindowSpec& s = plan.spec();
    const WindowRecord& r = got.records[k];
    const auto window = static_cast<std::uint32_t>(k);
    if (r.attempts == 0) {
        EXPECT_TRUE(completed) << "window " << k << " skipped before a final placement";
        ++coverage.skipped;
        return;
    }
    ++coverage.windows;
    EXPECT_FALSE(completed) << "window " << k << " decoded after a final placement";
    EXPECT_LE(r.attempts, plan.attempts(window));
    EXPECT_LE(r.attempts, s.max_deferrals + 1);
    EXPECT_EQ(r.flagged, !r.converged && s.on_failure != OnFailure::commit_anyway);
    if (!r.converged && s.on_failure == OnFailure::defer) {
        EXPECT_EQ(r.attempts, plan.attempts(window)) << "gave up with attempts left";
    }
    if (s.iteration_cap) {
        EXPECT_LE(r.iterations, std::uint64_t{*s.iteration_cap} * r.attempts);
    }
    coverage.not_converged += r.converged ? 0U : 1U;
    coverage.deferred += r.attempts > 1 ? 1U : 0U;
    coverage.flagged += r.flagged ? 1U : 0U;
    coverage.cap_hits += r.cap_hit ? 1U : 0U;
    coverage.virtual_commits += r.virtual_commits;
    const auto index = find_placement(plan, window, r.attempts - 1);
    ASSERT_TRUE(index);
    const Placement& p = plan.schedule()[*index];
    completed = completed || p.final;
    std::set<index_t> allowed;
    for (index_t l = 0; l < p.columns.size(); ++l) {
        if (plan.shape_of(p).commit()[l] != 0) {
            allowed.insert(p.columns[l]);
        }
    }
    for (const index_t j : got.faults[k]) {
        EXPECT_TRUE(allowed.contains(j)) << "window " << k << " committed " << j;
        ++committed[j];
    }
}

// The stream result against the global problem: every fault committed once, inside the commit
// set of the attempt that committed it; H·c ⊕ residual = σ and A·c = frame (exact boundary); the
// unexplained counts add up to the residual; a shot whose windows all converged leaves no
// residual; and the per-window bounds of the policies and the cap.
void check_against_problem(const Problem& problem, const WindowPlan& plan,
                           std::span<const Bit> sigma, const Collected& got, Coverage& coverage) {
    std::vector<int> committed(problem.num_columns, 0);
    std::uint64_t unexplained = 0;
    bool completed = false;
    for (std::size_t k = 0; k < got.records.size(); ++k) {
        unexplained += got.records[k].unexplained;
        check_window(plan, got, k, completed, committed, coverage);
    }
    const auto ones = static_cast<std::uint64_t>(std::ranges::count(got.residual, Bit{1}));
    EXPECT_EQ(unexplained, ones);
    if (got.summary.success) {
        ++coverage.all_converged_shots;
        EXPECT_EQ(ones, 0U) << "every window converged but detectors are left";
    }
    if (plan.spec().boundary != Boundary::exact) {
        return;
    }
    const auto supports = test::column_supports(problem);
    const auto classes = test::column_classes(problem);
    std::vector<Bit> hc(sigma.size(), 0);
    std::uint64_t frame = 0;
    for (index_t j = 0; j < problem.num_columns; ++j) {
        EXPECT_LE(committed[j], 1) << "column " << j << " committed twice";
        if (committed[j] == 0) {
            continue;
        }
        frame ^= classes[j];
        for (const index_t i : supports[j]) {
            hc[i] ^= Bit{1};
        }
    }
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        EXPECT_EQ(hc[i] ^ got.residual[i], sigma[i]) << "row " << i;
    }
    EXPECT_EQ(frame, got.summary.frame);
}

// Each inner call: the cap it was given, its iterations within the cap, cap_hit exactly when the
// uncapped decode needs more (and then the result is that decode cut at the cap), and a converged
// solution satisfying its shape's convergence rows (fewer than all rows when C′ < W).
template <class Uncapped>
void check_calls(const WindowPlan& plan, std::span<const test::InnerCall> calls,
                 std::vector<Uncapped>& uncapped, const ShapeColumns& columns,
                 Coverage& coverage) {
    const std::optional<std::uint32_t> cap = plan.spec().iteration_cap;
    for (const test::InnerCall& call : calls) {
        EXPECT_EQ(call.cap, cap);
        const Shape& shape = plan.shapes()[call.shape];
        if (cap) {
            EXPECT_LE(call.iterations, *cap);
            auto free = uncapped[call.shape].decode(call.syndrome, call.stream, DecodeLimits{});
            ASSERT_TRUE(free);
            EXPECT_EQ(call.cap_hit, free->iterations > *cap);
            if (call.cap_hit) {
                EXPECT_EQ(call.iterations, *cap);
            } else {
                EXPECT_EQ(call.iterations, free->iterations);
                EXPECT_EQ(call.success, free->success);
                EXPECT_TRUE(std::ranges::equal(call.support, free->support));
            }
        } else {
            EXPECT_FALSE(call.cap_hit);
        }
        if (!call.success) {
            continue;
        }
        std::vector<Bit> hx(shape.num_rows(), 0);
        for (const index_t l : call.support) {
            for (const index_t i : columns[call.shape][l]) {
                hx[i] ^= Bit{1};
            }
        }
        bool partial = false;
        for (index_t i = 0; i < shape.num_rows(); ++i) {
            if (shape.converge()[i] != 0) {
                EXPECT_EQ(hx[i], call.syndrome[i]) << "shape " << call.shape << " row " << i;
            } else {
                partial = true;
            }
        }
        coverage.partial_convergence_checks += partial ? 1U : 0U;
    }
}

struct RandomRelay {
    MinSumConfig min_sum;
    RelayConfig relay;
};

RandomRelay random_relay(std::mt19937_64& rng) {
    std::uniform_int_distribution<std::uint32_t> pre(2, 25);
    std::uniform_int_distribution<std::uint32_t> set(2, 15);
    std::uniform_int_distribution<std::uint32_t> sets(0, 6);
    std::uniform_int_distribution<std::uint32_t> rule(0, 2);
    std::uniform_int_distribution<std::uint32_t> count(1, 3);
    RandomRelay r;
    r.relay.pre_iter = pre(rng);
    r.relay.set_max_iter = set(rng);
    r.relay.num_sets = sets(rng);
    switch (rule(rng)) {
    case 0:
        r.relay.stopping = AfterLeg0{};
        break;
    case 1:
        r.relay.stopping = AfterNConverged{count(rng)};
        break;
    default:
        r.relay.stopping = AllLegs{};
        break;
    }
    r.min_sum.gamma0 = 0.125;
    if (std::bernoulli_distribution(0.3)(rng)) {
        r.min_sum.alpha = AdaptiveAlpha{1.0};
    }
    return r;
}

// Exact: any W up to Rt + 1 (the identity limit). Uniform: a bulk window must exist, i.e. window 1
// (starting at round C + 1) and its deferral attempts must end before round R = Rt − 1:
// W + (D + 1)·C ≤ Rt − 2.
WindowSpec random_spec(std::mt19937_64& rng, std::uint32_t rounds_total, Boundary boundary) {
    const auto pick = [&rng](std::uint32_t lo, std::uint32_t hi) {
        return std::uniform_int_distribution<std::uint32_t>(lo, hi)(rng);
    };
    const auto policy = static_cast<OnFailure>(std::discrete_distribution<int>({25, 45, 30})(rng));
    std::uint32_t deferrals = policy == OnFailure::defer ? pick(0, 3) : 0;
    std::uint32_t w = 0;
    std::uint32_t c = 0;
    if (boundary == Boundary::exact) {
        w = pick(2, rounds_total + 1);
        c = pick(1, w - 1);
    } else {
        deferrals = std::min(deferrals, 2U);
        c = pick(1, 3);
        while (c + 1 + (deferrals + 1) * c > rounds_total - 2) {
            if (deferrals > 0) {
                --deferrals;
            } else {
                --c;
            }
        }
        w = pick(c + 1, rounds_total - 2 - (deferrals + 1) * c);
    }
    const std::uint32_t converge = pick(c, w);
    std::optional<std::uint32_t> cap;
    if (std::bernoulli_distribution(0.5)(rng)) {
        cap = std::uniform_int_distribution<std::uint32_t>(1, 40)(rng);
    }
    return spec(w, c, converge, boundary, policy, deferrals, cap);
}

// A syndrome from faults drawn with their priors, or random detector bits (mostly not a syndrome
// of anything the window can explain, so windows fail to converge).
std::vector<Bit> random_syndrome(std::mt19937_64& rng, const test::OwnedProblem& owned,
                                 bool from_faults) {
    const index_t m = owned.rounds_total * owned.per_round;
    std::vector<Bit> s(m, 0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (from_faults) {
        for (std::size_t j = 0; j < owned.columns.size(); ++j) {
            if (unit(rng) < owned.priors[j]) {
                for (const index_t i : owned.columns[j]) {
                    s[i] ^= Bit{1};
                }
            }
        }
    } else {
        for (Bit& b : s) {
            b = unit(rng) < 0.15 ? Bit{1} : Bit{0};
        }
    }
    return s;
}

using Relay = CpuRelayDecoder<F64>;

std::expected<Relay, std::string> make_relay(const Shape& shape, const RandomRelay& config,
                                             const GammaSource* gammas) {
    auto backend = CpuBackend<F64>::create(shape.graph(), shape.priors());
    if (!backend) {
        return std::unexpected(backend.error().detail);
    }
    auto decoder = Relay::create(std::move(*backend), config.min_sum, config.relay,
                                 config.relay.num_sets > 0 ? gammas : nullptr);
    if (!decoder) {
        return std::unexpected(decoder.error().detail);
    }
    return std::move(*decoder);
}

// One random case: a problem, a spec, a relay configuration; several shots decoded in batch and
// streaming, against the reference and the global problem.
void run_random_case(std::uint64_t seed, Coverage& coverage) {
    std::mt19937_64 rng(seed);
    const bool uniform = std::bernoulli_distribution(0.35)(rng);
    const auto rounds_total = std::uniform_int_distribution<index_t>(uniform ? 6 : 2, 12)(rng);
    const auto per_round = std::uniform_int_distribution<index_t>(1, 4)(rng);
    const auto observables = std::uniform_int_distribution<index_t>(1, 6)(rng);
    const test::OwnedProblem owned =
        uniform ? test::translation_invariant_problem(seed, rounds_total, per_round, observables)
                : test::random_problem(seed, rounds_total, per_round, observables);
    const Problem problem = owned.view();
    const WindowSpec s =
        random_spec(rng, rounds_total, uniform ? Boundary::uniform : Boundary::exact);
    auto plan = WindowPlan::build(problem, s);
    if (!plan) {
        // Uniform plans need a bulk window; other rejections are covered by the plan tests.
        EXPECT_TRUE(uniform) << "seed " << seed << ": " << describe(plan.error());
        return;
    }
    ++coverage.cases;
    ++(uniform ? coverage.uniform : coverage.exact);
    const RandomRelay config = random_relay(rng);
    std::vector<UniformGammaGenerator> gammas;
    gammas.reserve(plan->shapes().size());
    for (const Shape& shape : plan->shapes()) {
        auto g = UniformGammaGenerator::create(seed, -0.24, 0.66, shape.num_columns());
        ASSERT_TRUE(g);
        gammas.push_back(*g);
    }
    std::vector<test::InnerCall> calls;
    auto decoder = StreamDecoder<test::Spy<Relay>>::create(
        *plan, [&](const Shape& shape) -> std::expected<test::Spy<Relay>, std::string> {
            auto relay = make_relay(shape, config, &gammas[shape.index()]);
            if (!relay) {
                return std::unexpected(relay.error());
            }
            return test::Spy<Relay>(std::move(*relay), shape.index(), &calls);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());
    std::vector<Relay> reference;
    std::vector<Relay> uncapped;
    for (const Shape& shape : plan->shapes()) {
        for (auto* target : {&reference, &uncapped}) {
            auto relay = make_relay(shape, config, &gammas[shape.index()]);
            ASSERT_TRUE(relay) << relay.error();
            relay->set_convergence_rows(shape.converge());
            target->push_back(std::move(*relay));
        }
    }
    const ShapeColumns columns = shape_columns(*plan);
    for (std::uint64_t shot = 0; shot < 4; ++shot) {
        const std::string where = std::format("seed {} shot {} ({}, {}, {}) {} {} D={} cap={}",
                                              seed, shot, s.width, s.commit, s.converge_rounds,
                                              to_string(s.boundary), to_string(s.on_failure),
                                              s.max_deferrals, s.iteration_cap.value_or(0));
        SCOPED_TRACE(where);
        const std::vector<Bit> sigma = random_syndrome(rng, owned, shot % 2 == 0);
        calls.clear();
        const auto got = batch(*decoder, sigma, shot);
        ASSERT_TRUE(got);
        const auto ref = reference_decode(*plan, reference, columns, sigma, shot);
        ASSERT_TRUE(ref);
        expect_matches_reference(*got, *ref, where);
        check_against_problem(problem, *plan, sigma, *got, coverage);
        check_calls(*plan, calls, uncapped, columns, coverage);
        const auto live = streamed(*decoder, sigma, shot);
        ASSERT_TRUE(live);
        expect_same_shot(*live, *got, where + " streamed");
        coverage.streaming_deferrals += live->deferred_returns;
        ++coverage.shots;
    }
}

TEST(WindowStreamProperties, RandomProblemsFollowTheRuleExactly) {
    Coverage coverage;
    for (std::uint64_t seed = 1; seed <= 200; ++seed) {
        run_random_case(seed, coverage);
        if (testing::Test::HasFatalFailure()) {
            return;
        }
    }
    std::cout << std::format(
        "[ stream   ] {} cases ({} exact, {} uniform), {} shots, {} windows: {} not converged, "
        "{} deferred ({} streaming waits), {} skipped, {} flagged, {} cap hits, {} partial "
        "convergence checks, {} virtual commits, {} shots all converged\n",
        coverage.cases, coverage.exact, coverage.uniform, coverage.shots, coverage.windows,
        coverage.not_converged, coverage.deferred, coverage.streaming_deferrals, coverage.skipped,
        coverage.flagged, coverage.cap_hits, coverage.partial_convergence_checks,
        coverage.virtual_commits, coverage.all_converged_shots);
    EXPECT_GT(coverage.uniform, 40U);
    EXPECT_GT(coverage.not_converged, 50U);
    EXPECT_GT(coverage.deferred, 10U);
    EXPECT_GT(coverage.streaming_deferrals, 5U);
    EXPECT_GT(coverage.skipped, 0U);
    EXPECT_GT(coverage.cap_hits, 20U);
    EXPECT_GT(coverage.partial_convergence_checks, 50U);
    EXPECT_GT(coverage.virtual_commits, 0U);
    EXPECT_GT(coverage.all_converged_shots, 50U);
}

// ---- Stream-layer cost on the gross code (R = 48, exact (12, 8)); opt-in. ---------------------

// Replays recorded inner results, so that a decode_next costs only the stream layer.
class ReplayInner {
public:
    ReplayInner(const Shape& shape, const std::vector<test::InnerCall>* calls)
        : rows_(shape.num_rows()), cols_(shape.num_columns()), calls_(calls) {}

    [[nodiscard]] index_t num_rows() const noexcept { return rows_; }
    [[nodiscard]] index_t num_columns() const noexcept { return cols_; }
    void set_convergence_rows(std::span<const Bit> /*mask*/) noexcept {}

    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> /*syndrome*/, std::uint64_t stream,
           DecodeLimits /*limits*/) noexcept {
        const test::InnerCall* found = nullptr;
        for (const test::InnerCall& call : *calls_) {
            if (call.stream == stream) {
                found = &call;
                break;
            }
        }
        if (found == nullptr) {
            return std::unexpected(DecodeError{
                .code = DecodeError::Code::syndrome_size_mismatch, .expected = 0, .found = 0});
        }
        return DecodeResult{.success = found->success,
                            .iterations = found->iterations,
                            .legs_executed = found->legs_executed,
                            .best_leg = found->best_leg,
                            .weight = found->weight,
                            .hard = {},
                            .legs = {},
                            .cap_hit = found->cap_hit,
                            .support = found->support};
    }

private:
    index_t rows_;
    index_t cols_;
    const std::vector<test::InnerCall>* calls_;
};

double median(std::vector<double> v) {
    std::ranges::sort(v);
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// RTD_STREAM_OVERHEAD=1 runs it on data/artifacts/gross_choi_p0.003_r48 and its shots: per window,
// the wall time of decode_next minus the time inside the inner decoder, first with the relay
// decoder (cold caches after a real decode), then replaying the same results (warm, the stream
// layer alone).
TEST(WindowStreamGross, StreamLayerOverheadR48) {
    if (std::getenv("RTD_STREAM_OVERHEAD") == nullptr) {
        GTEST_SKIP() << "set RTD_STREAM_OVERHEAD=1 to measure";
    }
    const fs::path data = fixture_root / ".." / ".." / "data";
    auto artifact = io::load_artifact(data / "artifacts" / "gross_choi_p0.003_r48");
    ASSERT_TRUE(artifact) << io::describe(artifact.error());
    auto shots = io::load_shots(data / "shots" / "gross_choi_p0.003_r48", *artifact);
    ASSERT_TRUE(shots) << io::describe(shots.error());
    const ArtifactProblem source(*artifact);
    auto plan = WindowPlan::build(source.problem(), spec(12, 8, 12));
    ASSERT_TRUE(plan) << describe(plan.error());
    std::vector<UniformGammaGenerator> gammas;
    for (const Shape& shape : plan->shapes()) {
        gammas.push_back(*UniformGammaGenerator::create(1, -0.24, 0.66, shape.num_columns()));
    }
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig relay{
        .pre_iter = 80, .set_max_iter = 60, .num_sets = 600, .stopping = AfterNConverged{5}};
    std::vector<test::InnerCall> calls;
    auto decoder = StreamDecoder<test::Spy<CpuRelayDecoder<F32>>>::create(
        *plan, [&](const Shape& shape) {
            auto backend = CpuBackend<F32>::create(shape.graph(), shape.priors());
            auto d = CpuRelayDecoder<F32>::create(std::move(*backend), min_sum, relay,
                                                  &gammas[shape.index()]);
            return test::Spy<CpuRelayDecoder<F32>>(std::move(*d), shape.index(), &calls);
        });
    ASSERT_TRUE(decoder) << describe(decoder.error());
    auto replay = StreamDecoder<ReplayInner>::create(
        *plan, [&](const Shape& shape) { return ReplayInner(shape, &calls); });
    ASSERT_TRUE(replay);

    const std::size_t count = std::min<std::size_t>(shots->count(), 8);
    std::vector<std::vector<double>> cold(plan->num_positions());
    std::vector<std::vector<double>> warm(plan->num_positions());
    std::vector<double> inner_ms;
    const auto time_windows = [](auto& d, std::span<const Bit> sigma, std::uint64_t shot,
                                 std::vector<std::vector<double>>& out) {
        const std::size_t m_round = d.plan().detectors_per_round();
        d.reset(shot);
        const std::uint32_t rounds = d.plan().rounds_total();
        for (std::uint32_t r = 0; r + 1 < rounds; ++r) {
            ASSERT_TRUE(d.push_round(sigma.subspan(r * m_round, m_round)));
        }
        ASSERT_TRUE(d.push_final(sigma.subspan((rounds - 1) * m_round, m_round)));
        while (!d.finished()) {
            const auto start = std::chrono::steady_clock::now();
            auto commit = d.decode_next();
            const auto wall = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            ASSERT_TRUE(commit);
            out[commit->window].push_back(
                wall - static_cast<double>(commit->record.decode_ns) / 1e3);
        }
    };
    for (std::size_t s = 0; s < count; ++s) {
        calls.clear();
        time_windows(*decoder, shots->syndrome(s), s, cold);
        for (const WindowRecord& r : decoder->records()) {
            inner_ms.push_back(static_cast<double>(r.decode_ns) / 1e6);
        }
        for (int rep = 0; rep < 200; ++rep) {
            time_windows(*replay, shots->syndrome(s), s, warm);
        }
    }
    for (std::uint32_t k = 0; k < plan->num_positions(); ++k) {
        const Placement* p = plan->placement(k, 0);
        ASSERT_NE(p, nullptr);
        std::cout << std::format("[ overhead ] window {} (shape {}, {} x {}): stream layer "
                                 "{:.2f} µs after a relay decode, {:.2f} µs replayed\n",
                                 k, p->shape, plan->shape_of(*p).num_rows(),
                                 plan->shape_of(*p).num_columns(), median(cold[k]),
                                 median(warm[k]));
    }
    std::cout << std::format("[ overhead ] {} shots, inner relay decode median {:.2f} ms per "
                             "window\n",
                             count, median(inner_ms));
}

} // namespace
