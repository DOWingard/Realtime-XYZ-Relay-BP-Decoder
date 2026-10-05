// Solution recording in the batch driver: the column classes, the support-keeping sink, the
// recorded arrays against a direct RelayDecoder decode with RecordingSink and against a naive
// recomputation from the saved supports, bit identity of every other output, the writer's dtypes
// and shapes, and the options a whole-shot run rejects.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "harness_support.hpp"
#include "rtd/core/decoder.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/harness/batch.hpp"
#include "rtd/harness/logger.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/spec.hpp"
#include "rtd/harness/writer.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/golden.hpp"
#include "rtd/io/npy.hpp"
#include "rtd/io/shots.hpp"

namespace {

using namespace rtd;
using namespace rtd::harness;
namespace fs = std::filesystem;

std::uint64_t bits_of(double x) { return std::bit_cast<std::uint64_t>(x); }

// ---- Naive recomputations from a support ----------------------------------------------------

// SplitMix64 written out again, independently of the library.
std::uint64_t naive_hash(std::span<const index_t> support) {
    std::uint64_t hash = 0;
    for (const index_t j : support) {
        std::uint64_t x = hash ^ j;
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
        hash = x ^ (x >> 31U);
    }
    return hash;
}

std::vector<Bit> dense(std::span<const index_t> support, index_t n) {
    std::vector<Bit> e(n, 0);
    for (const index_t j : support) {
        e[j] = 1;
    }
    return e;
}

// A·ê from the dense ê, packed with bit o = observable o.
std::uint64_t naive_class(const io::Artifact& artifact, std::span<const index_t> support) {
    std::vector<Bit> flips(artifact.num_observables());
    artifact.observables.apply(dense(support, artifact.num_columns()), flips);
    std::uint64_t mask = 0;
    for (std::size_t o = 0; o < flips.size(); ++o) {
        mask |= std::uint64_t{flips[o]} << o;
    }
    return mask;
}

// W(ê) = Σ λ_j over the support in ascending column order, skipping p = 0 columns.
double naive_weight(const io::Artifact& artifact, std::span<const index_t> support) {
    double weight = 0.0;
    for (const index_t j : support) {
        const double l = artifact.priors.llr()[j];
        if (std::isfinite(l)) {
            weight += l;
        }
    }
    return weight;
}

// H·ê computed column by column.
std::vector<Bit> naive_syndrome(const io::Artifact& artifact, std::span<const index_t> support) {
    std::vector<Bit> syndrome(artifact.num_detectors(), 0);
    for (const index_t j : support) {
        for (const index_t row : artifact.graph.column_rows(artifact.graph.internal_column(j))) {
            syndrome[row] ^= 1U;
        }
    }
    return syndrome;
}

// Column classes built from dense unit vectors, independently of column_classes().
std::vector<std::uint64_t> naive_column_classes(const io::Artifact& artifact) {
    std::vector<std::uint64_t> classes(artifact.num_columns());
    for (index_t j = 0; j < artifact.num_columns(); ++j) {
        const std::vector<index_t> unit{j};
        classes[j] = naive_class(artifact, unit);
    }
    return classes;
}

// ---- The bb18 fixture as a batch run --------------------------------------------------------

struct Fixture {
    std::unique_ptr<io::Artifact> artifact;
    std::unique_ptr<io::Golden> golden;
    std::unique_ptr<io::Shots> shots;
    fs::path root;
};

Fixture load_fixture(const std::string& golden_name, const std::string& tag) {
    Fixture f;
    auto artifact = io::load_artifact(test::fixture_root / "bb18_choi" / "artifact");
    EXPECT_TRUE(artifact) << io::describe(artifact.error());
    if (!artifact) {
        return f;
    }
    f.artifact = std::make_unique<io::Artifact>(std::move(*artifact));
    auto golden = io::load_golden(test::fixture_root / "bb18_choi" / golden_name,
                                  f.artifact->num_detectors(), f.artifact->num_columns());
    EXPECT_TRUE(golden) << io::describe(golden.error());
    if (!golden) {
        return f;
    }
    f.golden = std::make_unique<io::Golden>(std::move(*golden));
    f.root = test::scratch_dir(std::format("rtd_recording_{}_{}", golden_name, tag));
    auto shots = io::load_shots(test::write_shots(*f.artifact, *f.golden, f.root / "shots"),
                                *f.artifact, false);
    EXPECT_TRUE(shots) << io::describe(shots.error());
    if (shots) {
        f.shots = std::make_unique<io::Shots>(std::move(*shots));
    }
    return f;
}

DecoderSpec spec_of(const io::Golden& golden, unsigned team) {
    DecoderSpec spec;
    spec.policy = golden.float_type == "f64" ? Policy::f64 : Policy::f32;
    spec.executor.team_threads = team;
    spec.min_sum = golden.min_sum;
    spec.relay = golden.relay;
    return spec;
}

// Shots decoded per run: enough for every recording path (overflowing slots, non-converged
// shots, relay legs that land on the same solution), few enough for the sanitizer builds.
constexpr std::size_t run_shots = 40;

struct Recording {
    std::uint32_t slots = 0;
    bool supports = false;
    unsigned workers = 1;
    unsigned team = 0;
};

std::expected<ShotResults, HarnessError> run_batch(const Fixture& f, const Recording& r,
                                                   std::ostream& log_out) {
    JsonLogger logger(log_out, Level::warn, "test");
    const DecoderSpec spec = spec_of(*f.golden, r.team);
    RunOptions options{.first = 0,
                       .count = run_shots,
                       .workers = r.workers,
                       .cpus = {},
                       .warmup = 1,
                       .save_decodings = true,
                       .record_solutions = r.slots,
                       .save_solution_supports = r.supports,
                       .save_commits = false};
    const GammaSource* gammas = f.golden->gammas ? &*f.golden->gammas : nullptr;
    auto runner = BatchRunner::create(*f.artifact, *f.shots, spec, gammas, options, logger);
    if (!runner) {
        return std::unexpected(runner.error());
    }
    return runner->run();
}

// What a direct decode with RecordingSink reports for one shot.
struct Direct {
    std::vector<SolutionRecord> records;
    std::uint32_t found = 0;
    std::vector<index_t> returned;
    std::optional<std::uint32_t> best_leg;
};

template <MessageArithmetic A>
std::vector<Direct> direct_decodes(const Fixture& f, std::uint32_t slots,
                                   std::span<const std::uint64_t> column_class) {
    auto backend = CpuBackend<A, Serial>::create(f.artifact->graph, f.artifact->priors);
    EXPECT_TRUE(backend);
    const GammaSource* gammas = f.golden->gammas ? &*f.golden->gammas : nullptr;
    auto decoder = CpuRelayDecoder<A, Serial, RecordingSink>::create(
        std::move(*backend), f.golden->min_sum, f.golden->relay, gammas,
        RecordingSink(slots, column_class));
    EXPECT_TRUE(decoder);
    std::vector<Direct> out;
    for (std::size_t s = 0; s < run_shots; ++s) {
        const auto result = decoder->decode(f.golden->detectors.row(s), s);
        EXPECT_TRUE(result);
        const auto records = decoder->sink().records();
        out.push_back(Direct{.records = {records.begin(), records.end()},
                             .found = decoder->sink().found(),
                             .returned = {result->support.begin(), result->support.end()},
                             .best_leg = result->best_leg});
    }
    return out;
}

// ---- Column classes and the sink ------------------------------------------------------------

TEST(ColumnClasses, PackEachColumnOfA) {
    // A = [[1 0 1 0], [0 0 1 1], [1 0 0 1]]: column j's mask has bit o where A[o][j] = 1.
    const std::vector<index_t> row_ptr{0, 2, 4, 6};
    const std::vector<index_t> cols{0, 2, 2, 3, 0, 3};
    auto a = SparseBinaryMatrix::from_csr(3, 4, row_ptr, cols);
    ASSERT_TRUE(a);
    auto classes = column_classes(*a);
    ASSERT_TRUE(classes) << classes.error();
    EXPECT_EQ(*classes, (std::vector<std::uint64_t>{0b101, 0b000, 0b011, 0b110}));

    // k = 64 is the largest class that fits; observable 63 is the top bit.
    const std::vector<index_t> top_ptr = [] {
        std::vector<index_t> p(65, 0);
        p[64] = 1;
        return p;
    }();
    const std::vector<index_t> top_cols{1};
    auto wide = SparseBinaryMatrix::from_csr(64, 2, top_ptr, top_cols);
    ASSERT_TRUE(wide);
    classes = column_classes(*wide);
    ASSERT_TRUE(classes);
    EXPECT_EQ(*classes, (std::vector<std::uint64_t>{0, std::uint64_t{1} << 63U}));

    const std::vector<index_t> too_ptr(66, 0);
    auto too_wide = SparseBinaryMatrix::from_csr(65, 2, too_ptr, {});
    ASSERT_TRUE(too_wide);
    classes = column_classes(*too_wide);
    ASSERT_FALSE(classes);
    EXPECT_NE(classes.error().find("k = 65"), std::string::npos) << classes.error();
}

TEST(SolutionRecorder, KeepsTheSupportsOfStoredSolutionsOnly) {
    const std::vector<std::uint64_t> masks{0b01, 0b10, 0b11, 0b00, 0b01};
    SolutionRecorder recorder(RecordingSink(2, masks), true);
    EXPECT_TRUE(recorder.keeps_supports());
    const std::vector<index_t> a{0, 3};
    const std::vector<index_t> b{1, 2};
    const std::vector<index_t> c{2};
    recorder.on_decode_begin();
    EXPECT_TRUE(recorder.supports().empty());
    recorder.on_solution({.leg = 0, .cumulative_iterations = 5, .weight = 1.5, .support = a});
    recorder.on_solution({.leg = 2, .cumulative_iterations = 9, .weight = 0.5, .support = b});
    recorder.on_solution({.leg = 3, .cumulative_iterations = 12, .weight = 0.25, .support = c});
    EXPECT_EQ(recorder.found(), 3U);
    ASSERT_EQ(recorder.records().size(), 2U);
    EXPECT_EQ(recorder.records()[1].leg, 2U);
    EXPECT_EQ(recorder.records()[1].logical_class, 0b01U); // 0b10 ^ 0b11
    EXPECT_EQ(recorder.records()[1].hash, solution_hash(b));
    EXPECT_TRUE(std::ranges::equal(recorder.supports(), std::vector<index_t>{0, 3, 1, 2}));
    EXPECT_EQ(recorder.class_of(c), 0b11U);

    // A new decode starts empty.
    recorder.on_decode_begin();
    EXPECT_EQ(recorder.found(), 0U);
    EXPECT_TRUE(recorder.supports().empty());
    recorder.on_solution({.leg = 1, .cumulative_iterations = 3, .weight = 2.0, .support = c});
    EXPECT_TRUE(std::ranges::equal(recorder.supports(), c));

    SolutionRecorder counting(RecordingSink(1, masks), false);
    counting.on_decode_begin();
    counting.on_solution({.leg = 0, .cumulative_iterations = 5, .weight = 1.5, .support = a});
    EXPECT_EQ(counting.records().size(), 1U);
    EXPECT_TRUE(counting.supports().empty());
}

// ---- Recorded arrays against a direct decode and a naive recomputation ----------------------

struct DirectCase {
    std::string golden;
    Recording recording;
};

class RecordingVersusDirect : public testing::TestWithParam<DirectCase> {};

TEST_P(RecordingVersusDirect, RecordsWhatTheSinkSawAndTheSupportsAgree) {
    const DirectCase& c = GetParam();
    const Recording& rec = c.recording;
    const Fixture f = load_fixture(c.golden, std::format("direct_{}_{}_{}_{}", rec.slots,
                                                         rec.supports, rec.workers, rec.team));
    ASSERT_TRUE(f.artifact && f.golden && f.shots);
    std::ostringstream log;
    auto results = run_batch(f, rec, log);
    ASSERT_TRUE(results) << results.error().message;
    EXPECT_EQ(log.str(), "");

    const io::Artifact& artifact = *f.artifact;
    const std::vector<std::uint64_t> classes = naive_column_classes(artifact);
    const std::vector<Direct> direct = f.golden->float_type == "f64"
                                           ? direct_decodes<F64>(f, rec.slots, classes)
                                           : direct_decodes<F32>(f, rec.slots, classes);
    const std::size_t shots = run_shots;
    const std::size_t slots = rec.slots;
    ASSERT_EQ(results->windows, 1U);
    ASSERT_EQ(results->solution_slots, slots);
    ASSERT_EQ(results->sol_count.size(), shots);
    ASSERT_EQ(results->sol_leg.size(), shots * slots);
    if (rec.supports) {
        ASSERT_EQ(results->solsup_ptr.size(), (shots * slots) + 1);
        EXPECT_EQ(results->solsup_ptr.front(), 0U);
        EXPECT_EQ(results->solsup_ptr.back(), results->solsup_idx.size());
    } else {
        EXPECT_TRUE(results->solsup_ptr.empty());
        EXPECT_TRUE(results->solsup_idx.empty());
    }

    std::size_t recorded = 0;
    std::size_t overflowing = 0;
    std::size_t best_checked = 0;
    for (std::size_t s = 0; s < shots; ++s) {
        const Direct& d = direct[s];
        EXPECT_EQ(results->sol_count[s], d.found) << "shot " << s;
        EXPECT_EQ(results->returned_class[s], naive_class(artifact, d.returned)) << "shot " << s;
        overflowing += d.found > slots ? 1U : 0U;
        for (std::size_t t = 0; t < slots; ++t) {
            const std::size_t q = (s * slots) + t;
            if (t >= d.records.size()) {
                EXPECT_EQ(results->sol_weight[q], std::numeric_limits<double>::infinity());
                EXPECT_EQ(results->sol_leg[q] | results->sol_iterations[q] | results->sol_size[q],
                          0U);
                EXPECT_EQ(results->sol_class[q] | results->sol_hash[q], 0U);
                if (rec.supports) {
                    EXPECT_EQ(results->solsup_ptr[q], results->solsup_ptr[q + 1]);
                }
                continue;
            }
            ++recorded;
            const SolutionRecord& r = d.records[t];
            EXPECT_EQ(results->sol_leg[q], r.leg);
            EXPECT_EQ(results->sol_iterations[q], r.cumulative_iterations);
            EXPECT_EQ(bits_of(results->sol_weight[q]), bits_of(r.weight));
            EXPECT_EQ(results->sol_class[q], r.logical_class);
            EXPECT_EQ(results->sol_hash[q], r.hash);
            EXPECT_EQ(results->sol_size[q], r.size);
            if (d.best_leg && *d.best_leg == r.leg) {
                EXPECT_EQ(r.hash, naive_hash(d.returned));
                EXPECT_EQ(results->returned_class[s], r.logical_class);
                ++best_checked;
            }
            if (!rec.supports) {
                continue;
            }
            // The saved support reproduces every recorded field, and it is a solution: H·ê = σ.
            const std::span<const index_t> support =
                std::span(results->solsup_idx)
                    .subspan(results->solsup_ptr[q],
                             results->solsup_ptr[q + 1] - results->solsup_ptr[q]);
            EXPECT_TRUE(std::ranges::is_sorted(support));
            EXPECT_EQ(std::ranges::adjacent_find(support), support.end());
            EXPECT_EQ(support.size(), r.size);
            EXPECT_EQ(naive_hash(support), r.hash);
            EXPECT_EQ(naive_class(artifact, support), r.logical_class);
            EXPECT_EQ(bits_of(naive_weight(artifact, support)), bits_of(r.weight));
            EXPECT_TRUE(
                std::ranges::equal(naive_syndrome(artifact, support), f.golden->detectors.row(s)))
                << "shot " << s << " slot " << t;
        }
    }
    // The fixture must actually exercise the paths above.
    EXPECT_GT(recorded, 0U);
    if (f.golden->relay.num_sets > 0) {
        EXPECT_GT(best_checked, 0U);
    }
    if (c.golden == "relay_all_f32" && slots < 20) {
        EXPECT_GT(overflowing, 0U);
    }
    fs::remove_all(f.root);
}

INSTANTIATE_TEST_SUITE_P(
    Fixture, RecordingVersusDirect,
    testing::Values(
        DirectCase{"relay_f32", {.slots = 5, .supports = true, .workers = 1, .team = 0}},
        DirectCase{"relay_f32", {.slots = 1, .supports = true, .workers = 3, .team = 0}},
        DirectCase{"relay_f64", {.slots = 3, .supports = true, .workers = 1, .team = 2}},
        DirectCase{"relay_all_f32", {.slots = 4, .supports = true, .workers = 3, .team = 0}},
        DirectCase{"relay_all_f32", {.slots = 20, .supports = false, .workers = 1, .team = 2}},
        DirectCase{"min_sum_f32", {.slots = 2, .supports = true, .workers = 2, .team = 0}}),
    [](const testing::TestParamInfo<DirectCase>& param_info) {
        const DirectCase& c = param_info.param;
        return std::format("{}_n{}{}_w{}_t{}", c.golden, c.recording.slots,
                           c.recording.supports ? "s" : "", c.recording.workers, c.recording.team);
    });

// ---- Recording changes nothing else ---------------------------------------------------------

TEST(Recording, LeavesEveryPerShotOutputBitIdentical) {
    for (const char* golden : {"relay_all_f32", "relay_f64", "min_sum_f32"}) {
        const Fixture f = load_fixture(golden, "identity");
        ASSERT_TRUE(f.artifact && f.golden && f.shots);
        std::ostringstream log;
        auto off = run_batch(f, {.slots = 0, .supports = false, .workers = 1, .team = 0}, log);
        ASSERT_TRUE(off) << off.error().message;
        EXPECT_TRUE(off->sol_count.empty());
        EXPECT_EQ(off->solution_slots, 0U);
        for (const Recording& on :
             {Recording{.slots = 1, .supports = false, .workers = 2, .team = 0},
              Recording{.slots = 5, .supports = true, .workers = 3, .team = 0},
              Recording{.slots = 20, .supports = true, .workers = 1, .team = 2}}) {
            auto rec = run_batch(f, on, log);
            ASSERT_TRUE(rec) << rec.error().message;
            const auto weights = [](const ShotResults& r) {
                std::vector<std::uint64_t> w;
                std::ranges::transform(r.weight, std::back_inserter(w), bits_of);
                return w;
            };
            EXPECT_EQ(rec->success, off->success) << golden;
            EXPECT_EQ(rec->iterations, off->iterations) << golden;
            EXPECT_EQ(rec->legs, off->legs) << golden;
            EXPECT_EQ(rec->best_leg, off->best_leg) << golden;
            EXPECT_EQ(weights(*rec), weights(*off)) << golden;
            EXPECT_EQ(rec->predicted, off->predicted) << golden;
            EXPECT_EQ(rec->logical_failure, off->logical_failure) << golden;
            EXPECT_EQ(rec->decodings, off->decodings) << golden;
        }
        EXPECT_EQ(log.str(), "");
        fs::remove_all(f.root);
    }
}

// ---- The writer -----------------------------------------------------------------------------

std::set<std::string> files_in(const fs::path& dir) {
    std::set<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir)) {
        names.insert(entry.path().filename().string());
    }
    return names;
}

const std::set<std::string> per_shot_files{"success.npy",
                                           "iterations.npy",
                                           "legs.npy",
                                           "best_leg.npy",
                                           "weight.npy",
                                           "decode_ns.npy",
                                           "predicted_observables.npy",
                                           "logical_failure.npy",
                                           "decodings.npy",
                                           "run.json"};

const std::set<std::string> solution_files{
    "sol_count.npy", "sol_leg.npy",  "sol_iterations.npy", "sol_weight.npy",
    "sol_class.npy", "sol_hash.npy", "sol_size.npy",       "returned_class.npy"};

TEST(Recording, WriterWritesEveryArrayWithItsDtypeAndShape) {
    const Fixture f = load_fixture("relay_all_f32", "writer");
    ASSERT_TRUE(f.artifact && f.golden && f.shots);
    std::ostringstream log;
    const std::size_t shots = run_shots;
    constexpr std::uint32_t slots = 4;
    for (const bool supports : {true, false}) {
        auto results =
            run_batch(f, {.slots = slots, .supports = supports, .workers = 2, .team = 0}, log);
        ASSERT_TRUE(results) << results.error().message;
        const fs::path out = test::scratch_dir(std::format("rtd_recording_writer_{}", supports));
        ASSERT_TRUE(write_results(out, *results, {{"summary", nullptr}}));

        std::set<std::string> expected = per_shot_files;
        expected.insert(solution_files.begin(), solution_files.end());
        if (supports) {
            expected.insert({"solsup_ptr.npy", "solsup_idx.npy"});
        }
        EXPECT_EQ(files_in(out), expected);

        const std::vector<std::size_t> cells{shots, 1};
        const std::vector<std::size_t> slot_shape{shots, 1, slots};
        EXPECT_EQ(test::read_array<std::uint32_t>(out / "sol_count.npy", cells),
                  results->sol_count);
        EXPECT_EQ(test::read_array<std::uint32_t>(out / "sol_leg.npy", slot_shape),
                  results->sol_leg);
        EXPECT_EQ(test::read_array<std::uint32_t>(out / "sol_iterations.npy", slot_shape),
                  results->sol_iterations);
        const std::vector<double> weight =
            test::read_array<double>(out / "sol_weight.npy", slot_shape);
        ASSERT_EQ(weight.size(), results->sol_weight.size());
        for (std::size_t q = 0; q < weight.size(); ++q) {
            EXPECT_EQ(bits_of(weight[q]), bits_of(results->sol_weight[q]));
        }
        EXPECT_EQ(test::read_array<std::uint64_t>(out / "sol_class.npy", slot_shape),
                  results->sol_class);
        EXPECT_EQ(test::read_array<std::uint64_t>(out / "sol_hash.npy", slot_shape),
                  results->sol_hash);
        EXPECT_EQ(test::read_array<std::uint32_t>(out / "sol_size.npy", slot_shape),
                  results->sol_size);
        EXPECT_EQ(test::read_array<std::uint64_t>(out / "returned_class.npy", cells),
                  results->returned_class);
        if (supports) {
            const auto ptr =
                test::read_array<std::uint64_t>(out / "solsup_ptr.npy", {(shots * slots) + 1});
            EXPECT_EQ(ptr, results->solsup_ptr);
            EXPECT_TRUE(std::ranges::is_sorted(ptr));
            EXPECT_EQ(test::read_array<std::uint32_t>(out / "solsup_idx.npy",
                                                      {results->solsup_idx.size()}),
                      results->solsup_idx);
            EXPECT_GT(results->solsup_idx.size(), 0U);
        }
        fs::remove_all(out);
    }

    // Without recording the output set is exactly the one before recording existed.
    auto plain = run_batch(f, {.slots = 0, .supports = false, .workers = 1, .team = 0}, log);
    ASSERT_TRUE(plain);
    const fs::path out = test::scratch_dir("rtd_recording_writer_off");
    ASSERT_TRUE(write_results(out, *plain, {{"summary", nullptr}}));
    EXPECT_EQ(files_in(out), per_shot_files);
    fs::remove_all(out);
    fs::remove_all(f.root);
}

TEST(Recording, WriterHandlesRunsWithoutAnySolution) {
    ShotResults results;
    results.count = 2;
    results.num_observables = 1;
    results.success = {0, 0};
    results.iterations = {3, 4};
    results.legs = {1, 1};
    results.best_leg = {-1, -1};
    results.weight = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
    results.decode_ns = {1, 2};
    results.predicted = {0, 1};
    results.logical_failure = {0, 1};
    results.windows = 1;
    results.solution_slots = 3;
    results.sol_count = {0, 0};
    results.returned_class = {0, 1};
    results.sol_leg.assign(6, 0);
    results.sol_iterations.assign(6, 0);
    results.sol_weight.assign(6, std::numeric_limits<double>::infinity());
    results.sol_class.assign(6, 0);
    results.sol_hash.assign(6, 0);
    results.sol_size.assign(6, 0);
    results.solsup_ptr.assign(7, 0);
    const fs::path out = test::scratch_dir("rtd_recording_writer_empty");
    ASSERT_TRUE(write_results(out, results, {{"summary", nullptr}}));
    EXPECT_EQ(test::read_array<std::uint64_t>(out / "solsup_ptr.npy", {7}), results.solsup_ptr);
    EXPECT_TRUE(test::read_array<std::uint32_t>(out / "solsup_idx.npy", {0}).empty());
    EXPECT_EQ(test::read_array<std::uint64_t>(out / "returned_class.npy", {2, 1}),
              results.returned_class);

    const nlohmann::json summary = summarize(results, std::nullopt, std::nullopt);
    EXPECT_EQ(summary["solutions"]["slots"], 3);
    EXPECT_EQ(summary["solutions"]["decodes_with_unrecorded_solutions"], 0);
    EXPECT_DOUBLE_EQ(summary["solutions"]["found"]["max"].get<double>(), 0.0);
    fs::remove_all(out);
}

// ---- What a whole-shot run rejects -----------------------------------------------------------

TEST(Recording, UnsupportedOptionsAreRejectedBeforeDecoding) {
    const Fixture f = load_fixture("relay_f32", "reject");
    ASSERT_TRUE(f.artifact && f.golden && f.shots);
    std::ostringstream log;
    JsonLogger logger(log, Level::error, "test");
    const auto create = [&](const DecoderSpec& spec, RunOptions options) {
        options.count = 10;
        const GammaSource* gammas = f.golden->gammas ? &*f.golden->gammas : nullptr;
        return BatchRunner::create(*f.artifact, *f.shots, spec, gammas, std::move(options), logger);
    };
    const DecoderSpec whole = spec_of(*f.golden, 0);
    const auto message_of = [&](const DecoderSpec& spec, const RunOptions& options) -> std::string {
        auto checked = check_supported(spec, options);
        EXPECT_FALSE(checked);
        auto runner = create(spec, options);
        EXPECT_FALSE(runner);
        if (checked || runner) {
            return "";
        }
        EXPECT_EQ(runner.error().message, checked.error().message);
        return checked.error().message;
    };

    RunOptions too_many;
    too_many.record_solutions = RecordingSink::max_capacity + 1;
    EXPECT_NE(message_of(whole, too_many).find("[0, 20]"), std::string::npos);

    RunOptions supports_alone;
    supports_alone.save_solution_supports = true;
    EXPECT_NE(message_of(whole, supports_alone).find("--record-solutions"), std::string::npos);

    RunOptions commits;
    commits.save_commits = true;
    EXPECT_NE(message_of(whole, commits).find("sliding"), std::string::npos);

    DecoderSpec sliding = whole;
    sliding.window.mode = WindowSettings::Mode::sliding;
    sliding.window.sliding = window::WindowSpec{.width = 3,
                                                .commit = 1,
                                                .converge_rounds = 3,
                                                .boundary = window::Boundary::exact,
                                                .on_failure = window::OnFailure::flag,
                                                .max_deferrals = 0,
                                                .iteration_cap = std::nullopt};
    // A sliding spec is supported, but the runner needs its window plan (build_sliding).
    EXPECT_TRUE(check_supported(sliding, commits));
    auto without_plan = create(sliding, RunOptions{});
    ASSERT_FALSE(without_plan);
    EXPECT_NE(without_plan.error().message.find("window plan"), std::string::npos)
        << without_plan.error().message;

    RunOptions fine;
    fine.record_solutions = RecordingSink::max_capacity;
    fine.save_solution_supports = true;
    EXPECT_TRUE(check_supported(whole, fine));
    EXPECT_TRUE(create(whole, fine));
    fs::remove_all(f.root);
}

} // namespace
