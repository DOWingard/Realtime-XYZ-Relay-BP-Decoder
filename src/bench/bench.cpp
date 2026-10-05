// Benchmarks on a real decoding problem:
//
//   rtd_bench --artifact DIR --shots DIR [google-benchmark flags]
//
// check_pass / variable_pass time one pass of each kernel over the whole graph, from inputs
// captured after a genuine first iteration, for every edge layout and column order.
// iteration times plain min-sum decodes and reports seconds per iteration, per executor.
// relay5_decode times whole XYZ-Relay-BP-5 decodes (γ₀ = 0.125, 80 + 600 × 60 iterations,
// stop after 5 solutions).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <cstring>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/kernels.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/shots.hpp"
#ifdef RTD_BENCH_CAN_PIN
#include "rtd/harness/system.hpp"
#endif

namespace {

using namespace rtd;
using kernels::Source;
namespace fs = std::filesystem;

// Set once from the command line in main(), before any benchmark runs.
struct Inputs {
    fs::path artifact;
    fs::path shots;
};
Inputs& inputs() {
    static Inputs paths;
    return paths;
}

const io::Artifact& artifact_for(EdgeLayout layout, ColumnOrder order) {
    static std::map<std::pair<EdgeLayout, ColumnOrder>, std::unique_ptr<io::Artifact>> cache;
    auto& slot = cache[{layout, order}];
    if (!slot) {
        io::ArtifactOptions options;
        options.graph.layout = layout;
        options.graph.column_order = order;
        options.verify_checksums = false;
        auto loaded = io::load_artifact(inputs().artifact, options);
        if (!loaded) {
            std::println(stderr, "rtd_bench: {}", io::describe(loaded.error()));
            std::exit(2);
        }
        slot = std::make_unique<io::Artifact>(std::move(*loaded));
    }
    return *slot;
}

const io::Shots& shots() {
    static const std::unique_ptr<io::Shots> loaded = [] {
        auto s = io::load_shots(inputs().shots,
                                artifact_for(EdgeLayout::row_major, ColumnOrder::wavefront), false);
        if (!s) {
            std::println(stderr, "rtd_bench: {}", io::describe(s.error()));
            std::exit(2);
        }
        return std::make_unique<io::Shots>(std::move(*s));
    }();
    return *loaded;
}

// Buffers of one decoder, driven through the kernels directly.
struct PassFixture {
    const TannerGraph* graph;
    AlignedBuffer<float> lambda;
    AlignedBuffer<float> msg;
    AlignedBuffer<float> marginal;
    AlignedBuffer<float> gamma;
    AlignedBuffer<Bit> syndrome;
    AlignedBuffer<index_t> support;
    AlignedBuffer<std::uint64_t> bits;
    AlignedBuffer<float> scratch;
    AlignedBuffer<float> gather;
    AlignedBuffer<float> mu;  // messages entering a check pass
    AlignedBuffer<float> eta; // messages entering a variable pass

    explicit PassFixture(const io::Artifact& artifact)
        : graph(&artifact.graph), lambda(artifact.num_columns() + 65, F32::max_message()),
          msg(artifact.graph.num_slots() + 64, F32::max_message()),
          marginal(artifact.num_columns() + 64, 0.0F), gamma(artifact.num_columns() + 64, 0.0F),
          syndrome(artifact.num_detectors(), Bit{0}), support(artifact.num_columns()),
          bits((artifact.num_detectors() + 63) / 64 + 8, 0),
          scratch(artifact.graph.max_column_degree() + 16),
          gather(artifact.graph.max_row_degree() + 64, F32::max_message()) {
        const auto llr = artifact.priors.llr();
        for (index_t c = 0; c < artifact.num_columns(); ++c) {
            lambda[c] = F32::from_llr(llr[artifact.graph.external_column(c)]);
        }
        const auto sigma = shots().syndrome(0);
        std::ranges::copy(sigma, syndrome.begin());
        check(Source::priors);
        variable();
        mu = msg.clone();
        check(Source::messages);
        eta = msg.clone();
    }

    void check(Source source) {
        const index_t m = graph->num_rows();
        if (graph->layout() == EdgeLayout::row_major) {
            const auto view = graph->row_major_view();
            if (source == Source::priors) {
                kernels::check_rows<F32, true, Source::priors>(view, syndrome.data(), 1.0F,
                                                               lambda.data(), msg.data(), 0, m);
            } else {
                kernels::check_rows<F32, true, Source::messages>(view, syndrome.data(), 1.0F,
                                                                 lambda.data(), msg.data(), 0, m);
            }
        } else {
            const auto view = graph->column_blocked_view();
            if (source == Source::priors) {
                kernels::check_rows_blocked<F32, true, Source::priors>(
                    view, syndrome.data(), 1.0F, lambda.data(), msg.data(), 0, m, gather.data());
            } else {
                kernels::check_rows_blocked<F32, true, Source::messages>(
                    view, syndrome.data(), 1.0F, lambda.data(), msg.data(), 0, m, gather.data());
            }
        }
    }

    index_t variable() {
        std::ranges::fill(bits, std::uint64_t{0});
        kernels::HardDecisionSink sink{.support = support.data(),
                                       .count = 0,
                                       .syndrome_bits = bits.data(),
                                       .external = graph->external_columns().data()};
        for (const DegreeRun& run : graph->runs()) {
            if (graph->layout() == EdgeLayout::row_major) {
                kernels::variable_run<F32, false>(graph->row_major_view(), run, run.col_begin,
                                                  run.col_end, lambda.data(), gamma.data(),
                                                  marginal.data(), msg.data(), scratch.data(), sink);
            } else {
                kernels::variable_run_blocked<F32, false>(
                    graph->column_blocked_view(), run, run.col_begin, run.col_end, lambda.data(),
                    gamma.data(), marginal.data(), msg.data(), scratch.data(), sink);
            }
        }
        return sink.count;
    }
};

void set_edge_counters(benchmark::State& state, const TannerGraph& graph) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * graph.num_edges());
    state.counters["edges"] = graph.num_edges();
}

void check_pass(benchmark::State& state, EdgeLayout layout, ColumnOrder order) {
    PassFixture f(artifact_for(layout, order));
    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(f.msg.data(), f.mu.data(), f.msg.size() * sizeof(float));
        state.ResumeTiming();
        f.check(Source::messages);
        benchmark::DoNotOptimize(f.msg.data());
        benchmark::ClobberMemory();
    }
    set_edge_counters(state, *f.graph);
}

void variable_pass(benchmark::State& state, EdgeLayout layout, ColumnOrder order) {
    PassFixture f(artifact_for(layout, order));
    for (auto _ : state) {
        state.PauseTiming();
        std::memcpy(f.msg.data(), f.eta.data(), f.msg.size() * sizeof(float));
        state.ResumeTiming();
        benchmark::DoNotOptimize(f.variable());
        benchmark::ClobberMemory();
    }
    set_edge_counters(state, *f.graph);
}

// Team thread w runs on CPU w; the benchmark thread itself is pinned to CPU 0 in main().
Team pinned_team(unsigned threads) {
#ifdef RTD_BENCH_CAN_PIN
    return Team(threads, [](unsigned w) {
        if (auto pinned = harness::pin_current_thread(w); !pinned) {
            std::println(stderr, "rtd_bench: team thread {}: {}; running unpinned", w,
                         pinned.error());
        }
    });
#else
    return Team(threads);
#endif
}

template <class Executor>
void decode_loop(benchmark::State& state, Executor executor, const MinSumConfig& min_sum,
                 const RelayConfig& relay, const GammaSource* gammas, ColumnOrder order) {
    const io::Artifact& artifact = artifact_for(EdgeLayout::row_major, order);
    auto backend = CpuBackend<F32, Executor>::create(artifact.graph, artifact.priors,
                                                     std::move(executor));
    auto decoder = RelayDecoder<CpuBackend<F32, Executor>>::create(std::move(*backend), min_sum,
                                                                  relay, gammas);
    if (!decoder) {
        state.SkipWithError(decoder.error().detail);
        return;
    }
    const io::Shots& data = shots();
    std::size_t shot = 0;
    double iterations = 0;
    for (auto _ : state) {
        const auto result = decoder->decode(data.syndrome(shot), shot);
        iterations += result->iterations;
        shot = (shot + 1) % std::min<std::size_t>(data.count(), 256);
    }
    state.counters["iterations_per_decode"] =
        benchmark::Counter(iterations, benchmark::Counter::kAvgIterations);
    state.counters["s_per_iteration"] =
        benchmark::Counter(iterations, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

void iteration(benchmark::State& state, unsigned team, ColumnOrder order) {
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = std::nullopt};
    const RelayConfig relay{.pre_iter = 100, .set_max_iter = 0, .num_sets = 0,
                            .stopping = AfterLeg0{}};
    if (team == 0) {
        decode_loop(state, Serial{}, min_sum, relay, nullptr, order);
    } else {
        decode_loop(state, pinned_team(team), min_sum, relay, nullptr, order);
    }
}

void relay5_decode(benchmark::State& state, unsigned team) {
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig relay{.pre_iter = 80, .set_max_iter = 60, .num_sets = 600,
                            .stopping = AfterNConverged{5}};
    const io::Artifact& artifact = artifact_for(EdgeLayout::row_major, ColumnOrder::wavefront);
    const auto gammas = UniformGammaGenerator::create(1, -0.24, 0.66, artifact.num_columns());
    if (team == 0) {
        decode_loop(state, Serial{}, min_sum, relay, &*gammas, ColumnOrder::wavefront);
    } else {
        decode_loop(state, pinned_team(team), min_sum, relay, &*gammas, ColumnOrder::wavefront);
    }
}

std::string_view name(EdgeLayout layout) {
    return layout == EdgeLayout::row_major ? "rows" : "blocked";
}
std::string_view name(ColumnOrder order) {
    switch (order) {
    case ColumnOrder::wavefront:
        return "wavefront";
    case ColumnOrder::degree_classes:
        return "degree";
    case ColumnOrder::natural:
        return "natural";
    }
    return "?";
}

void register_benchmarks() {
    for (const EdgeLayout layout : {EdgeLayout::row_major, EdgeLayout::column_blocked}) {
        for (const ColumnOrder order :
             {ColumnOrder::wavefront, ColumnOrder::degree_classes, ColumnOrder::natural}) {
            if (layout == EdgeLayout::column_blocked && order == ColumnOrder::natural) {
                continue; // rejected by the graph builder
            }
            benchmark::RegisterBenchmark(std::format("check_pass/{}/{}", name(layout), name(order)),
                                         check_pass, layout, order)
                ->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark(
                std::format("variable_pass/{}/{}", name(layout), name(order)), variable_pass,
                layout, order)
                ->Unit(benchmark::kMicrosecond);
        }
    }
    for (const unsigned team : {0U, 2U, 3U, 6U}) {
        const std::string executor = team == 0 ? "serial" : std::format("team{}", team);
        for (const ColumnOrder order : {ColumnOrder::wavefront, ColumnOrder::degree_classes}) {
            benchmark::RegisterBenchmark(std::format("iteration/{}/{}", executor, name(order)),
                                         iteration, team, order)
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime();
        }
        benchmark::RegisterBenchmark(std::format("relay5_decode/{}", executor), relay5_decode, team)
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
    }
}

int run(int argc, char** argv) {
    // Take our own flags out before google-benchmark sees the rest.
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if ((arg == "--artifact" || arg == "--shots") && i + 1 < argc) {
            (arg == "--artifact" ? inputs().artifact : inputs().shots) = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    if (inputs().artifact.empty() || inputs().shots.empty()) {
        std::println(stderr, "usage: rtd_bench --artifact DIR --shots DIR [benchmark flags]");
        return 2;
    }
#ifdef RTD_BENCH_CAN_PIN
    if (auto pinned = harness::pin_current_thread(0); !pinned) {
        std::println(stderr, "rtd_bench: {}; running unpinned", pinned.error());
    }
#endif
    int count = static_cast<int>(rest.size());
    benchmark::Initialize(&count, rest.data());
    if (benchmark::ReportUnrecognizedArguments(count, rest.data())) {
        return 2;
    }
    register_benchmarks();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fputs("rtd_bench: unexpected exception: ", stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\n", stderr);
    } catch (...) {
        std::fputs("rtd_bench: unexpected non-standard exception\n", stderr);
    }
    return 1;
}
