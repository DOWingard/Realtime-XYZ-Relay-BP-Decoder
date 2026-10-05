// Fixed-point message formats against f32 on a real decoding problem:
//
//   rtd_bench_fixed --artifact DIR --shots DIR [google-benchmark flags]
//
// iteration/<format> times plain min-sum decodes (100 iterations, α = 1, no memory term) and
// reports seconds per iteration and per edge. relay5/<format> times Relay-BP-5 decodes with the
// FPGA settings of Maurer et al. (α = 1 − 2^−t, γ₀ = 0.125, 80 + 600 × 60 iterations, stop after
// 5 solutions), whose iteration count differs between formats, so seconds per iteration is the
// comparable figure. Everything runs on the calling thread; pin it from outside (taskset).

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtd/core/decoder.hpp"
#include "rtd/core/decoder_fixed.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/io/shots.hpp"

namespace {

using namespace rtd;
namespace fs = std::filesystem;

// One JSON line on stderr, the log format of the rtd tools.
void log_error(std::string_view message, std::string_view detail) {
    std::string escaped;
    for (const char c : detail) {
        if (c == '"' || c == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(c == '\n' ? ' ' : c);
    }
    std::fputs(std::format(R"({{"level": "error", "context": "rtd_bench_fixed", "message": "{}", "error": "{}"}})"
                           "\n",
                           message, escaped)
                   .c_str(),
               stderr);
}

struct Inputs {
    fs::path artifact;
    fs::path shots;
};
Inputs& inputs() {
    static Inputs paths;
    return paths;
}

const io::Artifact& artifact() {
    static const std::unique_ptr<io::Artifact> loaded = [] {
        io::ArtifactOptions options;
        options.graph.layout = EdgeLayout::row_major;
        options.graph.column_order = ColumnOrder::wavefront;
        options.verify_checksums = false;
        auto a = io::load_artifact(inputs().artifact, options);
        if (!a) {
            log_error("artifact not loaded", io::describe(a.error()));
            std::exit(2);
        }
        return std::make_unique<io::Artifact>(std::move(*a));
    }();
    return *loaded;
}

const io::Shots& shots() {
    static const std::unique_ptr<io::Shots> loaded = [] {
        auto s = io::load_shots(inputs().shots, artifact(), false);
        if (!s) {
            log_error("shots not loaded", io::describe(s.error()));
            std::exit(2);
        }
        return std::make_unique<io::Shots>(std::move(*s));
    }();
    return *loaded;
}

// Decodes shots 0..min(count, 256) in turn; the counters give iterations per decode and seconds
// per iteration (and per edge of the Tanner graph).
template <class A>
void decode_loop(benchmark::State& state, const MinSumConfig& min_sum, const RelayConfig& relay,
                 const GammaSource* gammas) {
    const io::Artifact& art = artifact();
    auto backend = CpuBackend<A, Serial>::create(art.graph, art.priors, Serial{});
    if (!backend) {
        state.SkipWithError(backend.error().detail);
        return;
    }
    auto decoder = RelayDecoder<CpuBackend<A, Serial>>::create(std::move(*backend), min_sum, relay,
                                                               gammas);
    if (!decoder) {
        state.SkipWithError(decoder.error().detail);
        return;
    }
    const io::Shots& data = shots();
    const std::size_t rotation = std::min<std::size_t>(data.count(), 256);
    std::size_t shot = 0;
    double iterations = 0;
    for (auto _ : state) {
        const auto result = decoder->decode(data.syndrome(shot), shot);
        if (!result) {
            state.SkipWithError(std::format("decode failed: {} (expected {}, found {})",
                                            to_string(result.error().code),
                                            result.error().expected, result.error().found));
            return;
        }
        iterations += result->iterations;
        shot = (shot + 1) % rotation;
    }
    const auto edges = static_cast<double>(art.graph.num_edges());
    state.counters["iterations_per_decode"] =
        benchmark::Counter(iterations, benchmark::Counter::kAvgIterations);
    state.counters["s_per_iteration"] =
        benchmark::Counter(iterations, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
    state.counters["s_per_edge_update"] = benchmark::Counter(
        iterations * edges, benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

template <class A>
void iteration(benchmark::State& state) {
    const MinSumConfig min_sum{.alpha = ConstantAlpha{1.0}, .gamma0 = std::nullopt};
    const RelayConfig relay{.pre_iter = 100, .set_max_iter = 0, .num_sets = 0,
                            .stopping = AfterLeg0{}};
    decode_loop<A>(state, min_sum, relay, nullptr);
}

template <class A>
void relay5(benchmark::State& state) {
    const MinSumConfig min_sum{.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig relay{.pre_iter = 80, .set_max_iter = 60, .num_sets = 600,
                            .stopping = AfterNConverged{5}};
    const auto gammas = UniformGammaGenerator::create(1, -0.24, 0.66, artifact().num_columns());
    if (!gammas) {
        state.SkipWithError("gamma generator not created");
        return;
    }
    decode_loop<A>(state, min_sum, relay, &*gammas);
}

template <class A>
void register_format() {
    const std::string format{A::name};
    benchmark::RegisterBenchmark(std::format("iteration/{}", format), iteration<A>)
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
    benchmark::RegisterBenchmark(std::format("relay5/{}", format), relay5<A>)
        ->Unit(benchmark::kMillisecond)
        ->UseRealTime();
}

int run(int argc, char** argv) {
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
        log_error("missing arguments",
                  "usage: rtd_bench_fixed --artifact DIR --shots DIR [benchmark flags]");
        return 2;
    }
    int count = static_cast<int>(rest.size());
    benchmark::Initialize(&count, rest.data());
    if (benchmark::ReportUnrecognizedArguments(count, rest.data())) {
        return 2;
    }
    register_format<F32>();
    register_format<Int4_2_8>();
    register_format<Int6_2_8>();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        // Plain writes: formatting could throw again here.
        std::fputs(R"({"level": "error", "context": "rtd_bench_fixed", "message": "unexpected exception", "error": ")",
                   stderr);
        std::fputs(e.what(), stderr);
        std::fputs("\"}\n", stderr);
    } catch (...) {
        std::fputs(
            R"({"level": "error", "context": "rtd_bench_fixed", "message": "unexpected non-standard exception"})"
            "\n",
            stderr);
    }
    return 1;
}
