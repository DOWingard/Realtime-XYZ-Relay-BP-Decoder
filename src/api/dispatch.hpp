#pragma once

// Compile-time decoder configurations chosen from a run-time spec: the number format, the
// executor and the solution sink are template parameters of the relay decoder, so a spec selects
// one of forty instantiations (five number formats × two executors × four sinks), the same set
// rtd_decode's harness builds from the same spec.

#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#include "rtd/api/error.hpp"
#include "rtd/core/arith.hpp"
#include "rtd/core/decoder.hpp"
#include "rtd/core/decoder_fixed.hpp"
#include "rtd/core/executor.hpp"
#include "rtd/core/fixed_arith.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/selection.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/harness/recording.hpp"
#include "rtd/harness/spec.hpp"

namespace rtd::api::detail {

// An executor for one decoder: the calling thread, or a team of T threads (unpinned).
template <class Executor> [[nodiscard]] Executor make_executor(unsigned team_threads) {
    if constexpr (std::same_as<Executor, Team>) {
        return Team(team_threads);
    } else {
        return Executor{};
    }
}

// What one decoder's solution sink is built from. `column_class` classifies a solution by the
// XOR of its columns' classes: the problem's observable classes for a whole shot, a window
// shape's commit classes for a window (the logical frame change the window would commit). With
// a selection policy, `llr` and `graph` are those of the problem the decoder solves: the policy's
// weights are Σ λ over a support, and Q_supp splits a support into components on that graph.
struct SinkInputs {
    std::uint32_t record_solutions = 0;
    std::span<const std::uint64_t> column_class;
    const SelectionConfig* selection = nullptr; // null: no selection policy
    std::span<const double> llr;
    const TannerGraph* graph = nullptr;
};

template <class Sink> inline constexpr bool is_selection_sink = false;
template <class Recorder> inline constexpr bool is_selection_sink<SelectionSink<Recorder>> = true;

// The solution sink of one decoder: nothing, a recorder of the first N converged legs, or a
// selection policy that sees every converged leg (and forwards it to a recorder when recording).
template <class Sink>
[[nodiscard]] std::expected<Sink, std::string> make_sink(const SinkInputs& in) {
    if constexpr (std::same_as<Sink, NoSink>) {
        return NoSink{};
    } else if constexpr (std::same_as<Sink, harness::SolutionRecorder>) {
        auto records = RecordingSink::create(in.record_solutions, in.column_class);
        if (!records) {
            return std::unexpected(records.error().detail);
        }
        return harness::SolutionRecorder(*records, false);
    } else {
        static_assert(is_selection_sink<Sink>, "a sink this API does not build");
        using Recorder = std::remove_cvref_t<decltype(std::declval<const Sink&>().recorder())>;
        if (in.selection == nullptr) {
            return std::unexpected(std::string("a selecting sink needs a selection policy"));
        }
        auto state = SelectionState::create(*in.selection, in.column_class, in.llr, in.graph);
        if (!state) {
            return std::unexpected(std::format("selection: {}: {}", to_string(state.error().code),
                                               state.error().detail));
        }
        auto recorder = make_sink<Recorder>(in);
        if (!recorder) {
            return std::unexpected(std::move(recorder.error()));
        }
        return Sink(std::move(*state), std::move(*recorder));
    }
}

// Calls f.template operator()<A, Executor, Sink>() for the configuration `spec` and
// `record_solutions` select: A is the spec's number format, Executor its executor, and Sink
// NoSink or SolutionRecorder without a selection policy, SelectionSink<NoSink> or
// SelectionSink<SolutionRecorder> with one.
template <class F>
[[nodiscard]] auto with_decoder_types(const harness::DecoderSpec& spec,
                                      std::uint32_t record_solutions, F&& f) {
    const bool recording = record_solutions > 0;
    const bool selecting = spec.selection.has_value();
    const auto with_sink = [&]<class A, class Executor>() {
        using Recorder = harness::SolutionRecorder;
        if (selecting) {
            return recording ? f.template operator()<A, Executor, SelectionSink<Recorder>>()
                             : f.template operator()<A, Executor, SelectionSink<NoSink>>();
        }
        return recording ? f.template operator()<A, Executor, Recorder>()
                         : f.template operator()<A, Executor, NoSink>();
    };
    const auto with_executor = [&]<class A>() {
        return spec.executor.team_threads > 0 ? with_sink.template operator()<A, Team>()
                                              : with_sink.template operator()<A, Serial>();
    };
    using Result = decltype(with_executor.template operator()<F32>());
    switch (spec.policy) {
    case harness::Policy::f32:
        return with_executor.template operator()<F32>();
    case harness::Policy::f64:
        return with_executor.template operator()<F64>();
    case harness::Policy::int4_2_8:
        return with_executor.template operator()<Int4_2_8>();
    case harness::Policy::int5_2_8:
        return with_executor.template operator()<Int5_2_8>();
    case harness::Policy::int6_2_8:
        return with_executor.template operator()<Int6_2_8>();
    }
    return Result(std::unexpect,
                  ApiError{.code = ApiError::Code::invalid_spec,
                           .detail = std::format("unknown number format {}",
                                                 static_cast<unsigned>(spec.policy))});
}

// Checks what every decoder built by this API needs of its spec and options. `classes_fit` says
// whether a solution's logical class fits one 64-bit word (at most 64 observables), which
// recording solutions and a selection policy both need.
[[nodiscard]] inline std::expected<void, ApiError>
check_decoder_setup(const harness::DecoderSpec& spec, unsigned workers,
                    std::uint32_t record_solutions, bool classes_fit) {
    if (spec.backend != harness::BackendKind::cpu) {
        return std::unexpected(ApiError{.code = ApiError::Code::invalid_spec,
                                        .detail = "this API runs the cpu backend only"});
    }
    if (workers == 0) {
        return std::unexpected(ApiError{.code = ApiError::Code::invalid_input,
                                        .detail = "a decoder needs at least one worker"});
    }
    if (record_solutions > RecordingSink::max_capacity) {
        return std::unexpected(
            ApiError{.code = ApiError::Code::invalid_input,
                     .detail = std::format("record_solutions must be in [0, {}], got {}",
                                           RecordingSink::max_capacity, record_solutions)});
    }
    if ((record_solutions > 0 || spec.selection) && !classes_fit) {
        return std::unexpected(ApiError{
            .code = ApiError::Code::invalid_input,
            .detail = std::format("{} packs a solution's logical class into 64 bits, and this "
                                  "problem has more than 64 observables",
                                  record_solutions > 0 ? "recording solutions"
                                                       : "the selection policy")});
    }
    return {};
}

} // namespace rtd::api::detail
