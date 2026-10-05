// rtd._native: the in-memory decoding API (rtd::api) for Python. Arrays cross the boundary
// without copies: inputs are read in place from C-contiguous numpy arrays, and every result array
// is a numpy array that owns the C++ buffer the decoder filled. The GIL is released while a batch
// or a window decodes. The Python package rtd wraps this module; its classes are the public API.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rtd/api/batch.hpp"
#include "rtd/api/decoder.hpp"
#include "rtd/api/error.hpp"
#include "rtd/api/problem.hpp"
#include "rtd/api/windowed.hpp"
#include "rtd/core/confidence.hpp"
#include "rtd/harness/confidence_outputs.hpp"
#include "rtd/window/history.hpp"
#include "rtd/window/spec.hpp"

namespace nb = nanobind;

namespace {

using rtd::Bit;
using rtd::index_t;
using rtd::api::ApiError;

template <class T, class Shape>
using Input = nb::ndarray<const T, Shape, nb::c_contig, nb::device::cpu>;
template <class T> using Output = nb::ndarray<nb::numpy, T>;

// Raises the Python exception that fits the error: ValueError for anything the caller passed in
// (spec, problem, input, window plan, γ tables), RuntimeError for failures of the decoder and for
// stream calls out of order.
[[noreturn]] void raise(const ApiError& error) {
    const std::string message = rtd::api::describe(error);
    switch (error.code) {
    case ApiError::Code::invalid_spec:
    case ApiError::Code::invalid_problem:
    case ApiError::Code::plan_rejected:
    case ApiError::Code::gamma_source:
    case ApiError::Code::invalid_input:
        throw nb::value_error(message.c_str());
    case ApiError::Code::construction_failed:
    case ApiError::Code::decode_failed:
    case ApiError::Code::stream_state:
        break;
    }
    throw std::runtime_error(message);
}

template <class T> T unwrap(std::expected<T, ApiError> value) {
    if (!value) {
        raise(value.error());
    }
    return std::move(*value);
}

inline void unwrap(const std::expected<void, ApiError>& value) {
    if (!value) {
        raise(value.error());
    }
}

template <class T> void delete_vector(void* pointer) noexcept {
    std::unique_ptr<std::vector<T>>(static_cast<std::vector<T>*>(pointer)).reset();
}

// A numpy array that takes ownership of `values`; its shape must cover exactly its elements.
template <class T>
Output<T> adopt(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    auto owned = std::make_unique<std::vector<T>>(std::move(values));
    const nb::capsule owner(owned.get(), &delete_vector<T>);
    std::vector<T>* vector = owned.release(); // the capsule owns it from here on
    return Output<T>(vector->data(), shape, owner);
}

// A numpy copy of a view that is only valid until the next call.
template <class T> Output<T> copy_of(std::span<const T> values) {
    return adopt(std::vector<T>(values.begin(), values.end()), {values.size()});
}

template <class T, class Shape> std::span<const T> view(const Input<T, Shape>& array) {
    return {array.data(), array.size()};
}

template <class T>
std::span<const T> optional_view(const std::optional<Input<T, nb::ndim<1>>>& array) {
    return array ? view(*array) : std::span<const T>{};
}

// ---- Spec ----------------------------------------------------------------------------------

struct Spec {
    rtd::harness::DecoderSpec value;
};

nb::dict window_dict(const rtd::harness::DecoderSpec& spec) {
    nb::dict window;
    if (!spec.window.is_sliding()) {
        window["mode"] = "whole_shot";
        return window;
    }
    const rtd::window::WindowSpec& w = spec.window.sliding;
    window["mode"] = "sliding";
    window["width"] = w.width;
    window["commit"] = w.commit;
    window["converge_rounds"] = w.converge_rounds;
    window["boundary"] = std::string(rtd::window::to_string(w.boundary));
    window["on_failure"] = std::string(rtd::window::to_string(w.on_failure));
    window["max_deferrals"] = w.max_deferrals;
    window["iteration_cap"] = w.iteration_cap ? nb::cast(*w.iteration_cap) : nb::none();
    return window;
}

// ---- Problem -------------------------------------------------------------------------------

struct ProblemHandle {
    std::shared_ptr<const rtd::api::Problem> value;
};

ProblemHandle make_problem(const Spec& spec, index_t num_rows, index_t num_columns,
                           index_t num_observables, const Input<index_t, nb::ndim<1>>& h_indptr,
                           const Input<index_t, nb::ndim<1>>& h_indices,
                           const Input<double, nb::ndim<1>>& priors,
                           const Input<index_t, nb::ndim<1>>& a_indptr,
                           const Input<index_t, nb::ndim<1>>& a_indices,
                           const std::optional<Input<std::int32_t, nb::ndim<1>>>& detector_round,
                           const std::optional<Input<Bit, nb::ndim<1>>>& syndrome_bias,
                           const std::optional<Input<Bit, nb::ndim<1>>>& observables_bias) {
    const rtd::api::ProblemArrays arrays{.num_rows = num_rows,
                                         .num_columns = num_columns,
                                         .num_observables = num_observables,
                                         .h_row_ptr = view(h_indptr),
                                         .h_col_indices = view(h_indices),
                                         .priors = view(priors),
                                         .a_row_ptr = view(a_indptr),
                                         .a_col_indices = view(a_indices),
                                         .detector_round = optional_view(detector_round),
                                         .syndrome_bias = optional_view(syndrome_bias),
                                         .observables_bias = optional_view(observables_bias)};
    return ProblemHandle{unwrap(rtd::api::Problem::create(arrays, spec.value.graph))};
}

// ---- Confidence ----------------------------------------------------------------------------

// One decode's confidence under a selection policy, as a dict (the fields of rtd::Confidence;
// gap_state 0 = no solution, 1 = every solution in one class, 2 = the gap is defined).
nb::dict confidence_dict(const rtd::Confidence& c) {
    nb::dict d;
    d["found"] = c.found;
    d["seen"] = c.seen;
    d["distinct"] = c.distinct;
    d["classes"] = c.classes;
    d["best_class"] = c.best_class;
    d["weight"] = c.weight;
    d["gap_state"] = static_cast<unsigned>(c.gap_state);
    d["gap"] = c.gap;
    d["second_class"] = c.second_class;
    d["agreement"] = c.agreement;
    d["first_legs"] = c.first_legs;
    d["first_iterations"] = c.first_iterations;
    d["class_sum_class"] = c.class_sum_class;
    d["class_sum_top"] = c.class_sum_top;
    d["agreement_class"] = c.agreement_class;
    d["q_supp"] = c.q_supp;
    d["q_sum_sq"] = c.q_sum_sq;
    d["q_total"] = c.q_total;
    d["components"] = c.components;
    d["syndrome_ones"] = c.syndrome_ones;
    d["syndrome_rows"] = c.syndrome_rows;
    d["density"] = c.density;
    d["decided_class"] = c.decided_class;
    d["decided_leg"] = c.decided_leg;
    d["score"] = c.score;
    d["low"] = c.low;
    d["stopped_early"] = c.stopped_early;
    d["extra_legs"] = c.extra_legs;
    return d;
}

// The conf_* arrays [shots, K] and, with a history, hist_<signal> [shots, K, L] and hist_state,
// under rtd_decode's names. The history values of every signal live in one buffer laid out
// [shots, K, G, L]; each hist_<signal> is a strided view of it, so nothing is copied.
void add_confidence(nb::dict& out, rtd::harness::ConfidenceOutputs c, std::size_t shots,
                    std::size_t windows) {
    out["conf_found"] = adopt(std::move(c.found), {shots, windows});
    out["conf_seen"] = adopt(std::move(c.seen), {shots, windows});
    out["conf_distinct"] = adopt(std::move(c.distinct), {shots, windows});
    out["conf_classes"] = adopt(std::move(c.classes), {shots, windows});
    out["conf_best_class"] = adopt(std::move(c.best_class), {shots, windows});
    out["conf_second_class"] = adopt(std::move(c.second_class), {shots, windows});
    out["conf_weight"] = adopt(std::move(c.weight), {shots, windows});
    out["conf_gap"] = adopt(std::move(c.gap), {shots, windows});
    out["conf_agreement"] = adopt(std::move(c.agreement), {shots, windows});
    out["conf_gap_state"] = adopt(std::move(c.gap_state), {shots, windows});
    out["conf_first_legs"] = adopt(std::move(c.first_legs), {shots, windows});
    out["conf_first_iterations"] = adopt(std::move(c.first_iterations), {shots, windows});
    out["conf_class_sum_class"] = adopt(std::move(c.class_sum_class), {shots, windows});
    out["conf_class_sum_top"] = adopt(std::move(c.class_sum_top), {shots, windows});
    out["conf_agreement_class"] = adopt(std::move(c.agreement_class), {shots, windows});
    out["conf_q_supp"] = adopt(std::move(c.q_supp), {shots, windows});
    out["conf_q_sum_sq"] = adopt(std::move(c.q_sum_sq), {shots, windows});
    out["conf_q_total"] = adopt(std::move(c.q_total), {shots, windows});
    out["conf_components"] = adopt(std::move(c.components), {shots, windows});
    out["conf_syndrome_ones"] = adopt(std::move(c.syndrome_ones), {shots, windows});
    out["conf_syndrome_rows"] = adopt(std::move(c.syndrome_rows), {shots, windows});
    out["conf_decided_class"] = adopt(std::move(c.decided_class), {shots, windows});
    out["conf_decided_leg"] = adopt(std::move(c.decided_leg), {shots, windows});
    out["conf_extra_legs"] = adopt(std::move(c.extra_legs), {shots, windows});
    out["conf_score"] = adopt(std::move(c.score), {shots, windows});
    out["conf_low"] = adopt(std::move(c.low), {shots, windows});
    out["conf_stopped_early"] = adopt(std::move(c.stopped_early), {shots, windows});
    if (c.sliding) {
        out["conf_low_deferrals"] = adopt(std::move(c.low_deferrals), {shots, windows});
    }
    const std::size_t signals = c.history_signals.size();
    const std::size_t lengths = c.history_lengths;
    if (signals == 0 || lengths == 0) {
        return;
    }
    auto owned = std::make_unique<std::vector<double>>(std::move(c.history));
    const nb::capsule owner(owned.get(), &delete_vector<double>);
    std::vector<double>* history = owned.release(); // the capsule owns it from here on
    const auto per_cell = static_cast<std::int64_t>(signals * lengths);
    const auto per_shot = static_cast<std::int64_t>(windows) * per_cell;
    for (std::size_t g = 0; g < signals; ++g) {
        const std::string name =
            std::string("hist_") + std::string(rtd::window::to_string(c.history_signals[g]));
        out[name.c_str()] = Output<double>(history->data() + (g * lengths), {shots, windows, lengths},
                                           owner, {per_shot, per_cell, 1});
    }
    out["hist_state"] = adopt(std::move(c.history_state), {shots, windows, lengths});
}

// ---- Batches -------------------------------------------------------------------------------

rtd::api::BatchInput batch_input(const Input<Bit, nb::ndim<2>>& detectors, bool bit_packed,
                                 std::uint64_t stream_offset) {
    return {.detectors = view(detectors),
            .shots = detectors.shape(0),
            .row_bytes = detectors.shape(1),
            .bit_packed = bit_packed,
            .stream_offset = stream_offset};
}

// The result arrays under rtd_decode's output names, each a numpy array owning its buffer.
nb::dict result_dict(rtd::api::BatchResult r, bool sliding) {
    const std::size_t shots = r.shots;
    const std::size_t windows = r.windows;
    const std::size_t slots = r.solution_slots;
    nb::dict out;
    out["predicted_observables"] = adopt(std::move(r.predicted), {shots, r.predicted_row_bytes});
    out["success"] = adopt(std::move(r.success), {shots});
    out["iterations"] = adopt(std::move(r.iterations), {shots});
    out["legs"] = adopt(std::move(r.legs), {shots});
    out["best_leg"] = adopt(std::move(r.best_leg), {shots});
    out["weight"] = adopt(std::move(r.weight), {shots});
    out["decode_ns"] = adopt(std::move(r.decode_ns), {shots});
    if (!r.decodings.empty()) {
        out["decodings"] = adopt(std::move(r.decodings), {shots, std::size_t{r.num_columns}});
    }
    if (sliding) {
        out["flagged"] = adopt(std::move(r.flagged), {shots});
        out["win_iterations"] = adopt(std::move(r.win_iterations), {shots, windows});
        out["win_legs"] = adopt(std::move(r.win_legs), {shots, windows});
        out["win_attempts"] = adopt(std::move(r.win_attempts), {shots, windows});
        out["win_converged"] = adopt(std::move(r.win_converged), {shots, windows});
        out["win_cap_hit"] = adopt(std::move(r.win_cap_hit), {shots, windows});
        out["win_weight"] = adopt(std::move(r.win_weight), {shots, windows});
        out["win_committed_weight"] = adopt(std::move(r.win_committed_weight), {shots, windows});
        out["win_unexplained"] = adopt(std::move(r.win_unexplained), {shots, windows});
        out["win_flagged"] = adopt(std::move(r.win_flagged), {shots, windows});
        out["win_virtual"] = adopt(std::move(r.win_virtual), {shots, windows});
        out["win_decode_ns"] = adopt(std::move(r.win_decode_ns), {shots, windows});
        if (!r.commit_ptr.empty()) {
            const std::size_t faults = r.commit_faults.size();
            out["commit_ptr"] = adopt(std::move(r.commit_ptr), {(shots * windows) + 1});
            out["commit_faults"] = adopt(std::move(r.commit_faults), {faults});
        }
    }
    if (slots > 0) {
        out["sol_count"] = adopt(std::move(r.sol_count), {shots, windows});
        out["sol_leg"] = adopt(std::move(r.sol_leg), {shots, windows, slots});
        out["sol_iterations"] = adopt(std::move(r.sol_iterations), {shots, windows, slots});
        out["sol_weight"] = adopt(std::move(r.sol_weight), {shots, windows, slots});
        out["sol_class"] = adopt(std::move(r.sol_class), {shots, windows, slots});
        out["sol_hash"] = adopt(std::move(r.sol_hash), {shots, windows, slots});
        out["sol_size"] = adopt(std::move(r.sol_size), {shots, windows, slots});
        out["returned_class"] = adopt(std::move(r.returned_class), {shots, windows});
    }
    if (r.confidence.enabled) {
        add_confidence(out, std::move(r.confidence), shots, windows);
    }
    return out;
}

rtd::api::BatchOptions batch_options(unsigned workers, bool pack_predictions, bool save_decodings,
                                     bool save_commits) {
    return {.workers = workers,
            .pack_predictions = pack_predictions,
            .save_decodings = save_decodings,
            .save_commits = save_commits};
}

// ---- Whole-shot decoder --------------------------------------------------------------------

struct WholeShot {
    std::unique_ptr<rtd::api::WholeShotDecoder> value;
};

WholeShot make_whole_shot(const ProblemHandle& problem, const Spec& spec, unsigned workers,
                          std::uint32_t record_solutions) {
    return WholeShot{unwrap(rtd::api::WholeShotDecoder::create(
        problem.value, spec.value, {.workers = workers, .record_solutions = record_solutions}))};
}

nb::dict whole_shot_batch(WholeShot& self, const Input<Bit, nb::ndim<2>>& detectors,
                          bool bit_packed, std::uint64_t stream_offset, unsigned workers,
                          bool pack_predictions, bool save_decodings) {
    std::expected<rtd::api::BatchResult, ApiError> result;
    {
        const nb::gil_scoped_release release;
        result = self.value->decode_batch(
            batch_input(detectors, bit_packed, stream_offset),
            batch_options(workers, pack_predictions, save_decodings, false));
    }
    return result_dict(unwrap(std::move(result)), false);
}

// ---- Windowed decoder and streams ----------------------------------------------------------

nb::dict record_dict(const rtd::window::WindowRecord& record) {
    nb::dict d;
    d["window"] = record.window;
    d["shape"] = record.shape == rtd::window::no_shape ? nb::none() : nb::cast(record.shape);
    d["attempts"] = record.attempts;
    d["iterations"] = record.iterations;
    d["legs"] = record.legs;
    d["converged"] = record.converged;
    d["cap_hit"] = record.cap_hit;
    d["flagged"] = record.flagged;
    d["weight"] = record.weight;
    d["committed_weight"] = record.committed_weight;
    d["unexplained"] = record.unexplained;
    d["virtual"] = record.virtual_commits;
    d["returned_class"] = record.returned_class;
    d["decode_ns"] = record.decode_ns;
    d["confidence"] = record.confidence ? nb::object(confidence_dict(*record.confidence)) : nb::none();
    d["low_confidence"] = record.low_confidence;
    d["low_confidence_deferrals"] = record.low_confidence_deferrals;
    return d;
}

nb::dict commit_dict(const rtd::window::Commit& commit) {
    nb::dict d;
    d["window"] = commit.window;
    d["first_round"] = commit.first_round;
    d["rounds"] = commit.rounds;
    d["deferred"] = commit.deferred;
    d["faults"] = copy_of(commit.faults);
    d["frame_delta"] = copy_of(commit.frame_delta);
    d["record"] = record_dict(commit.record);
    const std::size_t stored = commit.solutions.size();
    std::vector<std::uint32_t> leg(stored);
    std::vector<std::uint32_t> iterations(stored);
    std::vector<double> weight(stored);
    std::vector<std::uint64_t> logical_class(stored);
    std::vector<std::uint64_t> hash(stored);
    std::vector<std::uint32_t> size(stored);
    for (std::size_t s = 0; s < stored; ++s) {
        const rtd::SolutionRecord& solution = commit.solutions[s];
        leg[s] = solution.leg;
        iterations[s] = solution.cumulative_iterations;
        weight[s] = solution.weight;
        logical_class[s] = solution.logical_class;
        hash[s] = solution.hash;
        size[s] = solution.size;
    }
    nb::dict solutions;
    solutions["found"] = commit.solutions_found;
    solutions["leg"] = adopt(std::move(leg), {stored});
    solutions["iterations"] = adopt(std::move(iterations), {stored});
    solutions["weight"] = adopt(std::move(weight), {stored});
    solutions["class"] = adopt(std::move(logical_class), {stored});
    solutions["hash"] = adopt(std::move(hash), {stored});
    solutions["size"] = adopt(std::move(size), {stored});
    d["solutions"] = solutions;
    return d;
}

nb::dict summary_dict(const rtd::window::ShotSummary& summary) {
    nb::dict d;
    d["success"] = summary.success;
    d["flagged"] = summary.flagged;
    d["iterations"] = summary.iterations;
    d["legs"] = summary.legs;
    d["weight"] = summary.weight;
    d["frame"] = summary.frame;
    d["decode_ns"] = summary.decode_ns;
    return d;
}

nb::dict plan_dict(const rtd::window::WindowPlan& plan) {
    nb::dict d;
    d["rounds_total"] = plan.rounds_total();
    d["detectors_per_round"] = plan.detectors_per_round();
    d["positions"] = plan.num_positions();
    nb::list shapes;
    for (const rtd::window::Shape& shape : plan.shapes()) {
        nb::dict s;
        s["index"] = shape.index();
        s["rounds"] = shape.rounds();
        s["rows"] = shape.num_rows();
        s["columns"] = shape.num_columns();
        s["edges"] = shape.num_edges();
        s["merged_columns"] = shape.merged_columns();
        s["committed_columns"] = shape.committed_columns();
        shapes.append(s);
    }
    d["shapes"] = shapes;
    nb::list placements;
    for (const rtd::window::Placement& p : plan.schedule()) {
        nb::dict q;
        q["window"] = p.window;
        q["attempt"] = p.attempt;
        q["shape"] = p.shape;
        q["first_round"] = p.first_round;
        q["rounds"] = p.rounds;
        q["commit_rounds"] = p.commit_rounds;
        q["final"] = p.final;
        q["virtual_committed"] = p.virtual_committed;
        placements.append(q);
    }
    d["placements"] = placements;
    const rtd::window::PlanStats stats = plan.stats();
    d["merged_columns"] = stats.merged_columns;
    d["virtual_committed"] = stats.virtual_committed;
    d["memory_bytes_estimate"] = stats.memory_bytes;
    return d;
}

struct Windowed {
    std::unique_ptr<rtd::api::WindowedDecoder> value;
};

Windowed make_windowed(const ProblemHandle& problem, const Spec& spec, unsigned workers,
                       std::uint32_t record_solutions) {
    return Windowed{unwrap(rtd::api::WindowedDecoder::create(
        problem.value, spec.value, {.workers = workers, .record_solutions = record_solutions}))};
}

nb::dict windowed_batch(Windowed& self, const Input<Bit, nb::ndim<2>>& detectors, bool bit_packed,
                        std::uint64_t stream_offset, unsigned workers, bool pack_predictions,
                        bool save_decodings, bool save_commits) {
    std::expected<rtd::api::BatchResult, ApiError> result;
    {
        const nb::gil_scoped_release release;
        result = self.value->decode_batch(
            batch_input(detectors, bit_packed, stream_offset),
            batch_options(workers, pack_predictions, save_decodings, save_commits));
    }
    return result_dict(unwrap(std::move(result)), true);
}

// A stream behind a mutex: a Python caller may push from one thread and decode from another, and
// the stream decoder itself is single-threaded. The GIL is released before the mutex is taken,
// so a push waiting for a decode does not hold up the interpreter.
struct StreamHandle {
    std::unique_ptr<rtd::api::Stream> value;
    std::mutex mutex;
};

std::unique_ptr<StreamHandle> open_stream(const Windowed& self) {
    auto handle = std::make_unique<StreamHandle>();
    handle->value = unwrap(self.value->open_stream());
    return handle;
}

void push(StreamHandle& self, const Input<Bit, nb::ndim<1>>& detectors, bool final) {
    std::expected<void, ApiError> pushed;
    {
        const nb::gil_scoped_release release;
        const std::scoped_lock lock(self.mutex);
        pushed = final ? self.value->push_final(view(detectors))
                       : self.value->push_round(view(detectors));
    }
    unwrap(pushed);
}

nb::dict decode_next(StreamHandle& self) {
    std::expected<rtd::window::Commit, ApiError> commit;
    std::unique_lock<std::mutex> lock;
    {
        const nb::gil_scoped_release release;
        lock = std::unique_lock(self.mutex);
        commit = self.value->decode_next();
    }
    // The commit's views stay valid while the lock is held.
    return commit_dict(unwrap(std::move(commit)));
}

template <class F> auto locked(StreamHandle& self, F&& f) {
    const nb::gil_scoped_release release;
    const std::scoped_lock lock(self.mutex);
    return std::forward<F>(f)(*self.value);
}

// The per-window signal over the last L windows after the stream's last committed window, or
// None when the spec has no history: {"lengths": [...], "state": [L] u8, "<signal>": [L] f64}.
nb::object stream_history(StreamHandle& self) {
    struct Copy {
        std::vector<std::uint32_t> lengths;
        std::vector<rtd::window::HistorySignal> signals;
        std::vector<double> values; // [G, L]
        std::vector<std::uint8_t> state;
    };
    std::optional<Copy> copy = locked(self, [](rtd::api::Stream& s) -> std::optional<Copy> {
        const rtd::window::SignalHistory* history = s.history();
        if (history == nullptr) {
            return std::nullopt;
        }
        Copy c;
        c.lengths.assign(history->lengths().begin(), history->lengths().end());
        c.signals.assign(history->signals().begin(), history->signals().end());
        const std::size_t lengths = c.lengths.size();
        c.values.resize(c.signals.size() * lengths);
        c.state.resize(lengths);
        for (std::size_t g = 0; g < c.signals.size(); ++g) {
            for (std::size_t l = 0; l < lengths; ++l) {
                c.values[(g * lengths) + l] = history->value(g, l);
            }
        }
        for (std::size_t l = 0; l < lengths; ++l) {
            c.state[l] = static_cast<std::uint8_t>(history->state(l));
        }
        return c;
    });
    if (!copy) {
        return nb::none();
    }
    const std::size_t lengths = copy->lengths.size();
    nb::dict d;
    for (std::size_t g = 0; g < copy->signals.size(); ++g) {
        const std::string name(rtd::window::to_string(copy->signals[g]));
        d[name.c_str()] =
            copy_of(std::span<const double>(copy->values).subspan(g * lengths, lengths));
    }
    d["state"] = adopt(std::move(copy->state), {lengths});
    d["lengths"] = adopt(std::move(copy->lengths), {lengths});
    return d;
}

nb::dict stream_records(StreamHandle& self) {
    std::vector<rtd::window::WindowRecord> records = locked(self, [](rtd::api::Stream& s) {
        const auto view = s.records();
        return std::vector<rtd::window::WindowRecord>(view.begin(), view.end());
    });
    nb::list list;
    for (const rtd::window::WindowRecord& record : records) {
        list.append(record_dict(record));
    }
    nb::dict d;
    d["windows"] = list;
    return d;
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,misc-use-internal-linkage)
NB_MODULE(_native, m) {
    m.doc() = "In-memory Relay-BP and sliding-window decoders (rtd::api); use the rtd package.";

    nb::class_<Spec>(m, "Spec")
        .def_prop_ro("sliding", [](const Spec& s) { return s.value.window.is_sliding(); })
        .def_prop_ro("version", [](const Spec& s) { return s.value.version; })
        .def_prop_ro("arithmetic",
                     [](const Spec& s) { return std::string(rtd::harness::to_string(s.value.policy)); })
        .def_prop_ro("selection", [](const Spec& s) { return s.value.selection.has_value(); })
        .def_prop_ro("json", [](const Spec& s) { return s.value.raw.dump(); })
        .def_prop_ro("window", [](const Spec& s) { return window_dict(s.value); })
        .def_prop_ro("team_threads", [](const Spec& s) { return s.value.executor.team_threads; })
        .def_prop_ro("num_sets", [](const Spec& s) { return s.value.relay.num_sets; });
    m.def(
        "parse_spec",
        [](std::string_view text, const std::string& base) {
            return Spec{unwrap(rtd::api::parse_spec(text, std::filesystem::path(base)))};
        },
        nb::arg("text"), nb::arg("base_dir"),
        "Parses a decoder spec (version 2, or 3 with a selection policy); relative gamma paths "
        "resolve against base_dir.");

    nb::class_<ProblemHandle>(m, "Problem")
        .def(nb::new_(&make_problem), nb::arg("spec"), nb::arg("num_rows"), nb::arg("num_columns"),
             nb::arg("num_observables"), nb::arg("h_indptr").noconvert(),
             nb::arg("h_indices").noconvert(), nb::arg("priors").noconvert(),
             nb::arg("a_indptr").noconvert(), nb::arg("a_indices").noconvert(),
             nb::arg("detector_round").noconvert().none() = nb::none(),
             nb::arg("syndrome_bias").noconvert().none() = nb::none(),
             nb::arg("observables_bias").noconvert().none() = nb::none())
        .def_prop_ro("num_rows", [](const ProblemHandle& p) { return p.value->num_rows(); })
        .def_prop_ro("num_columns", [](const ProblemHandle& p) { return p.value->num_columns(); })
        .def_prop_ro("num_observables",
                     [](const ProblemHandle& p) { return p.value->num_observables(); })
        .def_prop_ro("num_edges",
                     [](const ProblemHandle& p) { return p.value->graph().num_edges(); });

    nb::class_<WholeShot>(m, "WholeShotDecoder")
        .def(nb::new_(&make_whole_shot), nb::arg("problem"), nb::arg("spec"), nb::arg("workers"),
             nb::arg("record_solutions"))
        .def_prop_ro("workers", [](const WholeShot& d) { return d.value->workers(); })
        .def_prop_ro("record_solutions",
                     [](const WholeShot& d) { return d.value->record_solutions(); })
        .def("decode_batch", &whole_shot_batch, nb::arg("detectors").noconvert(),
             nb::arg("bit_packed"), nb::arg("stream_offset"), nb::arg("workers"),
             nb::arg("pack_predictions"), nb::arg("save_decodings"));

    nb::class_<StreamHandle>(m, "Stream")
        .def(
            "reset",
            [](StreamHandle& s, std::uint64_t shot) {
                locked(s, [shot](rtd::api::Stream& stream) { stream.reset(shot); });
            },
            nb::arg("shot"))
        .def(
            "push_round",
            [](StreamHandle& s, const Input<Bit, nb::ndim<1>>& bits) { push(s, bits, false); },
            nb::arg("detectors").noconvert())
        .def(
            "push_final",
            [](StreamHandle& s, const Input<Bit, nb::ndim<1>>& bits) { push(s, bits, true); },
            nb::arg("detectors").noconvert())
        .def("window_ready",
             [](StreamHandle& s) {
                 return locked(s, [](rtd::api::Stream& stream) { return stream.window_ready(); });
             })
        .def("decode_next", &decode_next)
        .def("finished",
             [](StreamHandle& s) {
                 return locked(s, [](rtd::api::Stream& stream) { return stream.finished(); });
             })
        .def("closed",
             [](StreamHandle& s) {
                 return locked(s, [](rtd::api::Stream& stream) { return stream.closed(); });
             })
        .def("flagged",
             [](StreamHandle& s) {
                 return locked(s, [](rtd::api::Stream& stream) { return stream.flagged(); });
             })
        .def("rounds_received",
             [](StreamHandle& s) {
                 return locked(s,
                               [](rtd::api::Stream& stream) { return stream.rounds_received(); });
             })
        .def("frame",
             [](StreamHandle& s) {
                 auto bits = locked(s, [](rtd::api::Stream& stream) {
                     const auto frame = stream.frame();
                     return std::vector<Bit>(frame.begin(), frame.end());
                 });
                 const std::size_t k = bits.size();
                 return adopt(std::move(bits), {k});
             })
        .def("predicted",
             [](StreamHandle& s) {
                 auto bits = locked(s, [](rtd::api::Stream& stream) {
                     const auto predicted = stream.predicted();
                     return std::vector<Bit>(predicted.begin(), predicted.end());
                 });
                 const std::size_t k = bits.size();
                 return adopt(std::move(bits), {k});
             })
        .def("summary",
             [](StreamHandle& s) {
                 return summary_dict(
                     locked(s, [](rtd::api::Stream& stream) { return stream.summary(); }));
             })
        .def("records", &stream_records)
        .def("history", &stream_history);

    nb::class_<Windowed>(m, "WindowedDecoder")
        .def(nb::new_(&make_windowed), nb::arg("problem"), nb::arg("spec"), nb::arg("workers"),
             nb::arg("record_solutions"))
        .def_prop_ro("workers", [](const Windowed& d) { return d.value->workers(); })
        .def_prop_ro("record_solutions",
                     [](const Windowed& d) { return d.value->record_solutions(); })
        .def_prop_ro("num_positions",
                     [](const Windowed& d) { return d.value->plan().num_positions(); })
        .def("plan", [](const Windowed& d) { return plan_dict(d.value->plan()); })
        .def("decode_batch", &windowed_batch, nb::arg("detectors").noconvert(),
             nb::arg("bit_packed"), nb::arg("stream_offset"), nb::arg("workers"),
             nb::arg("pack_predictions"), nb::arg("save_decodings"), nb::arg("save_commits"))
        .def("open_stream", &open_stream);
}
