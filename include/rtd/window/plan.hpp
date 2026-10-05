#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <vector>

#include "rtd/core/graph.hpp"
#include "rtd/core/observables.hpp"
#include "rtd/core/priors.hpp"
#include "rtd/core/types.hpp"
#include "rtd/window/error.hpp"
#include "rtd/window/problem.hpp"
#include "rtd/window/spec.hpp"
#include "rtd/window/time_structure.hpp"

namespace rtd::window {

// Global column index of a uniform-boundary local column that has no counterpart in the true
// problem at a given window position (it would lie before round 1 or past the readout).
inline constexpr index_t virtual_column = std::numeric_limits<index_t>::max();

class PlanBuilder;

// One distinct window problem, shared by every placement whose local content is identical.
//
// Local rows are the window's detectors in ascending global order, so local row i of a placement
// is global row first_row + i. Local columns ascend by the global index of their representative.
// A local column is either one fault or, at the cut edge of a non-final window, a group of faults
// whose supports coincide once their rows past the window are removed: the window cannot tell
// them apart, so it decodes the parity of the group, with prior
// p = p₁(1 − p₂) + p₂(1 − p₁) folded over the members (the probability that an odd number of
// them occurred). Such groups lie in the window's last round and are never committed.
class Shape {
public:
    Shape(Shape&&) noexcept = default;
    Shape& operator=(Shape&&) noexcept = default;
    Shape(const Shape&) = delete;
    Shape& operator=(const Shape&) = delete;
    ~Shape() = default;

    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    // Rounds the shape spans; num_rows() = rounds() · M.
    [[nodiscard]] std::uint32_t rounds() const noexcept { return rounds_; }
    [[nodiscard]] index_t num_rows() const noexcept { return num_rows_; }
    [[nodiscard]] index_t num_columns() const noexcept {
        return static_cast<index_t>(commit_.size());
    }
    [[nodiscard]] index_t num_edges() const noexcept {
        return static_cast<index_t>(col_indices_.size());
    }

    // Local H in CSR form: row offsets [num_rows + 1] and, per row, strictly increasing local
    // columns.
    [[nodiscard]] std::span<const index_t> row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] std::span<const index_t> col_indices() const noexcept { return col_indices_; }
    // Local rows of a local column, ascending: the window syndrome bits that committing the
    // column flips.
    [[nodiscard]] std::span<const index_t> column_rows(index_t column) const noexcept {
        return by_column_.column(column);
    }

    // The Tanner graph of the local H and the local priors, for the shape's inner decoder.
    [[nodiscard]] const TannerGraph& graph() const noexcept { return graph_; }
    [[nodiscard]] const Priors& priors() const noexcept { return priors_; }

    // Per local column: 1 if the window decides it.
    [[nodiscard]] std::span<const Bit> commit() const noexcept { return commit_; }
    // Per local column: bit o set when the column flips logical observable o; 0 for columns the
    // window does not commit.
    [[nodiscard]] std::span<const std::uint64_t> commit_class() const noexcept {
        return commit_class_;
    }
    // Per local row: 1 if the row must satisfy H·ê = σ for the decode to count as converged.
    [[nodiscard]] std::span<const Bit> converge() const noexcept { return converge_; }

    // Local columns standing for two or more faults.
    [[nodiscard]] index_t merged_columns() const noexcept { return merged_columns_; }
    [[nodiscard]] index_t committed_columns() const noexcept { return committed_columns_; }
    // Bytes held by the shape's arrays and graph (an estimate: allocator overhead excluded).
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    friend class PlanBuilder;
    Shape(TannerGraph graph, Priors priors, SparseBinaryMatrix by_column) noexcept;

    std::uint32_t index_ = 0;
    std::uint32_t rounds_ = 0;
    index_t num_rows_ = 0;
    index_t merged_columns_ = 0;
    index_t committed_columns_ = 0;
    std::uint64_t hash_ = 0;
    std::vector<index_t> row_ptr_;
    std::vector<index_t> col_indices_;
    std::vector<double> probabilities_;
    std::vector<Bit> commit_;
    std::vector<std::uint64_t> commit_class_;
    std::vector<Bit> converge_;
    TannerGraph graph_;
    Priors priors_;
    SparseBinaryMatrix by_column_;
};

// One window decode at position `window` (k), deferral attempt `attempt` (a).
struct Placement {
    std::uint32_t window = 0;
    std::uint32_t attempt = 0;
    std::uint32_t shape = 0;
    // t_k = 1 + k·C.
    std::uint32_t first_round = 0;
    // Rounds covered: W + a·C, or Rt − t_k + 1 for a final placement. Under the uniform boundary
    // the rounds past Rt are virtual and read as zero.
    std::uint32_t rounds = 0;
    // Rounds whose faults are committed: C, or all `rounds` for a final placement.
    std::uint32_t commit_rounds = 0;
    // The placement reaches the readout round: it commits all of its columns, every row
    // converges and nothing is merged.
    bool final = false;
    // Global row of local row 0, (t_k − 1)·M.
    index_t first_row = 0;
    // Per local column: the global column it stands for. Exact boundary: the representative
    // (smallest member). Uniform boundary: the global column whose support is the bulk
    // representative's shifted by (t_k − t_1)·M rows, or virtual_column.
    std::vector<index_t> columns;
    // Exact boundary only: CSR over local columns of the global members, ascending.
    std::vector<index_t> members_ptr;
    std::vector<index_t> members;
    // Committed local columns that are virtual (uniform boundary only).
    index_t virtual_committed = 0;
};

struct PlanStats {
    std::uint32_t shapes = 0;
    std::uint32_t placements = 0; // schedule length, deferral attempts included
    std::uint32_t positions = 0;  // window positions K
    index_t merged_columns = 0;   // summed over shapes
    index_t virtual_committed = 0; // summed over placements
    std::size_t memory_bytes = 0;  // estimate, shapes and placements
};

// The immutable description of a sliding-window decode of one problem: the distinct window
// shapes and, for every window position and deferral attempt, which shape it uses and where it
// sits. Shared by const reference across every worker; inner decoders keep pointers to the
// shapes' graphs and priors, which stay in place for the plan's lifetime (moving the plan does
// not move them).
class WindowPlan {
public:
    // Builds the plan for `spec` over `problem`, with every shape's Tanner graph built with
    // `options`. Fails if the problem or the spec is invalid, if the problem has no time
    // structure, if k > 64, or (uniform boundary) if no bulk window exists.
    [[nodiscard]] static std::expected<WindowPlan, PlanError>
    build(const Problem& problem, const WindowSpec& spec, GraphOptions options = {});

    WindowPlan(WindowPlan&&) noexcept = default;
    WindowPlan& operator=(WindowPlan&&) noexcept = default;
    WindowPlan(const WindowPlan&) = delete;
    WindowPlan& operator=(const WindowPlan&) = delete;
    ~WindowPlan() = default;

    [[nodiscard]] const WindowSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::uint32_t rounds_total() const noexcept { return rounds_total_; }
    [[nodiscard]] index_t detectors_per_round() const noexcept { return detectors_per_round_; }
    [[nodiscard]] index_t num_detectors() const noexcept { return num_detectors_; }
    [[nodiscard]] index_t num_columns() const noexcept { return num_columns_; }
    [[nodiscard]] index_t num_observables() const noexcept { return num_observables_; }

    // Window positions K: exact, the number of windows up to the first final one; uniform,
    // ⌈Rt / C⌉.
    [[nodiscard]] std::uint32_t num_positions() const noexcept {
        return static_cast<std::uint32_t>(window_begin_.size() - 1);
    }
    [[nodiscard]] std::span<const Shape> shapes() const noexcept { return shapes_; }
    // Every placement, ordered by window position, then attempt.
    [[nodiscard]] std::span<const Placement> schedule() const noexcept { return schedule_; }
    // Placements available at position `window`: 1 + the deferral attempts it may make.
    [[nodiscard]] std::uint32_t attempts(std::uint32_t window) const noexcept {
        return window + 1 < window_begin_.size() ? window_begin_[window + 1] - window_begin_[window]
                                                 : 0;
    }
    // Placement (window, attempt), or nullptr if it does not exist.
    [[nodiscard]] const Placement* placement(std::uint32_t window,
                                             std::uint32_t attempt) const noexcept {
        return attempt < attempts(window) ? &schedule_[window_begin_[window] + attempt] : nullptr;
    }
    [[nodiscard]] const Shape& shape_of(const Placement& placement) const noexcept {
        return shapes_[placement.shape];
    }

    [[nodiscard]] PlanStats stats() const noexcept;

private:
    friend class PlanBuilder;
    WindowPlan() = default;

    WindowSpec spec_{};
    std::uint32_t rounds_total_ = 0;
    index_t detectors_per_round_ = 0;
    index_t num_detectors_ = 0;
    index_t num_columns_ = 0;
    index_t num_observables_ = 0;
    std::vector<Shape> shapes_;
    std::vector<Placement> schedule_;
    // schedule_ offsets of each window position's attempts, [K + 1].
    std::vector<std::uint32_t> window_begin_{0};
};

} // namespace rtd::window
