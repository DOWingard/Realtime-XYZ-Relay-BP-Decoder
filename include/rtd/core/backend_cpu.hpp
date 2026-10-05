#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "rtd/core/arith.hpp"
#include "rtd/core/backend.hpp"
#include "rtd/core/buffer.hpp"
#include "rtd/core/config.hpp"
#include "rtd/core/executor.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/kernels.hpp"
#include "rtd/core/priors.hpp"
#include "rtd/core/trace.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// The CPU implementation of one leg (see LegBackend).
//
// Owns every mutable buffer of one decoder instance, allocated once at construction; a decode
// allocates nothing. The graph and priors are shared, read-only, and must outlive the backend.
//
// Hard decisions are kept as a support list (the external indices of columns with M_j ≤ 0)
// rather than a dense vector: the variable pass appends to it and folds each such column into a
// bitset syndrome, so testing H·ê = σ costs O(|ê|·d) and no pass over the edges, and W(ê) is a
// sum over the sorted support. Dense ê is materialised only when asked for.
template <MessageArithmetic A, class Executor = Serial, class Tracer = NoTrace>
class CpuBackend {
public:
    using arith = A;
    using msg_t = A::msg_t;
    using acc_t = A::acc_t;

    [[nodiscard]] static std::expected<CpuBackend, BackendError>
    create(const TannerGraph& graph, const Priors& priors, Executor executor = Executor{},
           Tracer tracer = Tracer{}) {
        if (priors.size() != graph.num_columns()) {
            return std::unexpected(BackendError{
                .code = BackendError::Code::size_mismatch,
                .detail = std::format("graph has {} columns but priors have {} entries",
                                      graph.num_columns(), priors.size())});
        }
        return CpuBackend(graph, priors, std::move(executor), std::move(tracer));
    }

    CpuBackend(CpuBackend&&) noexcept = default;
    CpuBackend& operator=(CpuBackend&&) noexcept = default;
    CpuBackend(const CpuBackend&) = delete;
    CpuBackend& operator=(const CpuBackend&) = delete;
    ~CpuBackend() = default;

    [[nodiscard]] index_t num_rows() const noexcept { return graph_->num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return graph_->num_columns(); }
    [[nodiscard]] const TannerGraph& graph() const noexcept { return *graph_; }
    [[nodiscard]] unsigned workers() const noexcept { return exec_.size(); }
    [[nodiscard]] Tracer& tracer() noexcept { return tracer_; }

    // Only the rows with mask[i] = 1 then decide convergence; an empty mask (the default) requires
    // every row. Entries past the end of a short mask count as required and entries past row m are
    // ignored. The mask is folded into a bitset once here, so a decode reads it as one AND per
    // 64 rows of the convergence test, and not at all when every row is required.
    void set_convergence_rows(std::span<const Bit> mask) noexcept {
        converge_bits_.fill(~std::uint64_t{0});
        masked_ = false;
        const std::size_t rows = std::min<std::size_t>(mask.size(), num_rows());
        for (std::size_t i = 0; i < rows; ++i) {
            if (mask[i] == 0) {
                converge_bits_[i >> 6U] &= ~(std::uint64_t{1} << (i & 63U));
                masked_ = true;
            }
        }
    }

    void begin(std::span<const Bit> syndrome, bool init_marginals) noexcept {
        std::ranges::fill(syndrome_bits_.span(), std::uint64_t{0});
        for (index_t i = 0; i < num_rows(); ++i) {
            const Bit bit = syndrome[i] != 0 ? Bit{1} : Bit{0};
            syndrome_[i] = bit;
            syndrome_bits_[i >> 6U] |= std::uint64_t{bit} << (i & 63U);
        }
        if (init_marginals) {
            for (index_t c = 0; c < num_columns(); ++c) {
                marginal_[c] = A::widen(lambda_[c]);
            }
        }
        leg_ = 0;
        support_size_ = 0;
        support_sorted_ = true;
        best_size_ = 0;
        best_dense_valid_ = false;
    }

    void set_gamma(std::span<const double> gammas) noexcept {
        const std::span<const index_t> external = graph_->external_columns();
        for (index_t c = 0; c < num_columns(); ++c) {
            gamma_[c] = A::from_gamma(gammas[external[c]]);
        }
    }

    void set_gamma(double gamma) noexcept { gamma_.fill(A::from_gamma(gamma)); }

    [[nodiscard]] LegOutcome run_leg(const LegParams& params) noexcept {
        if (graph_->layout() == EdgeLayout::row_major) {
            return run_leg_for<EdgeLayout::row_major>(params);
        }
        return run_leg_for<EdgeLayout::column_blocked>(params);
    }

    void mark_best() noexcept {
        sort_support();
        std::copy_n(support_.data(), support_size_, best_.data());
        best_size_ = support_size_;
        best_dense_valid_ = false;
    }

    [[nodiscard]] std::span<const Bit> best_hard() noexcept {
        if (!best_dense_valid_) {
            materialise({best_.data(), best_size_}, best_dense_);
            best_dense_valid_ = true;
        }
        return best_dense_.span();
    }

    [[nodiscard]] std::span<const Bit> current_hard() noexcept {
        materialise({support_.data(), support_size_}, current_dense_);
        return current_dense_.span();
    }

    // External indices of the current ê, ascending.
    [[nodiscard]] std::span<const index_t> current_support() noexcept {
        sort_support();
        return {support_.data(), support_size_};
    }

    // External indices of the best ê, ascending.
    [[nodiscard]] std::span<const index_t> best_support() const noexcept {
        return {best_.data(), best_size_};
    }

    void read_marginals(std::span<double> out) const noexcept {
        const std::span<const index_t> external = graph_->external_columns();
        for (index_t c = 0; c < num_columns(); ++c) {
            out[external[c]] = A::to_double(marginal_[c]);
        }
    }

private:
    struct ColumnSlice {
        std::uint32_t run;
        index_t col_begin;
        index_t col_end;
    };

    // Check rows [row_begin, row_end), then update the columns in `ready`, all of whose rows are
    // by then updated.
    struct Step {
        index_t row_begin = 0;
        index_t row_end = 0;
        std::vector<ColumnSlice> ready;
    };

    // One worker's part of an iteration: its steps in order, then (after every worker has finished
    // its steps) the columns that depend on rows other workers update.
    struct Share {
        std::vector<Step> steps;
        std::vector<ColumnSlice> deferred;
        index_t support_offset = 0;
    };

    static constexpr std::size_t words_per_line = cache_line_bytes / sizeof(std::uint64_t);

    // Vector kernels may load up to one vector past the end of the message and per-column
    // arrays; this many entries of slack keep those loads inside the allocation.
    static constexpr std::size_t slack = 64;

    CpuBackend(const TannerGraph& graph, const Priors& priors, Executor executor, Tracer tracer)
        : graph_(&graph), priors_(&priors), exec_(std::move(executor)),
          tracer_(std::move(tracer)),
          lambda_(std::size_t{graph.num_columns()} + 1 + slack, A::max_message()),
          msg_(std::size_t{graph.num_slots()} + slack, A::max_message()),
          marginal_(std::size_t{graph.num_columns()} + slack, A::zero()),
          gamma_(std::size_t{graph.num_columns()} + slack, A::from_gamma(0.0)),
          syndrome_(graph.num_rows(), Bit{0}),
          words_((std::size_t{graph.num_rows()} + 63) / 64),
          words_stride_(round_up(std::max<std::size_t>(words_, 1), words_per_line)),
          syndrome_bits_(words_stride_, 0), converge_bits_(words_stride_, ~std::uint64_t{0}),
          support_(graph.num_columns()),
          best_(graph.num_columns()), best_dense_(graph.num_columns(), Bit{0}),
          current_dense_(graph.num_columns(), Bit{0}) {
        const std::span<const double> llr = priors.llr();
        for (index_t c = 0; c < graph.num_columns(); ++c) {
            lambda_[c] = A::from_llr(llr[graph.external_column(c)]);
        }
        // Padding slots name column n; reading its prior must be neutral for a check update.
        lambda_[graph.num_columns()] = A::max_message();
        if (graph.column_order() == ColumnOrder::wavefront) {
            build_wavefront_shares();
        } else {
            build_flooding_shares();
        }
        index_t offset = 0;
        for (Share& share : shares_) {
            share.support_offset = offset;
            for (const Step& step : share.steps) {
                for (const ColumnSlice& slice : step.ready) {
                    offset += slice.col_end - slice.col_begin;
                }
            }
            for (const ColumnSlice& slice : share.deferred) {
                offset += slice.col_end - slice.col_begin;
            }
        }
        const std::size_t workers = shares_.size();
        partial_bits_ = AlignedBuffer<std::uint64_t>(2 * workers * words_stride_, 0);
        support_counts_ = AlignedBuffer<index_t>(workers * index_stride, 0);
        scratch_stride_ = round_up(std::max<std::size_t>(graph.max_column_degree(), 1),
                                   std::max<std::size_t>(cache_line_bytes / sizeof(acc_t), 1));
        scratch_ = AlignedBuffer<acc_t>(workers * scratch_stride_);
        if (graph.layout() == EdgeLayout::column_blocked) {
            gather_stride_ = round_up(std::size_t{graph.max_row_degree()} + row_alignment_slots,
                                      cache_line_bytes);
            gather_ = AlignedBuffer<msg_t>(workers * gather_stride_, A::max_message());
        }
        if constexpr (Tracer::enabled) {
            trace_marginals_.resize(graph.num_columns());
        }
    }

    // Wavefront order: worker w gets a contiguous range of row blocks with about 1/W of the slots.
    // After checking a block it updates the block's columns whose rows all lie in its own range;
    // a column reaching back into the previous worker's rows is deferred until every worker has
    // finished its blocks. Within a run columns are sorted by first row, so the deferred ones form
    // a prefix.
    // Work of rows [0, i): padded slots in the row-major layout, edges in the blocked one.
    [[nodiscard]] const index_t* row_cost_prefix() const noexcept {
        return graph_->layout() == EdgeLayout::row_major ? graph_->row_major_view().row_slot_begin
                                                         : graph_->column_blocked_view().row_ptr;
    }

    void build_wavefront_shares() {
        const unsigned workers = exec_.size();
        const std::span<const RowBlock> blocks = graph_->row_blocks();
        const std::span<const DegreeRun> runs = graph_->runs();
        const index_t* cost = row_cost_prefix();
        const index_t* column_rows = graph_->column_blocked_view().column_rows;
        // Built in place rather than copied from Share{}: GCC's -Wnull-dereference flags the copy.
        shares_.clear();
        shares_.resize(workers);
        const std::uint64_t slots = cost[graph_->num_rows()];
        std::size_t block = 0;
        for (unsigned w = 0; w < workers; ++w) {
            const std::uint64_t target = slots * (w + 1) / workers;
            const std::size_t first_block = block;
            while (block < blocks.size() &&
                   (w + 1 == workers || block == first_block ||
                    cost[blocks[block].row_end] <= target)) {
                ++block;
            }
            if (first_block == block) {
                continue;
            }
            const index_t own_first_row = blocks[first_block].row_begin;
            for (std::size_t b = first_block; b < block; ++b) {
                Step step{.row_begin = blocks[b].row_begin, .row_end = blocks[b].row_end, .ready = {}};
                for (std::uint32_t r = blocks[b].run_begin; r < blocks[b].run_end; ++r) {
                    const DegreeRun& run = runs[r];
                    index_t split = run.col_begin;
                    if (run.degree > 0) {
                        while (split < run.col_end &&
                               column_rows[run.edge_begin + (split - run.col_begin) * run.degree] <
                                   own_first_row) {
                            ++split;
                        }
                    }
                    if (run.col_begin < split) {
                        shares_[w].deferred.push_back(ColumnSlice{r, run.col_begin, split});
                    }
                    if (split < run.col_end) {
                        step.ready.push_back(ColumnSlice{r, split, run.col_end});
                    }
                }
                shares_[w].steps.push_back(std::move(step));
            }
        }
    }

    // Other orders: the check pass is split into contiguous row ranges of equal slot count, and the
    // variable pass runs after all of it, so any split of the columns is correct. Each run is cut
    // into contiguous pieces, worker w getting as many of its columns as have their first row in
    // w's row range; when the run is ordered by first row (as a DEM's columns nearly are) these
    // are exactly those columns, and a worker's variable pass mostly reads messages its own check
    // pass wrote.
    void build_flooding_shares() {
        const unsigned workers = exec_.size();
        const index_t* cost = row_cost_prefix();
        const index_t* column_rows = graph_->column_blocked_view().column_rows;
        const index_t m = graph_->num_rows();
        shares_.clear();
        shares_.resize(workers);
        std::vector<index_t> row_bounds;
        row_bounds.reserve(std::size_t{workers} + 1);
        row_bounds.push_back(0);
        const std::uint64_t slots = cost[m];
        for (unsigned w = 1; w < workers; ++w) {
            const auto target = static_cast<index_t>(slots * w / workers);
            const auto* bound = std::lower_bound(cost, cost + m, target);
            row_bounds.push_back(std::max(row_bounds.back(), static_cast<index_t>(bound - cost)));
        }
        row_bounds.push_back(m);
        for (unsigned w = 0; w < workers; ++w) {
            shares_[w].steps.push_back(
                Step{.row_begin = row_bounds[w], .row_end = row_bounds[w + 1], .ready = {}});
        }

        const std::span<const DegreeRun> runs = graph_->runs();
        for (std::uint32_t r = 0; r < runs.size(); ++r) {
            const DegreeRun& run = runs[r];
            std::vector<index_t> cuts;
            cuts.reserve(std::size_t{workers} + 1);
            cuts.push_back(run.col_begin);
            for (unsigned w = 1; w < workers; ++w) {
                index_t below = 0;
                if (run.degree == 0) {
                    below = static_cast<index_t>(std::uint64_t{run.size()} * w / workers);
                } else {
                    for (index_t c = run.col_begin; c < run.col_end; ++c) {
                        const index_t home =
                            column_rows[run.edge_begin + (c - run.col_begin) * run.degree];
                        below += home < row_bounds[w] ? 1U : 0U;
                    }
                }
                cuts.push_back(std::max(cuts.back(), run.col_begin + below));
            }
            cuts.push_back(run.col_end);
            for (unsigned w = 0; w < workers; ++w) {
                if (cuts[w] < cuts[w + 1]) {
                    shares_[w].deferred.push_back(ColumnSlice{r, cuts[w], cuts[w + 1]});
                }
            }
        }
    }

    // Partial syndromes are double-buffered by iteration parity: in the wavefront schedule a
    // worker starts writing iteration t+1's bits before slower workers have read iteration t's.
    [[nodiscard]] std::uint64_t* partial_bits(std::size_t worker, std::uint32_t iteration) noexcept {
        return partial_bits_.data() + ((2 * worker + (iteration & 1U)) * words_stride_);
    }

    // H·ê = σ on the convergence rows: the XOR of σ and every worker's partial H·ê, restricted to
    // the mask when one is set, is zero.
    [[nodiscard]] bool syndrome_satisfied(std::uint32_t iteration) const noexcept {
        return masked_ ? residual_zero<true>(iteration) : residual_zero<false>(iteration);
    }

    template <bool Masked>
    [[nodiscard]] bool residual_zero(std::uint32_t iteration) const noexcept {
        const std::size_t workers = shares_.size();
        const std::size_t parity = iteration & 1U;
        for (std::size_t k = 0; k < words_; ++k) {
            std::uint64_t x = syndrome_bits_[k];
            for (std::size_t w = 0; w < workers; ++w) {
                x ^= partial_bits_[((2 * w + parity) * words_stride_) + k];
            }
            if constexpr (Masked) {
                x &= converge_bits_[k];
            }
            if (x != 0) {
                return false;
            }
        }
        return true;
    }

    template <EdgeLayout Layout>
    LegOutcome run_leg_for(const LegParams& params) noexcept {
        const bool unit = A::unit_alpha_is_identity && is_unit_alpha(params.alpha);
        if (unit) {
            return params.use_memory ? run_leg_impl<Layout, true, true>(params)
                                     : run_leg_impl<Layout, true, false>(params);
        }
        return params.use_memory ? run_leg_impl<Layout, false, true>(params)
                                 : run_leg_impl<Layout, false, false>(params);
    }

    template <EdgeLayout Layout, bool UnitAlpha, bool UseMemory>
    LegOutcome run_leg_impl(const LegParams& params) noexcept {
        using kernels::Source;
        constexpr bool row_major = Layout == EdgeLayout::row_major;
        const RowMajorView view = graph_->row_major_view();
        const ColumnBlockedView blocked = graph_->column_blocked_view();
        const std::span<const DegreeRun> runs = graph_->runs();
        const index_t* external = graph_->external_columns().data();
        std::uint32_t iterations = 0;
        bool converged = false;

        auto job = [&](WorkerContext& worker) noexcept {
            const unsigned w = worker.index();
            const Share& share = shares_[w];
            acc_t* scratch = scratch_.data() + (w * scratch_stride_);
            msg_t* gather = gather_.data() + (w * gather_stride_);
            for (std::uint32_t t = 0; t < params.max_iter; ++t) {
                msg_t alpha{};
                if constexpr (!UnitAlpha) {
                    alpha = A::from_alpha(alpha_at(params.alpha, t));
                }
                std::uint64_t* bits = partial_bits(w, t);
                std::fill_n(bits, words_, std::uint64_t{0});
                kernels::HardDecisionSink sink{.support = support_.data() + share.support_offset,
                                               .count = 0,
                                               .syndrome_bits = bits,
                                               .external = external};
                const auto update = [&](const ColumnSlice& slice) {
                    if constexpr (row_major) {
                        kernels::variable_run<A, UseMemory>(view, runs[slice.run], slice.col_begin,
                                                            slice.col_end, lambda_.data(),
                                                            gamma_.data(), marginal_.data(),
                                                            msg_.data(), scratch, sink);
                    } else {
                        kernels::variable_run_blocked<A, UseMemory>(
                            blocked, runs[slice.run], slice.col_begin, slice.col_end,
                            lambda_.data(), gamma_.data(), marginal_.data(), msg_.data(), scratch,
                            sink);
                    }
                };
                const auto check = [&]<Source Src>(const Step& step) {
                    if constexpr (row_major) {
                        kernels::check_rows<A, UnitAlpha, Src>(view, syndrome_.data(), alpha,
                                                               lambda_.data(), msg_.data(),
                                                               step.row_begin, step.row_end);
                    } else {
                        kernels::check_rows_blocked<A, UnitAlpha, Src>(
                            blocked, syndrome_.data(), alpha, lambda_.data(), msg_.data(),
                            step.row_begin, step.row_end, gather);
                    }
                };
                for (const Step& step : share.steps) {
                    if (t == 0) {
                        check.template operator()<Source::priors>(step);
                    } else {
                        check.template operator()<Source::messages>(step);
                    }
                    std::ranges::for_each(step.ready, update);
                }
                worker.sync();
                std::ranges::for_each(share.deferred, update);
                support_counts_[w * index_stride] = sink.count;
                worker.sync();

                const bool satisfied = syndrome_satisfied(t);
                if constexpr (Tracer::enabled) {
                    if (w == 0) {
                        read_marginals(trace_marginals_);
                        tracer_.on_iteration(leg_, t, trace_marginals_);
                    }
                    // The other workers must not start the next variable pass while worker 0
                    // is still reading M.
                    worker.sync();
                }
                if (w == 0) {
                    iterations = t + 1;
                    converged = satisfied;
                }
                if (satisfied) {
                    return;
                }
            }
        };
        exec_.run(job);
        ++leg_;

        // Compact the workers' support lists into one.
        index_t size = 0;
        for (std::size_t w = 0; w < shares_.size(); ++w) {
            const index_t count = support_counts_[w * index_stride];
            const index_t offset = shares_[w].support_offset;
            if (offset != size) {
                std::copy_n(support_.data() + offset, count, support_.data() + size);
            }
            size += count;
        }
        support_size_ = size;
        support_sorted_ = false;

        LegOutcome outcome{.converged = converged,
                           .iterations = iterations,
                           .weight = std::numeric_limits<double>::infinity()};
        if (converged) {
            sort_support();
            outcome.weight = leg_weight({support_.data(), support_size_});
        }
        return outcome;
    }

    // W(ê) of a sorted support: the policy's own rule when it has one (a fixed-point format
    // weighs solutions by the quantised priors it decodes with), else Σ λ_j in double.
    [[nodiscard]] double leg_weight(std::span<const index_t> support) const noexcept {
        if constexpr (requires {
                          { A::solution_weight(priors_->llr(), support) } -> std::same_as<double>;
                      }) {
            return A::solution_weight(priors_->llr(), support);
        } else {
            return kernels::solution_weight(priors_->llr(), support);
        }
    }

    void sort_support() noexcept {
        if (!support_sorted_) {
            std::sort(support_.data(), support_.data() + support_size_);
            support_sorted_ = true;
        }
    }

    static void materialise(std::span<const index_t> support, AlignedBuffer<Bit>& dense) noexcept {
        dense.fill(Bit{0});
        for (const index_t j : support) {
            dense[j] = Bit{1};
        }
    }

    // Workers' support counts live one cache line apart.
    static constexpr std::size_t index_stride = cache_line_bytes / sizeof(index_t);

    const TannerGraph* graph_;
    const Priors* priors_;
    Executor exec_;
    Tracer tracer_;

    AlignedBuffer<msg_t> lambda_;   // [n + 1] internal order; [n] is the padding sentinel
    AlignedBuffer<msg_t> msg_;      // [slots] μ before a check pass, η after it
    AlignedBuffer<acc_t> marginal_; // [n] M
    AlignedBuffer<msg_t> gamma_;    // [n] current leg's γ
    AlignedBuffer<Bit> syndrome_;   // [m] σ
    std::size_t words_;             // 64-bit words of an m-bit set
    std::size_t words_stride_;
    AlignedBuffer<std::uint64_t> syndrome_bits_; // σ as a bitset
    AlignedBuffer<std::uint64_t> converge_bits_; // rows that decide convergence, as a bitset
    bool masked_ = false;                        // some row is excluded by converge_bits_

    std::vector<Share> shares_;
    AlignedBuffer<std::uint64_t> partial_bits_; // per worker: its columns' share of H·ê
    AlignedBuffer<index_t> support_counts_;     // per worker, one cache line apart
    AlignedBuffer<acc_t> scratch_;              // per worker: prefix sums of the dynamic kernel
    std::size_t scratch_stride_ = 0;
    AlignedBuffer<msg_t> gather_;               // per worker: one gathered row (blocked layout)
    std::size_t gather_stride_ = 0;

    AlignedBuffer<index_t> support_; // current ê as external indices
    index_t support_size_ = 0;
    bool support_sorted_ = true;
    AlignedBuffer<index_t> best_; // best ê as external indices, ascending
    index_t best_size_ = 0;
    AlignedBuffer<Bit> best_dense_;
    bool best_dense_valid_ = false;
    AlignedBuffer<Bit> current_dense_;

    std::uint32_t leg_ = 0; // legs run since begin()
    std::vector<double> trace_marginals_;
};

} // namespace rtd
