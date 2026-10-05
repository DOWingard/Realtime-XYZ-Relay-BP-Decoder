#include "rtd/window/plan.hpp"

#include <algorithm>
#include <bit>
#include <compare>
#include <format>
#include <string>
#include <utility>

namespace rtd::window {

namespace {

constexpr index_t none = std::numeric_limits<index_t>::max();

std::unexpected<PlanError> fail(PlanError::Code code, std::string detail) {
    return std::unexpected(PlanError{.code = code, .detail = std::move(detail)});
}

// SplitMix64 finaliser; hashes a shape's content so that identical windows are found without
// comparing every pair in full.
constexpr std::uint64_t mix(std::uint64_t h, std::uint64_t x) noexcept {
    std::uint64_t z = h ^ (x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2));
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template <class T>
std::uint64_t mix_all(std::uint64_t h, std::span<const T> values) noexcept {
    h = mix(h, values.size());
    for (const T& v : values) {
        if constexpr (std::is_same_v<T, double>) {
            h = mix(h, std::bit_cast<std::uint64_t>(v));
        } else {
            h = mix(h, static_cast<std::uint64_t>(v));
        }
    }
    return h;
}

bool bitwise_equal(std::span<const double> a, std::span<const double> b) noexcept {
    return std::ranges::equal(a, b, [](double x, double y) {
        return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y);
    });
}

std::size_t graph_bytes(const TannerGraph& g) noexcept {
    const std::size_t m = g.num_rows();
    const std::size_t n = g.num_columns();
    const std::size_t e = g.num_edges();
    // Offsets, column view, per-slot or per-edge maps and the two column numberings.
    const std::size_t words = g.layout() == EdgeLayout::row_major
                                  ? (m + 1) + (n + 1) + 2 * e + (m + 1) + g.num_slots() + 2 * n
                                  : (m + 1) + (n + 1) + 3 * e + 2 * n;
    return words * sizeof(index_t) + g.runs().size() * sizeof(DegreeRun) +
           g.row_blocks().size() * sizeof(RowBlock);
}

template <class T>
std::size_t vector_bytes(const std::vector<T>& v) noexcept {
    return v.size() * sizeof(T);
}

} // namespace

Shape::Shape(TannerGraph graph, Priors priors, SparseBinaryMatrix by_column) noexcept
    : graph_(std::move(graph)), priors_(std::move(priors)), by_column_(std::move(by_column)) {}

std::size_t Shape::memory_bytes() const noexcept {
    const std::size_t n = num_columns();
    const std::size_t e = num_edges();
    return vector_bytes(row_ptr_) + vector_bytes(col_indices_) + vector_bytes(probabilities_) +
           vector_bytes(commit_) + vector_bytes(commit_class_) + vector_bytes(converge_) +
           2 * n * sizeof(double) + ((n + 1) + e) * sizeof(index_t) + graph_bytes(graph_);
}

PlanStats WindowPlan::stats() const noexcept {
    PlanStats stats{.shapes = static_cast<std::uint32_t>(shapes_.size()),
                    .placements = static_cast<std::uint32_t>(schedule_.size()),
                    .positions = num_positions(),
                    .merged_columns = 0,
                    .virtual_committed = 0,
                    .memory_bytes = 0};
    for (const Shape& shape : shapes_) {
        stats.merged_columns += shape.merged_columns();
        stats.memory_bytes += shape.memory_bytes();
    }
    for (const Placement& p : schedule_) {
        stats.virtual_committed += p.virtual_committed;
        stats.memory_bytes +=
            vector_bytes(p.columns) + vector_bytes(p.members_ptr) + vector_bytes(p.members);
    }
    return stats;
}

// Builds a WindowPlan: cuts every placement's local problem out of the global one, matches it
// against the shapes found so far and records where it sits.
class PlanBuilder {
public:
    PlanBuilder(const Problem& problem, const WindowSpec& spec, TimeStructure time,
                GraphOptions options)
        : problem_(&problem), spec_(spec), time_(std::move(time)), options_(options) {}

    [[nodiscard]] std::expected<WindowPlan, PlanError> build() {
        transpose();
        plan_.spec_ = spec_;
        plan_.rounds_total_ = time_.rounds_total();
        plan_.detectors_per_round_ = time_.detectors_per_round();
        plan_.num_detectors_ = problem_->num_rows;
        plan_.num_columns_ = problem_->num_columns;
        plan_.num_observables_ = problem_->num_observables;
        auto built = spec_.boundary == Boundary::exact ? build_exact() : build_uniform();
        if (!built) {
            return std::unexpected(std::move(built.error()));
        }
        return std::move(plan_);
    }

private:
    // The rounds of one placement and what it commits and requires to converge.
    struct Cut {
        std::uint32_t first_round;
        std::uint32_t rounds;
        std::uint32_t commit_rounds;
        std::uint32_t converge_rounds;
        bool final;
    };

    // A placement's local problem before it is matched against the shapes.
    struct Local {
        std::uint32_t rounds = 0;
        index_t num_rows = 0;
        std::vector<index_t> row_ptr;
        std::vector<index_t> col_indices;
        std::vector<double> probabilities;
        std::vector<Bit> commit;
        std::vector<std::uint64_t> commit_class;
        std::vector<Bit> converge;
        std::vector<index_t> representatives;
        std::vector<index_t> members_ptr;
        std::vector<index_t> members;
        index_t merged = 0;
        std::uint64_t hash = 0;
    };

    [[nodiscard]] std::span<const index_t> global_rows(index_t column) const noexcept {
        return {col_rows_.data() + col_ptr_[column], col_ptr_[column + 1] - col_ptr_[column]};
    }

    // Column view of H (rows ascending per column) and each column's observable mask.
    void transpose() {
        const index_t m = problem_->num_rows;
        const index_t n = problem_->num_columns;
        col_ptr_.assign(std::size_t{n} + 1, 0);
        col_rows_.resize(problem_->h_col_indices.size());
        for (const index_t j : problem_->h_col_indices) {
            ++col_ptr_[j + 1];
        }
        for (index_t j = 0; j < n; ++j) {
            col_ptr_[j + 1] += col_ptr_[j];
        }
        std::vector<index_t> fill(col_ptr_.begin(), col_ptr_.end() - 1);
        for (index_t i = 0; i < m; ++i) {
            for (index_t e = problem_->h_row_ptr[i]; e < problem_->h_row_ptr[i + 1]; ++e) {
                col_rows_[fill[problem_->h_col_indices[e]]++] = i;
            }
        }
        col_class_.assign(n, 0);
        for (index_t o = 0; o < problem_->num_observables; ++o) {
            for (index_t e = problem_->a_row_ptr[o]; e < problem_->a_row_ptr[o + 1]; ++e) {
                col_class_[problem_->a_col_indices[e]] |= std::uint64_t{1} << o;
            }
        }
        local_of_.assign(n, none);
        group_rep_.assign(n, none);
    }

    // Rows of column j before row `row_end`, i.e. inside a window whose last row is row_end − 1.
    [[nodiscard]] std::span<const index_t> rows_before(index_t j, index_t row_end) const noexcept {
        const auto rows = global_rows(j);
        const auto inside = std::ranges::lower_bound(rows, row_end);
        return rows.first(static_cast<std::size_t>(inside - rows.begin()));
    }

    // Groups the columns with s(j) = last by their rows inside the window. Sorted by that
    // restricted support, then by index, each run of equal supports starts with its smallest
    // member, which becomes the representative recorded in group_rep_. Returns the grouped
    // columns so the scratch can be cleared.
    [[nodiscard]] std::vector<index_t> group_cut_round(std::uint32_t last, index_t row_end) {
        const std::span<const std::uint32_t> s = time_.first_rounds();
        std::vector<index_t> cut;
        for (index_t j = 0; j < problem_->num_columns; ++j) {
            if (s[j] == last) {
                cut.push_back(j);
            }
        }
        std::ranges::sort(cut, [&](index_t a, index_t b) {
            const auto ra = rows_before(a, row_end);
            const auto rb = rows_before(b, row_end);
            const auto order =
                std::lexicographical_compare_three_way(ra.begin(), ra.end(), rb.begin(), rb.end());
            return std::is_neq(order) ? std::is_lt(order) : a < b;
        });
        for (std::size_t g = 0; g < cut.size();) {
            const index_t rep = cut[g];
            const auto rep_rows = rows_before(rep, row_end);
            for (; g < cut.size() && std::ranges::equal(rows_before(cut[g], row_end), rep_rows);
                 ++g) {
                group_rep_[cut[g]] = rep;
            }
        }
        return cut;
    }

    // Numbers the columns with t ≤ s(j) ≤ last in ascending order of representative (in local_of_)
    // and lists their representatives. A group's representative is its smallest member, so it is
    // numbered before any other member is reached. Returns every window column, ascending.
    [[nodiscard]] std::vector<index_t> number_columns(Local& local, std::uint32_t t,
                                                      std::uint32_t last) {
        const std::span<const std::uint32_t> s = time_.first_rounds();
        std::vector<index_t> window_columns;
        for (index_t j = 0; j < problem_->num_columns; ++j) {
            if (s[j] < t || s[j] > last) {
                continue;
            }
            window_columns.push_back(j);
            const index_t rep = group_rep_[j];
            if (rep != none && rep != j) {
                local_of_[j] = local_of_[rep];
                continue;
            }
            local_of_[j] = static_cast<index_t>(local.representatives.size());
            local.representatives.push_back(j);
        }
        return window_columns;
    }

    // Member lists (CSR, ascending), merged priors, commit flags and observable classes of the
    // local columns.
    void describe_columns(Local& local, std::span<const index_t> window_columns,
                          std::uint32_t commit_end) {
        const std::size_t columns = local.representatives.size();
        local.members_ptr.assign(columns + 1, 0);
        for (const index_t j : window_columns) {
            ++local.members_ptr[local_of_[j] + 1];
        }
        for (std::size_t l = 0; l < columns; ++l) {
            if (local.members_ptr[l + 1] > 1) {
                ++local.merged;
            }
            local.members_ptr[l + 1] += local.members_ptr[l];
        }
        local.members.resize(window_columns.size());
        std::vector<index_t> fill(local.members_ptr.begin(), local.members_ptr.end() - 1);
        for (const index_t j : window_columns) {
            local.members[fill[local_of_[j]]++] = j;
        }

        // The merged prior is the probability of an odd number of member faults, folded in
        // ascending member order exactly as written, so that every implementation rounds alike.
        const std::span<const std::uint32_t> s = time_.first_rounds();
        local.probabilities.resize(columns);
        local.commit.resize(columns);
        local.commit_class.resize(columns);
        for (std::size_t l = 0; l < columns; ++l) {
            const index_t first = local.members_ptr[l];
            double p = problem_->priors[local.members[first]];
            for (index_t k = first + 1; k < local.members_ptr[l + 1]; ++k) {
                const double q = problem_->priors[local.members[k]];
                p = p * (1.0 - q) + q * (1.0 - p);
            }
            local.probabilities[l] = p;
            const index_t rep = local.representatives[l];
            const bool committed = s[rep] < commit_end;
            local.commit[l] = committed ? 1 : 0;
            local.commit_class[l] = committed ? col_class_[rep] : 0;
        }
    }

    // The window's rows of H with their columns renumbered. Columns outside the window are
    // dropped: those touching the first round were committed by an earlier window and enter
    // through the carried-in syndrome. A row of the last round can list several members of one
    // group, and a later member maps back to its representative's smaller local index, so such
    // rows are sorted and deduplicated; every other row maps monotonically.
    void cut_rows(Local& local, index_t row_begin, index_t row_end) {
        local.row_ptr.reserve(std::size_t{row_end - row_begin} + 1);
        local.row_ptr.push_back(0);
        for (index_t i = row_begin; i < row_end; ++i) {
            const std::size_t start = local.col_indices.size();
            bool increasing = true;
            for (index_t e = problem_->h_row_ptr[i]; e < problem_->h_row_ptr[i + 1]; ++e) {
                const index_t l = local_of_[problem_->h_col_indices[e]];
                if (l == none) {
                    continue;
                }
                increasing = increasing &&
                             (local.col_indices.size() == start || l > local.col_indices.back());
                local.col_indices.push_back(l);
            }
            if (!increasing) {
                const auto row = local.col_indices.begin() + static_cast<std::ptrdiff_t>(start);
                std::sort(row, local.col_indices.end());
                local.col_indices.erase(std::unique(row, local.col_indices.end()),
                                        local.col_indices.end());
            }
            local.row_ptr.push_back(static_cast<index_t>(local.col_indices.size()));
        }
    }

    [[nodiscard]] static std::uint64_t content_hash(const Local& local) noexcept {
        std::uint64_t h = mix(0, local.num_rows);
        h = mix_all<index_t>(h, local.row_ptr);
        h = mix_all<index_t>(h, local.col_indices);
        h = mix_all<double>(h, local.probabilities);
        h = mix_all<Bit>(h, local.commit);
        h = mix_all<std::uint64_t>(h, local.commit_class);
        return mix_all<Bit>(h, local.converge);
    }

    // Cuts rounds [t, t + rounds) out of H: the rows of those rounds and the columns with
    // t ≤ s(j) < t + rounds, merging (non-final placements only) the columns of the last round
    // whose supports coincide inside the window.
    [[nodiscard]] Local slice(const Cut& cut) {
        const index_t per_round = time_.detectors_per_round();
        const std::uint32_t t = cut.first_round;
        const std::uint32_t last = t + cut.rounds - 1;
        const index_t row_begin = time_.first_row_of_round(t);
        const index_t row_end = last * per_round;

        const std::vector<index_t> cut_columns =
            cut.final ? std::vector<index_t>{} : group_cut_round(last, row_end);
        Local local;
        local.rounds = cut.rounds;
        local.num_rows = row_end - row_begin;
        const std::vector<index_t> window_columns = number_columns(local, t, last);
        describe_columns(local, window_columns, t + cut.commit_rounds);
        local.converge.assign(local.num_rows, 0);
        std::fill_n(local.converge.begin(), std::size_t{cut.converge_rounds} * per_round, Bit{1});
        cut_rows(local, row_begin, row_end);

        for (const index_t j : window_columns) {
            local_of_[j] = none;
        }
        for (const index_t j : cut_columns) {
            group_rep_[j] = none;
        }
        local.hash = content_hash(local);
        return local;
    }

    [[nodiscard]] static bool same_content(const Local& local, const Shape& shape) noexcept {
        return local.hash == shape.hash_ && local.num_rows == shape.num_rows_ &&
               std::ranges::equal(local.row_ptr, shape.row_ptr_) &&
               std::ranges::equal(local.col_indices, shape.col_indices_) &&
               bitwise_equal(local.probabilities, shape.probabilities_) &&
               std::ranges::equal(local.commit, shape.commit_) &&
               std::ranges::equal(local.commit_class, shape.commit_class_) &&
               std::ranges::equal(local.converge, shape.converge_);
    }

    // Index of the shape with the same content as `local`, adding one if there is none. With
    // `reuse` false a new shape is always added.
    [[nodiscard]] std::expected<std::uint32_t, PlanError> shape_for(Local& local, bool reuse) {
        if (reuse) {
            for (const Shape& shape : plan_.shapes_) {
                if (same_content(local, shape)) {
                    return shape.index_;
                }
            }
        }
        const auto columns = static_cast<index_t>(local.representatives.size());
        auto graph = TannerGraph::from_csr(local.num_rows, columns, local.row_ptr,
                                           local.col_indices, options_);
        if (!graph) {
            return fail(PlanError::Code::graph_failed,
                        std::format("window shape {}: {}: {}", plan_.shapes_.size(),
                                    to_string(graph.error().code), graph.error().detail));
        }
        auto priors = Priors::from_probabilities(local.probabilities);
        if (!priors) {
            return fail(PlanError::Code::invalid_prior,
                        std::format("window shape {}, local column {}: {}", plan_.shapes_.size(),
                                    priors.error().column, priors.error().detail));
        }
        auto by_column = SparseBinaryMatrix::from_csr(local.num_rows, columns, local.row_ptr,
                                                      local.col_indices);
        if (!by_column) {
            return fail(PlanError::Code::graph_failed,
                        std::format("window shape {}: {}: {}", plan_.shapes_.size(),
                                    to_string(by_column.error().code), by_column.error().detail));
        }
        Shape shape(std::move(*graph), std::move(*priors), std::move(*by_column));
        shape.index_ = static_cast<std::uint32_t>(plan_.shapes_.size());
        shape.rounds_ = local.rounds;
        shape.num_rows_ = local.num_rows;
        shape.merged_columns_ = local.merged;
        shape.committed_columns_ =
            static_cast<index_t>(std::ranges::count(local.commit, Bit{1}));
        shape.hash_ = local.hash;
        shape.row_ptr_ = std::move(local.row_ptr);
        shape.col_indices_ = std::move(local.col_indices);
        shape.probabilities_ = std::move(local.probabilities);
        shape.commit_ = std::move(local.commit);
        shape.commit_class_ = std::move(local.commit_class);
        shape.converge_ = std::move(local.converge);
        plan_.shapes_.push_back(std::move(shape));
        return plan_.shapes_.back().index_;
    }

    [[nodiscard]] Cut exact_cut(std::uint32_t t, std::uint32_t attempt) const noexcept {
        const std::uint64_t rounds_total = time_.rounds_total();
        const std::uint64_t width = std::uint64_t{spec_.width} + std::uint64_t{attempt} * spec_.commit;
        if (t + width - 1 >= rounds_total) {
            const auto rounds = static_cast<std::uint32_t>(rounds_total - t + 1);
            return {.first_round = t,
                    .rounds = rounds,
                    .commit_rounds = rounds,
                    .converge_rounds = rounds,
                    .final = true};
        }
        return {.first_round = t,
                .rounds = static_cast<std::uint32_t>(width),
                .commit_rounds = spec_.commit,
                .converge_rounds = spec_.converge_rounds + attempt * spec_.commit,
                .final = false};
    }

    [[nodiscard]] Placement placement_of(std::uint32_t k, std::uint32_t a, std::uint32_t shape,
                                         const Cut& cut) const noexcept {
        return {.window = k,
                .attempt = a,
                .shape = shape,
                .first_round = cut.first_round,
                .rounds = cut.rounds,
                .commit_rounds = cut.commit_rounds,
                .final = cut.final,
                .first_row = time_.first_row_of_round(cut.first_round),
                .columns = {},
                .members_ptr = {},
                .members = {},
                .virtual_committed = 0};
    }

    [[nodiscard]] std::uint32_t attempt_limit() const noexcept {
        return spec_.on_failure == OnFailure::defer ? spec_.max_deferrals : 0;
    }

    // Windows k = 0, 1, … until the first final one, each followed by its deferral attempts up
    // to the first that reaches the readout.
    [[nodiscard]] std::expected<void, PlanError> build_exact() {
        for (std::uint32_t k = 0;; ++k) {
            const auto t = static_cast<std::uint32_t>(1 + std::uint64_t{k} * spec_.commit);
            bool final_window = false;
            for (std::uint32_t a = 0; a <= attempt_limit(); ++a) {
                const Cut cut = exact_cut(t, a);
                Local local = slice(cut);
                auto shape = shape_for(local, true);
                if (!shape) {
                    return std::unexpected(std::move(shape.error()));
                }
                Placement placement = placement_of(k, a, *shape, cut);
                placement.columns = std::move(local.representatives);
                placement.members_ptr = std::move(local.members_ptr);
                placement.members = std::move(local.members);
                plan_.schedule_.push_back(std::move(placement));
                if (a == 0) {
                    final_window = cut.final;
                }
                if (cut.final) {
                    break;
                }
            }
            plan_.window_begin_.push_back(static_cast<std::uint32_t>(plan_.schedule_.size()));
            if (final_window) {
                return {};
            }
        }
    }

    // Columns indexed by their first (smallest) row, for finding a shifted support.
    void index_by_first_row() {
        const index_t m = problem_->num_rows;
        const index_t n = problem_->num_columns;
        start_ptr_.assign(std::size_t{m} + 1, 0);
        for (index_t j = 0; j < n; ++j) {
            ++start_ptr_[global_rows(j).front() + 1];
        }
        for (index_t i = 0; i < m; ++i) {
            start_ptr_[i + 1] += start_ptr_[i];
        }
        start_cols_.resize(n);
        std::vector<index_t> fill(start_ptr_.begin(), start_ptr_.end() - 1);
        for (index_t j = 0; j < n; ++j) {
            start_cols_[fill[global_rows(j).front()]++] = j;
        }
    }

    // The global column whose support is column g's shifted by `shift` rows, virtual_column if
    // there is none, an error if there are several.
    [[nodiscard]] std::expected<index_t, PlanError> shifted(index_t g, std::int64_t shift) const {
        const auto rows = global_rows(g);
        const std::int64_t lo = std::int64_t{rows.front()} + shift;
        const std::int64_t hi = std::int64_t{rows.back()} + shift;
        if (lo < 0 || hi >= std::int64_t{problem_->num_rows}) {
            return virtual_column;
        }
        const auto first = static_cast<index_t>(lo);
        index_t match = none;
        for (index_t c = start_ptr_[first]; c < start_ptr_[first + 1]; ++c) {
            const index_t j = start_cols_[c];
            const auto candidate = global_rows(j);
            if (candidate.size() != rows.size() ||
                !std::ranges::equal(candidate, rows, [&](index_t x, index_t y) {
                    return std::int64_t{x} == std::int64_t{y} + shift;
                })) {
                continue;
            }
            if (match != none) {
                return fail(PlanError::Code::ambiguous_mapping,
                            std::format("columns {} and {} both match column {} shifted by {} "
                                        "rows (identical supports)",
                                        match, j, g, shift));
            }
            match = j;
        }
        return match == none ? virtual_column : match;
    }

    // The bulk shapes B_a are the exact shapes of placement (1, a); every position k reuses them
    // and maps their columns to the global columns shifted by (k − 1)·C rounds.
    [[nodiscard]] std::expected<void, PlanError> build_uniform() {
        const std::uint64_t rounds_total = time_.rounds_total();
        const std::uint64_t t1 = 1 + std::uint64_t{spec_.commit};
        const std::uint64_t widest = spec_.width + std::uint64_t{attempt_limit()} * spec_.commit;
        // t_1 + W_a ≤ R keeps every bulk shape clear of round 1 and of the readout round's
        // faults, and implies that window 1 exists and is never final.
        if (t1 + widest > rounds_total - 1) {
            return fail(PlanError::Code::no_bulk_window,
                        std::format("the uniform boundary needs t_1 + W_a <= R for every attempt "
                                    "(t_1 = {}, widest W_a = {}, R = {})",
                                    t1, widest, rounds_total - 1));
        }
        std::vector<std::vector<index_t>> representatives;
        for (std::uint32_t a = 0; a <= attempt_limit(); ++a) {
            const Cut cut = exact_cut(static_cast<std::uint32_t>(t1), a);
            Local local = slice(cut);
            auto shape = shape_for(local, false);
            if (!shape) {
                return std::unexpected(std::move(shape.error()));
            }
            representatives.push_back(std::move(local.representatives));
        }

        index_by_first_row();
        const std::uint32_t commit = spec_.commit;
        const auto positions = static_cast<std::uint32_t>((rounds_total + commit - 1) / commit);
        const std::int64_t per_round = time_.detectors_per_round();
        std::vector<index_t> memo(problem_->num_columns, none);
        for (std::uint32_t k = 0; k < positions; ++k) {
            const std::uint32_t t = 1 + k * commit;
            const std::int64_t shift = (std::int64_t{k} - 1) * commit * per_round;
            std::ranges::fill(memo, none);
            for (std::uint32_t a = 0; a <= attempt_limit(); ++a) {
                const Shape& shape = plan_.shapes_[a];
                const Cut cut{.first_round = t,
                              .rounds = shape.rounds(),
                              .commit_rounds = commit,
                              .converge_rounds = 0,
                              .final = false};
                Placement placement = placement_of(k, a, a, cut);
                placement.columns.resize(representatives[a].size());
                for (std::size_t l = 0; l < representatives[a].size(); ++l) {
                    const index_t g = representatives[a][l];
                    if (memo[g] == none) {
                        auto mapped = shifted(g, shift);
                        if (!mapped) {
                            return std::unexpected(std::move(mapped.error()));
                        }
                        memo[g] = *mapped;
                    }
                    placement.columns[l] = memo[g];
                    if (memo[g] == virtual_column && shape.commit()[l] != 0) {
                        ++placement.virtual_committed;
                    }
                }
                plan_.schedule_.push_back(std::move(placement));
            }
            plan_.window_begin_.push_back(static_cast<std::uint32_t>(plan_.schedule_.size()));
        }
        return {};
    }

    const Problem* problem_;
    WindowSpec spec_;
    TimeStructure time_;
    GraphOptions options_;
    WindowPlan plan_;
    std::vector<index_t> col_ptr_;          // [n + 1] column view of H
    std::vector<index_t> col_rows_;         // [nnz]   rows per column, ascending
    std::vector<std::uint64_t> col_class_;  // [n]     observable mask per column
    std::vector<index_t> local_of_;         // [n]     scratch: local column of a window column
    std::vector<index_t> group_rep_;        // [n]     scratch: representative of a cut column
    std::vector<index_t> start_ptr_;        // [m + 1] columns grouped by their first row
    std::vector<index_t> start_cols_;       // [n]
};

std::expected<WindowPlan, PlanError> WindowPlan::build(const Problem& problem,
                                                       const WindowSpec& spec,
                                                       GraphOptions options) {
    if (auto valid = validate(spec); !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    if (auto valid = validate(problem); !valid) {
        return std::unexpected(std::move(valid.error()));
    }
    if (problem.num_observables > 64) {
        return fail(PlanError::Code::too_many_observables,
                    std::format("k = {} observables; a column's observable mask holds at most 64",
                                problem.num_observables));
    }
    auto time = TimeStructure::build(problem);
    if (!time) {
        return std::unexpected(std::move(time.error()));
    }
    PlanBuilder builder(problem, spec, std::move(*time), options);
    return builder.build();
}

} // namespace rtd::window
