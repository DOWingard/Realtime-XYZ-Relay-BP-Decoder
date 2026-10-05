#pragma once

// Inner decoders for testing the stream decoder: an exact minimum-weight decoder that enumerates
// every solution of a small window, a scripted wrapper that makes chosen (window, attempt) decodes
// fail, and a spy that records every call of any inner decoder. All satisfy SyndromeDecoder.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include "rtd/core/result.hpp"
#include "rtd/core/types.hpp"
#include "rtd/window/plan.hpp"
#include "rtd/window/stream.hpp"

namespace rtd::window::test {

// Window position and attempt of a γ stream shot + (k << 32) + (a << 56), for shots below 2^32.
struct StreamKey {
    std::uint64_t shot = 0;
    std::uint32_t window = 0;
    std::uint32_t attempt = 0;
    auto operator<=>(const StreamKey&) const = default;
};

inline StreamKey split_stream(std::uint64_t stream) {
    return {.shot = stream & 0xFFFFFFFFULL,
            .window = static_cast<std::uint32_t>((stream >> 32U) & 0xFFFFFFULL),
            .attempt = static_cast<std::uint32_t>(stream >> 56U)};
}

// The exact minimum-weight solution of H·x = s on the convergence rows, found by enumerating the
// particular solution plus every combination of a null-space basis (Gaussian elimination over
// F₂). Weights are summed over the support in ascending column order, skipping infinite λ, the
// same rule as the relay decoder's, so equal weights are recognised exactly. Reports how many
// solutions share the minimum. Refuses windows whose solution space has more than 2^max_free
// elements.
class BruteForceInner {
public:
    explicit BruteForceInner(const Shape& shape, std::uint32_t max_free = 22)
        : rows_(shape.num_rows()), cols_(shape.num_columns()),
          words_((std::size_t{shape.num_columns()} + 63) / 64),
          llr_(shape.priors().llr().begin(), shape.priors().llr().end()),
          row_ptr_(shape.row_ptr().begin(), shape.row_ptr().end()),
          col_(shape.col_indices().begin(), shape.col_indices().end()), mask_(rows_, Bit{1}),
          hard_(cols_, Bit{0}), max_free_(max_free) {}

    [[nodiscard]] index_t num_rows() const noexcept { return rows_; }
    [[nodiscard]] index_t num_columns() const noexcept { return cols_; }

    void set_convergence_rows(std::span<const Bit> mask) noexcept {
        for (index_t i = 0; i < rows_; ++i) {
            mask_[i] = i < mask.size() ? mask[i] : Bit{1};
        }
    }
    [[nodiscard]] std::span<const Bit> convergence_rows() const noexcept { return mask_; }

    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> syndrome, std::uint64_t /*stream*/, DecodeLimits /*limits*/) {
        if (syndrome.size() != rows_) {
            return std::unexpected(DecodeError{.code = DecodeError::Code::syndrome_size_mismatch,
                                               .expected = rows_,
                                               .found = syndrome.size()});
        }
        optimal_ = 0;
        too_large_ = false;
        support_.clear();
        std::ranges::fill(hard_, Bit{0});
        const std::optional<Solved> solved = solve(syndrome);
        bool success = solved.has_value();
        double best = std::numeric_limits<double>::infinity();
        if (solved) {
            const std::size_t free = solved->basis.size();
            if (free > max_free_) {
                too_large_ = true;
                success = false;
            } else {
                std::vector<std::uint64_t> x = solved->particular;
                std::vector<std::uint64_t> best_x = x;
                best = weight(x);
                optimal_ = 1;
                // Gray code: consecutive combinations differ in one basis vector.
                for (std::uint64_t g = 1; g < (std::uint64_t{1} << free); ++g) {
                    const auto& v = solved->basis[static_cast<std::size_t>(std::countr_zero(g))];
                    for (std::size_t w = 0; w < words_; ++w) {
                        x[w] ^= v[w];
                    }
                    const double wx = weight(x);
                    if (wx < best) {
                        best = wx;
                        best_x = x;
                        optimal_ = 1;
                    } else if (wx == best) {
                        ++optimal_;
                    }
                }
                for (index_t j = 0; j < cols_; ++j) {
                    if (bit(best_x, j)) {
                        support_.push_back(j);
                        hard_[j] = 1;
                    }
                }
            }
        }
        legs_[0] = LegRecord{.iterations = 1, .converged = success, .became_best = success,
                             .weight = success ? best : std::numeric_limits<double>::infinity()};
        return DecodeResult{.success = success,
                            .iterations = 1,
                            .legs_executed = 1,
                            .best_leg = success ? std::optional<std::uint32_t>(0) : std::nullopt,
                            .weight = success ? best : std::numeric_limits<double>::infinity(),
                            .hard = hard_,
                            .legs = legs_,
                            .cap_hit = false,
                            .support = support_};
    }

    // Solutions that share the minimum weight in the last decode (0 if there was none).
    [[nodiscard]] std::uint64_t optimal_count() const noexcept { return optimal_; }
    // The last decode was refused because the solution space was too large.
    [[nodiscard]] bool too_large() const noexcept { return too_large_; }

private:
    struct Solved {
        std::vector<std::uint64_t> particular;
        std::vector<std::vector<std::uint64_t>> basis;
    };

    [[nodiscard]] double weight(const std::vector<std::uint64_t>& x) const {
        double total = 0.0;
        for (index_t j = 0; j < cols_; ++j) {
            if (bit(x, j) && std::isfinite(llr_[j])) {
                total += llr_[j];
            }
        }
        return total;
    }

    // [H | s] restricted to the convergence rows, one bitset per equation.
    struct System {
        std::vector<std::vector<std::uint64_t>> eq;
        std::vector<Bit> rhs;
    };

    [[nodiscard]] static bool bit(const std::vector<std::uint64_t>& v, index_t j) {
        return ((v[j / 64] >> (j % 64)) & 1U) != 0;
    }
    static void set_bit(std::vector<std::uint64_t>& v, index_t j) {
        v[j / 64] |= std::uint64_t{1} << (j % 64);
    }

    [[nodiscard]] System system(std::span<const Bit> syndrome) const {
        System sys;
        for (index_t i = 0; i < rows_; ++i) {
            if (mask_[i] == 0) {
                continue;
            }
            std::vector<std::uint64_t> row(words_, 0);
            for (index_t e = row_ptr_[i]; e < row_ptr_[i + 1]; ++e) {
                set_bit(row, col_[e]);
            }
            sys.eq.push_back(std::move(row));
            sys.rhs.push_back(syndrome[i] != 0 ? Bit{1} : Bit{0});
        }
        return sys;
    }

    // Gauss-Jordan elimination: afterwards equation r < rank has its pivot in column pivots[r]
    // and no other equation has a one there. Returns the pivot columns.
    [[nodiscard]] std::vector<index_t> eliminate(System& sys) const {
        std::vector<index_t> pivots;
        auto& eq = sys.eq;
        for (index_t c = 0; c < cols_ && pivots.size() < eq.size(); ++c) {
            const std::size_t rank = pivots.size();
            std::size_t p = rank;
            while (p < eq.size() && !bit(eq[p], c)) {
                ++p;
            }
            if (p == eq.size()) {
                continue;
            }
            std::swap(eq[p], eq[rank]);
            std::swap(sys.rhs[p], sys.rhs[rank]);
            for (std::size_t r = 0; r < eq.size(); ++r) {
                if (r == rank || !bit(eq[r], c)) {
                    continue;
                }
                for (std::size_t w = 0; w < words_; ++w) {
                    eq[r][w] ^= eq[rank][w];
                }
                sys.rhs[r] ^= sys.rhs[rank];
            }
            pivots.push_back(c);
        }
        return pivots;
    }

    // Every solution is the particular one (free columns 0) plus a combination of the null-space
    // vectors, one per free column f: x_f = 1 and each pivot column set as f's coefficient in its
    // equation dictates.
    [[nodiscard]] std::optional<Solved> solve(std::span<const Bit> syndrome) const {
        System sys = system(syndrome);
        const std::vector<index_t> pivots = eliminate(sys);
        for (std::size_t r = pivots.size(); r < sys.eq.size(); ++r) {
            if (sys.rhs[r] != 0) {
                return std::nullopt; // 0 = 1: no solution
            }
        }
        Solved solved;
        solved.particular.assign(words_, 0);
        std::vector<bool> is_pivot(cols_, false);
        for (std::size_t r = 0; r < pivots.size(); ++r) {
            is_pivot[pivots[r]] = true;
            if (sys.rhs[r] != 0) {
                set_bit(solved.particular, pivots[r]);
            }
        }
        for (index_t f = 0; f < cols_; ++f) {
            if (is_pivot[f]) {
                continue;
            }
            std::vector<std::uint64_t> v(words_, 0);
            set_bit(v, f);
            for (std::size_t r = 0; r < pivots.size(); ++r) {
                if (bit(sys.eq[r], f)) {
                    set_bit(v, pivots[r]);
                }
            }
            solved.basis.push_back(std::move(v));
        }
        return solved;
    }

    index_t rows_;
    index_t cols_;
    std::size_t words_;
    std::vector<double> llr_;
    std::vector<index_t> row_ptr_;
    std::vector<index_t> col_;
    std::vector<Bit> mask_;
    std::vector<Bit> hard_;
    std::vector<index_t> support_;
    std::array<LegRecord, 1> legs_{};
    std::uint32_t max_free_;
    std::uint64_t optimal_ = 0;
    bool too_large_ = false;
};

// One call of an inner decoder as a spy saw it.
struct InnerCall {
    std::uint32_t shape = 0;
    std::uint64_t stream = 0;
    std::optional<std::uint32_t> cap;
    std::vector<Bit> syndrome;
    bool success = false;
    std::uint32_t iterations = 0;
    std::uint32_t legs_executed = 0;
    std::optional<std::uint32_t> best_leg;
    bool cap_hit = false;
    double weight = 0.0;
    std::vector<index_t> support;
    std::vector<LegRecord> legs;
    std::uint64_t optimal = 0; // brute force only
};

// Forwards to any inner decoder and appends every call, with copies of its inputs and outputs, to
// a log shared by the inner decoders of all shapes (so the log is in decode order).
template <SyndromeDecoder D>
class Spy {
public:
    Spy(D inner, std::uint32_t shape, std::vector<InnerCall>* log)
        : inner_(std::move(inner)), shape_(shape), log_(log) {}

    [[nodiscard]] index_t num_rows() const noexcept { return inner_.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return inner_.num_columns(); }
    void set_convergence_rows(std::span<const Bit> mask) {
        mask_.assign(mask.begin(), mask.end());
        inner_.set_convergence_rows(mask);
    }

    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> syndrome, std::uint64_t stream, DecodeLimits limits) {
        auto result = inner_.decode(syndrome, stream, limits);
        if (result) {
            InnerCall call{.shape = shape_,
                           .stream = stream,
                           .cap = limits.max_total_iterations,
                           .syndrome = {syndrome.begin(), syndrome.end()},
                           .success = result->success,
                           .iterations = result->iterations,
                           .legs_executed = result->legs_executed,
                           .best_leg = result->best_leg,
                           .cap_hit = result->cap_hit,
                           .weight = result->weight,
                           .support = {result->support.begin(), result->support.end()},
                           .legs = {result->legs.begin(), result->legs.end()},
                           .optimal = 0};
            if constexpr (requires { inner_.optimal_count(); }) {
                call.optimal = inner_.optimal_count();
            }
            log_->push_back(std::move(call));
        }
        return result;
    }

    [[nodiscard]] D& inner() noexcept { return inner_; }
    [[nodiscard]] std::span<const Bit> convergence_rows() const noexcept { return mask_; }

private:
    D inner_;
    std::uint32_t shape_;
    std::vector<InnerCall>* log_;
    std::vector<Bit> mask_;
};

// The exact decoder, except that the decodes whose (window, attempt) is in `fails` report no
// convergence: they return the exact solution as if it were leg 0's final ê, with +∞ weight and
// made-up iteration and leg counts (so that sums over attempts are visible).
class ScriptedInner {
public:
    static constexpr std::uint32_t failed_iterations = 7;
    static constexpr std::uint32_t failed_legs = 3;

    using Fails = std::set<std::pair<std::uint32_t, std::uint32_t>>;

    ScriptedInner(const Shape& shape, const Fails* fails) : exact_(shape), fails_(fails) {}

    [[nodiscard]] index_t num_rows() const noexcept { return exact_.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return exact_.num_columns(); }
    void set_convergence_rows(std::span<const Bit> mask) noexcept {
        exact_.set_convergence_rows(mask);
    }

    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> syndrome, std::uint64_t stream, DecodeLimits limits) {
        auto result = exact_.decode(syndrome, stream, limits);
        const StreamKey key = split_stream(stream);
        if (result && fails_->contains({key.window, key.attempt})) {
            result->success = false;
            result->best_leg = std::nullopt;
            result->weight = std::numeric_limits<double>::infinity();
            result->iterations = failed_iterations;
            result->legs_executed = failed_legs;
            result->legs = {};
        }
        return result;
    }

    [[nodiscard]] std::uint64_t optimal_count() const noexcept { return exact_.optimal_count(); }

private:
    BruteForceInner exact_;
    const Fails* fails_;
};

static_assert(SyndromeDecoder<BruteForceInner>);
static_assert(SyndromeDecoder<ScriptedInner>);
static_assert(SyndromeDecoder<Spy<BruteForceInner>>);

} // namespace rtd::window::test
