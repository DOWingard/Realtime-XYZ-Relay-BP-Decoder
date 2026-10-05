#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <experimental/simd>
#include <span>
#include <utility>

#include "rtd/core/arith.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/types.hpp"

// The min-sum kernels. One iteration of the flooding schedule is a check pass over every row
// followed by a variable pass over every column; both operate in place on a single edge-message
// array, which holds μ (variable → check) before a check pass and η (check → variable) after it.
//
// The single array is sound because rows partition the edges and so do columns: a check update
// reads and writes only its own row's slots, a variable update only its own column's slots. Every
// edge receives a fresh η before any variable update reads it, so no reset between passes is
// needed.
//
// Kernels take half-open ranges of rows or of columns, which is how an executor splits a pass
// across threads. Splitting changes no per-node arithmetic, so results do not depend on the
// thread count.

namespace rtd::kernels {

namespace stdx = std::experimental;

// Where a check pass reads its inputs. `priors` is the first iteration of a leg: every μ equals
// the prior of its variable, so the pass reads λ through the slot's column instead of first
// writing λ into every slot.
enum class Source : std::uint8_t { messages, priors };

// ---------------------------------------------------------------------------------------------
// Check update, per row i with inputs μ_e on its edges:
//   parity  = σ_i ⊕ (number of negative μ_e mod 2)
//   min1    = smallest |μ_e|, min2 = second smallest (equal to min1 if the minimum repeats),
//             both clamped to max_message(), so a row with one edge sends α·max_message()
//   η_e     = (parity ⊕ sign μ_e) · α · (|μ_e| == min1 ? min2 : min1)
// The minimum is identified by value, not by position. Min and parity are exact in any order,
// so vectorising within a row does not change a single bit.
// ---------------------------------------------------------------------------------------------

// Folds one magnitude into a running (min1, min2) pair. A NaN compares false everywhere and
// is ignored.
template <class T>
constexpr void fold_two_min(T magnitude, T& min1, T& min2) noexcept {
    const bool below = magnitude < min1;
    const T displaced = below ? min1 : magnitude;
    min1 = below ? magnitude : min1;
    min2 = displaced < min2 ? displaced : min2;
}

// Reference implementation for any arithmetic policy: one row whose inputs are `input(k)` for
// k < degree, whose outputs go to row[k].
template <MessageArithmetic A, bool UnitAlpha, class Input>
void check_row_scalar(typename A::msg_t* row, index_t degree, bool syndrome_bit,
                      typename A::msg_t alpha, const Input& input) noexcept {
    using msg_t = A::msg_t;
    bool parity = syndrome_bit;
    msg_t min1 = A::max_message();
    msg_t min2 = A::max_message();
    for (index_t k = 0; k < degree; ++k) {
        const msg_t x = input(k);
        parity ^= A::is_negative(x);
        fold_two_min(A::magnitude(x), min1, min2);
    }
    const msg_t out_min = UnitAlpha ? min1 : A::scale(min1, alpha);
    const msg_t out_second = UnitAlpha ? min2 : A::scale(min2, alpha);
    for (index_t k = 0; k < degree; ++k) {
        const msg_t x = input(k);
        const msg_t magnitude = A::magnitude(x) == min1 ? out_second : out_min;
        row[k] = A::with_sign(magnitude, parity != A::is_negative(x));
    }
}

// Rows [row_begin, row_end) of the row-major layout, any arithmetic policy.
template <MessageArithmetic A, bool UnitAlpha, Source Src>
void check_rows_scalar(const RowMajorView& g, const Bit* syndrome, typename A::msg_t alpha,
                       const typename A::msg_t* lambda, typename A::msg_t* msg, index_t row_begin,
                       index_t row_end) noexcept {
    for (index_t i = row_begin; i < row_end; ++i) {
        const index_t degree = g.row_ptr[i + 1] - g.row_ptr[i];
        if (degree == 0) {
            continue;
        }
        auto* row = msg + g.row_slot_begin[i];
        const index_t* columns = g.slot_column + g.row_slot_begin[i];
        check_row_scalar<A, UnitAlpha>(row, degree, syndrome[i] != 0, alpha, [&](index_t k) {
            if constexpr (Src == Source::priors) {
                return lambda[columns[k]];
            } else {
                return row[k];
            }
        });
    }
}

template <class A>
inline constexpr bool is_float_arith = false;
template <std::floating_point T>
inline constexpr bool is_float_arith<FloatArith<T>> = true;

// Vectorised check update of one row for IEEE policies, bit-identical to check_row_scalar.
//
// The row is `degree` values at `row`, padded to a multiple of two vectors with +max_message(),
// which is neutral for both reductions: its sign bit is clear and a magnitude of max_message()
// cannot lower a minimum that starts at max_message(). The row therefore runs over whole aligned
// vectors; the only care needed is to write max_message() back into padding lanes. With
// Src == priors the inputs are lambda[columns[k]] instead (padding columns name a sentinel whose
// prior is max_message()).
template <std::floating_point T, bool UnitAlpha, Source Src>
void check_row_simd(T* row, const index_t* columns, const T* lambda, index_t degree,
                    bool syndrome_bit, T alpha) noexcept {
    using V = stdx::native_simd<T>;
    using M = V::mask_type;
    constexpr index_t width = V::size();
    static_assert(row_alignment_slots % (2 * width) == 0,
                  "rows must hold a whole number of vector pairs");
    constexpr T max_msg = FloatArith<T>::max_message();
    const V lane_index([](auto lane) { return static_cast<T>(static_cast<int>(lane)); });

    const auto padded = static_cast<index_t>(round_up(degree, std::size_t{2} * width));
    const auto load = [&](index_t offset) {
        if constexpr (Src == Source::priors) {
            return V([&](auto lane) { return lambda[columns[offset + lane]]; });
        } else {
            return V(row + offset, stdx::vector_aligned);
        }
    };

    // Two independent accumulator sets halve the loop-carried dependency chain.
    V min1a(max_msg);
    V min2a(max_msg);
    V min1b(max_msg);
    V min2b(max_msg);
    M negative_a(false);
    M negative_b(false);
    const auto fold = [](const V& x, V& min1, V& min2, M& negative) {
        negative ^= stdx::signbit(x);
        const V magnitude = stdx::abs(x);
        const M below = magnitude < min1;
        V displaced = magnitude;
        stdx::where(below, displaced) = min1;
        stdx::where(below, min1) = magnitude;
        stdx::where(displaced < min2, min2) = displaced;
    };
    for (index_t offset = 0; offset < padded; offset += 2 * width) {
        fold(load(offset), min1a, min2a, negative_a);
        fold(load(offset + width), min1b, min2b, negative_b);
    }
    // Merge the two sets lane-wise, then the lanes.
    {
        const M below = min1b < min1a;
        V displaced = min1b;
        stdx::where(below, displaced) = min1a;
        stdx::where(below, min1a) = min1b;
        stdx::where(min2b < min2a, min2a) = min2b;
        stdx::where(displaced < min2a, min2a) = displaced;
    }
    T min1 = max_msg;
    T min2 = max_msg;
    for (std::size_t lane = 0; lane < width; ++lane) {
        fold_two_min(static_cast<T>(min1a[lane]), min1, min2);
        const T lane_second = min2a[lane];
        min2 = lane_second < min2 ? lane_second : min2;
    }
    const int negatives = stdx::popcount(negative_a) + stdx::popcount(negative_b);
    const bool parity = syndrome_bit != ((negatives & 1) != 0);

    const V out_min(UnitAlpha ? min1 : alpha * min1);
    const V out_second(UnitAlpha ? min2 : alpha * min2);
    // Vectors made only of padding are not rewritten; they keep their max_message().
    const M flip(parity);
    for (index_t offset = 0; offset < degree; offset += width) {
        const V x = load(offset);
        V out = out_min;
        stdx::where(stdx::abs(x) == V(min1), out) = out_second;
        stdx::where(stdx::signbit(x) != flip, out) = -out;
        if (degree - offset < width) {
            stdx::where(lane_index >= V(static_cast<T>(degree - offset)), out) = V(max_msg);
        }
        out.copy_to(row + offset, stdx::vector_aligned);
    }
}

// Rows [row_begin, row_end) of the row-major layout, vectorised within each row.
template <std::floating_point T, bool UnitAlpha, Source Src>
void check_rows_simd(const RowMajorView& g, const Bit* syndrome, T alpha, const T* lambda, T* msg,
                     index_t row_begin, index_t row_end) noexcept {
    for (index_t i = row_begin; i < row_end; ++i) {
        const index_t degree = g.row_ptr[i + 1] - g.row_ptr[i];
        if (degree != 0) {
            check_row_simd<T, UnitAlpha, Src>(msg + g.row_slot_begin[i],
                                              g.slot_column + g.row_slot_begin[i], lambda, degree,
                                              syndrome[i] != 0, alpha);
        }
    }
}

// Integer policies whose messages are two's complement bytes with the natural sign and magnitude
// (is_negative(x) = x < 0, magnitude(x) = |x|, with_sign(m, s) = s ? −m : m, |x| ≤ 127).
template <class A>
concept ByteMessages = std::same_as<typename A::msg_t, std::int8_t> && requires {
    requires A::byte_messages;
};

// Vectorised check update of one row for ByteMessages policies, bit-identical to
// check_row_scalar. Rows are padded to a multiple of row_alignment_slots with max_message(),
// which is neutral for both reductions (non-negative, and no smaller than any minimum), so one
// vector of row_alignment_slots bytes covers each block of the row with no tail. Minima and parity
// are exact in any order; the scaling α = 1 − 2^(−k) is applied once per row to the two minima.
template <MessageArithmetic A, bool UnitAlpha, Source Src>
    requires ByteMessages<A>
void check_row_simd_bytes(std::int8_t* row, const index_t* columns, const std::int8_t* lambda,
                          index_t degree, bool syndrome_bit, std::int8_t alpha) noexcept {
    using V = stdx::fixed_size_simd<std::int8_t, row_alignment_slots>;
    using M = V::mask_type;
    constexpr index_t width = row_alignment_slots;
    const std::int8_t max_msg = A::max_message();
    const V zero(std::int8_t{0});
    const V lane_index([](auto lane) { return static_cast<std::int8_t>(static_cast<int>(lane)); });
    const auto padded = static_cast<index_t>(round_up(degree, std::size_t{width}));
    const auto load = [&](index_t offset) {
        if constexpr (Src == Source::priors) {
            return V([&](auto lane) { return lambda[columns[offset + lane]]; });
        } else {
            return V(row + offset, stdx::element_aligned);
        }
    };
    const auto magnitude = [&](const V& x) {
        V m = x;
        stdx::where(x < zero, m) = -x;
        return m;
    };

    V min1v(max_msg);
    V min2v(max_msg);
    M negative(false);
    for (index_t offset = 0; offset < padded; offset += width) {
        const V x = load(offset);
        negative ^= x < zero;
        const V m = magnitude(x);
        const M below = m < min1v;
        V displaced = m;
        stdx::where(below, displaced) = min1v;
        stdx::where(below, min1v) = m;
        stdx::where(displaced < min2v, min2v) = displaced;
    }
    // Across lanes: min1 is the smallest lane minimum. When it occurs in two lanes it is also the
    // second smallest input; otherwise the second smallest is the smaller of the other lanes'
    // minima and every lane's second minimum.
    const std::int8_t min1 = stdx::hmin(min1v);
    const M at_min = min1v == V(min1);
    std::int8_t min2 = min1;
    if (stdx::popcount(at_min) == 1) {
        V others = min1v;
        stdx::where(at_min, others) = V(max_msg);
        min2 = std::min(stdx::hmin(others), stdx::hmin(min2v));
    }
    const bool parity = syndrome_bit != ((stdx::popcount(negative) & 1) != 0);

    const V out_min(UnitAlpha ? min1 : A::scale(min1, alpha));
    const V out_second(UnitAlpha ? min2 : A::scale(min2, alpha));
    const M flip(parity);
    for (index_t offset = 0; offset < degree; offset += width) {
        const V x = load(offset);
        V out = out_min;
        stdx::where(magnitude(x) == V(min1), out) = out_second;
        stdx::where((x < zero) != flip, out) = -out;
        if (degree - offset < width) {
            stdx::where(lane_index >= V(static_cast<std::int8_t>(degree - offset)), out) =
                V(max_msg);
        }
        out.copy_to(row + offset, stdx::element_aligned);
    }
}

// Rows [row_begin, row_end) of the row-major layout for ByteMessages policies.
template <MessageArithmetic A, bool UnitAlpha, Source Src>
    requires ByteMessages<A>
void check_rows_simd_bytes(const RowMajorView& g, const Bit* syndrome, std::int8_t alpha,
                           const std::int8_t* lambda, std::int8_t* msg, index_t row_begin,
                           index_t row_end) noexcept {
    for (index_t i = row_begin; i < row_end; ++i) {
        const index_t degree = g.row_ptr[i + 1] - g.row_ptr[i];
        if (degree != 0) {
            check_row_simd_bytes<A, UnitAlpha, Src>(msg + g.row_slot_begin[i],
                                                    g.slot_column + g.row_slot_begin[i], lambda,
                                                    degree, syndrome[i] != 0, alpha);
        }
    }
}

// Check pass over rows [row_begin, row_end), dispatching to the fastest exact implementation.
template <MessageArithmetic A, bool UnitAlpha, Source Src>
void check_rows(const RowMajorView& g, const Bit* syndrome, typename A::msg_t alpha,
                const typename A::msg_t* lambda, typename A::msg_t* msg, index_t row_begin,
                index_t row_end) noexcept {
    if constexpr (is_float_arith<A>) {
        check_rows_simd<typename A::msg_t, UnitAlpha, Src>(g, syndrome, alpha, lambda, msg,
                                                           row_begin, row_end);
    } else if constexpr (ByteMessages<A>) {
        check_rows_simd_bytes<A, UnitAlpha, Src>(g, syndrome, alpha, lambda, msg, row_begin,
                                                 row_end);
    } else {
        check_rows_scalar<A, UnitAlpha, Src>(g, syndrome, alpha, lambda, msg, row_begin, row_end);
    }
}

// Vectorised check update of one row of the column-blocked layout, whose inputs live at scattered
// slots. The first pass gathers them straight into registers (lanes past the row's degree read as
// max_message(), the neutral padding) and keeps a copy in `buffer` for the second pass, which
// scatters each outgoing message from the result vector. Same arithmetic as check_row_simd.
template <std::floating_point T, bool UnitAlpha, Source Src>
void check_row_gathered_simd(T* msg, const index_t* slots, const index_t* columns, const T* lambda,
                             index_t degree, bool syndrome_bit, T alpha, T* buffer) noexcept {
    using V = stdx::native_simd<T>;
    using M = V::mask_type;
    constexpr index_t width = V::size();
    constexpr T max_msg = FloatArith<T>::max_message();
    const auto input = [&](index_t k) {
        if constexpr (Src == Source::priors) {
            return lambda[columns[k]];
        } else {
            return msg[slots[k]];
        }
    };
    const auto gather = [&](index_t offset) {
        if (offset + width <= degree) [[likely]] {
            return V([&](auto lane) { return input(offset + static_cast<index_t>(lane)); });
        }
        return V([&](auto lane) {
            const index_t k = offset + static_cast<index_t>(lane);
            return k < degree ? input(k) : max_msg;
        });
    };

    const auto padded = static_cast<index_t>(round_up(degree, std::size_t{2} * width));
    V min1a(max_msg);
    V min2a(max_msg);
    V min1b(max_msg);
    V min2b(max_msg);
    M negative_a(false);
    M negative_b(false);
    const auto fold = [](const V& x, V& min1, V& min2, M& negative) {
        negative ^= stdx::signbit(x);
        const V magnitude = stdx::abs(x);
        const M below = magnitude < min1;
        V displaced = magnitude;
        stdx::where(below, displaced) = min1;
        stdx::where(below, min1) = magnitude;
        stdx::where(displaced < min2, min2) = displaced;
    };
    for (index_t offset = 0; offset < padded; offset += 2 * width) {
        const V xa = gather(offset);
        const V xb = gather(offset + width);
        xa.copy_to(buffer + offset, stdx::vector_aligned);
        xb.copy_to(buffer + offset + width, stdx::vector_aligned);
        fold(xa, min1a, min2a, negative_a);
        fold(xb, min1b, min2b, negative_b);
    }
    {
        const M below = min1b < min1a;
        V displaced = min1b;
        stdx::where(below, displaced) = min1a;
        stdx::where(below, min1a) = min1b;
        stdx::where(min2b < min2a, min2a) = min2b;
        stdx::where(displaced < min2a, min2a) = displaced;
    }
    T min1 = max_msg;
    T min2 = max_msg;
    for (std::size_t lane = 0; lane < width; ++lane) {
        fold_two_min(static_cast<T>(min1a[lane]), min1, min2);
        const T lane_second = min2a[lane];
        min2 = lane_second < min2 ? lane_second : min2;
    }
    const int negatives = stdx::popcount(negative_a) + stdx::popcount(negative_b);
    const bool parity = syndrome_bit != ((negatives & 1) != 0);

    const V out_min(UnitAlpha ? min1 : alpha * min1);
    const V out_second(UnitAlpha ? min2 : alpha * min2);
    const M flip(parity);
    for (index_t offset = 0; offset < degree; offset += width) {
        const V x(buffer + offset, stdx::vector_aligned);
        V out = out_min;
        stdx::where(stdx::abs(x) == V(min1), out) = out_second;
        stdx::where(stdx::signbit(x) != flip, out) = -out;
        const index_t lanes = std::min<index_t>(width, degree - offset);
        for (index_t lane = 0; lane < lanes; ++lane) {
            msg[slots[offset + lane]] = out[lane];
        }
    }
}

// Rows [row_begin, row_end) of the column-blocked layout. `buffer` is aligned and holds at least
// round_up(max_row_degree, row_alignment_slots) entries.
template <MessageArithmetic A, bool UnitAlpha, Source Src>
void check_rows_blocked(const ColumnBlockedView& g, const Bit* syndrome, typename A::msg_t alpha,
                        const typename A::msg_t* lambda, typename A::msg_t* msg, index_t row_begin,
                        index_t row_end, typename A::msg_t* buffer) noexcept {
    for (index_t i = row_begin; i < row_end; ++i) {
        const index_t first = g.row_ptr[i];
        const index_t degree = g.row_ptr[i + 1] - first;
        if (degree == 0) {
            continue;
        }
        const index_t* slots = g.row_edge_slot + first;
        const index_t* columns = g.row_edge_column + first;
        if constexpr (is_float_arith<A>) {
            check_row_gathered_simd<typename A::msg_t, UnitAlpha, Src>(
                msg, slots, columns, lambda, degree, syndrome[i] != 0, alpha, buffer);
        } else {
            for (index_t k = 0; k < degree; ++k) {
                buffer[k] = Src == Source::priors ? lambda[columns[k]] : msg[slots[k]];
            }
            check_row_scalar<A, UnitAlpha>(buffer, degree, syndrome[i] != 0, alpha,
                                           [&](index_t k) { return buffer[k]; });
            for (index_t k = 0; k < degree; ++k) {
                msg[slots[k]] = buffer[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Variable update, per column j with incoming η_1..η_d on its edges in ascending row order:
//   Λ   = mix(λ_j, M_j, γ_j) with memory, else λ_j
//   forward:  s = Λ;  for k = 1..d: μ_k = s, s = s + η_k;   M_j = s
//   reverse:  t = 0;  for k = d..1: μ_k = μ_k + t, t = t + η_k
// so μ_k = (Λ + η_1 + … + η_{k−1}) + (η_d + … + η_{k+1}), with the operations performed exactly
// in this order. The final "+ t" with t = +0 is kept on purpose: it turns a −0 into +0, and the
// sign of a zero message matters to the check update.
//
// The hard decision is fused in: when M_j is final, a column with M_j ≤ 0 is appended to the
// sink, which also flips the column's rows in a bitset syndrome. The convergence test H·ê = σ
// then costs O(|ê|·d) per iteration instead of a pass over every edge.
// ---------------------------------------------------------------------------------------------

// Collects the columns declared in error by one variable pass (or one thread's share of it).
struct HardDecisionSink {
    index_t* support;             // external column indices, in visiting order
    index_t count;                // entries of support written so far
    std::uint64_t* syndrome_bits; // H·ê accumulated as a bitset over rows
    const index_t* external;      // internal → external column map

    void flip_rows(const index_t* rows, index_t degree) const noexcept {
        for (index_t k = 0; k < degree; ++k) {
            syndrome_bits[rows[k] >> 6U] ^= std::uint64_t{1} << (rows[k] & 63U);
        }
    }
};

// Columns [col_begin, col_end) of one degree run, degree known at compile time: the slot
// gathers, both sums and the stores are fully unrolled and the loop has no data-dependent
// branch other than the (rarely taken) error flag.
template <MessageArithmetic A, index_t D, bool UseMemory>
void variable_run_fixed(const RowMajorView& g, const DegreeRun& run, index_t col_begin,
                        index_t col_end, const typename A::msg_t* lambda,
                        const typename A::msg_t* gamma, typename A::acc_t* marginal,
                        typename A::msg_t* msg, HardDecisionSink& sink) noexcept {
    using msg_t = A::msg_t;
    using acc_t = A::acc_t;
    const std::size_t first_edge =
        run.edge_begin + static_cast<std::size_t>(col_begin - run.col_begin) * D;
    const index_t* slots = g.column_slots + first_edge;
    const index_t* rows = g.column_rows + first_edge;
    for (index_t c = col_begin; c < col_end; ++c, slots += D, rows += D) {
        acc_t s = UseMemory ? A::mix(lambda[c], marginal[c], gamma[c]) : A::widen(lambda[c]);
        // Every element is written before it is read; zero-filling would cost a store per edge.
        std::array<msg_t, D> eta;    // NOLINT(cppcoreguidelines-pro-type-member-init)
        std::array<acc_t, D> prefix; // NOLINT(cppcoreguidelines-pro-type-member-init)
        for (index_t k = 0; k != D; ++k) {
            eta[k] = msg[slots[k]];
        }
        for (index_t k = 0; k != D; ++k) {
            prefix[k] = s;
            s = A::accumulate(s, eta[k]);
        }
        marginal[c] = s;
        acc_t t = A::zero();
        for (index_t k = D; k-- > 0;) {
            msg[slots[k]] = A::to_msg(A::combine(prefix[k], t));
            t = A::accumulate(t, eta[k]);
        }
        if (A::is_error(s)) [[unlikely]] {
            sink.support[sink.count++] = sink.external[c];
            sink.flip_rows(rows, D);
        }
    }
}

// Same computation for a degree only known at run time; `scratch` holds `degree` entries.
template <MessageArithmetic A, bool UseMemory>
void variable_run_dynamic(const RowMajorView& g, const DegreeRun& run, index_t col_begin,
                          index_t col_end, const typename A::msg_t* lambda,
                          const typename A::msg_t* gamma, typename A::acc_t* marginal,
                          typename A::msg_t* msg, typename A::acc_t* scratch,
                          HardDecisionSink& sink) noexcept {
    using acc_t = A::acc_t;
    const index_t d = run.degree;
    const std::size_t first_edge =
        run.edge_begin + static_cast<std::size_t>(col_begin - run.col_begin) * d;
    const index_t* slots = g.column_slots + first_edge;
    const index_t* rows = g.column_rows + first_edge;
    acc_t* prefix = scratch;
    for (index_t c = col_begin; c < col_end; ++c, slots += d, rows += d) {
        acc_t s = UseMemory ? A::mix(lambda[c], marginal[c], gamma[c]) : A::widen(lambda[c]);
        for (index_t k = 0; k < d; ++k) {
            prefix[k] = s;
            s = A::accumulate(s, msg[slots[k]]);
        }
        marginal[c] = s;
        // Reading η_k again after writing μ_{k+1..d} is safe: each slot is written only once,
        // after its own η has been consumed.
        acc_t t = A::zero();
        for (index_t k = d; k-- > 0;) {
            const auto eta = msg[slots[k]];
            msg[slots[k]] = A::to_msg(A::combine(prefix[k], t));
            t = A::accumulate(t, eta);
        }
        if (A::is_error(s)) [[unlikely]] {
            sink.support[sink.count++] = sink.external[c];
            sink.flip_rows(rows, d);
        }
    }
}

// Largest degree with a fully unrolled kernel; higher degrees use variable_run_dynamic.
inline constexpr index_t max_unrolled_degree = 16;

// Variable pass over columns [col_begin, col_end) ⊆ one degree run.
template <MessageArithmetic A, bool UseMemory>
void variable_run(const RowMajorView& g, const DegreeRun& run, index_t col_begin, index_t col_end,
                  const typename A::msg_t* lambda, const typename A::msg_t* gamma,
                  typename A::acc_t* marginal, typename A::msg_t* msg,
                  typename A::acc_t* scratch, HardDecisionSink& sink) noexcept {
    const bool unrolled = [&]<index_t... Ds>(std::integer_sequence<index_t, Ds...>) {
        return ((run.degree == Ds
                     ? (variable_run_fixed<A, Ds, UseMemory>(g, run, col_begin, col_end, lambda,
                                                             gamma, marginal, msg, sink),
                        true)
                     : false) ||
                ...);
    }(std::make_integer_sequence<index_t, max_unrolled_degree + 1>{});
    if (!unrolled) {
        variable_run_dynamic<A, UseMemory>(g, run, col_begin, col_end, lambda, gamma, marginal,
                                           msg, scratch, sink);
    }
}

// Column-blocked variable pass. The k-th edge of column c of `run` is the slot
// run.slot_begin + k·run.stride + (c − run.col_begin), so consecutive columns of a run have their
// k-th messages side by side and one vector holds the k-th message of `width` columns. Each lane
// performs its column's sums in exactly the scalar order; lanes never interact.
//
// Vector loads may run up to one vector past the end of a slice (into padding, the next plane or
// the arrays' slack), which is why the backend allocates the message and per-column arrays with
// a vector of slack. Stores to a partial vector are masked.

// Any arithmetic policy; degree known only at run time; `scratch` holds `degree` entries.
template <MessageArithmetic A, bool UseMemory>
void variable_run_blocked_scalar(const ColumnBlockedView& g, const DegreeRun& run,
                                 index_t col_begin, index_t col_end,
                                 const typename A::msg_t* lambda, const typename A::msg_t* gamma,
                                 typename A::acc_t* marginal, typename A::msg_t* msg,
                                 typename A::acc_t* scratch, HardDecisionSink& sink) noexcept {
    using acc_t = A::acc_t;
    const index_t d = run.degree;
    const std::size_t stride = run.stride;
    for (index_t c = col_begin; c < col_end; ++c) {
        auto* base = msg + run.slot_begin + (c - run.col_begin);
        acc_t s = UseMemory ? A::mix(lambda[c], marginal[c], gamma[c]) : A::widen(lambda[c]);
        for (index_t k = 0; k < d; ++k) {
            scratch[k] = s;
            s = A::accumulate(s, base[k * stride]);
        }
        marginal[c] = s;
        acc_t t = A::zero();
        for (index_t k = d; k-- > 0;) {
            const auto eta = base[k * stride];
            base[k * stride] = A::to_msg(A::combine(scratch[k], t));
            t = A::accumulate(t, eta);
        }
        if (A::is_error(s)) [[unlikely]] {
            sink.support[sink.count++] = sink.external[c];
            sink.flip_rows(g.column_rows + run.edge_begin +
                               static_cast<std::size_t>(c - run.col_begin) * d,
                           d);
        }
    }
}

// IEEE policies, degree known at compile time, `width` columns per step. A partial last vector
// is loaded and stored under a mask, so no lane outside the slice is touched: the memory beyond
// it may belong to another thread's columns.
template <std::floating_point T, index_t D, bool UseMemory>
void variable_run_blocked_simd(const ColumnBlockedView& g, const DegreeRun& run, index_t col_begin,
                               index_t col_end, const T* lambda, const T* gamma, T* marginal,
                               T* msg, HardDecisionSink& sink) noexcept {
    using V = stdx::native_simd<T>;
    using M = V::mask_type;
    constexpr index_t width = V::size();
    constexpr T max_msg = FloatArith<T>::max_message();
    const V lane_index([](auto lane) { return static_cast<T>(static_cast<int>(lane)); });
    const std::size_t stride = run.stride;

    const auto step = [&]<bool Full>(index_t c, const M& valid) {
        const auto load = [&](const T* p) {
            if constexpr (Full) {
                return V(p, stdx::element_aligned);
            } else {
                V v(T{0});
                stdx::where(valid, v).copy_from(p, stdx::element_aligned);
                return v;
            }
        };
        const auto store = [&](const V& v, T* p) {
            if constexpr (Full) {
                v.copy_to(p, stdx::element_aligned);
            } else {
                stdx::where(valid, v).copy_to(p, stdx::element_aligned);
            }
        };
        T* base = msg + run.slot_begin + (c - run.col_begin);
        const V prior = load(lambda + c);
        V s = prior;
        if constexpr (UseMemory) {
            const V g_c = load(gamma + c);
            const V previous = load(marginal + c);
            const V keep = V(T{1}) - g_c;
            const V prior_part = prior * keep;
            const V memory_part = previous * g_c;
            s = prior_part + memory_part;
            stdx::where(prior == V(max_msg), s) = prior;
        }
        std::array<V, D> eta;    // NOLINT(cppcoreguidelines-pro-type-member-init): as above
        std::array<V, D> prefix; // NOLINT(cppcoreguidelines-pro-type-member-init)
        for (index_t k = 0; k != D; ++k) {
            eta[k] = load(base + k * stride);
        }
        for (index_t k = 0; k != D; ++k) {
            prefix[k] = s;
            s = s + eta[k];
        }
        store(s, marginal + c);
        V t(T{0});
        for (index_t k = D; k-- > 0;) {
            store(prefix[k] + t, base + k * stride);
            t = t + eta[k];
        }
        const M error = (s <= V(T{0})) && valid;
        if (stdx::any_of(error)) [[unlikely]] {
            for (index_t lane = 0; lane < width; ++lane) {
                if (error[lane]) {
                    const index_t column = c + lane;
                    sink.support[sink.count++] = sink.external[column];
                    sink.flip_rows(g.column_rows + run.edge_begin +
                                       static_cast<std::size_t>(column - run.col_begin) * D,
                                   D);
                }
            }
        }
    };

    const M all(true);
    index_t c = col_begin;
    for (; c + width <= col_end; c += width) {
        step.template operator()<true>(c, all);
    }
    if (c < col_end) {
        step.template operator()<false>(c, lane_index < V(static_cast<T>(col_end - c)));
    }
}

// Variable pass over columns [col_begin, col_end) ⊆ one degree run, column-blocked layout.
template <MessageArithmetic A, bool UseMemory>
void variable_run_blocked(const ColumnBlockedView& g, const DegreeRun& run, index_t col_begin,
                          index_t col_end, const typename A::msg_t* lambda,
                          const typename A::msg_t* gamma, typename A::acc_t* marginal,
                          typename A::msg_t* msg, typename A::acc_t* scratch,
                          HardDecisionSink& sink) noexcept {
    bool done = false;
    if constexpr (is_float_arith<A>) {
        done = [&]<index_t... Ds>(std::integer_sequence<index_t, Ds...>) {
            return ((run.degree == Ds
                         ? (variable_run_blocked_simd<typename A::msg_t, Ds, UseMemory>(
                                g, run, col_begin, col_end, lambda, gamma, marginal, msg, sink),
                            true)
                         : false) ||
                    ...);
        }(std::make_integer_sequence<index_t, max_unrolled_degree + 1>{});
    }
    if (!done) {
        variable_run_blocked_scalar<A, UseMemory>(g, run, col_begin, col_end, lambda, gamma,
                                                  marginal, msg, scratch, sink);
    }
}

// ---------------------------------------------------------------------------------------------
// Utilities outside the iteration loop.
// ---------------------------------------------------------------------------------------------

// H·ê == σ, checked row by row with early exit; ê in external column order. For tests and
// assertions: the backend's fused bitset test is what runs in the loop.
[[nodiscard]] inline bool syndrome_matches(const TannerGraph& graph, std::span<const Bit> hard,
                                           std::span<const Bit> syndrome) noexcept {
    for (index_t i = 0; i < graph.num_rows(); ++i) {
        Bit parity = 0;
        for (const index_t c : graph.row_columns(i)) {
            parity ^= hard[graph.external_column(c)];
        }
        if (parity != syndrome[i]) {
            return false;
        }
    }
    return true;
}

// W(ê) = Σ λ_j over the support, skipping infinite λ, summed in the order given (the caller
// passes the support sorted ascending, which fixes the rounding).
[[nodiscard]] inline double solution_weight(std::span<const double> llr,
                                            std::span<const index_t> sorted_support) noexcept {
    double weight = 0.0;
    for (const index_t j : sorted_support) {
        const double l = llr[j];
        if (std::isfinite(l)) {
            weight += l;
        }
    }
    return weight;
}

} // namespace rtd::kernels
