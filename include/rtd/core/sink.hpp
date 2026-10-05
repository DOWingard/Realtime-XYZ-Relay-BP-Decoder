#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "rtd/core/types.hpp"

namespace rtd {

// One converged leg's solution ê, as the relay controller hands it to a SolutionSink.
struct SolutionEvent {
    std::uint32_t leg = 0; // 0 = leg 0
    // Iterations of every leg up to and including this one.
    std::uint32_t cumulative_iterations = 0;
    // W(ê) = Σ_{ê_j=1, λ_j finite} λ_j, exactly as the backend computed it for the leg.
    double weight = 0.0;
    // Columns j with ê_j = 1, ascending external indices; valid only during the call.
    std::span<const index_t> support;
};

// Observes every converged leg of a decode, in leg order (leg 0 included), each before the
// stopping rule is evaluated. Like the backend tracer it is a compile-time policy: when `enabled`
// is false the controller never fetches the support and the calls compile away. Both calls run
// inside a decode, so they must not allocate, throw, lock or log.
template <class S>
concept SolutionSink = std::movable<S> && requires(S s, const SolutionEvent& event) {
    typename std::bool_constant<S::enabled>;
    { s.on_decode_begin() } noexcept -> std::same_as<void>;
    { s.on_solution(event) } noexcept -> std::same_as<void>;
};

// The default sink: records nothing.
struct NoSink {
    static constexpr bool enabled = false;
    void on_decode_begin() noexcept {}
    void on_solution(const SolutionEvent& /*event*/) noexcept {}
};

// The SplitMix64 output function (Steele, Lea, Flood) applied to x + φ·2^64: a bijection of
// 64-bit words with full avalanche. Written out here, not taken from a library, because the Python
// analysis must reproduce the solution hashes bit for bit.
[[nodiscard]] constexpr std::uint64_t splitmix64_hash(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

// Identifies a solution by its support: h = 0, then h ← splitmix64_hash(h ⊕ j) for each column j
// in ascending order. Two legs that land on the same ê get the same hash, so distinct solutions
// can be counted without storing supports; distinct supports collide with probability ≈ 2^−64
// per pair.
[[nodiscard]] constexpr std::uint64_t solution_hash(std::span<const index_t> support) noexcept {
    std::uint64_t hash = 0;
    for (const index_t j : support) {
        hash = splitmix64_hash(hash ^ j);
    }
    return hash;
}

// The logical class A·ê packed into k ≤ 64 bits: bit o is the parity of observable o over the
// support, i.e. the XOR of the columns' masks. `column_class[j]` is column j of A as a bit mask.
[[nodiscard]] constexpr std::uint64_t solution_class(std::span<const std::uint64_t> column_class,
                                                     std::span<const index_t> support) noexcept {
    std::uint64_t mask = 0;
    for (const index_t j : support) {
        mask ^= column_class[j];
    }
    return mask;
}

// What RecordingSink keeps of one solution.
struct SolutionRecord {
    std::uint32_t leg = 0;
    std::uint32_t cumulative_iterations = 0;
    double weight = 0.0;
    std::uint64_t logical_class = 0; // solution_class of the support
    std::uint64_t hash = 0;          // solution_hash of the support
    std::uint32_t size = 0;          // |support|
};

struct SinkError {
    enum class Code : std::uint8_t { invalid_capacity };
    Code code;
    std::string detail;
};

[[nodiscard]] constexpr std::string_view to_string(SinkError::Code code) noexcept {
    switch (code) {
    case SinkError::Code::invalid_capacity:
        return "invalid_capacity";
    }
    return "unknown";
}

// Records up to `capacity` solutions per decode: leg, cumulative iterations, weight, logical class
// and hash, from one pass over each support. Solutions beyond the capacity are counted, not
// stored. The storage is a fixed array inside the object, so recording never allocates.
//
// `column_class` (one k-bit mask per column, external order) is borrowed and must outlive the
// sink; RelayDecoder::create checks that it has one entry per column.
class RecordingSink {
public:
    static constexpr bool enabled = true;
    static constexpr std::uint32_t max_capacity = 20;

    [[nodiscard]] static std::expected<RecordingSink, SinkError>
    create(std::uint32_t capacity, std::span<const std::uint64_t> column_class) {
        if (!valid_capacity(capacity)) {
            return std::unexpected(SinkError{
                .code = SinkError::Code::invalid_capacity,
                .detail = std::format("solution capacity must be in [1, {}], got {}",
                                      max_capacity, capacity)});
        }
        return RecordingSink(column_class, capacity);
    }

    // Throws std::invalid_argument unless 1 ≤ capacity ≤ max_capacity; create() reports the same
    // condition as a value.
    RecordingSink(std::uint32_t capacity, std::span<const std::uint64_t> column_class)
        : RecordingSink(column_class, checked(capacity)) {}

    void on_decode_begin() noexcept { found_ = 0; }

    void on_solution(const SolutionEvent& event) noexcept {
        if (found_ < capacity_) {
            std::uint64_t mask = 0;
            std::uint64_t hash = 0;
            for (const index_t j : event.support) {
                mask ^= column_class_[j];
                hash = splitmix64_hash(hash ^ j);
            }
            records_[found_] = SolutionRecord{
                .leg = event.leg,
                .cumulative_iterations = event.cumulative_iterations,
                .weight = event.weight,
                .logical_class = mask,
                .hash = hash,
                .size = static_cast<std::uint32_t>(event.support.size())};
        }
        ++found_;
    }

    // The solutions of the last decode, in leg order, at most capacity() of them.
    [[nodiscard]] std::span<const SolutionRecord> records() const noexcept {
        return {records_.data(), std::min(found_, capacity_)};
    }
    // Converged legs of the last decode, including those beyond the capacity.
    [[nodiscard]] std::uint32_t found() const noexcept { return found_; }
    // Solutions of the last decode that were counted but not stored.
    [[nodiscard]] std::uint32_t overflow() const noexcept {
        return found_ > capacity_ ? found_ - capacity_ : 0;
    }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] index_t num_columns() const noexcept {
        return static_cast<index_t>(column_class_.size());
    }
    [[nodiscard]] std::span<const std::uint64_t> column_class() const noexcept {
        return column_class_;
    }
    // Logical class of any support over this sink's columns (e.g. the returned ê when no leg
    // converged).
    [[nodiscard]] std::uint64_t class_of(std::span<const index_t> support) const noexcept {
        return solution_class(column_class_, support);
    }

private:
    RecordingSink(std::span<const std::uint64_t> column_class, std::uint32_t capacity) noexcept
        : column_class_(column_class), capacity_(capacity) {}

    [[nodiscard]] static constexpr bool valid_capacity(std::uint32_t capacity) noexcept {
        return capacity >= 1 && capacity <= max_capacity;
    }

    static std::uint32_t checked(std::uint32_t capacity) {
        if (!valid_capacity(capacity)) {
            throw std::invalid_argument(std::format(
                "RecordingSink capacity must be in [1, {}], got {}", max_capacity, capacity));
        }
        return capacity;
    }

    std::span<const std::uint64_t> column_class_;
    std::uint32_t capacity_;
    std::uint32_t found_ = 0;
    std::array<SolutionRecord, max_capacity> records_{};
};

static_assert(SolutionSink<NoSink>);
static_assert(SolutionSink<RecordingSink>);

} // namespace rtd
