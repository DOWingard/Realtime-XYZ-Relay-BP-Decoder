#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "rtd/core/types.hpp"

namespace rtd {

// Per-variable memory strengths γ_j for the relay legs.
//
// A source is immutable and safe to share between threads: every value is a pure function of
// (stream, leg, j). `stream` identifies the decoding problem (the harness passes the shot index),
// so a run is reproducible regardless of how shots are split across threads.
class GammaSource {
public:
    GammaSource() = default;
    GammaSource(const GammaSource&) = default;
    GammaSource& operator=(const GammaSource&) = default;
    GammaSource(GammaSource&&) = default;
    GammaSource& operator=(GammaSource&&) = default;
    virtual ~GammaSource() = default;

    // Number of variables n.
    [[nodiscard]] virtual index_t width() const noexcept = 0;

    // γ for relay leg `leg` ≥ 1 of problem `stream`, in external column order. Returns either a
    // view of the source's own storage or `scratch` (of size width()) after filling it.
    [[nodiscard]] virtual std::span<const double> gammas(std::uint64_t stream, std::uint32_t leg,
                                                         std::span<double> scratch) const = 0;
};

struct GammaError {
    enum class Code : std::uint8_t { empty_table, shape_mismatch, not_finite, invalid_interval };
    Code code;
    std::string detail;
};

// A fixed table of T rows × n columns; relay leg r uses row r mod T for every problem.
// This is the input format for bit-identity checks against golden outputs.
class ExplicitGammaTable final : public GammaSource {
public:
    [[nodiscard]] static std::expected<ExplicitGammaTable, GammaError>
    create(std::vector<double> table, std::size_t rows, std::size_t width);

    [[nodiscard]] index_t width() const noexcept override { return width_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    // True when num_sets legs need more rows than the table has, so rows are reused.
    [[nodiscard]] bool reuses_rows(std::uint32_t num_sets) const noexcept {
        return num_sets > rows_;
    }

    [[nodiscard]] std::span<const double> gammas(std::uint64_t stream, std::uint32_t leg,
                                                 std::span<double> scratch) const override;

private:
    ExplicitGammaTable(std::vector<double> table, std::size_t rows, index_t width)
        : table_(std::move(table)), rows_(rows), width_(width) {}

    std::vector<double> table_;
    std::size_t rows_;
    index_t width_;
};

// γ_j drawn independently and uniformly from [low, high) for every (stream, leg, j).
//
// Generator: each (seed, stream, leg) triple is hashed by SplitMix64 into the 256-bit state of a
// xoshiro256+ generator (Blackman and Vigna), whose output x gives u = (x >> 11)·2^−53 ∈ [0, 1)
// and γ = low + (high − low)·u, nudged below `high` if rounding lands on it. Both algorithms are
// fully specified, so the values are portable across compilers and platforms.
class UniformGammaGenerator final : public GammaSource {
public:
    [[nodiscard]] static std::expected<UniformGammaGenerator, GammaError>
    create(std::uint64_t seed, double low, double high, index_t width);

    [[nodiscard]] index_t width() const noexcept override { return width_; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }
    [[nodiscard]] double low() const noexcept { return low_; }
    [[nodiscard]] double high() const noexcept { return high_; }

    [[nodiscard]] std::span<const double> gammas(std::uint64_t stream, std::uint32_t leg,
                                                 std::span<double> scratch) const override;

private:
    UniformGammaGenerator(std::uint64_t seed, double low, double high, index_t width)
        : seed_(seed), low_(low), high_(high), width_(width) {}

    std::uint64_t seed_;
    double low_;
    double high_;
    index_t width_;
};

} // namespace rtd
