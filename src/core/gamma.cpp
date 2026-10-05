#include "rtd/core/gamma.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <format>
#include <limits>

namespace rtd {

std::expected<ExplicitGammaTable, GammaError>
ExplicitGammaTable::create(std::vector<double> table, std::size_t rows, std::size_t width) {
    using Code = GammaError::Code;
    if (rows == 0 || width == 0) {
        return std::unexpected(
            GammaError{.code = Code::empty_table,
                       .detail = std::format("gamma table is {} x {}", rows, width)});
    }
    if (width > std::numeric_limits<index_t>::max() || table.size() != rows * width) {
        return std::unexpected(
            GammaError{.code = Code::shape_mismatch,
                       .detail = std::format("gamma table has {} values, expected {} x {}",
                                             table.size(), rows, width)});
    }
    if (const auto bad = std::ranges::find_if(table, [](double g) { return !std::isfinite(g); });
        bad != table.end()) {
        const auto at = static_cast<std::size_t>(bad - table.begin());
        return std::unexpected(GammaError{.code = Code::not_finite,
                                          .detail = std::format("gamma[{}][{}] = {} is not finite",
                                                                at / width, at % width, *bad)});
    }
    return ExplicitGammaTable(std::move(table), rows, static_cast<index_t>(width));
}

std::span<const double> ExplicitGammaTable::gammas(std::uint64_t /*stream*/, std::uint32_t leg,
                                                   std::span<double> /*scratch*/) const {
    const std::size_t row = leg % rows_;
    return {table_.data() + row * width_, width_};
}

namespace {

constexpr std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

class Xoshiro256Plus {
public:
    explicit constexpr Xoshiro256Plus(std::uint64_t key) noexcept {
        for (std::uint64_t& word : s_) {
            word = splitmix64(key);
        }
    }

    constexpr std::uint64_t next() noexcept {
        const std::uint64_t result = s_[0] + s_[3];
        const std::uint64_t t = s_[1] << 17U;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

private:
    std::array<std::uint64_t, 4> s_{};
};

} // namespace

std::expected<UniformGammaGenerator, GammaError>
UniformGammaGenerator::create(std::uint64_t seed, double low, double high, index_t width) {
    if (!std::isfinite(low) || !std::isfinite(high) || low >= high) {
        return std::unexpected(
            GammaError{.code = GammaError::Code::invalid_interval,
                       .detail = std::format("gamma interval [{}, {}) is empty or not "
                                             "finite",
                                             low, high)});
    }
    if (width == 0) {
        return std::unexpected(
            GammaError{.code = GammaError::Code::empty_table, .detail = "gamma width is 0"});
    }
    return UniformGammaGenerator(seed, low, high, width);
}

std::span<const double> UniformGammaGenerator::gammas(std::uint64_t stream, std::uint32_t leg,
                                                      std::span<double> scratch) const {
    std::uint64_t key = seed_;
    key = splitmix64(key) ^ stream;
    key = splitmix64(key) ^ leg;
    Xoshiro256Plus rng(splitmix64(key));
    const double span = high_ - low_;
    const double below_high = std::nextafter(high_, low_);
    constexpr double unit = 0x1.0p-53;
    for (double& g : scratch.first(width_)) {
        const double u = static_cast<double>(rng.next() >> 11U) * unit;
        g = std::min(low_ + span * u, below_high);
    }
    return scratch.first(width_);
}

} // namespace rtd
