#pragma once

#include <cstddef>
#include <cstdint>

namespace rtd {

// Row, column, edge and slot indices. The gross-code XYZ problem has n = 71,280 columns and
// E = 419,904 edges: too many for 16 bits, far below 2^32.
using index_t = std::uint32_t;

// One GF(2) value per byte (syndrome bits σ_i, hard decisions ê_j).
using Bit = std::uint8_t;

// Cache-line size used for alignment and false-sharing padding.
inline constexpr std::size_t cache_line_bytes = 64;

// Every row of the row-major message layout starts on a multiple of this many slots, so the check
// pass runs on whole aligned vectors (16 floats = one cache line; also a whole number of AVX-512
// vectors) and never needs a scalar tail.
inline constexpr index_t row_alignment_slots = 16;

[[nodiscard]] constexpr std::size_t round_up(std::size_t value, std::size_t multiple) noexcept {
    return (value + multiple - 1) / multiple * multiple;
}

} // namespace rtd
