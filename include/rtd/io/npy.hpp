#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <mdspan>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "rtd/core/buffer.hpp"
#include "rtd/io/error.hpp"

// NumPy .npy files: a short ASCII header (a Python dict literal giving dtype, memory order and
// shape) followed by the raw array. Only little-endian C-order arrays are accepted.

namespace rtd::io {

enum class DType : std::uint8_t { u1, b1, i4, u4, i8, u8, f4, f8 };

[[nodiscard]] std::string_view to_string(DType dtype) noexcept;
[[nodiscard]] std::size_t item_size(DType dtype) noexcept;

template <class T>
struct dtype_of;
template <>
struct dtype_of<std::uint8_t> {
    static constexpr DType value = DType::u1;
};
template <>
struct dtype_of<std::int32_t> {
    static constexpr DType value = DType::i4;
};
template <>
struct dtype_of<std::uint32_t> {
    static constexpr DType value = DType::u4;
};
template <>
struct dtype_of<std::int64_t> {
    static constexpr DType value = DType::i8;
};
template <>
struct dtype_of<std::uint64_t> {
    static constexpr DType value = DType::u8;
};
template <>
struct dtype_of<float> {
    static constexpr DType value = DType::f4;
};
template <>
struct dtype_of<double> {
    static constexpr DType value = DType::f8;
};

template <class T>
concept NpyElement = requires { dtype_of<T>::value; };

struct NpyHeader {
    DType dtype;
    std::vector<std::size_t> shape;
    std::size_t data_offset; // bytes from the start of the file to the array
    [[nodiscard]] std::size_t count() const noexcept;
};

// Parses the header of an .npy file held in `bytes` (the whole file or at least its header).
[[nodiscard]] std::expected<NpyHeader, IoError> parse_npy_header(std::span<const std::byte> bytes,
                                                                 std::string_view path);

[[nodiscard]] std::expected<NpyHeader, IoError> read_npy_header(const std::filesystem::path& path);

// An array read from an .npy file, in C order.
template <NpyElement T>
struct NpyArray {
    AlignedBuffer<T> data;
    std::vector<std::size_t> shape;

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
    [[nodiscard]] std::span<const T> span() const noexcept { return data.span(); }
    [[nodiscard]] std::size_t rows() const noexcept { return shape.empty() ? 0 : shape[0]; }
    [[nodiscard]] std::size_t cols() const noexcept { return shape.size() < 2 ? 1 : shape[1]; }

    // Row r of a 2-D array.
    [[nodiscard]] std::span<const T> row(std::size_t r) const noexcept {
        return {data.data() + r * cols(), cols()};
    }
    [[nodiscard]] std::mdspan<const T, std::dextents<std::size_t, 2>> matrix() const noexcept {
        return std::mdspan<const T, std::dextents<std::size_t, 2>>(data.data(), rows(), cols());
    }
};

// Reads an array of element type T. The file's dtype must be T's (a numpy bool array also reads
// as uint8). With `rank`, the number of dimensions must match; each entry of `extents` that is
// set must match that dimension.
template <NpyElement T>
[[nodiscard]] std::expected<NpyArray<T>, IoError>
read_npy(const std::filesystem::path& path, std::optional<std::size_t> rank = std::nullopt,
         std::span<const std::optional<std::size_t>> extents = {});

template <NpyElement T>
[[nodiscard]] std::expected<void, IoError> write_npy(const std::filesystem::path& path,
                                                     std::span<const T> data,
                                                     std::span<const std::size_t> shape);

} // namespace rtd::io
