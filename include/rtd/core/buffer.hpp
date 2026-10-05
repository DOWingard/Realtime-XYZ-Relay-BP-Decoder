#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

#include "rtd/core/types.hpp"

namespace rtd {

// Owning, fixed-size, cache-line-aligned array of a trivially copyable type.
//
// Used for every buffer the kernels touch: alignment makes whole-vector loads legal and keeps two
// threads' data off the same cache line. Construction without a fill value leaves the contents
// uninitialised, so allocating the message array or a large shot file costs no page-touching
// memset. Move-only; copies are explicit through clone().
template <class T>
    requires std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>
class AlignedBuffer {
public:
    static constexpr std::size_t alignment = std::max(cache_line_bytes, alignof(T));

    AlignedBuffer() noexcept = default;

    explicit AlignedBuffer(std::size_t size) : data_(allocate(size)), size_(size) {}

    AlignedBuffer(std::size_t size, const T& value) : AlignedBuffer(size) {
        std::fill_n(data_.get(), size_, value);
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(std::move(other.data_)), size_(std::exchange(other.size_, 0)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        data_ = std::move(other.data_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    ~AlignedBuffer() = default;

    [[nodiscard]] AlignedBuffer clone() const {
        AlignedBuffer copy(size_);
        std::copy_n(data_.get(), size_, copy.data_.get());
        return copy;
    }

    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::span<T> span() noexcept { return {data_.get(), size_}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {data_.get(), size_}; }

    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    [[nodiscard]] T* begin() noexcept { return data_.get(); }
    [[nodiscard]] T* end() noexcept { return data_.get() + size_; }
    [[nodiscard]] const T* begin() const noexcept { return data_.get(); }
    [[nodiscard]] const T* end() const noexcept { return data_.get() + size_; }

    void fill(const T& value) noexcept { std::fill_n(data_.get(), size_, value); }

private:
    struct Deleter {
        void operator()(T* p) const noexcept { ::operator delete(p, std::align_val_t{alignment}); }
    };
    // The array form of unique_ptr is what owns a runtime-sized block with a custom deleter.
    using Storage = std::unique_ptr<T[], Deleter>; // NOLINT(*-avoid-c-arrays)

    // ::operator new implicitly creates objects of implicit-lifetime types, which every
    // trivially copyable, trivially destructible T is.
    static Storage allocate(std::size_t size) {
        if (size == 0) {
            return nullptr;
        }
        if (size > (static_cast<std::size_t>(-1) - alignment) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        // Round the allocation up to whole cache lines so a vector load that starts inside the
        // buffer never crosses into memory the allocator handed to someone else.
        const std::size_t bytes = round_up(size * sizeof(T), alignment);
        return Storage(static_cast<T*>(::operator new(bytes, std::align_val_t{alignment})));
    }

    Storage data_;
    std::size_t size_ = 0;
};

} // namespace rtd
