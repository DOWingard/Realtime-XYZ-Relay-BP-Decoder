// Replaces the global allocation functions with ones that count every allocation, so that a test
// can check that a piece of code allocates nothing. They forward to malloc / aligned_alloc and
// free, as the default ones do; the nothrow and array forms reach them through the standard
// library's defaults. They live in a translation unit of their own so that the compiler never
// inlines them into code whose new / delete pairs it then checks. Sanitized builds keep the
// sanitizer's own functions (see the header).

#include "unit/allocation_counter.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {

std::atomic<std::uint64_t>& counter() noexcept {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

} // namespace

std::uint64_t rtd::window::test::allocations() noexcept { return counter().load(); }

#ifndef RTD_TEST_SANITIZED_ALLOCATOR

void* operator new(std::size_t size) {
    counter().fetch_add(1, std::memory_order_relaxed);
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
    if (void* p = std::malloc(std::max<std::size_t>(size, 1))) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    counter().fetch_add(1, std::memory_order_relaxed);
    const auto align = static_cast<std::size_t>(alignment);
    const std::size_t bytes = (std::max<std::size_t>(size, 1) + align - 1) / align * align;
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
    if (void* p = std::aligned_alloc(align, bytes)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept {
    std::free(p); // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
}

void operator delete(void* p, std::size_t /*size*/) noexcept {
    std::free(p); // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
}

void operator delete(void* p, std::align_val_t /*alignment*/) noexcept {
    std::free(p); // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
}

void operator delete(void* p, std::size_t /*size*/, std::align_val_t /*alignment*/) noexcept {
    std::free(p); // NOLINT(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
}

#endif
