#pragma once

#include <cstdint>

// AddressSanitizer and ThreadSanitizer supply the global allocation functions themselves and check
// through them that every allocation is freed by the matching form (new by delete, new[] by
// delete[]). Replacing them would switch those checks off for the whole test program, so
// sanitized builds keep theirs and count nothing.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define RTD_TEST_SANITIZED_ALLOCATOR
#elifdef __has_feature
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define RTD_TEST_SANITIZED_ALLOCATOR
#endif
#endif

namespace rtd::window::test {

#ifdef RTD_TEST_SANITIZED_ALLOCATOR
inline constexpr bool allocations_counted = false;
#else
inline constexpr bool allocations_counted = true;
#endif

// Heap allocations made by the test program so far, through any form of global operator new
// (allocation_counter.cpp replaces them with counting forwarders to malloc / aligned_alloc).
// Always 0 when allocations_counted is false.
[[nodiscard]] std::uint64_t allocations() noexcept;

} // namespace rtd::window::test
