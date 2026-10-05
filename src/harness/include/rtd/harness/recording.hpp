#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "rtd/core/buffer.hpp"
#include "rtd/core/decoder.hpp"
#include "rtd/core/observables.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/core/types.hpp"

namespace rtd::harness {

// Column j of the observable matrix A as a k-bit mask, bit o set when fault j flips observable o.
// The logical class A·ê of a correction is then the XOR of the masks over its support. Fails
// when k > 64, since a class must fit one 64-bit word.
[[nodiscard]] std::expected<std::vector<std::uint64_t>, std::string>
column_classes(const ObservableMatrix& observables);

// The batch driver's solution sink: RecordingSink's record of every solution (leg, cumulative
// iterations, weight, logical class, hash, size) and, when asked, a copy of each recorded
// solution's support. The supports of one decode are stored back to back in a buffer allocated at
// construction for `capacity` supports of up to n columns each, so a decode never allocates; the
// buffer is left uninitialised, and pages that no support reaches are never touched.
class SolutionRecorder {
public:
    static constexpr bool enabled = true;

    SolutionRecorder(RecordingSink records, bool keep_supports)
        : records_(records),
          supports_(keep_supports ? std::size_t{records.capacity()} * records.num_columns() : 0),
          keep_supports_(keep_supports) {}

    void on_decode_begin() noexcept {
        records_.on_decode_begin();
        stored_ = 0;
    }

    void on_solution(const SolutionEvent& event) noexcept {
        const bool stored = records_.found() < records_.capacity();
        records_.on_solution(event);
        if (keep_supports_ && stored) {
            // Each support has at most n entries and at most `capacity` are stored, so the
            // buffer cannot overflow.
            const std::size_t begin = ends_[stored_];
            std::ranges::copy(event.support, supports_.begin() + begin);
            ends_[stored_ + 1] = begin + event.support.size();
            ++stored_;
        }
    }

    // The solutions of the last decode, in leg order, at most capacity() of them.
    [[nodiscard]] std::span<const SolutionRecord> records() const noexcept {
        return records_.records();
    }
    // Converged legs of the last decode, including those beyond the capacity.
    [[nodiscard]] std::uint32_t found() const noexcept { return records_.found(); }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return records_.capacity(); }
    [[nodiscard]] index_t num_columns() const noexcept { return records_.num_columns(); }
    [[nodiscard]] bool keeps_supports() const noexcept { return keep_supports_; }
    // Logical class of any support over this sink's columns (the returned ê included).
    [[nodiscard]] std::uint64_t class_of(std::span<const index_t> support) const noexcept {
        return records_.class_of(support);
    }
    // The supports of records(), concatenated in record order (record s has records()[s].size
    // entries); empty unless supports are kept.
    [[nodiscard]] std::span<const index_t> supports() const noexcept {
        return {supports_.data(), ends_[stored_]};
    }

private:
    RecordingSink records_;
    AlignedBuffer<index_t> supports_;
    // ends_[s + 1] is where stored support s ends in supports_.
    std::array<std::size_t, RecordingSink::max_capacity + 1> ends_{};
    std::uint32_t stored_ = 0;
    bool keep_supports_;
};

static_assert(SolutionSink<SolutionRecorder>);

} // namespace rtd::harness

// Compiled once, in recording.cpp.
namespace rtd {
extern template class RelayDecoder<CpuBackend<F32, Serial>, harness::SolutionRecorder>;
extern template class RelayDecoder<CpuBackend<F64, Serial>, harness::SolutionRecorder>;
extern template class RelayDecoder<CpuBackend<F32, Team>, harness::SolutionRecorder>;
extern template class RelayDecoder<CpuBackend<F64, Team>, harness::SolutionRecorder>;
} // namespace rtd
