#pragma once

// The fixed-point decoder configurations compiled into rtd_core, next to the floating-point ones
// of decoder.hpp. Including this header instead of instantiating the kernels again in every
// translation unit keeps build times down.

#include "rtd/core/decoder.hpp"
#include "rtd/core/fixed_arith.hpp"

namespace rtd {

extern template class CpuBackend<Int4_2_8, Serial>;
extern template class CpuBackend<Int5_2_8, Serial>;
extern template class CpuBackend<Int6_2_8, Serial>;
extern template class CpuBackend<Int4_2_8, Team>;
extern template class CpuBackend<Int5_2_8, Team>;
extern template class CpuBackend<Int6_2_8, Team>;
extern template class RelayDecoder<CpuBackend<Int4_2_8, Serial>, NoSink>;
extern template class RelayDecoder<CpuBackend<Int5_2_8, Serial>, NoSink>;
extern template class RelayDecoder<CpuBackend<Int6_2_8, Serial>, NoSink>;
extern template class RelayDecoder<CpuBackend<Int4_2_8, Team>, NoSink>;
extern template class RelayDecoder<CpuBackend<Int5_2_8, Team>, NoSink>;
extern template class RelayDecoder<CpuBackend<Int6_2_8, Team>, NoSink>;

static_assert(LegBackend<CpuBackend<Int4_2_8, Serial>>);
static_assert(LegBackend<CpuBackend<Int6_2_8, Team>>);

} // namespace rtd
