#pragma once

// The decoder configurations compiled into rtd_core. Including this header instead of the
// individual ones avoids re-instantiating the kernels in every translation unit.

#include "rtd/core/arith.hpp"
#include "rtd/core/backend_cpu.hpp"
#include "rtd/core/executor.hpp"
#include "rtd/core/relay.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/core/trace.hpp"

namespace rtd {

template <MessageArithmetic A, class Executor = Serial, class Sink = NoSink>
using CpuRelayDecoder = RelayDecoder<CpuBackend<A, Executor>, Sink>;

extern template class CpuBackend<F32, Serial>;
extern template class CpuBackend<F64, Serial>;
extern template class CpuBackend<F32, Team>;
extern template class CpuBackend<F64, Team>;
extern template class RelayDecoder<CpuBackend<F32, Serial>, NoSink>;
extern template class RelayDecoder<CpuBackend<F64, Serial>, NoSink>;
extern template class RelayDecoder<CpuBackend<F32, Team>, NoSink>;
extern template class RelayDecoder<CpuBackend<F64, Team>, NoSink>;
extern template class RelayDecoder<CpuBackend<F32, Serial>, RecordingSink>;
extern template class RelayDecoder<CpuBackend<F64, Serial>, RecordingSink>;
extern template class RelayDecoder<CpuBackend<F32, Team>, RecordingSink>;
extern template class RelayDecoder<CpuBackend<F64, Team>, RecordingSink>;

static_assert(LegBackend<CpuBackend<F32, Serial>>);
static_assert(LegBackend<CpuBackend<F64, Team>>);

} // namespace rtd
