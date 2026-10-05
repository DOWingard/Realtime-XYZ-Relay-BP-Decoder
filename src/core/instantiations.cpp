#include "rtd/core/decoder.hpp"

namespace rtd {

template class CpuBackend<F32, Serial>;
template class CpuBackend<F64, Serial>;
template class CpuBackend<F32, Team>;
template class CpuBackend<F64, Team>;
template class RelayDecoder<CpuBackend<F32, Serial>, NoSink>;
template class RelayDecoder<CpuBackend<F64, Serial>, NoSink>;
template class RelayDecoder<CpuBackend<F32, Team>, NoSink>;
template class RelayDecoder<CpuBackend<F64, Team>, NoSink>;
template class RelayDecoder<CpuBackend<F32, Serial>, RecordingSink>;
template class RelayDecoder<CpuBackend<F64, Serial>, RecordingSink>;
template class RelayDecoder<CpuBackend<F32, Team>, RecordingSink>;
template class RelayDecoder<CpuBackend<F64, Team>, RecordingSink>;

} // namespace rtd
