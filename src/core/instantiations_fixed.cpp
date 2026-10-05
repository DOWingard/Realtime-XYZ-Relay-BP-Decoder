#include "rtd/core/decoder_fixed.hpp"

namespace rtd {

template class CpuBackend<Int4_2_8, Serial>;
template class CpuBackend<Int5_2_8, Serial>;
template class CpuBackend<Int6_2_8, Serial>;
template class CpuBackend<Int4_2_8, Team>;
template class CpuBackend<Int5_2_8, Team>;
template class CpuBackend<Int6_2_8, Team>;
template class RelayDecoder<CpuBackend<Int4_2_8, Serial>, NoSink>;
template class RelayDecoder<CpuBackend<Int5_2_8, Serial>, NoSink>;
template class RelayDecoder<CpuBackend<Int6_2_8, Serial>, NoSink>;
template class RelayDecoder<CpuBackend<Int4_2_8, Team>, NoSink>;
template class RelayDecoder<CpuBackend<Int5_2_8, Team>, NoSink>;
template class RelayDecoder<CpuBackend<Int6_2_8, Team>, NoSink>;

} // namespace rtd
