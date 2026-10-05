#include "rtd/core/backend_cuda.hpp"

#include "rtd/core/relay.hpp"
#include "rtd/core/sink.hpp"

namespace rtd {

// The reserved device backend must keep satisfying the controller's requirements, with and
// without solution recording.
static_assert(LegBackend<CudaBackend<F32>>);
static_assert(LegBackend<CudaBackend<F64>>);
template class RelayDecoder<CudaBackend<F32>>;
template class RelayDecoder<CudaBackend<F32>, RecordingSink>;

} // namespace rtd
