#include "rtd/harness/recording.hpp"

#include <format>

namespace rtd {

template class RelayDecoder<CpuBackend<F32, Serial>, harness::SolutionRecorder>;
template class RelayDecoder<CpuBackend<F64, Serial>, harness::SolutionRecorder>;
template class RelayDecoder<CpuBackend<F32, Team>, harness::SolutionRecorder>;
template class RelayDecoder<CpuBackend<F64, Team>, harness::SolutionRecorder>;

} // namespace rtd

namespace rtd::harness {

std::expected<std::vector<std::uint64_t>, std::string>
column_classes(const ObservableMatrix& observables) {
    constexpr index_t max_observables = 64;
    if (observables.num_rows() > max_observables) {
        return std::unexpected(std::format(
            "a logical class packs the observables into one 64-bit mask, but the problem has "
            "k = {} observables",
            observables.num_rows()));
    }
    std::vector<std::uint64_t> classes(observables.num_columns(), 0);
    for (index_t j = 0; j < observables.num_columns(); ++j) {
        for (const index_t o : observables.column(j)) {
            classes[j] ^= std::uint64_t{1} << o;
        }
    }
    return classes;
}

} // namespace rtd::harness
