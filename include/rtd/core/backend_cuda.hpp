#pragma once

#include <expected>
#include <span>

#include "rtd/core/arith.hpp"
#include "rtd/core/backend.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/priors.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// The device backend's interface, reserved so the relay controller and the harness can be written
// against it now. The device implementation will hold the messages, marginals, memory strengths
// and hard decisions in device memory and run a leg either as one launch per iteration or as one
// persistent kernel that tests convergence on the device.
//
// Until it exists, construction always fails with BackendError::Code::unavailable, so no object
// of this type is ever live and the members below are never reached.
template <MessageArithmetic A>
class CudaBackend {
public:
    using arith = A;

    [[nodiscard]] static std::expected<CudaBackend, BackendError>
    create(const TannerGraph& /*graph*/, const Priors& /*priors*/) {
        return std::unexpected(BackendError{.code = BackendError::Code::unavailable,
                                            .detail = "the CUDA backend is not implemented yet"});
    }

    [[nodiscard]] index_t num_rows() const noexcept { return 0; }
    [[nodiscard]] index_t num_columns() const noexcept { return 0; }
    void set_convergence_rows(std::span<const Bit> /*mask*/) noexcept {}
    void begin(std::span<const Bit> /*syndrome*/, bool /*init_marginals*/) noexcept {}
    void set_gamma(std::span<const double> /*gammas*/) noexcept {}
    void set_gamma(double /*gamma*/) noexcept {}
    [[nodiscard]] LegOutcome run_leg(const LegParams& /*params*/) noexcept { return {}; }
    void mark_best() noexcept {}
    [[nodiscard]] std::span<const Bit> best_hard() noexcept { return {}; }
    [[nodiscard]] std::span<const Bit> current_hard() noexcept { return {}; }
    [[nodiscard]] std::span<const index_t> best_support() const noexcept { return {}; }
    [[nodiscard]] std::span<const index_t> current_support() noexcept { return {}; }
    void read_marginals(std::span<double> /*out*/) const noexcept {}

private:
    CudaBackend() = default;
};

} // namespace rtd
