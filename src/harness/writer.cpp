#include "rtd/harness/writer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <span>
#include <vector>

#include "rtd/io/npy.hpp"
#include "rtd/window/spec.hpp"

namespace rtd::harness {

Proportion wilson(std::size_t hits, std::size_t trials, double z) {
    Proportion p{.hits = hits, .trials = trials, .estimate = 0.0, .low = 0.0, .high = 1.0};
    if (trials == 0) {
        return p;
    }
    const auto n = static_cast<double>(trials);
    const double phat = static_cast<double>(hits) / n;
    const double z2 = z * z;
    const double denominator = 1.0 + (z2 / n);
    const double centre = (phat + (z2 / (2.0 * n))) / denominator;
    const double half = z * std::sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))) / denominator;
    p.estimate = phat;
    // The bounds are exactly 0 and 1 at the extremes; the formula leaves rounding residue there.
    p.low = hits == 0 ? 0.0 : std::max(0.0, centre - half);
    p.high = hits == trials ? 1.0 : std::min(1.0, centre + half);
    return p;
}

namespace {
// False for NaN as well as for values outside [0, 1].
bool is_probability(double p) noexcept { return p >= 0.0 && p <= 1.0; }
} // namespace

// Both rates are small differences from 1; log1p and expm1 keep their relative precision.
std::optional<double> ler_per_cycle(double block_error_rate, std::uint32_t rounds) {
    if (rounds == 0 || !is_probability(block_error_rate)) {
        return std::nullopt;
    }
    return -std::expm1(std::log1p(-block_error_rate) / rounds);
}

std::optional<double> ler_per_qubit_per_cycle(double block_error_rate,
                                              std::uint32_t logical_qubits, std::uint32_t rounds) {
    if (logical_qubits == 0 || rounds == 0 || !is_probability(block_error_rate)) {
        return std::nullopt;
    }
    // f = 2(1 − P)^(1/k) − 1 is the per-qubit fidelity 1 − 2q over all R cycles.
    const double fidelity_minus_one =
        2.0 * std::expm1(std::log1p(-block_error_rate) / logical_qubits);
    if (fidelity_minus_one <= -1.0) {
        return std::nullopt;
    }
    return -std::expm1(std::log1p(fidelity_minus_one) / rounds) / 2.0;
}

template <class T>
T nearest_rank(std::span<const T> sorted, double q) {
    if (sorted.empty()) {
        return T{};
    }
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    return sorted[std::clamp<std::size_t>(rank, 1, sorted.size()) - 1];
}

template std::uint32_t nearest_rank(std::span<const std::uint32_t>, double);
template std::uint64_t nearest_rank(std::span<const std::uint64_t>, double);

namespace {

template <class T>
nlohmann::json distribution(std::vector<T> values, double scale = 1.0) {
    if (values.empty()) {
        return nullptr;
    }
    std::ranges::sort(values);
    const std::span<const T> sorted(values);
    const double sum = std::accumulate(values.begin(), values.end(), 0.0,
                                       [](double acc, T v) { return acc + static_cast<double>(v); });
    const auto at = [&](double q) { return static_cast<double>(nearest_rank(sorted, q)) * scale; };
    return {{"mean", sum / static_cast<double>(values.size()) * scale},
            {"min", static_cast<double>(sorted.front()) * scale},
            {"p50", at(0.5)},
            {"p90", at(0.9)},
            {"p99", at(0.99)},
            {"p999", at(0.999)},
            {"max", static_cast<double>(sorted.back()) * scale}};
}

nlohmann::json to_json(const Proportion& p) {
    return {{"count", p.hits},
            {"of", p.trials},
            {"rate", p.estimate},
            {"wilson95", {p.low, p.high}}};
}

// Per-window statistics of a sliding-window run, over the window positions that ran a decode
// (a position an earlier final window already decided has attempts 0 and is only counted).
nlohmann::json window_summary(const ShotResults& results) {
    const std::size_t windows = results.windows;
    const std::size_t cells = results.count * windows;
    std::vector<std::uint32_t> iterations;
    std::vector<std::uint32_t> legs;
    std::vector<std::uint64_t> decode_ns;
    std::vector<std::uint32_t> max_per_shot(results.count, 0);
    iterations.reserve(cells);
    legs.reserve(cells);
    decode_ns.reserve(cells);
    std::size_t skipped = 0;
    std::size_t converged = 0;
    std::size_t cap_hit = 0;
    std::size_t deferred = 0;
    std::uint64_t deferral_attempts = 0;
    std::size_t flagged_windows = 0;
    std::size_t with_unexplained = 0;
    std::uint64_t unexplained = 0;
    std::uint64_t virtual_commits = 0;
    for (std::size_t q = 0; q < cells; ++q) {
        const std::uint8_t attempts = results.win_attempts[q];
        if (attempts == 0) {
            ++skipped;
            continue;
        }
        iterations.push_back(results.win_iterations[q]);
        legs.push_back(results.win_legs[q]);
        decode_ns.push_back(results.win_decode_ns[q]);
        max_per_shot[q / windows] = std::max(max_per_shot[q / windows], results.win_iterations[q]);
        converged += results.win_converged[q];
        cap_hit += results.win_cap_hit[q];
        deferred += attempts > 1 ? 1U : 0U;
        deferral_attempts += attempts - 1U;
        flagged_windows += results.win_flagged[q];
        with_unexplained += results.win_unexplained[q] != 0 ? 1U : 0U;
        unexplained += results.win_unexplained[q];
        virtual_commits += results.win_virtual[q];
    }
    const std::size_t decoded = cells - skipped;
    const auto flagged_shots = static_cast<std::size_t>(std::ranges::count(results.flagged, 1));
    // Under defer, a window is committed without converging only once its attempts ran out.
    const std::size_t limit_reached =
        results.on_failure == window::OnFailure::defer ? decoded - converged : 0;
    return {{"positions", windows},
            {"decoded", decoded},
            {"skipped_after_final", skipped},
            {"iterations", distribution(std::move(iterations))},
            {"legs", distribution(std::move(legs))},
            {"max_window_iterations_per_shot", distribution(std::move(max_per_shot))},
            {"decode_time_us", distribution(std::move(decode_ns), 1e-3)},
            {"converged", to_json(wilson(converged, decoded))},
            {"cap_hit", to_json(wilson(cap_hit, decoded))},
            {"deferred", to_json(wilson(deferred, decoded))},
            {"deferral_attempts", deferral_attempts},
            {"deferral_limit_reached", limit_reached},
            {"flagged_windows", flagged_windows},
            {"flagged_shots", to_json(wilson(flagged_shots, results.count))},
            {"windows_with_unexplained", with_unexplained},
            {"unexplained_total", unexplained},
            {"virtual_commits_total", virtual_commits}};
}

} // namespace

nlohmann::json summarize(const ShotResults& results, std::optional<std::uint32_t> logical_qubits,
                         std::optional<std::uint32_t> rounds) {
    const std::size_t shots = results.count;
    std::size_t failures = 0;
    std::size_t not_converged = 0;
    std::size_t not_converged_but_correct = 0;
    for (std::size_t i = 0; i < shots; ++i) {
        const bool failed = results.logical_failure[i] != 0;
        const bool converged = results.success[i] != 0;
        failures += failed ? 1U : 0U;
        not_converged += converged ? 0U : 1U;
        not_converged_but_correct += (!converged && !failed) ? 1U : 0U;
    }
    const Proportion block = wilson(failures, shots);

    nlohmann::json summary = {
        {"shots", shots},
        {"block_error", to_json(block)},
        {"not_converged", to_json(wilson(not_converged, shots))},
        {"not_converged_but_correct", not_converged_but_correct},
        {"iterations", distribution(results.iterations)},
        {"legs", distribution(results.legs)},
        {"decode_time_us", distribution(results.decode_ns, 1e-3)},
        {"wall_seconds", results.wall_seconds},
        // Aggregate over all workers running concurrently; not a per-decode latency.
        {"throughput_shots_per_s_amortized",
         results.wall_seconds > 0 ? static_cast<double>(shots) / results.wall_seconds : 0.0},
    };
    // Both rates are monotone in P_block, so the interval maps through endpoint by endpoint.
    const auto rate = [&](auto convert) -> nlohmann::json {
        const auto as_json = [&](double p) {
            const std::optional<double> value = convert(p);
            return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
        };
        return {{"rate", as_json(block.estimate)},
                {"wilson95", {as_json(block.low), as_json(block.high)}}};
    };
    if (rounds) {
        nlohmann::json per_cycle = rate([&](double p) { return ler_per_cycle(p, *rounds); });
        per_cycle["rounds"] = *rounds;
        per_cycle["formula"] = "p_L = 1 - (1 - P_block)^(1/R)";
        summary["ler_per_cycle"] = std::move(per_cycle);
    }
    if (results.sliding) {
        summary["windows"] = window_summary(results);
    }
    if (results.solution_slots > 0) {
        const auto beyond = static_cast<std::size_t>(std::ranges::count_if(
            results.sol_count, [&](std::uint32_t c) { return c > results.solution_slots; }));
        summary["solutions"] = {
            {"slots", results.solution_slots},
            {"windows", results.windows},
            // Converged legs per decode, counted beyond the slots too.
            {"found", distribution(results.sol_count)},
            {"decodes_with_unrecorded_solutions", beyond},
        };
    }
    if (rounds && logical_qubits) {
        nlohmann::json per_qubit = rate(
            [&](double p) { return ler_per_qubit_per_cycle(p, *logical_qubits, *rounds); });
        per_qubit["logical_qubits"] = *logical_qubits;
        per_qubit["rounds"] = *rounds;
        per_qubit["formula"] = "p_L = [1 - (2(1 - P_block)^(1/k) - 1)^(1/R)] / 2";
        summary["ler_per_qubit_per_cycle"] = std::move(per_qubit);
    }
    if (results.confidence.enabled) {
        summary["confidence"] = confidence_summary(results.confidence);
    }
    return summary;
}

namespace {

// The per-window and per-solution arrays: [S, K] per window, [S, K, N] per solution slot, and the
// supports as CSR over the S·K·N slots.
std::expected<void, io::IoError> write_solutions(const std::filesystem::path& directory,
                                                 const ShotResults& results) {
    const std::array<std::size_t, 2> cells{results.count, results.windows};
    const std::array<std::size_t, 3> slots{results.count, results.windows, results.solution_slots};
    const auto write = [&]<class T>(const char* name, const std::vector<T>& data,
                                    std::span<const std::size_t> dims) {
        return io::write_npy<T>(directory / name, std::span<const T>(data), dims);
    };
    std::expected<void, io::IoError> ok = write("sol_count.npy", results.sol_count, cells);
    ok = ok.and_then([&] { return write("sol_leg.npy", results.sol_leg, slots); })
             .and_then([&] { return write("sol_iterations.npy", results.sol_iterations, slots); })
             .and_then([&] { return write("sol_weight.npy", results.sol_weight, slots); })
             .and_then([&] { return write("sol_class.npy", results.sol_class, slots); })
             .and_then([&] { return write("sol_hash.npy", results.sol_hash, slots); })
             .and_then([&] { return write("sol_size.npy", results.sol_size, slots); })
             .and_then([&] { return write("returned_class.npy", results.returned_class, cells); });
    if (ok && !results.solsup_ptr.empty()) {
        const std::array<std::size_t, 1> ptr_shape{results.solsup_ptr.size()};
        const std::array<std::size_t, 1> idx_shape{results.solsup_idx.size()};
        ok = write("solsup_ptr.npy", results.solsup_ptr, ptr_shape).and_then([&] {
            return write("solsup_idx.npy", results.solsup_idx, idx_shape);
        });
    }
    return ok;
}

// The sliding-window arrays: every window record field as [S, K], the flag per shot, and with
// commits the committed faults as CSR over the S·K cells.
std::expected<void, io::IoError> write_windows(const std::filesystem::path& directory,
                                               const ShotResults& results) {
    const std::array<std::size_t, 2> cells{results.count, results.windows};
    const std::array<std::size_t, 1> shots{results.count};
    const auto write = [&]<class T>(const char* name, const std::vector<T>& data,
                                    std::span<const std::size_t> dims) {
        return io::write_npy<T>(directory / name, std::span<const T>(data), dims);
    };
    std::expected<void, io::IoError> ok =
        write("win_iterations.npy", results.win_iterations, cells);
    ok = ok.and_then([&] { return write("win_legs.npy", results.win_legs, cells); })
             .and_then([&] { return write("win_attempts.npy", results.win_attempts, cells); })
             .and_then([&] { return write("win_converged.npy", results.win_converged, cells); })
             .and_then([&] { return write("win_cap_hit.npy", results.win_cap_hit, cells); })
             .and_then([&] { return write("win_weight.npy", results.win_weight, cells); })
             .and_then([&] {
                 return write("win_committed_weight.npy", results.win_committed_weight, cells);
             })
             .and_then([&] { return write("win_unexplained.npy", results.win_unexplained, cells); })
             .and_then([&] { return write("win_flagged.npy", results.win_flagged, cells); })
             .and_then([&] { return write("win_virtual.npy", results.win_virtual, cells); })
             .and_then([&] { return write("win_decode_ns.npy", results.win_decode_ns, cells); })
             .and_then([&] { return write("flagged.npy", results.flagged, shots); });
    if (ok && !results.commit_ptr.empty()) {
        const std::array<std::size_t, 1> ptr_shape{results.commit_ptr.size()};
        const std::array<std::size_t, 1> faults_shape{results.commit_faults.size()};
        ok = write("commit_ptr.npy", results.commit_ptr, ptr_shape).and_then([&] {
            return write("commit_faults.npy", results.commit_faults, faults_shape);
        });
    }
    return ok;
}

} // namespace

std::expected<void, io::IoError> write_results(const std::filesystem::path& directory,
                                               const ShotResults& results,
                                               const nlohmann::json& record) {
    const std::array<std::size_t, 1> shape{results.count};
    const std::array<std::size_t, 2> observables_shape{results.count, results.num_observables};
    const std::array<std::size_t, 2> decodings_shape{results.count, results.num_columns};

    const auto write = [&]<class T>(const char* name, const std::vector<T>& data,
                                    std::span<const std::size_t> dims)
        -> std::expected<void, io::IoError> {
        return io::write_npy<T>(directory / name, std::span<const T>(data), dims);
    };
    std::expected<void, io::IoError> ok = write("success.npy", results.success, shape);
    ok = ok.and_then([&] { return write("iterations.npy", results.iterations, shape); })
             .and_then([&] { return write("legs.npy", results.legs, shape); })
             .and_then([&] { return write("best_leg.npy", results.best_leg, shape); })
             .and_then([&] { return write("weight.npy", results.weight, shape); })
             .and_then([&] { return write("decode_ns.npy", results.decode_ns, shape); })
             .and_then([&] {
                 return write("predicted_observables.npy", results.predicted, observables_shape);
             })
             .and_then([&] { return write("logical_failure.npy", results.logical_failure, shape); });
    if (ok && !results.decodings.empty()) {
        ok = write("decodings.npy", results.decodings, decodings_shape);
    }
    if (ok && results.sliding) {
        ok = write_windows(directory, results);
    }
    if (ok && results.solution_slots > 0) {
        ok = write_solutions(directory, results);
    }
    if (ok && results.confidence.enabled) {
        ok = write_confidence(directory, results.confidence, results.count, results.windows);
    }
    if (!ok) {
        return ok;
    }

    const std::filesystem::path final_path = directory / "run.json";
    std::filesystem::path partial = final_path;
    partial += ".partial";
    {
        std::ofstream out(partial, std::ios::trunc);
        out << record.dump(2) << '\n';
        out.flush();
        if (!out) {
            return std::unexpected(io::IoError{.code = io::IoError::Code::write_failed,
                                               .path = partial.string(),
                                               .expected = "a writable file",
                                               .found = "a write error"});
        }
    }
    std::error_code ec;
    std::filesystem::rename(partial, final_path, ec);
    if (ec) {
        return std::unexpected(io::IoError{.code = io::IoError::Code::write_failed,
                                           .path = final_path.string(),
                                           .expected = "a successful rename",
                                           .found = ec.message()});
    }
    return {};
}

} // namespace rtd::harness
