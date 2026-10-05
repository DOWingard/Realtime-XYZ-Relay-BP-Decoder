#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "rtd/core/backend.hpp"
#include "rtd/core/buffer.hpp"
#include "rtd/core/config.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/core/result.hpp"
#include "rtd/core/sink.hpp"
#include "rtd/core/types.hpp"

namespace rtd {

// The relay controller (Relay-BP, Müller et al., arXiv:2506.01779, Algorithm 1).
//
// Leg 0 runs min-sum with memory strength γ₀ from the priors. Each relay leg r = 1..num_sets then
// takes a fresh γ vector, resets every edge message to its prior but keeps the marginals the
// previous leg ended with, and runs again. Every leg that converges contributes a solution; the one
// with the lowest weight W(ê) = Σ_{ê_j=1} λ_j (most probable under the priors) is returned, the
// earliest winning ties. The stopping rule decides how many solutions are enough.
//
// The controller never touches an edge or a marginal: everything numerical happens in the Backend,
// so the same schedule drives any number format and any device.
//
// Plain min-sum and Mem-BP are the num_sets = 0 configurations of this one code path.
//
// A decode may be given a total iteration budget (DecodeLimits): the leg that would cross it is cut
// short and no further leg starts, so the result is exactly what the uncapped decode had found by
// that many iterations. Every converged leg is reported to the Sink, in leg order, before the
// stopping rule looks at it; the default sink compiles away. A selecting sink (confidence.hpp)
// additionally decides which solution is returned, may stop the decode early, and may extend a
// low-confidence decode by a few legs; it also attaches the decode's confidence to the result.
template <LegBackend Backend, SolutionSink Sink = NoSink>
class RelayDecoder {
public:
    // `gammas` may be null only when num_sets = 0; it must outlive the decoder. A sink that
    // declares num_columns() must match the problem's column count.
    [[nodiscard]] static std::expected<RelayDecoder, ConfigError>
    create(Backend backend, MinSumConfig min_sum, RelayConfig relay, const GammaSource* gammas,
           Sink sink = Sink{}) {
        if (auto valid = validate(min_sum, relay); !valid) {
            return std::unexpected(std::move(valid.error()));
        }
        if (relay.num_sets > 0 && gammas == nullptr) {
            return std::unexpected(
                ConfigError{.code = ConfigError::Code::missing_gamma_source,
                            .detail = std::format("num_sets = {} relay legs need a gamma "
                                                  "source",
                                                  relay.num_sets)});
        }
        if (gammas != nullptr && gammas->width() != backend.num_columns()) {
            return std::unexpected(ConfigError{
                ConfigError::Code::gamma_width_mismatch,
                std::format("gamma source has width {} but the problem has {} columns",
                            gammas->width(), backend.num_columns())});
        }
        if constexpr (requires {
                          { std::as_const(sink).num_columns() } -> std::same_as<index_t>;
                      }) {
            if (sink.num_columns() != backend.num_columns()) {
                return std::unexpected(ConfigError{
                    ConfigError::Code::gamma_width_mismatch,
                    std::format("solution sink has column classes for {} columns but the problem "
                                "has {} columns",
                                sink.num_columns(), backend.num_columns())});
            }
        }
        return RelayDecoder(std::move(backend), min_sum, relay, gammas, std::move(sink));
    }

    // Decodes one syndrome. `gamma_stream` selects the relay γ draws of this problem (a
    // GammaSource defines it; an explicit table ignores it). `limits` caps the iterations summed
    // over all legs: before each leg, remaining = cap − used; if nothing remains the decode stops,
    // otherwise the leg runs at most min(its budget, remaining) iterations.
    [[nodiscard]] std::expected<DecodeResult, DecodeError>
    decode(std::span<const Bit> syndrome, std::uint64_t gamma_stream = 0,
           DecodeLimits limits = {}) noexcept {
        if (syndrome.size() != backend_.num_rows()) {
            return std::unexpected(DecodeError{DecodeError::Code::syndrome_size_mismatch,
                                               backend_.num_rows(), syndrome.size()});
        }
        if constexpr (Sink::enabled) {
            sink_.on_decode_begin();
        }
        const bool memory = min_sum_.gamma0.has_value();
        backend_.begin(syndrome, memory);
        if (memory) {
            backend_.set_gamma(*min_sum_.gamma0);
        }
        const std::optional<std::uint32_t> cap = limits.max_total_iterations;
        if (cap.has_value() && *cap == 0) {
            // No leg may start: the answer is ê = 0, and leg 0 would have run.
            return finish(result(0, 0, std::nullopt, std::numeric_limits<double>::infinity(), true),
                          Extension{}, syndrome);
        }

        const std::uint32_t leg0_budget = cap ? std::min(relay_.pre_iter, *cap) : relay_.pre_iter;
        const LegOutcome first = backend_.run_leg(LegParams{
            .max_iter = leg0_budget, .alpha = min_sum_.alpha, .use_memory = memory});
        // Leg 0's final ê is both the first candidate and the answer if nothing converges.
        backend_.mark_best();
        std::uint32_t legs = 1;
        std::uint32_t iterations = first.iterations;
        std::uint32_t converged_legs = 0;
        std::optional<std::uint32_t> best_leg;
        double best_weight = std::numeric_limits<double>::infinity();
        trace_[0] = LegRecord{.iterations = first.iterations,
                              .converged = first.converged,
                              .became_best = first.converged,
                              .weight = first.weight};
        if (first.converged) {
            converged_legs = 1;
            best_leg = 0;
            best_weight = first.weight;
            report(0, iterations, first.weight);
        }

        // Set when the budget, not the stopping rule or the end of the schedule, ends the decode.
        // A low-confidence extension only runs once the stopping rule is met, so the budget
        // ending one is not a cap hit.
        bool cap_hit = false;
        Extension extension;
        const bool stop_after_leg0 =
            first.converged && stop_here(0,
                                         std::holds_alternative<AfterLeg0>(relay_.stopping) ||
                                             reached_count(converged_legs),
                                         extension);
        if (!first.converged && leg0_budget < relay_.pre_iter) {
            cap_hit = true; // leg 0 itself was cut short
        } else if (!stop_after_leg0) {
            for (std::uint32_t leg = 1; leg <= relay_.num_sets; ++leg) {
                if (extension_spent(leg, extension)) {
                    break;
                }
                std::uint32_t budget = relay_.set_max_iter;
                if (cap) {
                    const std::uint32_t remaining = *cap - iterations;
                    if (remaining == 0) {
                        cap_hit = !extension.active; // this leg would have run
                        break;
                    }
                    budget = std::min(budget, remaining);
                }
                backend_.set_gamma(gammas_->gammas(gamma_stream, leg, gamma_scratch_.span()));
                const LegOutcome outcome = backend_.run_leg(
                    LegParams{.max_iter = budget, .alpha = min_sum_.alpha, .use_memory = true});
                ++legs;
                iterations += outcome.iterations;
                LegRecord& record = trace_[leg];
                record = LegRecord{.iterations = outcome.iterations,
                                   .converged = outcome.converged,
                                   .became_best = false,
                                   .weight = outcome.weight};
                if (!outcome.converged) {
                    if (budget < relay_.set_max_iter) {
                        cap_hit = !extension.active; // cut short with the budget exhausted
                        break;
                    }
                    continue;
                }
                ++converged_legs;
                report(leg, iterations, outcome.weight);
                if (outcome.weight < best_weight) {
                    best_weight = outcome.weight;
                    best_leg = leg;
                    record.became_best = true;
                    backend_.mark_best();
                }
                if (stop_here(leg, reached_count(converged_legs), extension)) {
                    break;
                }
            }
        }
        return finish(result(iterations, legs, best_leg, best_weight, cap_hit), extension,
                      syndrome);
    }

    // Restricts the convergence test to the rows with mask[i] = 1 (one entry per row); an empty
    // mask restores the default, every row. Applies to every later decode.
    void set_convergence_rows(std::span<const Bit> mask) noexcept {
        backend_.set_convergence_rows(mask);
    }

    [[nodiscard]] index_t num_rows() const noexcept { return backend_.num_rows(); }
    [[nodiscard]] index_t num_columns() const noexcept { return backend_.num_columns(); }
    [[nodiscard]] Backend& backend() noexcept { return backend_; }
    [[nodiscard]] Sink& sink() noexcept { return sink_; }
    [[nodiscard]] const Sink& sink() const noexcept { return sink_; }
    [[nodiscard]] const MinSumConfig& min_sum_config() const noexcept { return min_sum_; }
    [[nodiscard]] const RelayConfig& relay_config() const noexcept { return relay_; }

private:
    RelayDecoder(Backend backend, MinSumConfig min_sum, RelayConfig relay,
                 const GammaSource* gammas, Sink sink)
        : backend_(std::move(backend)), sink_(std::move(sink)), min_sum_(min_sum), relay_(relay),
          gammas_(gammas), trace_(max_legs(relay)),
          gamma_scratch_(gammas != nullptr ? gammas->width() : 0) {}

    [[nodiscard]] bool reached_count(std::uint32_t converged_legs) const noexcept {
        const auto* rule = std::get_if<AfterNConverged>(&relay_.stopping);
        return rule != nullptr && converged_legs >= rule->count;
    }

    // A selecting sink decides the returned solution and may end the decode before, or run legs
    // after, the relay stopping rule; any other sink leaves the controller exactly as it is.
    static constexpr bool selecting = SelectingSink<Sink>;

    // Legs a low-confidence decode may run after the relay rule was met at leg `start`, and
    // whether the sink's stopping rule ended the decode before the relay rule was met.
    struct Extension {
        bool active = false;
        std::uint32_t start = 0;
        std::uint32_t last_leg = 0;
        bool early_stop = false;
    };

    // Whether leg `leg` lies past the legs a low-confidence extension allows.
    [[nodiscard]] static bool extension_spent(std::uint32_t leg,
                                              const Extension& extension) noexcept {
        if constexpr (selecting) {
            return extension.active && leg > extension.last_leg;
        } else {
            (void)leg;
            (void)extension;
            return false;
        }
    }

    // After converged leg `leg`, given the relay rule's verdict `rule_met`: whether the decode
    // ends here.
    [[nodiscard]] bool stop_here(std::uint32_t leg, bool rule_met,
                                 Extension& extension) const noexcept {
        if constexpr (selecting) {
            if (sink_.stop_requested()) {
                extension.early_stop = !rule_met && !extension.active;
                return true;
            }
            if (extension.active) {
                return !sink_.low_confidence(); // extra legs end once confidence is restored
            }
            if (!rule_met) {
                return false;
            }
            const std::uint32_t extra = sink_.low_confidence() ? sink_.extra_legs() : 0;
            if (extra == 0 || leg >= relay_.num_sets) {
                return true;
            }
            extension = Extension{.active = true, .start = leg, .last_leg = leg + extra};
            return false;
        } else {
            (void)leg;
            (void)extension;
            return rule_met;
        }
    }

    // Lets a selecting sink attach its confidence and, if its rule chose another solution, put
    // that solution in place of the lowest-weight one.
    [[nodiscard]] DecodeResult finish(DecodeResult out, const Extension& extension,
                                      std::span<const Bit> syndrome) noexcept {
        if constexpr (selecting) {
            std::uint32_t ones = 0;
            for (const Bit bit : syndrome) {
                ones += bit != 0 ? 1U : 0U;
            }
            const std::uint32_t last_leg = out.legs_executed == 0 ? 0 : out.legs_executed - 1;
            const Selection chosen = sink_.finish(DecodeFacts{
                .success = out.success,
                .iterations = out.iterations,
                .legs_executed = out.legs_executed,
                .best_leg = out.best_leg.value_or(0),
                .best_weight = out.weight,
                .best_support = out.support,
                .stopped_early = extension.early_stop,
                .extra_legs = extension.active ? last_leg - extension.start : 0,
                .syndrome_ones = ones,
                .syndrome_rows = static_cast<std::uint32_t>(syndrome.size())});
            if (chosen.replaces) {
                out.best_leg = chosen.leg;
                out.weight = chosen.weight;
                out.support = chosen.support;
                out.hard = chosen.hard;
            }
            out.confidence = chosen.confidence;
        } else {
            (void)extension;
            (void)syndrome;
        }
        return out;
    }

    // Hands the leg that just converged to the sink; the support is fetched only if it listens.
    void report(std::uint32_t leg, std::uint32_t cumulative_iterations, double weight) noexcept {
        if constexpr (Sink::enabled) {
            sink_.on_solution(SolutionEvent{.leg = leg,
                                            .cumulative_iterations = cumulative_iterations,
                                            .weight = weight,
                                            .support = backend_.current_support()});
        }
    }

    [[nodiscard]] DecodeResult result(std::uint32_t iterations, std::uint32_t legs,
                                      std::optional<std::uint32_t> best_leg, double weight,
                                      bool cap_hit) noexcept {
        return DecodeResult{.success = best_leg.has_value(),
                            .iterations = iterations,
                            .legs_executed = legs,
                            .best_leg = best_leg,
                            .weight = weight,
                            .hard = backend_.best_hard(),
                            .legs = std::span<const LegRecord>(trace_.data(), legs),
                            .cap_hit = cap_hit,
                            .support = backend_.best_support()};
    }

    Backend backend_;
    Sink sink_;
    MinSumConfig min_sum_;
    RelayConfig relay_;
    const GammaSource* gammas_;
    std::vector<LegRecord> trace_;
    AlignedBuffer<double> gamma_scratch_;
};

} // namespace rtd
