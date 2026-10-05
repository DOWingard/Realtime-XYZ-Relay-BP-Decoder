// The fixed-point arithmetic intN.S.M: its operations one by one (including the worked example of
// the shift-and-add multiplier from Maurer et al., App. C), and whole relay decodes of the
// optimised backend against a naive integer decoder written from the format's definition, in
// every layout, column order and executor.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rtd/core/decoder_fixed.hpp"
#include "rtd/core/fixed_arith.hpp"
#include "rtd/core/kernels.hpp"
#include "support.hpp"

namespace {

using namespace rtd;

TEST(FixedArith, NamesSpellTheFormat) {
    EXPECT_EQ(Int4_2_8::name, "int4.2.8");
    EXPECT_EQ(Int5_2_8::name, "int5.2.8");
    EXPECT_EQ(Int6_2_8::name, "int6.2.8");
    EXPECT_EQ((FixedArith<7, 16, 32>::name), "int7.16.32");
    EXPECT_EQ(Int4_2_8::max_message(), 15);
    EXPECT_EQ(Int6_2_8::max_message(), 63);
}

TEST(FixedArith, PriorsRoundHalfAwayFromZeroAndSaturate) {
    using A = Int4_2_8;
    EXPECT_EQ(A::from_llr(2.25), 5);  // 4.5 rounds up
    EXPECT_EQ(A::from_llr(2.2), 4);   // 4.4
    EXPECT_EQ(A::from_llr(-2.25), -5); // half away from zero
    EXPECT_EQ(A::from_llr(0.0), 0);
    EXPECT_EQ(A::from_llr(7.24), 14);
    EXPECT_EQ(A::from_llr(7.25), 15);
    EXPECT_EQ(A::from_llr(40.0), 15);
    EXPECT_EQ(A::from_llr(-40.0), -15);
    EXPECT_EQ(A::from_llr(std::numeric_limits<double>::infinity()), 15);
    // The largest double below 0.5 must not round up (a naive floor(x + 0.5) does).
    EXPECT_EQ(A::round_half_away(std::nextafter(0.5, 0.0)), 0.0);
    EXPECT_EQ(A::round_half_away(0.5), 1.0);
    EXPECT_EQ(A::round_half_away(-0.5), -1.0);
    // Gross-code priors at p = 3e-3: ln((1 - p)/p) = 5.806 -> 11.6 -> 12.
    EXPECT_EQ(A::from_llr(std::log((1.0 - 0.003) / 0.003)), 12);
    EXPECT_EQ(Int6_2_8::from_llr(40.0), 63);
}

TEST(FixedArith, MemoryStrengthsBecomeBetaTimesM) {
    using A = Int4_2_8;
    EXPECT_EQ(A::from_gamma(0.125), 7); // gamma0 of Relay-BP: beta = 0.875
    EXPECT_EQ(A::from_gamma(-0.24), 10); // 1.24 * 8 = 9.92
    EXPECT_EQ(A::from_gamma(0.66), 3);  // 0.34 * 8 = 2.72
    EXPECT_EQ(A::from_gamma(0.0), 8);
    EXPECT_EQ(A::from_gamma(1.0), 0);
    EXPECT_EQ(A::from_gamma(-1.0), 16);
    EXPECT_EQ(A::from_gamma(-3.0), 16); // clipped to beta = 2
    EXPECT_EQ(A::from_gamma(2.0), 0);   // clipped to beta = 0
    EXPECT_EQ(A::from_gamma(0.0625), 8); // 7.5 rounds half away from zero
}

TEST(FixedArith, ShiftAndAddMultiplierReproducesTheWorkedExample) {
    using A = Int4_2_8;
    // 15 x 7 at M = 8: partial products 56, 28, 14, 7 divided by 8 and truncated: 7, 3, 1, 0.
    EXPECT_EQ(A::product(15, 7), 11);
    EXPECT_EQ(A::product(8, 7), 7);
    EXPECT_EQ(A::product(4, 7), 3);
    EXPECT_EQ(A::product(2, 7), 1);
    EXPECT_EQ(A::product(1, 7), 0);
    EXPECT_EQ(A::product(-15, 7), -11); // sign and magnitude
    for (int x = -15; x <= 15; ++x) {
        EXPECT_EQ(A::product(x, 8), x) << x; // beta = 1 is exact
        EXPECT_EQ(A::product(x, 0), 0) << x;
        EXPECT_EQ(A::product(x, 16), 2 * x) << x; // beta = 2 is a shift, exact too
        // Truncation always shrinks the magnitude.
        for (int beta = 0; beta <= 16; ++beta) {
            EXPECT_LE(std::abs(A::product(x, beta)), std::abs(x) * beta / 8) << x << " " << beta;
        }
    }
}

TEST(FixedArith, BiasMixesPriorAndMarginalAndSaturates) {
    using A = Int4_2_8;
    // gamma = 0 (beta = M): no memory.
    EXPECT_EQ(A::mix(12, -9, 8), 12);
    // gamma = 1 (beta = 0): memory only, the marginal clipped to the message range.
    EXPECT_EQ(A::mix(12, -9, 0), -9);
    EXPECT_EQ(A::mix(12, -40, 0), -15);
    // M = lambda gives lambda exactly for every beta: the two products cancel.
    for (int beta = 0; beta <= 16; ++beta) {
        EXPECT_EQ(A::mix(12, 12, static_cast<A::msg_t>(beta)), 12) << beta;
    }
    // beta = 7: 12 = 8 + 4 gives 7 + 3 = 10; -6 = -(4 + 2) gives -(3 + 1) = -4;
    // bias = 10 + (-6) - (-4) = 8.
    EXPECT_EQ(A::product(12, 7), 10);
    EXPECT_EQ(A::product(-6, 7), -4);
    EXPECT_EQ(A::mix(12, -6, 7), 8);
    // beta = 16 (gamma = -1): 2 lambda - M, clipped.
    EXPECT_EQ(A::mix(12, 3, 16), 15);
    EXPECT_EQ(A::mix(3, 12, 16), -6);
}

TEST(FixedArith, ScalingIsShiftAndSubtract) {
    using A = Int4_2_8;
    EXPECT_EQ(A::from_alpha(0.5), 1);
    EXPECT_EQ(A::from_alpha(0.75), 2);
    EXPECT_EQ(A::from_alpha(1.0 - std::exp2(-5.0)), 5);
    EXPECT_EQ(A::from_alpha(1.0), A::identity_shift);
    EXPECT_EQ(A::from_alpha(1.0 - std::exp2(-60.0)), A::identity_shift);
    EXPECT_EQ(A::scale(15, 1), 8); // 15 - 7: the truncated half is subtracted, so it rounds up
    EXPECT_EQ(A::scale(7, 1), 4);
    EXPECT_EQ(A::scale(15, 2), 12);
    EXPECT_EQ(A::scale(1, 1), 1);
    EXPECT_EQ(A::scale(0, 1), 0);
    for (int x = 0; x <= 15; ++x) {
        EXPECT_EQ(A::scale(static_cast<A::msg_t>(x), A::from_alpha(1.0)), x);
    }
    // The adaptive rule with scaling 1 walks k = 1, 2, 3, ...
    for (std::uint32_t t = 0; t < 40; ++t) {
        const int expected = std::min<int>(static_cast<int>(t) + 1, A::identity_shift);
        EXPECT_EQ(A::from_alpha(alpha_at(AdaptiveAlpha{1.0}, t)), expected) << t;
    }
}

TEST(FixedArith, AccumulatorIsExactAndMessagesSaturate) {
    using A = Int6_2_8;
    EXPECT_EQ(A::to_msg(1000), 63);
    EXPECT_EQ(A::to_msg(-1000), -63);
    EXPECT_EQ(A::to_msg(-12), -12);
    EXPECT_TRUE(A::is_error(0));
    EXPECT_TRUE(A::is_error(-1));
    EXPECT_FALSE(A::is_error(1));
    EXPECT_FALSE(A::is_negative(0));
    EXPECT_EQ(A::with_sign(0, true), 0);
    EXPECT_EQ(A::magnitude(-63), 63);
    EXPECT_DOUBLE_EQ(A::to_double(25), 12.5);
    const std::vector<double> llr{5.806, 7.3, std::numeric_limits<double>::infinity(), 40.0};
    const std::vector<index_t> support{0, 1, 2, 3};
    // 12 + 15 + 63 + 63 = 153, over S = 2.
    EXPECT_EQ(A::solution_weight(llr, support), 76.5);
    EXPECT_EQ(Int4_2_8::solution_weight(llr, support), (12.0 + 15 + 15 + 15) / 2);
}

TEST(FixedArith, AlphaRulesAFormatCanRun) {
    EXPECT_FALSE(fixed_alpha_problem(ConstantAlpha{1.0}));
    EXPECT_FALSE(fixed_alpha_problem(ConstantAlpha{0.5}));
    EXPECT_FALSE(fixed_alpha_problem(ConstantAlpha{0.875}));
    EXPECT_FALSE(fixed_alpha_problem(AdaptiveAlpha{1.0}));
    EXPECT_TRUE(fixed_alpha_problem(ConstantAlpha{0.8}));
    EXPECT_TRUE(fixed_alpha_problem(ConstantAlpha{0.25}));
    EXPECT_TRUE(fixed_alpha_problem(ConstantAlpha{1.5}));
    EXPECT_TRUE(fixed_alpha_problem(AdaptiveAlpha{2.0}));
}

// ---- The check update of the kernels with an integer policy. --------------------------------

TEST(FixedKernels, CheckRowSendsScaledExclusiveMinimaWithExclusiveSigns) {
    using A = Int4_2_8;
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 3000; ++trial) {
        const auto degree = static_cast<index_t>(1 + rng() % 12);
        const bool syndrome = (rng() & 1U) != 0;
        const auto shift = static_cast<A::msg_t>(1 + rng() % 5);
        std::vector<A::msg_t> input(degree);
        for (auto& x : input) {
            x = static_cast<A::msg_t>(static_cast<int>(rng() % 31) - 15); // repeats are common
        }
        std::vector<A::msg_t> out(degree);
        kernels::check_row_scalar<A, false>(out.data(), degree, syndrome, shift,
                                            [&](index_t k) { return input[k]; });
        for (index_t k = 0; k < degree; ++k) {
            int others_min = A::max_magnitude;
            bool negative = syndrome;
            for (index_t q = 0; q < degree; ++q) {
                if (q != k) {
                    others_min = std::min(others_min, std::abs(int{input[q]}));
                    negative = negative != (input[q] < 0);
                }
            }
            const int magnitude = others_min - (others_min >> shift);
            const int expected = negative ? -magnitude : magnitude;
            ASSERT_EQ(int{out[k]}, expected) << "trial " << trial << " edge " << k;
        }
    }
}

// The byte-vector check row against the scalar one, on padded rows as the backend lays them out:
// every degree up to several vectors, both input sources, unit and shift scaling.
template <class A>
void expect_byte_rows_match_scalar(std::uint64_t seed) {
    static_assert(kernels::ByteMessages<A>);
    using msg_t = A::msg_t;
    std::mt19937_64 rng(seed);
    const auto max = A::max_message();
    for (int trial = 0; trial < 4000; ++trial) {
        const auto degree = static_cast<index_t>(1 + rng() % 70);
        const auto padded = static_cast<index_t>(round_up(degree, std::size_t{row_alignment_slots}));
        const bool syndrome = (rng() & 1U) != 0;
        const auto shift = static_cast<msg_t>(1 + rng() % 5);
        const bool from_priors = (rng() & 1U) != 0;
        const auto random_message = [&] {
            // Small magnitudes make repeated minima common; the extremes are included.
            const int range = (rng() & 3U) == 0 ? 2 * max + 1 : 7;
            return static_cast<msg_t>(static_cast<int>(rng() % static_cast<unsigned>(range)) -
                                      range / 2);
        };
        const index_t n = 50;
        std::vector<msg_t> lambda(n + 1);
        for (auto& l : lambda) {
            l = random_message();
        }
        lambda[n] = max; // the padding sentinel
        std::vector<index_t> columns(padded, n);
        std::vector<msg_t> row(padded, max);
        for (index_t k = 0; k < degree; ++k) {
            columns[k] = static_cast<index_t>(rng() % n);
            row[k] = random_message();
        }
        const auto input = [&](index_t k) { return from_priors ? lambda[columns[k]] : row[k]; };
        std::vector<msg_t> expected(padded, max);
        const bool unit = (trial % 3) == 0;
        if (unit) {
            kernels::check_row_scalar<A, true>(expected.data(), degree, syndrome, shift, input);
        } else {
            kernels::check_row_scalar<A, false>(expected.data(), degree, syndrome, shift, input);
        }
        std::vector<msg_t> got = row;
        const auto run = [&]<bool Unit, kernels::Source Src> {
            kernels::check_row_simd_bytes<A, Unit, Src>(got.data(), columns.data(), lambda.data(),
                                                         degree, syndrome, shift);
        };
        if (from_priors) {
            unit ? run.template operator()<true, kernels::Source::priors>()
                 : run.template operator()<false, kernels::Source::priors>();
        } else {
            unit ? run.template operator()<true, kernels::Source::messages>()
                 : run.template operator()<false, kernels::Source::messages>();
        }
        ASSERT_EQ(got, expected) << A::name << " trial " << trial << " degree " << degree;
    }
}

TEST(FixedKernels, ByteVectorCheckRowEqualsTheScalarRow) {
    expect_byte_rows_match_scalar<Int4_2_8>(11);
    expect_byte_rows_match_scalar<Int5_2_8>(12);
    expect_byte_rows_match_scalar<Int6_2_8>(13);
}

// ---- Whole decodes against a naive decoder written from the definition. ---------------------

// Relay-BP in intN.S.M straight from its definition: separate μ and η per (row, column) pair,
// exclusive minima and signs computed edge by edge, the shift-and-add multiplier bit by bit, a
// dense ê and a full H·ê product. Nothing is shared with the policy or the kernels.
class NaiveFixedDecoder {
public:
    NaiveFixedDecoder(const test::Csr& h, std::span<const double> llr, int bits, int scale,
                      int memory_scale)
        : h_(h), max_(( 1 << bits) - 1), bits_(bits), scale_(scale), m_(memory_scale) {
        col_rows_.resize(h.cols);
        for (index_t i = 0; i < h.rows; ++i) {
            for (index_t k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) {
                col_rows_[h.col_idx[k]].push_back(i);
            }
        }
        for (const double l : llr) {
            const double v = std::isinf(l) && l > 0 ? max_ : std::round(l * scale);
            lambda_.push_back(static_cast<int>(std::clamp<double>(v, -max_, max_)));
        }
    }

    test::ReferenceResult decode(std::span<const Bit> syndrome, const MinSumConfig& ms,
                                 const RelayConfig& rc, const GammaSource* gammas,
                                 std::uint64_t stream) {
        const bool memory = ms.gamma0.has_value();
        std::vector<int> beta(h_.cols, beta_of(memory ? *ms.gamma0 : 0.0));
        marginal_ = lambda_;
        test::ReferenceResult out;
        const Leg first = run_leg(syndrome, rc.pre_iter, ms.alpha, memory, beta);
        out.legs = 1;
        out.iterations = first.iterations;
        out.leg_iterations.push_back(first.iterations);
        out.leg_converged.push_back(first.converged);
        out.hard = first.hard;
        std::uint32_t converged = 0;
        if (first.converged) {
            converged = 1;
            out.best_leg = 0;
            out.weight = weight(first.hard);
        }
        const auto* nconv = std::get_if<AfterNConverged>(&rc.stopping);
        const bool done = first.converged && (std::holds_alternative<AfterLeg0>(rc.stopping) ||
                                              (nconv != nullptr && converged >= nconv->count));
        if (!done) {
            std::vector<double> scratch(h_.cols);
            for (std::uint32_t leg = 1; leg <= rc.num_sets; ++leg) {
                const auto g = gammas->gammas(stream, leg, scratch);
                for (index_t j = 0; j < h_.cols; ++j) {
                    beta[j] = beta_of(g[j]);
                }
                const Leg result = run_leg(syndrome, rc.set_max_iter, ms.alpha, true, beta);
                ++out.legs;
                out.iterations += result.iterations;
                out.leg_iterations.push_back(result.iterations);
                out.leg_converged.push_back(result.converged);
                if (result.converged) {
                    ++converged;
                    const double w = weight(result.hard);
                    if (w < out.weight) {
                        out.weight = w;
                        out.best_leg = leg;
                        out.hard = result.hard;
                    }
                    if (nconv != nullptr && converged >= nconv->count) {
                        break;
                    }
                }
            }
        }
        out.success = out.best_leg.has_value();
        for (const int v : marginal_) {
            out.marginals.push_back(static_cast<double>(v) / scale_);
        }
        return out;
    }

private:
    struct Leg {
        bool converged = false;
        std::uint32_t iterations = 0;
        std::vector<Bit> hard;
    };

    [[nodiscard]] int sat(long long x) const {
        return static_cast<int>(std::clamp<long long>(x, -max_, max_));
    }

    [[nodiscard]] int beta_of(double gamma) const {
        return static_cast<int>(std::clamp(std::round((1.0 - gamma) * m_), 0.0, 2.0 * m_));
    }

    // Each set bit of |x| contributes beta·2^b / M with its fraction dropped.
    [[nodiscard]] int times_beta(int x, int beta) const {
        const int magnitude = std::abs(x);
        int sum = 0;
        for (int b = 0; b < bits_; ++b) {
            if ((magnitude & (1 << b)) != 0) {
                sum += (beta * (1 << b)) / m_;
            }
        }
        return x < 0 ? -sum : sum;
    }

    // The shift k of α = 1 − 2^(−k) for iteration t; 0 means α = 1.
    static int shift_of(const AlphaRule& rule, std::uint32_t t) {
        if (const auto* adaptive = std::get_if<AdaptiveAlpha>(&rule)) {
            EXPECT_EQ(adaptive->scaling, 1.0);
            return static_cast<int>(t) + 1;
        }
        const double alpha = std::get<ConstantAlpha>(rule).value;
        if (alpha == 1.0) {
            return 0;
        }
        return static_cast<int>(std::lround(-std::log2(1.0 - alpha)));
    }

    [[nodiscard]] double weight(const std::vector<Bit>& hard) const {
        long long w = 0;
        for (index_t j = 0; j < h_.cols; ++j) {
            if (hard[j] != 0) {
                w += lambda_[j];
            }
        }
        return static_cast<double>(w) / scale_;
    }

    using EdgeMap = std::map<std::pair<index_t, index_t>, int>;

    // μ_ij = sign · α(min over the row's other ν), the sign being σ_i times the other ν's signs.
    void check_update(std::span<const Bit> syndrome, int shift, EdgeMap& nu, EdgeMap& mu) const {
        for (index_t i = 0; i < h_.rows; ++i) {
            for (index_t k = h_.row_ptr[i]; k < h_.row_ptr[i + 1]; ++k) {
                const index_t j = h_.col_idx[k];
                int least = max_;
                bool negative = syndrome[i] != 0;
                for (index_t q = h_.row_ptr[i]; q < h_.row_ptr[i + 1]; ++q) {
                    if (q != k) {
                        const int x = nu[{i, h_.col_idx[q]}];
                        least = std::min(least, std::abs(x));
                        negative = negative != (x < 0);
                    }
                }
                const int scaled = shift == 0 || shift >= 31 ? least : least - (least >> shift);
                mu[{i, j}] = negative ? -scaled : scaled;
            }
        }
    }

    // Bias from prior and stored marginal, exact total, saturated outgoing ν, hard decision.
    void variable_update(bool memory, const std::vector<int>& beta, EdgeMap& mu, EdgeMap& nu,
                         std::vector<Bit>& hard) {
        for (index_t j = 0; j < h_.cols; ++j) {
            int bias = lambda_[j];
            if (memory) {
                const int stored = marginal_[j];
                bias = sat(times_beta(lambda_[j], beta[j]) + stored - times_beta(stored, beta[j]));
            }
            long long total = bias;
            for (const index_t i : col_rows_[j]) {
                total += mu[{i, j}];
            }
            for (const index_t i : col_rows_[j]) {
                nu[{i, j}] = sat(total - mu[{i, j}]);
            }
            marginal_[j] = sat(total);
            hard[j] = total <= 0 ? 1 : 0;
        }
    }

    Leg run_leg(std::span<const Bit> syndrome, std::uint32_t max_iter, const AlphaRule& alpha,
                bool memory, const std::vector<int>& beta) {
        EdgeMap nu;
        EdgeMap mu;
        for (index_t i = 0; i < h_.rows; ++i) {
            for (index_t k = h_.row_ptr[i]; k < h_.row_ptr[i + 1]; ++k) {
                nu[{i, h_.col_idx[k]}] = lambda_[h_.col_idx[k]];
            }
        }
        Leg leg;
        leg.hard.assign(h_.cols, 0);
        const std::vector<Bit> target(syndrome.begin(), syndrome.end());
        for (std::uint32_t t = 0; t < max_iter; ++t) {
            check_update(syndrome, shift_of(alpha, t), nu, mu);
            variable_update(memory, beta, mu, nu, leg.hard);
            leg.iterations = t + 1;
            if (test::syndrome_of(h_, leg.hard) == target) {
                leg.converged = true;
                break;
            }
        }
        return leg;
    }

    test::Csr h_; // a copy: the test graphs are small
    int max_;
    int bits_;
    int scale_;
    int m_;
    std::vector<std::vector<index_t>> col_rows_;
    std::vector<int> lambda_;
    std::vector<int> marginal_;
};

struct Problem {
    test::Csr h;
    std::vector<double> p;
    std::vector<std::vector<Bit>> syndromes;
};

// Priors spread over the range where the formats saturate or not, a few p = 0 columns, and
// syndromes of errors drawn from the priors.
Problem random_problem(std::uint64_t seed, index_t m, index_t n, std::size_t shots) {
    Problem problem{.h = test::random_csr(m, n, 1, 6, seed), .p = {}, .syndromes = {}};
    std::mt19937_64 rng(seed * 104729);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (index_t j = 0; j < n; ++j) {
        problem.p.push_back(unit(rng) < 0.03 ? 0.0 : 0.0005 + 0.12 * unit(rng));
    }
    for (std::size_t s = 0; s < shots; ++s) {
        std::vector<Bit> error(n, 0);
        for (index_t j = 0; j < n; ++j) {
            error[j] = unit(rng) < problem.p[j] ? 1 : 0;
        }
        problem.syndromes.push_back(test::syndrome_of(problem.h, error));
    }
    return problem;
}

struct Scenario {
    const char* name;
    MinSumConfig min_sum;
    RelayConfig relay;
};

const std::vector<Scenario> scenarios{
    {.name = "min_sum",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "halved",
     .min_sum = {.alpha = ConstantAlpha{0.5}, .gamma0 = {}},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "mem_bp_adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 25, .set_max_iter = 0, .num_sets = 0, .stopping = AfterLeg0{}}},
    {.name = "relay_nconv",
     .min_sum = {.alpha = ConstantAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 12, .set_max_iter = 8, .num_sets = 15, .stopping = AfterNConverged{3}}},
    {.name = "relay_all_adaptive",
     .min_sum = {.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125},
     .relay = {.pre_iter = 6, .set_max_iter = 6, .num_sets = 8, .stopping = AllLegs{}}},
    {.name = "relay_negative_gamma0",
     .min_sum = {.alpha = ConstantAlpha{0.75}, .gamma0 = -0.3},
     .relay = {.pre_iter = 5, .set_max_iter = 5, .num_sets = 6, .stopping = AfterNConverged{2}}},
};

struct Coverage {
    std::size_t decodes = 0;
    std::size_t converged = 0;
    std::size_t failed = 0;
    std::size_t best_after_leg0 = 0;
};

template <class A, class MakeExecutor>
void compare_with_naive(const Problem& problem, const GraphOptions& options,
                        MakeExecutor make_executor, const std::string& label, Coverage& coverage) {
    using Exec = decltype(make_executor());
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, options);
    ASSERT_TRUE(graph) << graph.error().detail;
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(5, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    NaiveFixedDecoder naive(problem.h, priors->llr(), static_cast<int>(A::magnitude_bits),
                            static_cast<int>(A::llr_scale), static_cast<int>(A::memory_scale));
    std::vector<double> marginals(problem.h.cols);
    for (const Scenario& scenario : scenarios) {
        auto backend = CpuBackend<A, Exec>::create(*graph, *priors, make_executor());
        ASSERT_TRUE(backend);
        auto decoder = RelayDecoder<CpuBackend<A, Exec>>::create(
            std::move(*backend), scenario.min_sum, scenario.relay, &*gammas);
        ASSERT_TRUE(decoder) << decoder.error().detail;
        for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
            const auto& syndrome = problem.syndromes[s];
            const test::ReferenceResult want =
                naive.decode(syndrome, scenario.min_sum, scenario.relay, &*gammas, s);
            auto got = decoder->decode(syndrome, s);
            ASSERT_TRUE(got);
            const std::string where = std::format("{} {} {} shot {}", A::name, label,
                                                  scenario.name, s);
            ASSERT_EQ(got->success, want.success) << where;
            ASSERT_EQ(got->iterations, want.iterations) << where;
            ASSERT_EQ(got->legs_executed, want.legs) << where;
            ASSERT_EQ(got->best_leg, want.best_leg) << where;
            ASSERT_EQ(std::bit_cast<std::uint64_t>(got->weight),
                      std::bit_cast<std::uint64_t>(want.weight))
                << where;
            ASSERT_TRUE(std::ranges::equal(got->hard, want.hard)) << where;
            for (std::uint32_t leg = 0; leg < got->legs_executed; ++leg) {
                ASSERT_EQ(got->legs[leg].iterations, want.leg_iterations[leg]) << where;
                ASSERT_EQ(got->legs[leg].converged, want.leg_converged[leg]) << where;
            }
            decoder->backend().read_marginals(marginals);
            for (index_t j = 0; j < problem.h.cols; ++j) {
                // The backend stores σ and clips it when it is next read; the naive decoder
                // clips when it stores. The clipped values must agree.
                const double limit = static_cast<double>(A::max_magnitude) / A::llr_scale;
                ASSERT_EQ(std::clamp(marginals[j], -limit, limit), want.marginals[j])
                    << where << " column " << j;
            }
            ++coverage.decodes;
            coverage.converged += want.success ? 1U : 0U;
            coverage.failed += want.success ? 0U : 1U;
            coverage.best_after_leg0 += want.best_leg.value_or(0) > 0 ? 1U : 0U;
        }
    }
}

template <class A>
class FixedDecode : public testing::Test {};
using Formats = testing::Types<Int4_2_8, Int5_2_8, Int6_2_8>;
TYPED_TEST_SUITE(FixedDecode, Formats);

TYPED_TEST(FixedDecode, BackendMatchesTheNaiveDecoderEverywhere) {
    using A = TypeParam;
    Coverage coverage;
    for (const std::uint64_t seed : {1U, 2U, 3U}) {
        const Problem problem = random_problem(seed, 30, 70, 12);
        for (const EdgeLayout layout : {EdgeLayout::row_major, EdgeLayout::column_blocked}) {
            for (const ColumnOrder order :
                 {ColumnOrder::wavefront, ColumnOrder::degree_classes, ColumnOrder::natural}) {
                if (layout == EdgeLayout::column_blocked && order == ColumnOrder::natural) {
                    continue;
                }
                const GraphOptions options{.layout = layout, .column_order = order, .block_rows = 8};
                const std::string label = std::format("seed {} layout {} order {}", seed,
                                                      static_cast<int>(layout),
                                                      static_cast<int>(order));
                compare_with_naive<A>(problem, options, [] { return Serial{}; }, label, coverage);
                if (order == ColumnOrder::wavefront) {
                    compare_with_naive<A>(problem, options, [] { return Team(3); },
                                          label + " team", coverage);
                }
            }
        }
    }
    EXPECT_GT(coverage.converged, coverage.decodes / 4);
    EXPECT_GT(coverage.failed, 0U);
    EXPECT_GT(coverage.best_after_leg0, 0U);
}

TEST(FixedDecode, LowPrecisionChangesTheOutcomeSomewhere) {
    // The formats must not all collapse onto the same decode: over a batch of syndromes int4
    // and int6 differ in some iteration count or solution (otherwise the quantisation is not
    // reaching the decoder).
    const Problem problem = random_problem(9, 40, 90, 40);
    auto graph = TannerGraph::from_csr(problem.h.rows, problem.h.cols, problem.h.row_ptr,
                                       problem.h.col_idx, GraphOptions{});
    ASSERT_TRUE(graph);
    auto priors = Priors::from_probabilities(problem.p);
    ASSERT_TRUE(priors);
    auto gammas = UniformGammaGenerator::create(5, -0.24, 0.66, problem.h.cols);
    ASSERT_TRUE(gammas);
    const MinSumConfig ms{.alpha = AdaptiveAlpha{1.0}, .gamma0 = 0.125};
    const RelayConfig rc{.pre_iter = 20, .set_max_iter = 10, .num_sets = 10,
                         .stopping = AfterNConverged{2}};
    auto low = CpuRelayDecoder<Int4_2_8>::create(
        std::move(*CpuBackend<Int4_2_8>::create(*graph, *priors)), ms, rc, &*gammas);
    auto high = CpuRelayDecoder<Int6_2_8>::create(
        std::move(*CpuBackend<Int6_2_8>::create(*graph, *priors)), ms, rc, &*gammas);
    ASSERT_TRUE(low && high);
    std::size_t differ = 0;
    for (std::size_t s = 0; s < problem.syndromes.size(); ++s) {
        const auto a = low->decode(problem.syndromes[s], s);
        const auto b = high->decode(problem.syndromes[s], s);
        ASSERT_TRUE(a && b);
        differ += a->iterations != b->iterations || !std::ranges::equal(a->hard, b->hard) ? 1U : 0U;
    }
    EXPECT_GT(differ, 0U);
}

} // namespace
