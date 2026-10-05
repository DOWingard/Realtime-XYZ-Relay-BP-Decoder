#pragma once

// Test helpers: small matrices, random graphs, and a deliberately naive reference decoder.
//
// The reference decoder implements the decode specification in the most direct way possible:
// separate μ and η arrays indexed by (row, column) pairs, a dense ê vector recomputed every
// iteration, and a full H·ê product for the convergence test. It shares no code with the
// optimised kernels, so agreement between the two is evidence for both.

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "rtd/core/config.hpp"
#include "rtd/core/gamma.hpp"
#include "rtd/core/graph.hpp"
#include "rtd/core/types.hpp"

namespace rtd::test {

struct Csr {
    index_t rows = 0;
    index_t cols = 0;
    std::vector<index_t> row_ptr{0};
    std::vector<index_t> col_idx;
};

inline Csr csr_from_dense(const std::vector<std::vector<int>>& dense) {
    Csr csr;
    csr.rows = static_cast<index_t>(dense.size());
    csr.cols = dense.empty() ? 0 : static_cast<index_t>(dense[0].size());
    for (const auto& row : dense) {
        for (index_t j = 0; j < row.size(); ++j) {
            if (row[j] != 0) {
                csr.col_idx.push_back(j);
            }
        }
        csr.row_ptr.push_back(static_cast<index_t>(csr.col_idx.size()));
    }
    return csr;
}

// A random m×n matrix: each column gets a degree in [min_degree, max_degree] (so degree runs are
// short and varied) and distinct random rows. Some rows may be empty.
inline Csr random_csr(index_t m, index_t n, index_t min_degree, index_t max_degree,
                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<index_t> degree_dist(min_degree, max_degree);
    std::uniform_int_distribution<index_t> row_dist(0, m - 1);
    std::vector<std::vector<index_t>> rows(m);
    for (index_t j = 0; j < n; ++j) {
        const index_t degree = std::min(degree_dist(rng), m);
        std::vector<index_t> chosen;
        while (chosen.size() < degree) {
            const index_t r = row_dist(rng);
            if (std::ranges::find(chosen, r) == chosen.end()) {
                chosen.push_back(r);
            }
        }
        for (const index_t r : chosen) {
            rows[r].push_back(j);
        }
    }
    Csr csr;
    csr.rows = m;
    csr.cols = n;
    for (auto& row : rows) {
        std::ranges::sort(row);
        csr.col_idx.insert(csr.col_idx.end(), row.begin(), row.end());
        csr.row_ptr.push_back(static_cast<index_t>(csr.col_idx.size()));
    }
    return csr;
}

inline std::vector<Bit> syndrome_of(const Csr& h, std::span<const Bit> error) {
    std::vector<Bit> s(h.rows, 0);
    for (index_t i = 0; i < h.rows; ++i) {
        for (index_t k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) {
            s[i] ^= error[h.col_idx[k]];
        }
    }
    return s;
}

struct ReferenceResult {
    bool success = false;
    std::uint32_t iterations = 0;
    std::uint32_t legs = 0;
    std::optional<std::uint32_t> best_leg;
    double weight = std::numeric_limits<double>::infinity();
    std::vector<Bit> hard;
    std::vector<std::uint32_t> leg_iterations;
    std::vector<bool> leg_converged;
    std::vector<double> marginals; // M after the last iteration run
};

// Relay-BP straight from the specification, in number type T.
template <class T>
class ReferenceDecoder {
public:
    ReferenceDecoder(const Csr& h, std::span<const double> llr) : h_(h), llr_(llr.begin(), llr.end()) {
        col_rows_.resize(h.cols);
        for (index_t i = 0; i < h.rows; ++i) {
            for (index_t k = h.row_ptr[i]; k < h.row_ptr[i + 1]; ++k) {
                col_rows_[h.col_idx[k]].push_back(i); // ascending rows
            }
        }
        lambda_.resize(h.cols);
        for (index_t j = 0; j < h.cols; ++j) {
            lambda_[j] = llr_[j] == std::numeric_limits<double>::infinity()
                             ? std::numeric_limits<T>::max()
                             : static_cast<T>(llr_[j]);
        }
    }

    ReferenceResult decode(std::span<const Bit> syndrome, const MinSumConfig& ms,
                           const RelayConfig& rc, const GammaSource* gammas,
                           std::uint64_t stream = 0) {
        const bool memory = ms.gamma0.has_value();
        std::vector<T> gamma(h_.cols, memory ? static_cast<T>(*ms.gamma0) : T{0});
        marginal_.assign(h_.cols, T{0});
        if (memory) {
            marginal_ = lambda_;
        }
        ReferenceResult out;
        const Leg first = run_leg(syndrome, rc.pre_iter, ms.alpha, memory, gamma);
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
        const bool done_after_leg0 =
            first.converged && (std::holds_alternative<AfterLeg0>(rc.stopping) ||
                                (nconv != nullptr && converged >= nconv->count));
        if (!done_after_leg0) {
            std::vector<double> scratch(h_.cols);
            for (std::uint32_t leg = 1; leg <= rc.num_sets; ++leg) {
                const auto g = gammas->gammas(stream, leg, scratch);
                for (index_t j = 0; j < h_.cols; ++j) {
                    gamma[j] = static_cast<T>(g[j]);
                }
                const Leg result = run_leg(syndrome, rc.set_max_iter, ms.alpha, true, gamma);
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
        out.marginals.assign(marginal_.begin(), marginal_.end());
        return out;
    }

private:
    struct Leg {
        bool converged = false;
        std::uint32_t iterations = 0;
        std::vector<Bit> hard;
    };

    double weight(const std::vector<Bit>& hard) const {
        double w = 0.0;
        for (index_t j = 0; j < h_.cols; ++j) {
            if (hard[j] != 0 && std::isfinite(llr_[j])) {
                w += llr_[j];
            }
        }
        return w;
    }

    Leg run_leg(std::span<const Bit> syndrome, std::uint32_t max_iter, const AlphaRule& alpha_rule,
                bool memory, const std::vector<T>& gamma) {
        // μ[(i, j)] variable → check and η[(i, j)] check → variable.
        std::map<std::pair<index_t, index_t>, T> mu;
        std::map<std::pair<index_t, index_t>, T> eta;
        for (index_t i = 0; i < h_.rows; ++i) {
            for (index_t k = h_.row_ptr[i]; k < h_.row_ptr[i + 1]; ++k) {
                mu[{i, h_.col_idx[k]}] = lambda_[h_.col_idx[k]];
                eta[{i, h_.col_idx[k]}] = T{0};
            }
        }
        Leg leg;
        leg.hard.assign(h_.cols, 0);
        const T max = std::numeric_limits<T>::max();
        for (std::uint32_t t = 0; t < max_iter; ++t) {
            const T alpha = static_cast<T>(alpha_at(alpha_rule, t));
            for (index_t i = 0; i < h_.rows; ++i) {
                const index_t begin = h_.row_ptr[i];
                const index_t end = h_.row_ptr[i + 1];
                if (begin == end) {
                    continue;
                }
                bool parity = syndrome[i] != 0;
                std::vector<T> magnitudes;
                for (index_t k = begin; k < end; ++k) {
                    const T x = mu[{i, h_.col_idx[k]}];
                    parity ^= std::signbit(x);
                    magnitudes.push_back(std::fabs(x));
                }
                // Two smallest non-NaN magnitudes of the multiset, each clamped to max.
                T min1 = max;
                T min2 = max;
                for (const T a : magnitudes) {
                    if (a < min1) {
                        min2 = min1;
                        min1 = a;
                    } else if (a < min2) {
                        min2 = a;
                    }
                }
                for (index_t k = begin; k < end; ++k) {
                    const T x = mu[{i, h_.col_idx[k]}];
                    const T mag = std::fabs(x) == min1 ? alpha * min2 : alpha * min1;
                    eta[{i, h_.col_idx[k]}] = (parity != std::signbit(x)) ? -mag : mag;
                }
            }
            for (index_t j = 0; j < h_.cols; ++j) {
                T big_lambda = lambda_[j];
                if (memory && lambda_[j] != max) {
                    const T keep = T{1} - gamma[j];
                    const T a = lambda_[j] * keep;
                    const T b = marginal_[j] * gamma[j];
                    big_lambda = a + b;
                }
                const auto& rows = col_rows_[j];
                std::vector<T> prefix(rows.size());
                T s = big_lambda;
                for (std::size_t k = 0; k < rows.size(); ++k) {
                    prefix[k] = s;
                    s = s + eta[{rows[k], j}];
                }
                marginal_[j] = s;
                T suffix = T{0};
                for (std::size_t k = rows.size(); k-- > 0;) {
                    mu[{rows[k], j}] = prefix[k] + suffix;
                    suffix = suffix + eta[{rows[k], j}];
                }
                leg.hard[j] = s <= T{0} ? 1 : 0;
            }
            leg.iterations = t + 1;
            if (syndrome_of(h_, leg.hard) == std::vector<Bit>(syndrome.begin(), syndrome.end())) {
                leg.converged = true;
                break;
            }
        }
        return leg;
    }

    const Csr& h_;
    std::vector<double> llr_;
    std::vector<std::vector<index_t>> col_rows_;
    std::vector<T> lambda_;
    std::vector<T> marginal_;
};

} // namespace rtd::test
