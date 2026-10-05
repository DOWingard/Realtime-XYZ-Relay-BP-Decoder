#pragma once

// Header-only bridge from a loaded artifact to the window layer's Problem. It needs rtd_io, which
// rtd_window itself does not link; include it only from targets that link both.

#include <span>
#include <vector>

#include "rtd/core/types.hpp"
#include "rtd/io/artifact.hpp"
#include "rtd/window/problem.hpp"

namespace rtd::window {

// The CSR arrays of H and A in the artifact's own column numbering, rebuilt from the loaded
// graph (whose internal column order may differ) and the column-stored observable matrix. The
// priors and detector rounds are viewed in place, so the artifact must outlive this object and
// every Problem it hands out.
class ArtifactProblem {
public:
    explicit ArtifactProblem(const io::Artifact& artifact)
        : artifact_(&artifact), h_row_ptr_(std::size_t{artifact.num_detectors()} + 1, 0),
          a_row_ptr_(std::size_t{artifact.num_observables()} + 1, 0) {
        const TannerGraph& graph = artifact.graph;
        const index_t m = graph.num_rows();
        const index_t n = graph.num_columns();
        // Visiting columns in ascending external index leaves every row's columns ascending.
        for (index_t i = 0; i < m; ++i) {
            h_row_ptr_[i + 1] = h_row_ptr_[i] + graph.row_degree(i);
        }
        h_col_indices_.resize(graph.num_edges());
        std::vector<index_t> fill(h_row_ptr_.begin(), h_row_ptr_.end() - 1);
        for (index_t j = 0; j < n; ++j) {
            for (const index_t row : graph.column_rows(graph.internal_column(j))) {
                h_col_indices_[fill[row]++] = j;
            }
        }
        const ObservableMatrix& a = artifact.observables;
        for (index_t j = 0; j < n; ++j) {
            for (const index_t o : a.column(j)) {
                ++a_row_ptr_[o + 1];
            }
        }
        for (index_t o = 0; o < a.num_rows(); ++o) {
            a_row_ptr_[o + 1] += a_row_ptr_[o];
        }
        a_col_indices_.resize(a.num_nonzeros());
        fill.assign(a_row_ptr_.begin(), a_row_ptr_.end() - 1);
        for (index_t j = 0; j < n; ++j) {
            for (const index_t o : a.column(j)) {
                a_col_indices_[fill[o]++] = j;
            }
        }
    }

    [[nodiscard]] Problem problem() const noexcept {
        return {.num_rows = artifact_->num_detectors(),
                .num_columns = artifact_->num_columns(),
                .num_observables = artifact_->num_observables(),
                .h_row_ptr = h_row_ptr_,
                .h_col_indices = h_col_indices_,
                .priors = artifact_->priors.probabilities(),
                .a_row_ptr = a_row_ptr_,
                .a_col_indices = a_col_indices_,
                .detector_round = artifact_->detector_round};
    }

private:
    const io::Artifact* artifact_;
    std::vector<index_t> h_row_ptr_;
    std::vector<index_t> h_col_indices_;
    std::vector<index_t> a_row_ptr_;
    std::vector<index_t> a_col_indices_;
};

} // namespace rtd::window
