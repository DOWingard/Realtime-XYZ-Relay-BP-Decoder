"""rtd.dem: the conversion of a detector error model into (H, A, priors, rounds).

It must reproduce the committed artifacts exactly, which were exported through relay_bp's
from_dem: the bb18 R = 9 fixture and, when the data directory holds it, the gross-code R = 12
artifact. Separately: decomposed mechanisms combine by symmetric difference, decided mechanisms
are pruned with relay_bp's rule (and its biases), rounds come from coordinates, and a time
structure that sliding windows cannot use is reported.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import stim

from rtd.dem import (
    check_time_structure,
    detector_rounds,
    parse_dem,
    problem_from_artifact,
    problem_from_dem,
)

REPO = Path(__file__).resolve().parents[2]
BB18_R9 = REPO / "test" / "fixtures" / "bb18_choi_r9"
GROSS_ARTIFACT = REPO / "data" / "artifacts" / "gross_choi_p0.003"
GROSS_CIRCUIT = REPO / "data" / "shots" / "gross_choi_p0.003_s12345" / "circuit.stim"


def _expect_artifact(problem, artifact: Path) -> None:
    for name, ours in (
        ("H_indptr.npy", problem.h_indptr),
        ("H_indices.npy", problem.h_indices),
        ("A_indptr.npy", problem.a_indptr),
        ("A_indices.npy", problem.a_indices),
        ("col_to_dem.npy", problem.col_to_dem),
        ("det_round.npy", problem.det_round),
    ):
        stored = np.load(artifact / name)
        assert ours.dtype == stored.dtype, name
        assert np.array_equal(ours, stored), name
    stored_priors = np.load(artifact / "priors.npy")
    assert problem.priors.dtype == stored_priors.dtype == np.float64
    # Bitwise, not approximately.
    assert np.array_equal(problem.priors.view(np.uint64), stored_priors.view(np.uint64))
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert problem.num_detectors == manifest["num_detectors"]
    assert problem.num_columns == manifest["num_columns"]
    assert problem.num_observables == manifest["num_observables"]
    assert problem.syndrome_bias is None and not (artifact / "syndrome_bias.npy").exists()


def test_reproduces_the_bb18_r9_artifact():
    circuit = stim.Circuit.from_file(BB18_R9 / "shots" / "circuit.stim")
    problem = problem_from_dem(circuit.detector_error_model(decompose_errors=False), round_of=1)
    _expect_artifact(problem, BB18_R9 / "artifact")
    assert check_time_structure(problem.det_round) == (10, 18)
    assert not problem.decomposed


@pytest.mark.skipif(not (GROSS_ARTIFACT.is_dir() and GROSS_CIRCUIT.is_file()), reason="gross R = 12 data not present")
def test_reproduces_the_gross_r12_artifact():
    circuit = stim.Circuit.from_file(GROSS_CIRCUIT)
    problem = problem_from_dem(circuit.detector_error_model(decompose_errors=False), round_of=1)
    _expect_artifact(problem, GROSS_ARTIFACT)
    assert (problem.num_detectors, problem.num_columns) == (1872, 71280)
    assert check_time_structure(problem.det_round) == (13, 144)


def test_artifact_loader_round_trips():
    problem = problem_from_artifact(BB18_R9 / "artifact")
    circuit = stim.Circuit.from_file(BB18_R9 / "shots" / "circuit.stim")
    converted = problem_from_dem(circuit.detector_error_model(decompose_errors=False), round_of=1)
    for field in ("h_indptr", "h_indices", "a_indptr", "a_indices", "priors", "col_to_dem", "det_round"):
        assert np.array_equal(getattr(problem, field), getattr(converted, field)), field
    assert problem_from_artifact(BB18_R9 / "artifact", rounds=False).det_round is None
    with pytest.raises(FileNotFoundError):
        problem_from_artifact(BB18_R9)


def test_decomposed_mechanisms_combine_by_symmetric_difference(caplog):
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1 ^ D1 D2 L0
        error(0.2) D2
        detector(0, 1) D0
        detector(0, 1) D1
        detector(0, 2) D2
        """
    )
    h, a, p, decomposed = parse_dem(dem)
    assert decomposed
    assert h.toarray().tolist() == [[1, 0], [0, 0], [1, 1]]
    assert a.toarray().tolist() == [[1, 0]]
    assert p.tolist() == [0.1, 0.2]
    with caplog.at_level("WARNING", logger="rtd.dem"):
        problem = problem_from_dem(dem)
    assert problem.decomposed
    assert any("decomposed" in r.getMessage() for r in caplog.records)
    # D1 is flipped by no mechanism once the components are combined; its row is empty.
    assert problem.h_indptr.tolist() == [0, 1, 1, 3]


def test_repeat_blocks_and_detector_shifts_are_unrolled():
    dem = stim.DetectorErrorModel(
        """
        error(0.01) D0
        repeat 3 {
            error(0.02) D0 D1
            detector(0, 1) D0
            shift_detectors(0, 1) 1
        }
        detector(0, 1) D0
        """
    )
    problem = problem_from_dem(dem, round_of=1)
    # Coordinates shift with the detectors: the last detector sits at round 1 + 3.
    assert problem.num_columns == 4
    assert problem.h_matrix().toarray().tolist() == [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    assert problem.det_round.tolist() == [1, 2, 3, 4]
    assert check_time_structure(problem.det_round) == (4, 1)


def test_decided_mechanisms_are_pruned_into_biases():
    dem = stim.DetectorErrorModel(
        """
        error(0) D0 L0
        error(0.1) D0 D1
        error(1) D1 D2 L1
        error(0.2) D2 L1
        error(1) D2
        """
    )
    problem = problem_from_dem(dem)
    assert problem.col_to_dem.tolist() == [1, 3]
    assert (problem.pruned_p0, problem.pruned_p1) == (1, 2)
    # The two certain mechanisms flip D1, D2 and D2: D2 cancels.
    assert problem.syndrome_bias.tolist() == [0, 1, 0]
    assert problem.observables_bias.tolist() == [0, 1]
    assert problem.priors.tolist() == [0.1, 0.2]
    # A threshold also prunes the mechanisms within it of 0 or 1.
    thresholded = problem_from_dem(dem, prune_threshold=0.15)
    assert thresholded.col_to_dem.tolist() == [3]
    assert thresholded.syndrome_bias.tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        problem_from_dem(dem, prune_threshold=0.5)


def test_pruning_matches_relay_bp():
    relay = pytest.importorskip("relay_bp.stim.sinter.check_matrices", reason="needs the reference group")
    dem = stim.DetectorErrorModel(
        """
        error(0) D0 L0
        error(0.1) D0 D1
        error(1) D1 D2 L1
        error(0.2) D2 L1
        error(0.3) D0 D2
        """
    )
    ours = problem_from_dem(dem)
    theirs = relay.CheckMatrices.from_dem(dem, prune_decided_errors=False).prune_decided_errors()
    assert np.array_equal(ours.h_matrix().toarray(), np.asarray(theirs.check_matrix.todense()))
    assert np.array_equal(ours.priors, theirs.error_priors)
    assert np.array_equal(ours.syndrome_bias, np.asarray(theirs.syndrome_bias).reshape(-1))
    assert np.array_equal(ours.observables_bias, np.asarray(theirs.observables_bias).reshape(-1))


def test_rounds_come_from_coordinates():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1
        error(0.1) D1 D2 D3
        detector(5, 0, 1) D0
        detector(6, 0, 1) D1
        detector(5, 1, 0) D2
        detector(6, 1, 0) D3
        """
    )
    assert detector_rounds(dem, 1).tolist() == [0, 0, 1, 1]
    assert detector_rounds(dem, lambda c: int(c[1]) + 1).tolist() == [1, 1, 2, 2]
    assert detector_rounds(dem, -1).tolist() == [1, 1, 0, 0]
    with pytest.raises(ValueError, match="numbered from 1"):
        check_time_structure(detector_rounds(dem, 1))
    with pytest.raises(ValueError, match="grouped by round"):
        check_time_structure(np.array([1, 2, 1, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="no coordinate"):
        detector_rounds(dem, 3)
    with pytest.raises(ValueError, match="not an integer"):
        detector_rounds(dem, lambda c: c[0] / 2 + 0.25)
    missing = stim.DetectorErrorModel("error(0.1) D0 D1\ndetector(0, 1) D0")
    with pytest.raises(ValueError, match="no coordinates"):
        detector_rounds(missing, 1)
    unequal = np.array([1, 1, 2, 3, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="equal numbers"):
        check_time_structure(unequal)


def test_summary_and_matrices_agree():
    circuit = stim.Circuit.from_file(BB18_R9 / "shots" / "circuit.stim")
    dem = circuit.detector_error_model(decompose_errors=False)
    problem = problem_from_dem(dem, round_of=1)
    h, a, p, _ = parse_dem(dem)
    assert (problem.h_matrix() != h.tocsr()).nnz == 0
    assert (problem.a_matrix() != a.tocsr()).nnz == 0
    summary = problem.summary()
    assert summary["nnz_H"] == h.nnz and summary["rounds"] == 10 and summary["columns_in_model"] == p.size
