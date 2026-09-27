"""Detector-error-model parity with the reference circuits shipped in IBM's relay_bp repository.

Set RTD_RELAY_TESTDATA to relay_bp's tests/testdata/bicycle_bivariate directory to run these.
"""

import os
from pathlib import Path

import pytest
import stim

from parity import compare
from rtd.bb_code import get_code
from rtd.circuit import build_memory_circuit
from rtd.noise import NoiseModel

TESTDATA = os.environ.get("RTD_RELAY_TESTDATA")
pytestmark = pytest.mark.skipif(not TESTDATA, reason="RTD_RELAY_TESTDATA not set")

REFERENCE_NAMES = {"bb18": "18_4_3", "bb72": "72_12_6", "gross": "144_12_12"}
REFERENCE_KINDS = {"z": "memory_Z", "x": "memory_X", "choi": "memory_choi_XZ"}


def _reference(code_name: str, experiment: str, p: float) -> stim.Circuit:
    tag = f"bicycle_bivariate_{REFERENCE_NAMES[code_name]}_{REFERENCE_KINDS[experiment]},"
    matches = [f for f in Path(TESTDATA).iterdir() if tag in f.name and f"error_rate={p}," in f.name]
    assert len(matches) == 1, f"expected one reference circuit for {tag} p={p}, found {len(matches)}"
    return stim.Circuit.from_file(matches[0])


@pytest.mark.parametrize("experiment", sorted(REFERENCE_KINDS))
@pytest.mark.parametrize("code_name", sorted(REFERENCE_NAMES))
@pytest.mark.parametrize("p", [0.001, 0.003])
def test_compat_mode_matches_reference_exactly(code_name, experiment, p):
    code = get_code(code_name)
    mine = build_memory_circuit(code, code.distance, experiment, NoiseModel.uniform(p), relay_bp_compat=True)
    report = compare(mine.circuit, _reference(code_name, experiment, p), num_code_qubits=2 * code.n)
    assert report.identical, report


@pytest.mark.parametrize("experiment", sorted(REFERENCE_KINDS))
def test_default_mode_differs_only_in_final_cycle_probabilities(experiment):
    code = get_code("bb72")
    mine = build_memory_circuit(code, code.distance, experiment, NoiseModel.uniform(0.003))
    report = compare(mine.circuit, _reference("bb72", experiment, 0.003), num_code_qubits=2 * code.n)
    assert report.only_mine == report.only_reference == 0
    assert report.observables_equivalent
    assert 0 < report.probability_mismatches < report.errors_mine // 50
