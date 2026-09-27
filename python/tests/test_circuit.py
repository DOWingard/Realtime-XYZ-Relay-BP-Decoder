import numpy as np
import pytest

from rtd.bb_code import get_code
from rtd.circuit import DETECTOR_X, DETECTOR_Z, EXPERIMENTS, build_memory_circuit
from rtd.noise import NoiseModel
from rtd.sample import check_deterministic


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("name", ["bb18", "bb72", "gross"])
def test_noiseless_circuit_is_deterministic(name, experiment):
    code = get_code(name)
    mc = build_memory_circuit(code, code.distance, experiment, NoiseModel.uniform(1e-3))
    check_deterministic(mc.circuit)


@pytest.mark.parametrize(
    "experiment, detectors, observables",
    [("z", 1728, 12), ("x", 1728, 12), ("choi", 1872, 24)],
)
def test_gross_counts(experiment, detectors, observables):
    code = get_code("gross")
    mc = build_memory_circuit(code, 12, experiment, NoiseModel.uniform(3e-3))
    assert mc.circuit.num_detectors == detectors
    assert mc.circuit.num_observables == observables
    assert len(mc.detectors.round) == detectors


def test_choi_detector_metadata_layers():
    code = get_code("gross")
    mc = build_memory_circuit(code, 12, "choi", NoiseModel.uniform(3e-3))
    info = mc.detectors
    # Every layer (12 noisy rounds + final perfect cycle) holds all 72 X and 72 Z stabilizers.
    for r in range(1, 14):
        in_round = info.round == r
        assert np.sum(in_round & (info.kind == DETECTOR_X)) == 72
        assert np.sum(in_round & (info.kind == DETECTOR_Z)) == 72


def test_noise_model_rejects_bad_probability():
    with pytest.raises(ValueError):
        NoiseModel.uniform(1.5)


def test_rounds_must_be_positive():
    with pytest.raises(ValueError):
        build_memory_circuit(get_code("bb18"), 0, "z", NoiseModel.uniform(1e-3))
