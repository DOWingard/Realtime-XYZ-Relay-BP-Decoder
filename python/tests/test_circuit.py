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


def _dem_by_detector_set(dem, relabel=None):
    """{(sorted detectors, sorted observables): summed probability} of a flattened detector error model."""
    out: dict = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = sorted((relabel[t.val] if relabel is not None else t.val) for t in inst.targets_copy() if t.is_relative_detector_id())
        obs = sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id())
        key = (tuple(dets), tuple(obs))
        out[key] = out.get(key, 0.0) + inst.args_copy()[0]
    return out


@pytest.mark.parametrize("relay_bp_compat", [False, True])
def test_choi_readout_in_round_order(relay_bp_compat):
    code = get_code("bb18")
    noise = NoiseModel.uniform(3e-3)
    native = build_memory_circuit(code, 3, "choi", noise, relay_bp_compat=relay_bp_compat)
    ordered = build_memory_circuit(code, 3, "choi", noise, relay_bp_compat=relay_bp_compat, readout_in_round_order=True)
    check_deterministic(ordered.circuit)
    info, h = ordered.detectors, code.half
    # Every round, the final layer included, lists (type, check) in the same order: Z checks, then X checks.
    key = (info.kind.astype(int) * h + info.check).reshape(4, 2 * h)
    assert (key == key[0]).all() and list(key[0][:h]) == [DETECTOR_Z * h + i for i in range(h)]
    assert not (native.detectors.kind.reshape(4, 2 * h) == native.detectors.kind.reshape(4, 2 * h)[0]).all()
    # Same detector error model up to relabelling the final layer: native detector d is ordered detector perm[d].
    m = native.circuit.num_detectors
    pos = {(int(t), int(c), int(r)): i for i, (t, c, r) in enumerate(zip(info.kind, info.check, info.round, strict=True))}
    perm = [pos[(int(t), int(c), int(r))] for t, c, r in zip(native.detectors.kind, native.detectors.check, native.detectors.round, strict=True)]
    assert sorted(perm) == list(range(m)) and perm[: 3 * 2 * h] == list(range(3 * 2 * h))
    a = _dem_by_detector_set(native.circuit.detector_error_model(), relabel=perm)
    b = _dem_by_detector_set(ordered.circuit.detector_error_model())
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-15 for k in a)
    # The default order is unchanged (existing circuits keep their checksums).
    assert str(native.circuit) == str(build_memory_circuit(code, 3, "choi", noise, relay_bp_compat=relay_bp_compat).circuit)


def test_readout_in_round_order_needs_choi():
    with pytest.raises(ValueError, match="choi"):
        build_memory_circuit(get_code("bb18"), 3, "z", NoiseModel.uniform(1e-3), readout_in_round_order=True)
