"""rtd.sinter_decoders(): the sinter adapter decodes what the Python decoders decode.

A compiled decoder numbers the shots it has seen 0, 1, 2, ... across batches, so feeding it one
file of shots in pieces reproduces a single decode_batch of the file bit for bit, records
included; it pickles into sinter's worker processes; it refuses to run with another build of the
extension module than the one it was pinned to; and a small sinter.collect run finishes with a
plausible error count.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import stim

pytest.importorskip("rtd._native", reason="the extension module is not built (RTD_ENABLE_PYTHON)")
sinter = pytest.importorskip("sinter")

import rtd  # noqa: E402
from rtd.decoders import RELAY_BP5  # noqa: E402
from rtd.sample import build_circuit  # noqa: E402
from rtd.sinter_adapter import RtdSinterDecoder, native_module_sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
R9 = REPO / "test" / "fixtures" / "bb18_choi_r9"

SMALL = {
    **RELAY_BP5,
    "pre_iter": 20,
    "set_max_iter": 15,
    "num_sets": 40,
    "stopping": {"rule": "after_n_converged", "count": 3},
    "gamma_source": {"type": "uniform", "seed": 7, "low": -0.24, "high": 0.66},
}


def _dem() -> stim.DetectorErrorModel:
    return stim.Circuit.from_file(R9 / "shots" / "circuit.stim").detector_error_model(decompose_errors=False)


def _packed_shots() -> np.ndarray:
    return np.packbits(np.load(R9 / "shots" / "detectors.npy"), axis=1, bitorder="little")


def _decode_in_pieces(compiled, packed: np.ndarray, cuts: tuple[int, ...]) -> np.ndarray:
    edges = (0, *cuts, packed.shape[0])
    return np.concatenate(
        [compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed[a:b]) for a, b in zip(edges, edges[1:], strict=False)]
    )


def test_whole_shot_decoder_matches_decode_batch_across_batches():
    dem = _dem()
    decoders = rtd.sinter_decoders(SMALL, windows=())
    assert sorted(decoders) == ["rtd_relay"]
    compiled = decoders["rtd_relay"].compile_decoder_for_dem(dem=dem)
    packed = _packed_shots()
    ours = _decode_in_pieces(compiled, packed, (30, 31, 77))
    reference = rtd.RelayDecoder.from_detector_error_model(dem, SMALL).decode_batch(packed, bit_packed_shots=True)
    assert ours.dtype == np.uint8 and ours.shape == (packed.shape[0], 1)
    assert np.array_equal(np.unpackbits(ours, axis=1, count=dem.num_observables, bitorder="little"), reference)
    assert compiled.next_stream == packed.shape[0]


def test_windowed_decoder_matches_decode_batch_across_batches():
    dem = _dem()
    decoders = rtd.sinter_decoders(SMALL, windows=((4, 2), (5, 2)), on_failure="defer", max_deferrals=1)
    assert sorted(decoders) == ["rtd_relay", "rtd_window_4_2", "rtd_window_5_2"]
    compiled = decoders["rtd_window_4_2"].compile_decoder_for_dem(dem=dem)
    packed = _packed_shots()
    ours = _decode_in_pieces(compiled, packed, (50,))
    reference = rtd.WindowedDecoder.from_detector_error_model(
        dem, width=4, commit=2, config=SMALL, round_of=1, on_failure="defer", max_deferrals=1
    ).decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True)
    assert np.array_equal(ours, reference)


def test_kept_records_are_the_records_of_one_decode_batch():
    dem = _dem()
    packed = _packed_shots()
    # A fixed-point format and a selection policy go through sinter like any spec.
    spec = {key: value for key, value in SMALL.items() if key != "policy"}
    spec = {**spec, "arithmetic": "int6.2.8", "version": 3,
            "selection": {"rule": "class_sum", "stop": {"rule": "fixed"}, "capacity": 5,
                          "confidence": None, "history": None}}
    decoders = rtd.sinter_decoders(spec, windows=((4, 2),))
    for name, reference in (
        ("rtd_relay", rtd.RelayDecoder.from_detector_error_model(dem, spec)),
        ("rtd_window_4_2", rtd.WindowedDecoder.from_detector_error_model(dem, width=4, commit=2, config=spec, round_of=1)),
    ):
        compiled = decoders[name].compile_decoder_for_dem(dem=dem)
        compiled.keep_records = True
        _decode_in_pieces(compiled, packed, (41, 42))
        assert [first for first, _ in compiled.batches] == [0, 41, 42]
        windowed = name == "rtd_window_4_2"
        options = {"commits": True} if windowed else {}
        _, expected = reference.decode_batch(packed, bit_packed_shots=True, records=True, **options)
        for key in ("iterations", "legs", "success", "conf_decided_class"):
            joined = np.concatenate([records[key] for _, records in compiled.batches])
            assert np.array_equal(joined, expected[key]), (name, key)
        joined_weight = np.concatenate([records["weight"] for _, records in compiled.batches])
        assert np.array_equal(joined_weight.view(np.uint64), expected["weight"].view(np.uint64))
        if windowed:
            faults = np.concatenate([records["commit_faults"] for _, records in compiled.batches])
            assert np.array_equal(faults, expected["commit_faults"])


def test_a_decoder_pinned_to_another_module_build_refuses_to_compile(caplog):
    dem = _dem()
    sha = native_module_sha256()
    assert len(sha) == 64
    pinned = rtd.sinter_decoders(SMALL, windows=(), require_native_sha256=sha)["rtd_relay"]
    pinned.compile_decoder_for_dem(dem=dem)
    wrong = pickle.loads(pickle.dumps(rtd.sinter_decoders(SMALL, windows=(), require_native_sha256="0" * 64)["rtd_relay"]))
    with pytest.raises(RuntimeError, match="pinned"):
        wrong.compile_decoder_for_dem(dem=dem)
    assert any(r.levelname == "ERROR" and "pinned" in r.getMessage() for r in caplog.records)


def test_decoders_pickle_and_resolve_relative_gamma_paths(tmp_path):
    table = np.random.default_rng(3).uniform(-0.24, 0.66, size=(4, 1)).astype(np.float64)
    (tmp_path / "tables").mkdir()
    np.save(tmp_path / "tables" / "gammas.npy", table)
    spec = {**SMALL, "gamma_source": {"type": "explicit", "path": "tables/gammas.npy"}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    decoder = RtdSinterDecoder(tmp_path / "spec.json")
    restored = pickle.loads(pickle.dumps(decoder))
    assert restored.spec == spec
    assert restored._config()["gamma_source"]["path"] == str(tmp_path / "tables" / "gammas.npy")
    # The explicit table has one column where the problem needs n; the spec is read, the table refused.
    with pytest.raises(ValueError, match="gamma"):
        restored.compile_decoder_for_dem(dem=_dem())


def test_a_bad_detector_error_model_is_refused():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D1 D2")
    with pytest.raises(ValueError, match="round"):
        rtd.sinter_decoders(SMALL)["rtd_window_12_8"].compile_decoder_for_dem(dem=dem)


def test_sinter_collect_runs_both_decoders():
    # bb18 at p = 1e-3 over 3 rounds, where Relay-BP fails on the order of 1% of shots.
    _, _, memory = build_circuit("bb18", "choi", 0.001, 3, False)
    circuit = memory.circuit
    dem = circuit.detector_error_model(decompose_errors=False)
    tasks = [
        sinter.Task(circuit=circuit, detector_error_model=dem, decoder=name, json_metadata={"decoder": name})
        for name in ("rtd_relay", "rtd_window_4_2")
    ]
    stats = sinter.collect(
        num_workers=1,
        tasks=tasks,
        custom_decoders=rtd.sinter_decoders(SMALL, windows=((4, 2),)),
        max_shots=300,
        max_errors=300,
    )
    assert sorted(s.decoder for s in stats) == ["rtd_relay", "rtd_window_4_2"]
    for s in stats:
        assert s.shots == 300
        assert s.errors < 0.1 * s.shots
