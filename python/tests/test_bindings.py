"""The Python decoders (rtd.RelayDecoder, rtd.WindowedDecoder, streams) over the extension module.

Bit-exact against the committed goldens: the bb18 whole-shot relay golden (relay_bp's own
outputs with an explicit gamma table), every window golden of the bb18 fixtures (the windowing
reference with relay_bp inside, explicit gamma tables per window shape) and every fixed-point
golden (the same reference with the integer emulator inside). Then the properties the API
promises: the same predictions for any number of workers, from bit-packed input, from a detector
error model and from the artifact it was exported to; a stream fed round by round commits what a
batch commits; a selection policy's confidence and history arrive in the batch records and in
the stream alike; returned arrays own their memory; and bad input is refused with ValueError,
calls out of order with RuntimeError, each logged.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import stim

pytest.importorskip("rtd._native", reason="the extension module is not built (RTD_ENABLE_PYTHON)")

import rtd  # noqa: E402
from rtd.decoders import RELAY_BP5, load_config, sliding_window  # noqa: E402
from rtd.dem import problem_from_artifact, problem_from_dem  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "test" / "fixtures"
R9 = FIXTURES / "bb18_choi_r9"

# A small Relay-BP configuration (bb18 needs few legs).
SMALL = {
    **RELAY_BP5,
    "pre_iter": 20,
    "set_max_iter": 15,
    "num_sets": 40,
    "stopping": {"rule": "after_n_converged", "count": 3},
    "gamma_source": {"type": "uniform", "seed": 7, "low": -0.24, "high": 0.66},
}


def _shots():
    return np.load(R9 / "shots" / "detectors.npy"), np.load(R9 / "shots" / "observables.npy")


def _goldens(prefix):
    return sorted(
        d for fixture in ("bb18_choi", "bb18_choi_r9") for d in (FIXTURES / fixture).iterdir()
        if d.name.startswith(prefix) and (d / "spec.json").is_file()
    )


def _window_goldens():
    return _goldens("window_")


# Windows (4, 2) deferred while the gap of the committed class is below 2, with every history
# signal over the last 1, 2 and 3 windows (3 relay solutions + 3 extra legs <= capacity 10).
DEFERRING_SELECTION = {
    "rule": "largest_agreement",
    "stop": {"rule": "agree", "count": 2},
    "capacity": 10,
    "confidence": {"signal": "gap", "threshold": 2.0, "single_class": "high", "extra_legs": 3, "on_low": "defer"},
    "history": {
        "lengths": [1, 2, 3],
        "signals": ["gap", "agreement", "weight", "first_legs", "first_iterations", "q_supp", "density",
                    "commit_weight", "commit_q_supp"],
    },
}


def test_reproduces_the_whole_shot_relay_golden():
    golden = FIXTURES / "bb18_choi" / "relay_f32"
    config = json.loads((golden / "manifest.json").read_text())["config"]
    spec = {
        **SMALL,
        "gamma0": config["gamma0"],
        "pre_iter": config["pre_iter"],
        "set_max_iter": config["set_max_iter"],
        "num_sets": config["num_sets"],
        "stopping": {"rule": "after_n_converged", "count": config["stop_nconv"]},
        "gamma_source": {"type": "explicit", "path": str(golden / "gammas.npy")},
    }
    decoder = rtd.RelayDecoder.from_artifact(FIXTURES / "bb18_choi" / "artifact", spec, workers=2)
    detectors = np.load(golden / "detectors.npy")
    predictions, records = decoder.decode_batch(detectors, records=True, decodings=True)
    assert np.array_equal(records["decodings"], np.load(golden / "decoding.npy"))
    assert np.array_equal(records["success"], np.load(golden / "success.npy"))
    assert np.array_equal(records["iterations"], np.load(golden / "iterations.npy"))
    assert np.array_equal(records["weight"].view(np.uint64), np.load(golden / "weight.npy").view(np.uint64))
    a = decoder.problem.a_matrix()
    assert np.array_equal(predictions, (records["decodings"].astype(np.int64) @ a.T.toarray().astype(np.int64)) % 2)


@pytest.mark.parametrize("golden", _window_goldens(), ids=lambda d: f"{d.parent.name}-{d.name}")
def test_reproduces_every_window_golden(golden):
    manifest = json.loads((golden / "manifest.json").read_text())
    spec, base = load_config(golden / "spec.json")
    window = spec["window"]
    problem = problem_from_artifact(REPO / manifest["artifact"]["path"])
    # The spec's explicit_shapes directory is relative to the golden directory.
    spec["gamma_source"]["directory"] = str(base / spec["gamma_source"]["directory"])
    decoder = rtd.WindowedDecoder(
        problem,
        spec,
        width=window["width"],
        commit=window["commit"],
        converge_rounds=window["converge_rounds"],
        boundary=window["boundary"],
        on_failure=window["on_failure"],
        max_deferrals=window["max_deferrals"],
        iteration_cap=window["iteration_cap"],
        workers=3,
    )
    assert decoder.num_positions == manifest["num_positions"]
    predictions, records = decoder.decode_batch(np.load(golden / "detectors.npy"), records=True, commits=True)
    assert np.array_equal(predictions, np.load(golden / "predicted_observables.npy"))
    for name in (
        "flagged",
        "win_iterations",
        "win_legs",
        "win_attempts",
        "win_converged",
        "win_unexplained",
        "win_flagged",
        "win_virtual",
        "commit_ptr",
        "commit_faults",
    ):
        expected = np.load(golden / f"{name}.npy")
        assert records[name].dtype == expected.dtype, name
        assert np.array_equal(records[name], expected), name
    for name in ("win_weight", "win_committed_weight"):
        assert np.array_equal(records[name].view(np.uint64), np.load(golden / f"{name}.npy").view(np.uint64)), name
    failures = np.any(predictions != np.load(golden / "observables.npy"), axis=1)
    assert np.array_equal(failures.astype(np.uint8), np.load(golden / "logical_failure.npy"))


@pytest.mark.parametrize("golden", _goldens("fixed_"), ids=lambda d: d.name)
def test_reproduces_every_fixed_point_golden(golden):
    manifest = json.loads((golden / "manifest.json").read_text())
    spec, base = load_config(golden / "spec.json")
    assert spec["arithmetic"].startswith("int")
    window = spec["window"]
    spec["gamma_source"]["directory"] = str(base / spec["gamma_source"]["directory"])
    decoder = rtd.WindowedDecoder(
        problem_from_artifact(REPO / manifest["artifact"]["path"]),
        spec,
        width=window["width"],
        commit=window["commit"],
        converge_rounds=window["converge_rounds"],
        boundary=window["boundary"],
        on_failure=window["on_failure"],
        max_deferrals=window["max_deferrals"],
        iteration_cap=window["iteration_cap"],
        workers=3,
    )
    predictions, records = decoder.decode_batch(np.load(golden / "detectors.npy"), records=True, commits=True)
    assert np.array_equal(predictions, np.load(golden / "predicted_observables.npy"))
    for name in ("success", "iterations", "legs", "flagged", "win_iterations", "win_legs", "win_attempts",
                 "win_converged", "win_cap_hit", "win_unexplained", "win_flagged", "win_virtual", "commit_ptr",
                 "commit_faults"):
        assert np.array_equal(records[name], np.load(golden / f"{name}.npy")), name
    for name in ("weight", "win_weight", "win_committed_weight"):
        assert np.array_equal(records[name].view(np.uint64), np.load(golden / f"{name}.npy").view(np.uint64)), name


def test_a_selection_policy_reports_its_confidence_in_batches_and_streams():
    detectors, _ = _shots()
    whole = {**SMALL, "version": 3, "selection": {**DEFERRING_SELECTION, "history": None,
                                                  "confidence": {**DEFERRING_SELECTION["confidence"], "on_low": "none"}}}
    decoder = rtd.RelayDecoder.from_artifact(R9 / "artifact", whole, workers=2, record_solutions=4)
    predictions, records = decoder.decode_batch(detectors[:30], records=True)
    assert records["conf_found"].shape == (30, 1) and records["conf_gap"].dtype == np.float64
    assert "conf_low_deferrals" not in records and "hist_state" not in records
    # The decided class is the class of the returned solution, whose flips are the prediction.
    classes = np.packbits(predictions, axis=1, bitorder="little").view(np.uint8)[:, 0]
    assert np.array_equal(records["conf_decided_class"][:, 0], classes.astype(np.uint64))
    assert np.array_equal(records["returned_class"][:, 0], records["conf_decided_class"][:, 0])

    # A fixed-point format under the same policy, with windows and a history.
    windowed_spec = {**SMALL, "version": 3, "selection": DEFERRING_SELECTION}
    windowed_spec.pop("policy")
    windowed_spec["arithmetic"] = "int5.2.8"
    windowed = rtd.WindowedDecoder.from_artifact(
        R9 / "artifact", windowed_spec, width=4, commit=2, on_failure="defer", max_deferrals=2, workers=2
    )
    predictions, records = windowed.decode_batch(detectors[:20], records=True)
    k_windows = windowed.num_positions
    names = DEFERRING_SELECTION["history"]["signals"]
    for name in names:
        hist = records[f"hist_{name}"]
        assert hist.shape == (20, k_windows, 3) and hist.dtype == np.float64
        # A view of the one history buffer: not a copy, and kept alive by the arrays.
        assert not hist.flags.owndata and hist.base is not None
    assert records["hist_state"].shape == (20, k_windows, 3)
    assert records["conf_low_deferrals"].sum() > 0 and records["conf_low"].sum() > 0

    stream = windowed.stream()
    m = windowed.detectors_per_round
    assert stream.history() is not None
    for s in range(20):
        stream.reset(s)
        for r in range(windowed.rounds_total):
            (stream.push_round if r + 1 < windowed.rounds_total else stream.push_final)(detectors[s, r * m:(r + 1) * m])
            while stream.window_ready():
                commit = stream.decode_next()
                if commit.deferred:
                    continue
                k = commit.window
                record = commit.record
                assert record["low_confidence_deferrals"] == records["conf_low_deferrals"][s, k]
                if record["confidence"] is not None:
                    assert record["confidence"]["low"] == bool(records["conf_low"][s, k])
                history = stream.history()
                assert history["lengths"].tolist() == [1, 2, 3]
                for name in names:
                    assert np.array_equal(history[name].view(np.uint64), records[f"hist_{name}"][s, k].view(np.uint64))
                assert np.array_equal(history["state"], records["hist_state"][s, k])
        assert np.array_equal(stream.predicted(), predictions[s])
    assert windowed.totals["low_confidence"] == int(records["conf_low"].sum())
    # Without a history the stream has none.
    plain = rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=4, commit=2)
    assert plain.stream().history() is None


def test_workers_and_input_forms_do_not_change_predictions():
    detectors, _ = _shots()
    decoder = rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL, workers=3, record_solutions=2)
    one = decoder.decode_batch(detectors, workers=1)
    three = decoder.decode_batch(detectors)
    assert np.array_equal(one, three)
    assert one.dtype == np.uint8 and one.shape == (detectors.shape[0], 8)
    # Bool input, a strided view and bit-packed input and output.
    assert np.array_equal(decoder.decode_batch(detectors.astype(bool)), one)
    assert np.array_equal(decoder.decode_batch(np.asfortranarray(detectors)), one)
    packed = np.packbits(detectors, axis=1, bitorder="little")
    packed_out = decoder.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True)
    assert packed_out.shape == (detectors.shape[0], 1)
    assert np.array_equal(np.unpackbits(packed_out, axis=1, count=8, bitorder="little"), one)
    # One shot at a time, with its stream, is the batch.
    for s in (0, 17, 99):
        assert np.array_equal(decoder.decode(detectors[s], stream=s), one[s])
    prediction, record = decoder.decode(detectors[3], stream=3, return_record=True)
    _, records = decoder.decode_batch(detectors[3:4], stream_offset=3, records=True, decodings=True)
    assert np.array_equal(record["support"], np.flatnonzero(records["decodings"][0]))
    assert record["iterations"] == records["iterations"][0]
    assert records["sol_count"].shape == (1, 1) and records["sol_leg"].shape == (1, 1, 2)
    assert decoder.totals["shots"] >= 5 * detectors.shape[0]


def test_stream_offset_selects_the_gamma_streams():
    detectors, _ = _shots()
    decoder = rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL)
    _, all_records = decoder.decode_batch(detectors, records=True)
    _, tail = decoder.decode_batch(detectors[40:], stream_offset=40, records=True)
    assert np.array_equal(tail["iterations"], all_records["iterations"][40:])
    _, shifted = decoder.decode_batch(detectors[40:], stream_offset=0, records=True)
    # Other gamma draws take other paths through the relay legs on some shot.
    assert not np.array_equal(shifted["legs"], all_records["legs"][40:])


def test_the_detector_error_model_and_its_artifact_decode_alike():
    circuit = stim.Circuit.from_file(R9 / "shots" / "circuit.stim")
    dem = circuit.detector_error_model(decompose_errors=False)
    detectors, _ = _shots()
    from_dem = rtd.RelayDecoder.from_detector_error_model(dem, SMALL)
    from_artifact = rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL)
    assert np.array_equal(from_dem.decode_batch(detectors), from_artifact.decode_batch(detectors))
    windowed_dem = rtd.WindowedDecoder.from_detector_error_model(dem, width=4, commit=2, config=SMALL, round_of=1)
    windowed_artifact = rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=4, commit=2)
    predictions = windowed_dem.decode_batch(detectors)
    assert np.array_equal(predictions, windowed_artifact.decode_batch(detectors))


def test_a_stream_commits_what_a_batch_commits():
    detectors, _ = _shots()
    decoder = rtd.WindowedDecoder.from_artifact(
        R9 / "artifact", SMALL, width=4, commit=2, on_failure="defer", max_deferrals=2, record_solutions=3
    )
    predictions, records = decoder.decode_batch(detectors[:25], records=True, commits=True)
    k_windows = decoder.num_positions
    m = decoder.detectors_per_round
    stream = decoder.stream()
    for s in range(25):
        stream.reset(s)
        decoded = 0
        for r in range(decoder.rounds_total):
            bits = detectors[s, r * m:(r + 1) * m]
            (stream.push_round if r + 1 < decoder.rounds_total else stream.push_final)(bits)
            while stream.window_ready():
                commit = stream.decode_next()
                if commit.deferred:
                    continue
                cell = s * k_windows + commit.window
                lo, hi = records["commit_ptr"][cell], records["commit_ptr"][cell + 1]
                assert np.array_equal(commit.faults, records["commit_faults"][lo:hi])
                assert commit.record["iterations"] == records["win_iterations"][s, commit.window]
                assert commit.solutions["found"] == records["sol_count"][s, commit.window]
                assert commit.solutions["leg"].size == min(3, commit.solutions["found"])
                decoded += 1
        assert decoded == k_windows and stream.finished() and stream.closed()
        assert np.array_equal(stream.predicted(), predictions[s])
        assert stream.summary()["iterations"] == records["iterations"][s]
        assert len(stream.records()) == k_windows


def test_stream_calls_out_of_order_are_refused(caplog):
    decoder = rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=4, commit=2)
    stream = decoder.stream()
    m = decoder.detectors_per_round
    stream.reset(0)
    with caplog.at_level(logging.ERROR, logger="rtd.decoders"):
        with pytest.raises(RuntimeError, match="not_ready"):
            stream.decode_next()
        with pytest.raises(ValueError, match="wrong_round_size"):
            stream.push_round(np.zeros(m - 1, dtype=np.uint8))
        for _ in range(decoder.rounds_total - 1):
            stream.push_round(np.zeros(m, dtype=bool))
        stream.push_final(np.zeros(m, dtype=np.uint8))
        with pytest.raises(RuntimeError, match="stream_closed"):
            stream.push_round(np.zeros(m, dtype=np.uint8))
    assert sum(r.levelname == "ERROR" for r in caplog.records) == 3
    while not stream.finished():
        assert stream.decode_next().faults.size == 0
    assert not stream.flagged() and stream.frame().tolist() == [0] * 8
    with pytest.raises(ValueError):
        stream.reset(-1)


def test_returned_arrays_own_their_memory():
    detectors, _ = _shots()
    decoder = rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=5, commit=2, boundary="uniform")
    predictions, records = decoder.decode_batch(detectors[:10], records=True, commits=True)
    for array in (predictions, *records.values()):
        assert isinstance(array, np.ndarray)
        # Wrapped, not copied: numpy does not own the buffer, the capsule behind it does.
        assert not array.flags.owndata and array.base is not None
        assert array.flags.c_contiguous
    del decoder
    assert predictions.sum() >= 0 and records["win_iterations"].sum() > 0


def test_bad_input_is_refused_and_logged(caplog):
    detectors, _ = _shots()
    decoder = rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL, workers=2)
    with pytest.raises(ValueError, match="columns per shot"):
        decoder.decode_batch(detectors[:, :-1])
    with pytest.raises(ValueError, match="bytes per shot"):
        decoder.decode_batch(detectors, bit_packed_shots=True)
    with caplog.at_level(logging.ERROR, logger="rtd.decoders"):
        with pytest.raises(ValueError, match="workers"):
            decoder.decode_batch(detectors, workers=3)
        with pytest.raises(ValueError, match="invalid_spec"):
            rtd.RelayDecoder.from_artifact(R9 / "artifact", {**SMALL, "pre_iter": "eighty"})
        with pytest.raises(ValueError, match="plan_rejected"):
            rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=12, commit=2, boundary="uniform")
    messages = [r.getMessage() for r in caplog.records]
    assert "batch decode failed" in messages
    assert "decoder spec rejected" in messages
    assert "decoder construction failed" in messages
    # Version 3 with or without a policy is accepted; a policy the relay schedule cannot serve is
    # refused (3 solutions + 8 extra legs exceed the capacity of 10), as rtd_decode refuses it.
    rtd.RelayDecoder.from_artifact(R9 / "artifact", {**SMALL, "version": 3, "selection": None})
    too_many = {**DEFERRING_SELECTION, "history": None,
                "confidence": {**DEFERRING_SELECTION["confidence"], "on_low": "none", "extra_legs": 8}}
    with pytest.raises(ValueError, match="extra_legs"):
        rtd.RelayDecoder.from_artifact(R9 / "artifact", {**SMALL, "version": 3, "selection": too_many})
    # "policy" names the IEEE formats; a fixed-point format goes under "arithmetic".
    with pytest.raises(ValueError, match="invalid_spec"):
        rtd.RelayDecoder.from_artifact(R9 / "artifact", {**SMALL, "policy": "int4.2.8"})
    fixed = {key: value for key, value in SMALL.items() if key != "policy"}
    rtd.RelayDecoder.from_artifact(R9 / "artifact", {**fixed, "arithmetic": "int4.2.8"})
    with pytest.raises(ValueError, match="alpha"):
        rtd.RelayDecoder.from_artifact(
            R9 / "artifact", {**fixed, "arithmetic": "int4.2.8", "alpha": {"rule": "constant", "value": 0.8}}
        )
    with pytest.raises(ValueError, match="preset"):
        load_config("no_such_preset")
    with pytest.raises(ValueError, match="round_of"):
        rtd.WindowedDecoder(problem_from_artifact(R9 / "artifact", rounds=False), SMALL, width=4, commit=2)
    with pytest.raises(ValueError, match="workers"):
        rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL, workers=0)


def test_record_arrays_need_records_and_bad_spec_files_are_logged(tmp_path, caplog):
    detectors, _ = _shots()
    decoder = rtd.RelayDecoder.from_artifact(R9 / "artifact", SMALL)
    windowed = rtd.WindowedDecoder.from_artifact(R9 / "artifact", SMALL, width=4, commit=2)
    with caplog.at_level(logging.ERROR, logger="rtd.decoders"):
        with pytest.raises(ValueError, match="records=True"):
            decoder.decode_batch(detectors[:2], decodings=True)
        with pytest.raises(ValueError, match="records=True"):
            windowed.decode_batch(detectors[:2], commits=True)
        (tmp_path / "broken.json").write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_config(tmp_path / "broken.json")
        (tmp_path / "list.json").write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            load_config(tmp_path / "list.json")
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("record arrays asked for without records=True") == 2
    assert "decoder spec is not valid JSON; nothing is built" in messages
    assert "decoder spec is not a JSON object; nothing is built" in messages
    # records=False returns the predictions alone, whatever else is asked.
    assert isinstance(windowed.decode_batch(detectors[:2]), np.ndarray)


def test_flagged_shots_and_cap_hits_are_logged_once_per_batch(caplog):
    detectors, _ = _shots()
    windowed = rtd.WindowedDecoder.from_artifact(
        R9 / "artifact", SMALL, width=4, commit=2, on_failure="flag", iteration_cap=25
    )
    with caplog.at_level(logging.WARNING, logger="rtd.decoders"):
        _, records = windowed.decode_batch(detectors[:40], records=True)
    assert records["flagged"].sum() > 0 and records["win_cap_hit"].sum() > 0
    flagged = [r for r in caplog.records if r.getMessage() == "shots were flagged in this batch"]
    capped = [r for r in caplog.records if r.getMessage() == "windows hit the iteration cap in this batch"]
    assert len(flagged) == 1 and flagged[0].levelname == "ERROR"
    assert flagged[0].flagged == int(records["flagged"].sum()) and flagged[0].on_failure == "flag"
    assert len(capped) == 1 and capped[0].levelname == "WARNING"
    assert capped[0].windows_cap_hit == int(records["win_cap_hit"].sum()) and capped[0].iteration_cap == 25
    assert windowed.totals["flagged"] == int(records["flagged"].sum())


def test_presets_and_windows_build_valid_specs():
    circuit = stim.Circuit.from_file(R9 / "shots" / "circuit.stim")
    dem = circuit.detector_error_model(decompose_errors=False)
    for preset in rtd.PRESETS:
        decoder = rtd.RelayDecoder.from_detector_error_model(dem, preset)
        assert decoder.spec["window"] == {"mode": "whole_shot"}
    window = sliding_window(4, 2, on_failure="flag", iteration_cap=500)
    assert window["converge_rounds"] == 4 and window["iteration_cap"] == 500
    problem = problem_from_dem(dem, round_of=1)
    decoder = rtd.WindowedDecoder(problem, "min_sum", width=4, commit=2, on_failure="flag", iteration_cap=500)
    assert decoder.spec["window"] == window
    assert decoder.plan["positions"] == decoder.num_positions
    assert pickle.loads(pickle.dumps(decoder.spec)) == decoder.spec
