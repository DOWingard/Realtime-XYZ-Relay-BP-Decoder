import json
from pathlib import Path

import numpy as np
import pytest

from rtd import records
from rtd.records import RecordsError, arm_output_dir

K = 3  # windows per shot in the fake per-window arrays


def _write_half(directory: Path, rng: np.random.Generator, shots: int, extra: dict | None = None) -> dict:
    directory.mkdir(parents=True)
    commits_per_window = rng.integers(0, 4, size=shots * K)
    arrays = {
        "logical_failure": rng.integers(0, 2, shots, dtype=np.uint8),
        "success": rng.integers(0, 2, shots, dtype=np.uint8),
        "iterations": rng.integers(1, 500, shots, dtype=np.uint32),
        "legs": rng.integers(1, 7, shots, dtype=np.uint32),
        "decode_ns": rng.integers(1000, 10**6, shots, dtype=np.uint64),
        "weight": rng.uniform(1.0, 50.0, shots),
        "win_iterations": rng.integers(1, 100, (shots, K), dtype=np.uint32),
        "commit_ptr": np.concatenate([[0], np.cumsum(commits_per_window)]).astype(np.uint64),
        "commit_faults": rng.integers(0, 1000, int(commits_per_window.sum()), dtype=np.uint32),
        **(extra or {}),
    }
    for name, arr in arrays.items():
        np.save(directory / f"{name}.npy", arr)
    (directory / "run.json").write_text(json.dumps({"summary": {"shots": shots}}))
    return arrays


def _fake_campaign(tmp_path: Path, mode: str, shots_per_chunk: list[int], complete: list[bool], arm: str = "a", name: str = "camp"):
    campaign = tmp_path / name
    campaign.mkdir()
    (campaign / "campaign.json").write_text(json.dumps({"mode": mode}))
    rng = np.random.default_rng(0)
    written = []
    for chunk, (shots, done) in enumerate(zip(shots_per_chunk, complete, strict=True)):
        per_half = {}
        for half in records.MODES[mode]:
            per_half[half] = _write_half(arm_output_dir(campaign, chunk, arm, half), rng, shots)
        (campaign / "chunks" / str(chunk) / "chunk.json").write_text(json.dumps({"chunk": chunk, "shots": shots}))
        if done:
            (campaign / "chunks" / str(chunk) / arm / "done.json").write_text(json.dumps({"chunk": chunk}))
        written.append(per_half)
    return campaign, written


def test_load_concatenates_completed_chunks_in_order(tmp_path):
    campaign, written = _fake_campaign(tmp_path, "xz", [5, 3, 4, 2], [True, True, False, True])
    rec = records.load(campaign, "a")
    assert rec.chunks == [0, 1, 3]
    assert rec.shots == 10
    assert rec.chunk.tolist() == [0] * 5 + [1] * 3 + [3] * 2
    assert rec.shot_in_chunk.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 0, 1]
    for half in ("x", "z"):
        got = rec.half(half)
        expect = [written[c][half] for c in (0, 1, 3)]
        assert np.array_equal(got["iterations"], np.concatenate([e["iterations"] for e in expect]))
        assert got["win_iterations"].shape == (10, K)
        # Ragged commits: the concatenated pointer still indexes the concatenated faults.
        ptr, faults = got["commit_ptr"], got["commit_faults"]
        assert ptr.size == 10 * K + 1 and int(ptr[-1]) == faults.size
        row = 0
        for e in expect:
            for i in range(e["commit_ptr"].size - 1):
                mine = faults[ptr[row] : ptr[row + 1]]
                theirs = e["commit_faults"][e["commit_ptr"][i] : e["commit_ptr"][i + 1]]
                assert np.array_equal(mine, theirs)
                row += 1


def test_xz_outcome_combination(tmp_path):
    campaign, _ = _fake_campaign(tmp_path, "xz", [6, 6], [True, True])
    rec = records.load(campaign, "a")
    x, z = rec.half("x"), rec.half("z")
    assert np.array_equal(rec["logical_failure"], (x["logical_failure"] | z["logical_failure"]).astype(np.uint8))
    assert np.array_equal(rec["success"], (x["success"] & z["success"]).astype(np.uint8))
    assert np.array_equal(rec["iterations_sum"], x["iterations"].astype(np.uint64) + z["iterations"])
    assert np.array_equal(rec["iterations_max"], np.maximum(x["iterations"], z["iterations"]))
    assert np.allclose(rec["weight_sum"], x["weight"] + z["weight"])
    assert "iterations" not in rec.combined


def test_combine_halves_truth_table():
    per_half = {
        "x": {"logical_failure": np.array([0, 1, 0, 1], np.uint8), "success": np.array([1, 1, 0, 0], np.uint8)},
        "z": {"logical_failure": np.array([0, 0, 1, 1], np.uint8), "success": np.array([1, 0, 1, 0], np.uint8)},
    }
    combined = records.combine_halves(per_half)
    assert combined["logical_failure"].tolist() == [0, 1, 1, 1]
    assert combined["success"].tolist() == [1, 0, 0, 0]


@pytest.mark.parametrize("mode", ["xyz", "z_only"])
def test_single_half_modes_keep_the_arrays(tmp_path, mode):
    campaign, written = _fake_campaign(tmp_path, mode, [4, 3], [True, True])
    (half,) = records.MODES[mode]
    rec = records.load(campaign, "a")
    assert rec.halves == (half,)
    expect = {name: np.concatenate([w[half][name] for w in written]) for name in written[0][half] if name not in records.PTR_PAIRS}
    # Every array of the only half is in `combined`, plus the names split modes use.
    assert set(rec.half(half)) <= set(rec.combined)
    for name in ("iterations", "logical_failure", "weight", "win_iterations", "commit_faults"):
        assert np.array_equal(rec[name], expect[name])
    assert np.array_equal(rec["commit_ptr"], rec.half(half)["commit_ptr"])
    assert np.array_equal(rec["iterations_sum"], expect["iterations"])
    assert np.array_equal(rec["iterations_max"], expect["iterations"])
    assert np.array_equal(rec["weight_sum"], expect["weight"])


def test_combined_names_common_to_every_mode(tmp_path):
    # Arrays as rtd_decode writes them for a whole-shot decode.
    extra = {"best_leg": np.zeros(0, np.int32), "flagged": np.zeros(0, np.uint8), "predicted_observables": np.zeros((0, 2), np.uint8)}
    common = {"logical_failure", "success", "flagged", "iterations_sum", "iterations_max", "legs_sum", "legs_max", "decode_ns_sum", "decode_ns_max", "weight_sum"}
    loaded = {}
    for mode in records.MODES:
        campaign, _ = _fake_campaign(tmp_path, mode, [4, 3], [True, True], name=f"camp_{mode}")
        for chunk, shots in ((0, 4), (1, 3)):
            for half in records.MODES[mode]:
                directory = arm_output_dir(campaign, chunk, "a", half)
                for name, arr in extra.items():
                    np.save(directory / f"{name}.npy", np.zeros((shots, *arr.shape[1:]), arr.dtype))
        loaded[mode] = records.load(campaign, "a")
    for mode, rec in loaded.items():
        assert common <= set(rec.combined), mode
        for name in common:
            assert rec[name].shape[0] == 7
        (first,) = rec.halves if len(rec.halves) == 1 else (None,)
        if first is not None:
            # One half: its own arrays too, the very same objects, and the sums equal the half's values.
            assert set(rec.combined) == set(rec.half(first)) | common
            assert rec["iterations"] is rec.half(first)["iterations"]
            assert np.array_equal(rec["legs_max"], rec.half(first)["legs"]) and np.array_equal(rec["weight_sum"], rec.half(first)["weight"])
        else:
            assert set(rec.combined) == common


def test_gamma_seeds(tmp_path):
    campaign, _ = _fake_campaign(tmp_path, "xz", [3, 2, 4], [True, True, True])
    seeds = {0: 11, 1: 2**64 - 1, 2: None}
    for chunk, seed in seeds.items():
        done = {"chunk": chunk} if seed is None else {"chunk": chunk, "gamma": {"per_chunk": True, "seed": seed}}
        (campaign / "chunks" / str(chunk) / "a" / "done.json").write_text(json.dumps(done))
    rec = records.load(campaign, "a", chunks=[0, 1])
    assert rec.gamma_seeds == {0: 11, 1: 2**64 - 1}
    assert rec.gamma_seed.dtype == np.uint64 and rec.gamma_seed.tolist() == [11] * 3 + [2**64 - 1] * 2
    with pytest.raises(RecordsError, match=r"chunks \[2\] have no recorded gamma seed"):
        _ = records.load(campaign, "a").gamma_seed


def test_unreadable_markers_count_as_absent(tmp_path, caplog):
    campaign, _ = _fake_campaign(tmp_path, "xz", [3, 3, 3], [True, True, True])
    (campaign / "chunks" / "1" / "a" / "done.json").write_text("")  # truncated by a full disk
    (campaign / "chunks" / "2" / "chunk.json").write_text('{"chunk": 2, "sh')
    with caplog.at_level("WARNING", logger="rtd.records"):
        assert records.completed_chunks(campaign, "a") == [0]
        rec = records.load(campaign, "a")
    assert rec.chunks == [0] and rec.shots == 3
    assert any("not valid JSON" in r.getMessage() for r in caplog.records)
    with pytest.raises(RecordsError, match="missing or unreadable"):
        records.load_chunk(campaign, "a", 1)
    # Readers leave the files alone; with quarantine the marker is renamed and reported once.
    marker = campaign / "chunks" / "1" / "a" / "done.json"
    assert marker.is_file()
    assert records.read_json_marker(marker, quarantine=True) is None
    assert not marker.exists() and len(list(marker.parent.glob("done.json.corrupt-*"))) == 1
    assert records.read_json_marker(marker, quarantine=True) is None
    (campaign / "x.json").write_text("[1, 2]")
    assert records.read_json_marker(campaign / "x.json") is None


def test_names_and_chunk_selection(tmp_path):
    campaign, _ = _fake_campaign(tmp_path, "z_only", [3, 3, 3], [True, True, True])
    rec = records.load(campaign, "a", names=("logical_failure",), chunks=[2, 0])
    assert rec.chunks == [0, 2]
    assert set(rec.half("z")) == {"logical_failure"}
    with pytest.raises(RecordsError, match="has not completed chunks"):
        records.load(campaign, "a", chunks=[5])
    with pytest.raises(RecordsError, match="has no arrays"):
        records.load(campaign, "a", names=("not_an_array",))
    with pytest.raises(RecordsError, match="not decoded in mode"):
        records.load(campaign, "a", halves=("x",))
    empty = records.load(campaign, "unknown_arm")
    assert empty.shots == 0 and empty.combined == {}


def test_inconsistent_chunks_are_rejected(tmp_path):
    campaign, _ = _fake_campaign(tmp_path, "xyz", [3, 3], [True, True])
    np.save(arm_output_dir(campaign, 1, "a", "xyz") / "extra.npy", np.zeros(3))
    with pytest.raises(RecordsError, match="decoded with different options"):
        records.load(campaign, "a")


def test_missing_campaign(tmp_path):
    with pytest.raises(RecordsError, match="not a campaign directory"):
        records.load(tmp_path, "a")


# ---------------------------------------------------------------------------------------------
# Sliding-window arms: per-window [S, K] and per-solution [S, K, N] arrays


def _write_windowed_half(directory: Path, rng: np.random.Generator, shots: int, windows: int, slots: int | None) -> dict:
    """The arrays rtd_decode writes for a sliding-window decode (with recorded solutions when
    `slots` is given), with random contents."""
    directory.mkdir(parents=True)
    attempts = rng.integers(0, 3, (shots, windows), dtype=np.uint8)
    arrays = {
        "logical_failure": rng.integers(0, 2, shots, dtype=np.uint8),
        "success": rng.integers(0, 2, shots, dtype=np.uint8),
        "iterations": rng.integers(1, 500, shots, dtype=np.uint32),
        "flagged": rng.integers(0, 2, shots, dtype=np.uint8),
        "win_iterations": rng.integers(0, 100, (shots, windows), dtype=np.uint32),
        "win_attempts": attempts,
        "win_converged": rng.integers(0, 2, (shots, windows), dtype=np.uint8),
        "win_weight": rng.uniform(1.0, 9.0, (shots, windows)),
        "win_decode_ns": rng.integers(1, 10**6, (shots, windows), dtype=np.uint64),
    }
    if slots is not None:
        arrays["sol_count"] = rng.integers(0, slots + 2, (shots, windows), dtype=np.uint32)
        arrays["returned_class"] = rng.integers(0, 2**63, (shots, windows), dtype=np.uint64)
        arrays["sol_leg"] = rng.integers(0, 50, (shots, windows, slots), dtype=np.uint32)
        arrays["sol_weight"] = rng.uniform(1.0, 9.0, (shots, windows, slots))
        arrays["sol_class"] = rng.integers(0, 2**63, (shots, windows, slots), dtype=np.uint64)
    for name, arr in arrays.items():
        np.save(directory / f"{name}.npy", arr)
    (directory / "run.json").write_text(json.dumps({"summary": {"shots": shots}}))
    return arrays


def _windowed_campaign(tmp_path: Path, mode: str, shots_per_chunk: list[int], windows: list[int], slots: int | None, arm: str = "w"):
    campaign = tmp_path / f"camp_{mode}_{arm}"
    campaign.mkdir(exist_ok=True)
    (campaign / "campaign.json").write_text(json.dumps({"mode": mode}))
    rng = np.random.default_rng(3)
    written = []
    for chunk, (shots, k) in enumerate(zip(shots_per_chunk, windows, strict=True)):
        written.append(
            {half: _write_windowed_half(arm_output_dir(campaign, chunk, arm, half), rng, shots, k, slots) for half in records.MODES[mode]}
        )
        (campaign / "chunks" / str(chunk) / "chunk.json").write_text(json.dumps({"chunk": chunk, "shots": shots}))
        (campaign / "chunks" / str(chunk) / arm / "done.json").write_text(json.dumps({"chunk": chunk}))
    return campaign, written


def test_window_and_solution_arrays_keep_their_axes(tmp_path):
    campaign, written = _windowed_campaign(tmp_path, "xz", [5, 3], [4, 4], slots=2)
    rec = records.load(campaign, "w")
    assert rec.shots == 8
    for half in ("x", "z"):
        arrays = rec.half(half)
        assert arrays["win_attempts"].shape == (8, 4) and arrays["win_attempts"].dtype == np.uint8
        assert arrays["win_weight"].shape == (8, 4) and arrays["win_weight"].dtype == np.float64
        assert arrays["sol_leg"].shape == (8, 4, 2) and arrays["sol_class"].dtype == np.uint64
        assert arrays["returned_class"].shape == (8, 4) and arrays["sol_count"].shape == (8, 4)
        for name in ("win_iterations", "sol_weight", "sol_class", "flagged"):
            assert np.array_equal(arrays[name], np.concatenate([w[half][name] for w in written]))
        assert rec.windows(half) == 4 and rec.solution_slots(half) == 2
    # Flags combine over the halves like failures; per-window arrays stay per half.
    assert np.array_equal(rec["flagged"], (rec.half("x")["flagged"] | rec.half("z")["flagged"]).astype(np.uint8))
    assert "win_iterations" not in rec.combined
    with pytest.raises(RecordsError, match="name one"):
        rec.windows()

    single, _ = _windowed_campaign(tmp_path, "z_only", [4], [3], slots=None)
    rec = records.load(single, "w")
    assert rec.windows() == 3 and rec.solution_slots() is None
    assert rec["win_iterations"] is rec.half("z")["win_iterations"]


def test_optional_arrays_are_loaded_where_the_arm_has_them(tmp_path):
    campaign, _ = _windowed_campaign(tmp_path, "xyz", [4, 2], [3, 3], slots=None)
    # A whole-shot arm of the same campaign: no per-window arrays.
    rng = np.random.default_rng(5)
    for chunk, shots in ((0, 4), (1, 2)):
        _write_half(arm_output_dir(campaign, chunk, "whole", "xyz"), rng, shots)
        for name in ("win_iterations", "commit_ptr", "commit_faults"):
            (arm_output_dir(campaign, chunk, "whole", "xyz") / f"{name}.npy").unlink()
        (campaign / "chunks" / str(chunk) / "whole" / "done.json").write_text(json.dumps({"chunk": chunk}))
    optional = ("flagged", *records.WINDOW_ARRAYS, "sol_count")
    windowed = records.load(campaign, "w", names=("logical_failure",), optional=optional)
    assert set(windowed.half("xyz")) == {"logical_failure", "flagged", "win_iterations", "win_attempts", "win_converged", "win_weight", "win_decode_ns"}
    assert windowed.windows() == 3
    whole = records.load(campaign, "whole", names=("logical_failure", "iterations"), optional=optional)
    assert set(whole.half("xyz")) == {"logical_failure", "iterations"}
    assert whole.windows() is None and whole.solution_slots() is None
    # Required names are still required.
    with pytest.raises(RecordsError, match="has no arrays"):
        records.load(campaign, "whole", names=("win_iterations",), optional=optional)


def test_chunks_decoded_with_another_window_count_are_rejected(tmp_path):
    campaign, _ = _windowed_campaign(tmp_path, "xyz", [3, 3], [3, 4], slots=None)
    with pytest.raises(RecordsError, match="different window plans or solution slots"):
        records.load(campaign, "w")
    campaign, _ = _windowed_campaign(tmp_path, "z_only", [3, 3], [2, 2], slots=2, arm="n")
    np.save(arm_output_dir(campaign, 1, "n", "z") / "sol_leg.npy", np.zeros((3, 2, 5), np.uint32))
    with pytest.raises(RecordsError, match="sol_leg has per-shot shapes"):
        records.load(campaign, "n")
