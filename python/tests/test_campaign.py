import copy
import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from rtd import campaign as cp
from rtd import records

pytest.importorskip("relay_bp", reason="campaign init exports artifacts; needs `uv sync --group reference`")

REAL_BINARY = cp.DEFAULT_BINARY
CPUS = "10,11"

FAKE_DECODER = r'''#!{python}
# Stand-in for rtd_decode: a shot "fails" when its detector weight mod fake_fail_mod equals
# fake_fail_offset (both read from the spec), which makes arms differ in a controlled way.
import json, os, sys, time
from pathlib import Path
import numpy as np

args = sys.argv[1:]
if "--help" in args:
    print("usage: fake --artifact --shots --config --out --workers --cpus --first --count "
          "--record-solutions --save-commits --warmup")
    sys.exit(0)
opts, i = {{}}, 0
while i < len(args):
    if args[i] in ("--save-commits", "--save-solution-supports", "--overwrite"):
        opts[args[i]] = True
        i += 1
    else:
        opts[args[i]] = args[i + 1]
        i += 2
if os.environ.get("FAKE_RTD_CALLS"):
    with open(os.environ["FAKE_RTD_CALLS"], "a") as calls:
        calls.write(opts["--out"] + "\n")
if os.environ.get("FAKE_RTD_PIDFILE"):
    Path(os.environ["FAKE_RTD_PIDFILE"]).write_text(str(os.getpid()))
if os.environ.get("FAKE_RTD_GRANDCHILD"):
    # A helper in the decoder's process group, as a multi-process decoder would have.
    import subprocess
    helper = subprocess.Popen(["sleep", "120"])
    Path(os.environ["FAKE_RTD_GRANDCHILD"]).write_text(str(helper.pid))
if os.environ.get("FAKE_RTD_FAIL"):
    print("forced failure", file=sys.stderr)
    sys.exit(5)
marker = os.environ.get("FAKE_RTD_FAIL_ONCE")
if marker and not Path(marker).exists():
    Path(marker).write_text("failed once")
    print("failing once", file=sys.stderr)
    sys.exit(5)
slow_marker = os.environ.get("FAKE_RTD_SLEEP_ONCE")
if slow_marker and not Path(slow_marker).exists():
    Path(slow_marker).write_text("slept once")
    time.sleep(120)
time.sleep(float(os.environ.get("FAKE_RTD_SLEEP", "0")))
if os.environ.get("FAKE_RTD_FAIL_LATE"):
    print("failing after the sleep", file=sys.stderr)
    sys.exit(5)
config = Path(opts["--config"])
spec = json.loads(config.read_text())
# Files a spec names by relative path resolve against the spec's directory, as in rtd_decode.
for key in ("aux", "gamma_source"):
    ref = (spec.get(key) or {{}}).get("path")
    if ref is not None:
        np.load(config.parent / ref)
shots = Path(opts["--shots"])
dets, obs = np.load(shots / "detectors.npy"), np.load(shots / "observables.npy")
weight = dets.sum(axis=1).astype(np.int64)
fail = (weight % spec["fake_fail_mod"] == spec["fake_fail_offset"]).astype(np.uint8)
out = Path(opts["--out"])
out.mkdir(parents=True, exist_ok=True)
n = dets.shape[0]
np.save(out / "logical_failure.npy", fail)
np.save(out / "success.npy", np.ones(n, np.uint8))
np.save(out / "iterations.npy", (weight + 1).astype(np.uint32))
np.save(out / "legs.npy", np.ones(n, np.uint32))
np.save(out / "decode_ns.npy", (weight * 1000 + 1).astype(np.uint64))
np.save(out / "weight.npy", weight.astype(np.float64))
np.save(out / "predicted_observables.npy", obs ^ fail[:, None])
summary = {{"shots": n, "block_error": {{"count": int(fail.sum())}}, "not_converged": {{"count": 0}},
           "iterations": {{"mean": float(weight.mean() + 1) if n else 0.0}}, "wall_seconds": 0.0}}
import hashlib
inputs = {{"config": str(config), "detectors_sha256": hashlib.sha256(np.ascontiguousarray(dets).tobytes()).hexdigest()}}
(out / "run.json").write_text(json.dumps({{"run_id": "fake", "argv": args, "summary": summary, "decoder": spec, "inputs": inputs}}))
print(json.dumps({{"level": "info", "message": "fake decode finished"}}), file=sys.stderr)
'''


@pytest.fixture
def fake_binary(tmp_path) -> Path:
    path = tmp_path / "fake_rtd_decode"
    path.write_text(FAKE_DECODER.format(python=sys.executable))
    path.chmod(0o755)
    return path


def _config(name: str, mode: str = "xyz", chunk_shots: int = 100, max_shots: int | None = 300, arms=None, binary: Path | None = None, **stop):
    config = {
        "name": name,
        "circuit": {"code": "bb18", "experiment": "choi", "p": 0.004, "rounds": 3, "relay_bp_compat": False},
        "mode": mode,
        "seed": 11,
        "chunk_shots": chunk_shots,
        "arms": arms
        or {
            "a": {
                "spec": "configs/xz_relay5_f32.json",
                "workers": 1,
                "cpus": None,
                "record_solutions": 0,
                "save_commits": False,
                "save_solution_supports": False,
                "extra_args": [],
            }
        },
        "stop": {"max_shots": max_shots, "max_wall_seconds": None, "min_failures": None, "failure_arm": None, "paired": None, **stop},
        "keep_raw": False,
    }
    if binary is not None:
        config["rtd_decode"] = str(binary)
    return config


def _fake_arm(tmp_path: Path, name: str, mod: int, offset: int = 0, spec_dir: Path | None = None, **fields) -> dict:
    spec = (spec_dir or tmp_path) / f"spec_{name}.json"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({"fake_fail_mod": mod, "fake_fail_offset": offset, **fields}))
    return {"spec": str(spec), "workers": 1, "cpus": None, "record_solutions": 0, "save_commits": False, "save_solution_supports": False, "extra_args": []}


UNIFORM = {"type": "uniform", "seed": 1, "low": -0.24, "high": 0.66}


def _log_events(directory: Path, message: str) -> list[dict]:
    lines = (directory / "campaign.log").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["message"] == message]


def _init(tmp_path: Path, config: dict, file_name: str = "config.json") -> Path:
    file = tmp_path / file_name
    file.write_text(json.dumps(config, indent=1))
    return cp.init_campaign(file, tmp_path / "campaigns", tmp_path / "artifacts")


# ---------------------------------------------------------------------------------------------
# Configuration


def test_config_validation():
    good = _config("ok")
    parsed = cp.parse_config(good)
    assert parsed.halves == ("xyz",)
    assert parsed.circuit.artifact_name == "bb18_choi_p0.004_r3"
    assert parsed.rtd_decode == cp.DEFAULT_BINARY
    assert parsed.arms["a"].spec == cp.REPO_ROOT / "configs" / "xz_relay5_f32.json"

    def bad(mutate, match):
        c = copy.deepcopy(good)
        mutate(c)
        with pytest.raises(cp.CampaignError, match=match):
            cp.parse_config(c)

    bad(lambda c: c.pop("seed"), "lacks required fields")
    bad(lambda c: c.update(typo=1), "unknown fields")
    bad(lambda c: c["arms"]["a"].pop("extra_args"), "lacks required fields")
    bad(lambda c: c.update(mode="xy"), "mode")
    bad(lambda c: c["circuit"].update(code="bb99"), "circuit.code")
    bad(lambda c: c.update(chunk_shots=0), "chunk_shots")
    bad(lambda c: c.update(seed=True), "seed must be")
    bad(lambda c: c["arms"]["a"].update(extra_args=["--first", "3"]), "driver sets them")
    bad(lambda c: c["arms"]["a"].update(record_solutions=21), "record_solutions")
    bad(lambda c: c["stop"].update(max_shots=None), "bounded")
    bad(lambda c: c["stop"].update(min_failures=10), "failure_arm")
    bad(lambda c: c["stop"].update(paired={"reference": "a", "test": "a", "rel_halfwidth": 0.25}), "two different arms")
    bad(lambda c: c["arms"].update(shots=c["arms"]["a"]), "arm name")


def test_chunk_seed():
    assert cp.chunk_seed(7, 0) == 7 * 1_000_003
    assert cp.chunk_seed(7, 5) == 7 * 1_000_003 + 5
    assert cp.chunk_seed(2**62, 3) == (2**62 * 1_000_003 + 3) % 2**63


def test_sampling_is_deterministic():
    mc = cp._circuit_for(cp.parse_config(_config("det")))
    d1, o1, _ = cp.sample_chunk(mc, 123, 250, 100)
    d2, o2, _ = cp.sample_chunk(mc, 123, 250, 100)
    d3, _, _ = cp.sample_chunk(mc, 124, 250, 100)
    assert d1.dtype == np.uint8 and d1.shape == (250, mc.circuit.num_detectors)
    assert d1.tobytes() == d2.tobytes() and o1.tobytes() == o2.tobytes()
    assert d1.tobytes() != d3.tobytes()


# ---------------------------------------------------------------------------------------------
# Driver behaviour, with a stand-in decoder


def test_run_resume_and_regenerated_chunk_bytes(tmp_path, fake_binary):
    config = _config("fake_xz", mode="xz", chunk_shots=120, max_shots=300, arms={"a": _fake_arm(tmp_path, "a", 3)}, binary=fake_binary)
    config["keep_raw"] = True
    directory = _init(tmp_path, config)
    assert (directory / "bin" / "rtd_decode").is_file()
    init = json.loads((directory / "init.json").read_text())
    assert init["binary"]["sha256"] == cp.sha256_file(fake_binary)
    assert set(init["artifacts"]) == {"xyz", "x", "z"}

    first = cp.run_campaign(directory, max_chunks=1)
    assert first["reason"] == "--chunks reached" and first["chunks"] == 1
    second = cp.run_campaign(directory)
    assert second["reason"] == "criteria met: max_shots"
    assert [cp.load_campaign(directory).chunk_record(c)["shots"] for c in (0, 1, 2)] == [120, 120, 60]
    again = cp.run_campaign(directory)
    assert again["chunks_processed"] == 0

    # The same config in another place gives the same shots and the same records.
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "config.json").write_text(json.dumps(config))
    other = cp.init_campaign(other_root / "config.json", other_root / "campaigns", tmp_path / "artifacts")
    cp.run_campaign(other)
    for c in (0, 1, 2):
        a, b = (json.loads((d / "chunks" / str(c) / "chunk.json").read_text()) for d in (directory, other))
        assert a["detectors_sha256"] == b["detectors_sha256"] and a["seed"] == b["seed"] == cp.chunk_seed(11, c)
    ra, rb = records.load(directory, "a"), records.load(other, "a")
    for half in ("x", "z"):
        for name in ("logical_failure", "iterations", "predicted_observables"):
            assert np.array_equal(ra.half(half)[name], rb.half(half)[name])

    # Regenerating a chunk reproduces the written shot files byte for byte.
    shots = directory / "chunks" / "1" / "shots"
    before = {p.relative_to(shots): p.read_bytes() for p in shots.rglob("*.npy")}
    shutil.rmtree(shots)
    campaign = cp.load_campaign(directory)
    cp._Runner(campaign, cp._Stop()).write_shots(1, dict(campaign.chunk_record(1)))
    after = {p.relative_to(shots): p.read_bytes() for p in shots.rglob("*.npy")}
    assert before == after


def test_failed_chunk_is_retried_once(tmp_path, fake_binary, monkeypatch):
    directory = _init(tmp_path, _config("retry", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    monkeypatch.setenv("FAKE_RTD_FAIL_ONCE", str(tmp_path / "marker"))
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: max_shots"
    log_lines = [json.loads(line) for line in (directory / "campaign.log").read_text().splitlines()]
    failed = [e for e in log_lines if e["message"] == "chunk failed"]
    assert len(failed) == 1 and failed[0]["attempt"] == 1 and failed[0]["recovery"] == "retrying once"
    assert "exit 5" in failed[0]["error"]


def test_chunk_failing_twice_stops_the_run(tmp_path, fake_binary, monkeypatch):
    directory = _init(tmp_path, _config("fail", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    monkeypatch.setenv("FAKE_RTD_FAIL", "1")
    with pytest.raises(cp.CampaignError, match="chunk 0 failed twice"):
        cp.run_campaign(directory)
    assert not (directory / "chunks" / "0" / "a").exists()
    assert (directory / "chunks" / "0" / "chunk.json").is_file()
    monkeypatch.delenv("FAKE_RTD_FAIL")
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"


def _start_run(directory: Path, env: dict, ignore_sighup: bool = False) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "rtd.campaign", "run", str(directory)],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # What nohup does before starting a program.
        preexec_fn=(lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)) if ignore_sighup else None,
    )


def _wait_for(path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.05)


def test_first_signal_finishes_the_chunk(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("finish", chunk_shots=50, max_shots=200, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    pid_file = tmp_path / "pid"
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "2", "FAKE_RTD_PIDFILE": str(pid_file)})
    _wait_for(pid_file)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=60) == 0
    assert records.completed_chunks(directory, "a") == [0]
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert records.completed_chunks(directory, "a") == [0, 1, 2, 3]


def test_second_signal_abandons_the_chunk(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("abandon", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    pid_file = tmp_path / "pid"
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "60", "FAKE_RTD_PIDFILE": str(pid_file)})
    _wait_for(pid_file)
    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    started = time.monotonic()
    assert proc.wait(timeout=30) == 130
    assert time.monotonic() - started < 20
    child = int(pid_file.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    assert not (directory / "chunks" / "0" / "a").exists()
    assert records.completed_chunks(directory, "a") == []
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert records.completed_chunks(directory, "a") == [0, 1]


def test_adding_an_arm_regenerates_and_verifies_chunks(tmp_path, fake_binary):
    config = _config("grow", mode="z_only", chunk_shots=60, max_shots=120, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary)
    directory = _init(tmp_path, config)
    cp.run_campaign(directory)
    assert not list((directory / "chunks").glob("*/shots"))  # raw shots deleted
    grown = copy.deepcopy(config)
    grown["arms"]["b"] = _fake_arm(tmp_path, "b", 3)
    assert _init(tmp_path, grown, "grown.json") == directory
    result = cp.run_campaign(directory)
    assert result["chunks"] == 2 and records.completed_chunks(directory, "b") == [0, 1]
    log_text = (directory / "campaign.log").read_text()
    assert log_text.count("chunk regenerated and verified") == 2

    changed = copy.deepcopy(grown)
    changed["arms"]["a"]["workers"] = 2
    with pytest.raises(cp.CampaignError, match="removed or changed"):
        _init(tmp_path, changed, "changed.json")
    reseeded = copy.deepcopy(grown)
    reseeded["seed"] = 12
    with pytest.raises(cp.CampaignError, match="differs from the initialized campaign"):
        _init(tmp_path, reseeded, "reseeded.json")

    # A regenerated chunk that no longer matches its checksums is refused.
    record_file = directory / "chunks" / "1" / "chunk.json"
    record = json.loads(record_file.read_text())
    record["detectors_sha256"] = "0" * 64
    record_file.write_text(json.dumps(record))
    grown["arms"]["c"] = _fake_arm(tmp_path, "c", 5)
    _init(tmp_path, grown, "grown2.json")
    with pytest.raises(cp.CampaignError, match="chunk 1 failed twice"):
        cp.run_campaign(directory)


def test_pins_are_verified(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("pins", chunk_shots=50, max_shots=50, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    spec = cp.load_campaign(directory).spec("a")
    assert spec == directory / "specs" / "a" / "spec_a.json"
    spec.write_text(json.dumps({"fake_fail_mod": 7, "fake_fail_offset": 0}))
    with pytest.raises(cp.CampaignError, match="changed since init"):
        cp.run_campaign(directory)


def test_arm_specs_are_pinned_separately(tmp_path, fake_binary):
    # Two specs in different directories that use the same relative name for different tables.
    arms = {}
    for name, value in (("a", 1.0), ("b", 2.0)):
        golden = tmp_path / f"golden_{name}"
        (golden / "gammas").mkdir(parents=True)
        np.save(golden / "gammas" / "shape_0.npy", np.full((2, 3), value))
        spec = {"fake_fail_mod": 2, "fake_fail_offset": 0, "gamma_source": {"type": "explicit_shapes", "directory": "gammas"}}
        (golden / "spec.json").write_text(json.dumps(spec))
        arms[name] = {**_fake_arm(tmp_path, name, 2), "spec": str(golden / "spec.json")}
    directory = _init(tmp_path, _config("pin_dirs", chunk_shots=50, max_shots=50, arms=arms, binary=fake_binary))
    campaign = cp.load_campaign(directory)
    for name, value in (("a", 1.0), ("b", 2.0)):
        spec = campaign.spec(name)
        assert spec == directory / "specs" / name / "spec.json"
        assert np.all(np.load(spec.parent / "gammas" / "shape_0.npy") == value)
        pinned = campaign.init["specs"][name]
        assert pinned["reference_sha256"] == {"gammas/shape_0.npy": cp.sha256_file(tmp_path / f"golden_{name}" / "gammas" / "shape_0.npy")}
    np.save(directory / "specs" / "a" / "gammas" / "shape_0.npy", np.full((2, 3), 3.0))
    with pytest.raises(cp.CampaignError, match="referenced by the pinned spec of arm a"):
        cp.run_campaign(directory)


def test_an_interrupted_init_can_be_completed(tmp_path, fake_binary):
    config = _config("partial", chunk_shots=50, max_shots=50, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary)
    directory = tmp_path / "campaigns" / "partial"
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "rtd_decode").write_text("left over")
    assert _init(tmp_path, config) == directory
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    (directory / "campaign.json").unlink()
    with pytest.raises(cp.CampaignError, match="holds chunks"):
        _init(tmp_path, config)


def test_unsupported_flags_are_refused_at_init(tmp_path, fake_binary):
    arm = _fake_arm(tmp_path, "a", 2)
    arm["save_solution_supports"] = True
    with pytest.raises(cp.CampaignError, match="does not support"):
        _init(tmp_path, _config("flags", arms={"a": arm}, binary=fake_binary))

    # A binary without windowed decoding would read a sliding window spec as whole-shot.
    old_binary = tmp_path / "old_rtd_decode"
    old_binary.write_text(fake_binary.read_text().replace(" --save-commits", ""))
    old_binary.chmod(0o755)
    sliding = {"mode": "sliding", "width": 3, "commit": 1, "converge_rounds": 3, "boundary": "exact", "on_failure": "flag", "max_deferrals": 0, "iteration_cap": None}
    windowed = _fake_arm(tmp_path, "w", 2, version=2, window=sliding)
    with pytest.raises(cp.CampaignError, match="predates windowed decoding"):
        _init(tmp_path, _config("old_window", arms={"w": windowed}, binary=old_binary), "old_window.json")
    whole = _fake_arm(tmp_path, "h", 2, version=2, window={"mode": "whole_shot"})
    _init(tmp_path, _config("old_whole", arms={"h": whole}, binary=old_binary), "old_whole.json")
    _init(tmp_path, _config("new_window", arms={"w": windowed}, binary=fake_binary), "new_window.json")


def test_min_failures_and_paired_stopping(tmp_path, fake_binary):
    arms = {"ref": _fake_arm(tmp_path, "ref", 3), "test": _fake_arm(tmp_path, "test", 2)}
    config = _config("stop_fail", chunk_shots=100, max_shots=5000, arms=arms, binary=fake_binary, min_failures=40, failure_arm="ref")
    directory = _init(tmp_path, config)
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: min_failures"
    failures = records.load(directory, "ref")["logical_failure"]
    assert failures.sum() >= 40 and failures[: -100].sum() < 40  # stopped after the first chunk that reached it

    paired = {"reference": "ref", "test": "test", "rel_halfwidth": 0.2}
    directory = _init(tmp_path, _config("stop_pair", chunk_shots=100, max_shots=20000, arms=arms, binary=fake_binary, paired=paired), "pair.json")
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: paired"
    criterion = result["criteria"]["paired"]
    assert criterion["rel_halfwidth"] <= 0.2 < cp.evaluate_stop(cp.load_campaign(directory))["criteria"]["paired"]["threshold"] + 1
    ref, test = cp.paired_failures(cp.load_campaign(directory), "ref", "test")
    n11 = int(np.sum(ref & test))
    assert criterion["n11"] == n11 and criterion["shots"] == ref.size < 20000

    summary = cp.summarize(directory, bootstrap=50)
    assert "gamma_streams" in summary["caveats"]
    assert summary["paired"][0]["n11"] == n11
    assert summary["paired"][0]["ratio"]["method"] == "log-normal"
    written = json.loads((directory / "summary.json").read_text())
    assert written["arms"]["ref"]["combined"]["failures"] == int(ref.sum())


# ---------------------------------------------------------------------------------------------
# Crashes, full disks and signals


def test_full_disk_while_writing_a_marker_leaves_no_truncated_file(tmp_path, fake_binary, monkeypatch):
    directory = _init(tmp_path, _config("enospc", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if ".done.json.tmp" in os.readlink(f"/proc/self/fd/{fd}"):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(cp.CampaignError, match="chunk 0 failed twice"):
        cp.run_campaign(directory)
    arm_dir = directory / "chunks" / "0" / "a"
    assert not (arm_dir / "done.json").exists() and not list(arm_dir.glob(".done.json.tmp-*"))
    monkeypatch.setattr(os, "fsync", real_fsync)  # disk freed
    assert cp.status(directory)["arms"]["a"]["chunks_done"] == 0
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert records.completed_chunks(directory, "a") == [0, 1]
    cp.summarize(directory, bootstrap=20)


def test_unreadable_markers_are_redone(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("corrupt", chunk_shots=50, max_shots=150, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    cp.run_campaign(directory)
    chunks = directory / "chunks"
    last_sha = json.loads((chunks / "2" / "chunk.json").read_text())["detectors_sha256"]
    before = records.load(directory, "a")["logical_failure"]
    # As a crash or a full disk leaves them when a marker is written in place.
    (chunks / "0" / "a" / "done.json").write_text("")
    (chunks / "2" / "chunk.json").write_text('{"chunk": 2, "se')
    status = cp.status(directory)
    assert status["arms"]["a"]["chunks_done"] == 1 and status["stop"]["shots_sampled"] == 100
    assert len(list(chunks.glob("0/a/done.json.corrupt-*"))) == 1 and len(list(chunks.glob("2/chunk.json.corrupt-*"))) == 1
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: max_shots" and result["chunks_processed"] == 2
    assert records.completed_chunks(directory, "a") == [0, 1, 2]
    assert json.loads((chunks / "2" / "chunk.json").read_text())["detectors_sha256"] == last_sha
    assert np.array_equal(records.load(directory, "a")["logical_failure"], before)


def test_failed_chunk_is_not_retried_after_a_stop_request(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("stop_fail", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    pid_file, calls = tmp_path / "pid", tmp_path / "calls"
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "2", "FAKE_RTD_FAIL_LATE": "1", "FAKE_RTD_PIDFILE": str(pid_file), "FAKE_RTD_CALLS": str(calls)})
    _wait_for(pid_file)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=60) == 2
    assert len(calls.read_text().splitlines()) == 1
    failed = [json.loads(line) for line in (directory / "campaign.log").read_text().splitlines() if '"chunk failed"' in line]
    assert len(failed) == 1 and failed[0]["recovery"].startswith("not retrying because a stop was requested")
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"


def test_second_signal_is_honoured_between_phases(tmp_path, fake_binary, monkeypatch):
    directory = _init(tmp_path, _config("phases", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    calls = tmp_path / "calls"
    monkeypatch.setenv("FAKE_RTD_CALLS", str(calls))
    campaign = cp.load_campaign(directory)
    stop = cp._Stop()
    stop.count = 2  # both signals arrived while the chunk was being sampled
    with pytest.raises(cp.Abandoned, match="before decoding arm a"):
        cp._Runner(campaign, stop).process_chunk(0, list(campaign.config.arms.values()), 50)
    assert not calls.exists()  # rtd_decode was never started
    assert (directory / "chunks" / "0" / "chunk.json").is_file() and not (directory / "chunks" / "0" / "a").exists()
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"


def test_hangup_finishes_the_chunk_unless_ignored(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("hup", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    pid_file = tmp_path / "pid"
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "2", "FAKE_RTD_PIDFILE": str(pid_file)})
    _wait_for(pid_file)
    proc.send_signal(signal.SIGHUP)
    assert proc.wait(timeout=60) == 0
    assert records.completed_chunks(directory, "a") == [0]

    # Under nohup a hangup is ignored and the run carries on to its stopping criterion.
    directory = _init(tmp_path, _config("nohup", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary), "nohup.json")
    pid_file.unlink()
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "1", "FAKE_RTD_PIDFILE": str(pid_file)}, ignore_sighup=True)
    _wait_for(pid_file)
    proc.send_signal(signal.SIGHUP)
    assert proc.wait(timeout=60) == 0
    assert records.completed_chunks(directory, "a") == [0, 1]


def test_killed_driver_leaves_no_decoder_running(tmp_path, fake_binary):
    directory = _init(tmp_path, _config("killed", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    pid_file = tmp_path / "pid"
    proc = _start_run(directory, {"FAKE_RTD_SLEEP": "30", "FAKE_RTD_PIDFILE": str(pid_file)})
    _wait_for(pid_file)
    _wait_for(directory / "chunks" / "0" / "logs" / "a.xyz.pid")
    proc.kill()
    proc.wait(timeout=30)
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while cp._running(child, None):
        assert time.monotonic() < deadline, "rtd_decode outlived its killed driver"
        time.sleep(0.05)
    assert (directory / "chunks" / "0" / "logs" / "a.xyz.pid").is_file()  # the driver could not clean up
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert not list((directory / "chunks").glob("*/logs/*.pid"))


def test_stale_decoders_are_terminated(tmp_path):
    logs = tmp_path / "chunks" / "0" / "logs"
    logs.mkdir(parents=True)
    stale = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        ticks = cp._proc_stat(stale.pid)[1]
        (logs / "a.x.pid").write_text(json.dumps({"pid": stale.pid, "start_ticks": ticks}))
        # The same pid number with another start time is a different process: left alone.
        (logs / "a.z.pid").write_text(json.dumps({"pid": os.getpid(), "start_ticks": "1"}))
        (logs / "b.x.pid").write_text("")
        assert cp.reap_stale_decoders(tmp_path) == 1
        assert stale.wait(timeout=15) == -signal.SIGTERM
        assert not list(logs.glob("*.pid"))
    finally:
        if stale.poll() is None:
            stale.kill()
            stale.wait()


# ---------------------------------------------------------------------------------------------
# Per-chunk gamma seeds


def test_splitmix64_and_chunk_gamma_seed():
    # Same known answers as the C++ solution-hash test.
    assert cp.splitmix64(0) == 0xE220A8397B1DCDAF
    assert cp.splitmix64(3) == 0x1D0B14E4DB018FED
    assert cp.splitmix64(cp.splitmix64(0) ^ 1) == 0x08B4FDA8C892B50E
    assert cp.chunk_gamma_seed(1, 0) == cp.splitmix64(1 ^ cp.splitmix64(1))
    assert cp.chunk_gamma_seed(0, 4) == cp.splitmix64(cp.splitmix64(5))
    top = 2**64 - 1
    assert cp.chunk_gamma_seed(top, 7) == cp.splitmix64(top ^ cp.splitmix64(8))
    seeds = [cp.chunk_gamma_seed(1, c) for c in range(5000)]
    assert len(set(seeds)) == 5000 and all(0 <= s <= top for s in seeds)
    assert cp.chunk_gamma_seed(1, 3) != cp.chunk_gamma_seed(2, 3)
    with pytest.raises(ValueError):
        cp.chunk_gamma_seed(-1, 0)
    with pytest.raises(ValueError):
        cp.chunk_gamma_seed(2**64, 0)


def test_gamma_plan_and_chunk_spec_for_both_spec_versions(tmp_path):
    v1 = json.loads((cp.REPO_ROOT / "configs" / "xz_relay5_f32.json").read_text())
    v1.pop("version", None)
    v1.pop("window", None)
    v2 = {"version": 2, **v1, "window": {"mode": "sliding", "width": 3, "commit": 1, "converge_rounds": 3, "boundary": "exact",
                                         "on_failure": "flag", "max_deferrals": 0, "iteration_cap": None}}
    for spec in (v1, v2):
        plan = cp.gamma_plan(spec, True, "test")
        assert (plan.source, plan.spec_seed, plan.per_chunk) == ("uniform", 1, True)
        assert plan.seed(9) == cp.chunk_gamma_seed(1, 9)
        assert plan.record(9) == {"source": "uniform", "per_chunk": True, "spec_seed": 1, "seed": cp.chunk_gamma_seed(1, 9), "rule": cp.GAMMA_SEED_RULE}
        pinned = tmp_path / f"v{spec.get('version', 1)}" / "spec.json"
        pinned.parent.mkdir()
        pinned.write_text(json.dumps(spec, indent=1))
        copy_ = cp.write_chunk_spec(pinned, 9, plan.seed(9))
        assert copy_.parent == pinned.parent and copy_.name == ".chunk-9.spec.json"
        got = json.loads(copy_.read_text())
        assert got["gamma_source"]["seed"] == cp.chunk_gamma_seed(1, 9)
        got["gamma_source"]["seed"] = 1
        assert got == spec and list(got) == list(spec)  # everything else unchanged, key order too
    off = cp.gamma_plan(v1, False, "test")
    assert off.seed(9) == 1 and off.record(9)["seed"] == 1 and "rule" not in off.record(9)
    explicit = cp.gamma_plan({**v1, "gamma_source": {"type": "explicit_shapes", "directory": "gammas"}}, True, "test")
    assert (explicit.source, explicit.seed(3), explicit.per_chunk) == ("explicit_shapes", None, False)
    assert cp.gamma_plan({"fake_fail_mod": 2}, True, "test").source == "absent"
    for bad in (-1, 2**64, 1.5, True, None):
        with pytest.raises(cp.CampaignError, match="unsigned 64-bit"):
            cp.gamma_plan({**v1, "gamma_source": {**UNIFORM, "seed": bad}}, True, "test")


def test_chunks_get_their_own_gamma_seeds(tmp_path, fake_binary):
    # Two arms with the same uniform seed (one also names a file by relative path), one arm with
    # an explicit table, one with another seed; XZ, so each chunk is decoded as two halves.
    specs = tmp_path / "specs_src"
    specs.mkdir()
    np.save(specs / "aux.npy", np.arange(3))
    np.save(specs / "table.npy", np.ones((2, 4)))
    arms = {
        "a": _fake_arm(tmp_path, "a", 3, spec_dir=specs, gamma_source=UNIFORM, aux={"path": "aux.npy"}),
        "b": _fake_arm(tmp_path, "b", 4, spec_dir=specs, gamma_source=UNIFORM),
        "c": _fake_arm(tmp_path, "c", 5, spec_dir=specs, gamma_source={"type": "explicit", "path": "table.npy"}),
        "d": _fake_arm(tmp_path, "d", 5, spec_dir=specs, gamma_source={**UNIFORM, "seed": 2**64 - 5}),
    }
    directory = _init(tmp_path, _config("gamma", mode="xz", chunk_shots=40, max_shots=120, arms=arms, binary=fake_binary))
    init = json.loads((directory / "init.json").read_text())
    assert init["gamma_seed_per_chunk"] is True and init["gamma_seed_rule"] == cp.GAMMA_SEED_RULE
    assert init["specs"]["a"]["gamma"] == {"source": "uniform", "spec_seed": 1, "per_chunk": True}
    assert init["specs"]["c"]["gamma"] == {"source": "explicit", "spec_seed": None, "per_chunk": False}
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"

    campaign = cp.load_campaign(directory)
    for arm, spec_seed in (("a", 1), ("b", 1), ("d", 2**64 - 5)):
        pinned = json.loads(campaign.spec(arm).read_text())
        rec = records.load(directory, arm)
        expected = {c: cp.chunk_gamma_seed(spec_seed, c) for c in (0, 1, 2)}
        assert rec.gamma_seeds == expected
        assert rec.gamma_seed.dtype == np.uint64 and rec.gamma_seed.tolist() == [expected[c] for c in rec.chunk.tolist()]
        for c in (0, 1, 2):
            done = campaign.done_record(c, arm)
            assert done["gamma"] == {"source": "uniform", "per_chunk": True, "spec_seed": spec_seed, "seed": expected[c], "rule": cp.GAMMA_SEED_RULE}
            for half in ("x", "z"):
                run = rec.runs[half][c]
                assert run["decoder"]["gamma_source"]["seed"] == expected[c]  # both halves: the same seed
                used = dict(run["decoder"])
                used["gamma_source"] = {**used["gamma_source"], "seed": spec_seed}
                assert used == pinned  # nothing else changed
                assert Path(run["inputs"]["config"]).parent == campaign.spec(arm).parent
    # Arms with the same spec seed see the same draws on every shot.
    assert np.array_equal(records.load(directory, "a").gamma_seed, records.load(directory, "b").gamma_seed)
    explicit = records.load(directory, "c")
    assert explicit.gamma_seeds == {0: None, 1: None, 2: None}
    assert all(Path(r["inputs"]["config"]) == campaign.spec("c") for r in explicit.runs["x"])
    with pytest.raises(records.RecordsError, match="no recorded gamma seed"):
        _ = explicit.gamma_seed
    assert not list((directory / "specs").glob("*/.chunk-*"))  # per-chunk copies removed
    cp.verify_pins(campaign)
    summary = cp.summarize(directory, bootstrap=20)
    assert summary["gamma_seed_per_chunk"] is True and "splitmix64" in summary["caveats"]["gamma_streams"]
    assert cp.status(directory)["gamma_seed_per_chunk"] is True


def test_gamma_seed_switch_off_and_campaigns_from_before_the_switch(tmp_path, fake_binary):
    arms = {"a": _fake_arm(tmp_path, "a", 3, gamma_source={**UNIFORM, "seed": 42})}
    config = _config("gamma_off", chunk_shots=40, max_shots=80, arms=arms, binary=fake_binary)
    config["gamma_seed_per_chunk"] = False
    directory = _init(tmp_path, config)
    assert json.loads((directory / "init.json").read_text())["gamma_seed_per_chunk"] is False
    cp.run_campaign(directory)
    assert records.load(directory, "a").gamma_seeds == {0: 42, 1: 42}
    assert "row i of every chunk" in cp.summarize(directory, bootstrap=20)["caveats"]["gamma_streams"]
    switched = copy.deepcopy(config)
    switched["gamma_seed_per_chunk"] = True
    with pytest.raises(cp.CampaignError, match="gamma_seed_per_chunk"):
        _init(tmp_path, switched, "switched.json")
    with pytest.raises(cp.CampaignError, match="gamma_seed_per_chunk must be"):
        cp.parse_config({**config, "gamma_seed_per_chunk": 1})

    # A campaign initialized before the switch existed keeps the spec seed for its later chunks.
    old = _config("gamma_old", chunk_shots=40, max_shots=80, arms=arms, binary=fake_binary)
    directory = _init(tmp_path, old, "old.json")
    init_file = directory / "init.json"
    init = json.loads(init_file.read_text())
    del init["gamma_seed_per_chunk"], init["gamma_seed_rule"]
    init_file.write_text(json.dumps(init))
    cp.run_campaign(directory)
    assert records.load(directory, "a").gamma_seeds == {0: 42, 1: 42}
    assert len(_log_events(directory, "gamma draws repeat across chunks: every chunk uses its spec's seed")) == 1


# ---------------------------------------------------------------------------------------------
# Stopping criteria while resuming, watchdog, leftovers, verified reuse


def test_criteria_are_checked_while_resuming(tmp_path, fake_binary, monkeypatch):
    arms = {"a": _fake_arm(tmp_path, "a", 2), "b": _fake_arm(tmp_path, "b", 3)}
    config = _config("resume_stop", chunk_shots=50, max_shots=300, arms=arms, binary=fake_binary, min_failures=20, failure_arm="b")
    directory = _init(tmp_path, config)
    assert cp.run_campaign(directory, arms=["a"])["reason"] == "criteria met: max_shots"  # b pending on 6 chunks
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: min_failures"
    done_b = records.completed_chunks(directory, "b")
    failures = records.load(directory, "b")["logical_failure"]
    assert 1 <= len(done_b) < 6 and failures.sum() >= 20 and failures[:-50].sum() < 20
    assert _log_events(directory, "stopping criteria evaluated after a resumed chunk")[-1]["newly_met"] == ["min_failures"]

    # A criterion already met when the run starts does not stop the catch-up of an added arm.
    grown = copy.deepcopy(config)
    grown["arms"]["c"] = _fake_arm(tmp_path, "c", 5)
    _init(tmp_path, grown, "grown.json")
    assert cp.run_campaign(directory, arms=["c"])["reason"] == "criteria met: max_shots, min_failures"
    assert records.completed_chunks(directory, "c") == [0, 1, 2, 3, 4, 5]

    # Campaign compute time reaching max_wall_seconds during the catch-up stops it after that chunk.
    budget = copy.deepcopy(grown)
    budget["arms"]["d"] = _fake_arm(tmp_path, "d", 7)
    budget["stop"]["max_wall_seconds"] = cp.compute_seconds(cp.load_campaign(directory)) + 0.5
    _init(tmp_path, budget, "budget.json")
    monkeypatch.setenv("FAKE_RTD_SLEEP", "1")
    result = cp.run_campaign(directory, arms=["d"])
    assert result["reason"] == "criteria met: max_wall_seconds" and records.completed_chunks(directory, "d") == [0]


def test_decode_timeout_kills_the_process_group(tmp_path, fake_binary, monkeypatch):
    arm = {**_fake_arm(tmp_path, "a", 2), "decode_timeout_seconds": 1.5}
    config = _config("watchdog", chunk_shots=50, max_shots=100, arms={"a": arm}, binary=fake_binary)
    directory = _init(tmp_path, config)
    helper_file = tmp_path / "helper"
    monkeypatch.setenv("FAKE_RTD_SLEEP", "120")
    monkeypatch.setenv("FAKE_RTD_GRANDCHILD", str(helper_file))
    monkeypatch.setenv("FAKE_RTD_PIDFILE", str(tmp_path / "pid"))
    started = time.monotonic()
    with pytest.raises(cp.CampaignError, match="chunk 0 failed twice"):
        cp.run_campaign(directory)
    assert time.monotonic() - started < 40
    for pid_file in (tmp_path / "pid", helper_file):
        assert not cp._running(int(pid_file.read_text()), None)  # decoder and its helper both gone
    timeouts = _log_events(directory, "rtd_decode exceeded decode_timeout_seconds; killing its process group")
    assert len(timeouts) == 2 and timeouts[0]["timeout_seconds"] == 1.5 and timeouts[0]["arm"] == "a"
    failed = _log_events(directory, "chunk failed")
    assert [e["recovery"] for e in failed] == ["retrying once", "stopping the run"] and "decode_timeout_seconds" in failed[0]["error"]
    assert not (directory / "chunks" / "0" / "a").exists()

    # Only the first decode hangs: the retry completes.
    monkeypatch.delenv("FAKE_RTD_SLEEP")
    monkeypatch.delenv("FAKE_RTD_GRANDCHILD")
    monkeypatch.setenv("FAKE_RTD_SLEEP_ONCE", str(tmp_path / "slept"))
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert len(_log_events(directory, "rtd_decode exceeded decode_timeout_seconds; killing its process group")) == 3

    # The limit is operational: it may change on an existing arm; it must be positive.
    relaxed = copy.deepcopy(config)
    relaxed["arms"]["a"]["decode_timeout_seconds"] = None
    relaxed["arms"]["b"] = _fake_arm(tmp_path, "b", 3)
    assert _init(tmp_path, relaxed, "relaxed.json") == directory
    for bad in (0, -1, float("inf"), "10"):
        broken = copy.deepcopy(config)
        broken["arms"]["a"]["decode_timeout_seconds"] = bad
        with pytest.raises(cp.CampaignError, match="decode_timeout_seconds"):
            cp.parse_config(broken)


def test_leftovers_are_removed_at_the_start_of_a_run(tmp_path, fake_binary):
    config = _config("leftovers", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2, gamma_source=UNIFORM)}, binary=fake_binary)
    directory = _init(tmp_path, config)
    cp.run_campaign(directory)
    campaign = cp.load_campaign(directory)
    runner = cp._Runner(campaign, cp._Stop())
    # As a driver killed between decoding and deleting leaves them.
    runner.write_shots(0, dict(campaign.chunk_record(0)))
    partial = directory / "chunks" / "1" / ".shots.partial-999999"
    (partial / "xyz").mkdir(parents=True)
    (partial / "xyz" / "detectors.npy").write_bytes(b"partial")
    stale_spec = cp.write_chunk_spec(campaign.spec("a"), 7, 123)
    # A chunk that an added arm still needs keeps its shots.
    grown = copy.deepcopy(config)
    grown["arms"]["b"] = _fake_arm(tmp_path, "b", 3)
    _init(tmp_path, grown, "grown.json")
    runner.write_shots(1, dict(campaign.chunk_record(1)))
    removed = cp.clean_leftovers(cp.load_campaign(directory))
    assert removed["raw_shot_dirs"] == 0 and removed["partial_dirs"] == 1 and removed["chunk_specs"] == 1
    assert (directory / "chunks" / "0" / "shots").is_dir() and not partial.exists() and not stale_spec.exists()
    assert cp.run_campaign(directory, arms=["b"], max_chunks=1)["chunks_processed"] == 1  # chunk 0: b decoded, shots deleted
    assert not (directory / "chunks" / "0" / "shots").exists() and (directory / "chunks" / "1" / "shots").is_dir()
    # Once every arm of chunk 1 is done its leftover shots go at the start of the next run.
    b_done = cp.load_campaign(directory).chunk_dir(1) / "b"
    shutil.copytree(cp.load_campaign(directory).chunk_dir(0) / "b", b_done)
    result = cp.run_campaign(directory)
    assert result["chunks_processed"] == 0 and not (directory / "chunks" / "1" / "shots").exists()
    assert _log_events(directory, "deleted leftover raw shots of a finished chunk")[-1]["chunk"] == 1
    kept = copy.deepcopy(grown)
    kept["keep_raw"] = True
    _init(tmp_path, kept, "kept.json")
    runner = cp._Runner(cp.load_campaign(directory), cp._Stop())
    runner.write_shots(0, dict(campaign.chunk_record(0)))
    assert cp.clean_leftovers(cp.load_campaign(directory))["raw_shot_dirs"] == 0  # keep_raw keeps them


@pytest.mark.parametrize("mode", ["xyz", "xz"])
def test_reused_shots_are_verified(tmp_path, fake_binary, monkeypatch, mode):
    directory = _init(tmp_path, _config(f"verify_{mode}", mode=mode, chunk_shots=60, max_shots=60, arms={"a": _fake_arm(tmp_path, "a", 2)}, binary=fake_binary))
    campaign = cp.load_campaign(directory)
    halves = campaign.config.halves
    # A failed first decode keeps the shots; the retry reuses them after checking them.
    monkeypatch.setenv("FAKE_RTD_FAIL_ONCE", str(tmp_path / "failed"))
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    assert len(_log_events(directory, "raw shots reused after verification")) == 1
    record = campaign.chunk_record(0)
    assert set(record["shots_sha256"]) == set(halves)
    if mode == "xyz":
        assert record["shots_sha256"]["xyz"] == {"detectors": record["detectors_sha256"], "observables": record["observables_sha256"]}

    # Shots on disk that differ from chunk.json are regenerated before an added arm decodes them.
    grown = _config(f"verify_{mode}", mode=mode, chunk_shots=60, max_shots=60, arms={"a": _fake_arm(tmp_path, "a", 2), "b": _fake_arm(tmp_path, "b", 3)}, binary=fake_binary)
    _init(tmp_path, grown, "grown.json")
    runner = cp._Runner(cp.load_campaign(directory), cp._Stop())
    runner.write_shots(0, dict(record))
    target = directory / "chunks" / "0" / "shots" / halves[0] / "detectors.npy"
    tampered = np.load(target)
    tampered[0, 0] ^= 1
    np.save(target, tampered)
    assert not runner.verify_shots(0, record)
    cp.run_campaign(directory)
    mismatch = _log_events(directory, "raw shots on disk do not match the checksums in chunk.json")  # the run's check
    assert len(mismatch) == 1 and mismatch[0]["half"] == halves[0] and mismatch[0]["array"] == "detectors"
    for half in halves:
        run = records.load(directory, "b").runs[half][0]
        assert run["inputs"]["detectors_sha256"] == record["shots_sha256"][half]["detectors"]

    # chunk.json without per-half checksums (an older driver): xyz shots are checked against the
    # sampled arrays' checksums; split shots cannot be checked and are regenerated.
    legacy = {k: v for k, v in record.items() if k != "shots_sha256"}
    runner.write_shots(0, dict(record))
    assert runner.verify_shots(0, legacy) == (mode == "xyz")
    (directory / "chunks" / "0" / "shots" / halves[-1] / "observables.npy").write_bytes(b"not an array")
    assert not runner.verify_shots(0, record)


# ---------------------------------------------------------------------------------------------
# Queue runner (CONTEXT/tmp/campaigns/queue.py)

QUEUE_SCRIPT = cp.REPO_ROOT / "CONTEXT" / "tmp" / "campaigns" / "queue.py"
needs_queue = pytest.mark.skipif(not QUEUE_SCRIPT.is_file(), reason=f"{QUEUE_SCRIPT} not present")

FAKE_CAMPAIGN = r'''
# Stand-in for `rtd-campaign run DIR [options]`: behaviour chosen by the directory's name.
import json, os, signal, sys, time
from pathlib import Path

signals = []
signal.signal(signal.SIGTERM, lambda s, f: signals.append(s))
signal.signal(signal.SIGINT, lambda s, f: signals.append(s))
assert sys.argv[1] == "run"
directory, options = Path(sys.argv[2]), sys.argv[3:]
with open(os.environ["FAKE_QUEUE_CALLS"], "a") as f:
    f.write(json.dumps({"dir": directory.name, "options": options, "pid": os.getpid()}) + "\n")
name = directory.name
if name.startswith("append"):
    with open(os.environ["FAKE_QUEUE_FILE"], "a") as q:
        q.write("data/campaigns/appended_job\n")
if name.startswith("fail"):
    print("campaign stopped", file=sys.stderr)
    sys.exit(2)
if name.startswith("slow") and not os.environ.get("FAKE_QUEUE_FAST"):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not signals:
        time.sleep(0.05)
    time.sleep(0.5)  # a second signal in this window abandons the chunk
    if len(signals) >= 2:
        sys.exit(130)
    if signals:
        print(json.dumps({"reason": "signal SIGTERM"}))
        sys.exit(0)
print(json.dumps({"reason": "criteria met: max_shots"}))
'''


def _load_queue_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("campaign_queue", QUEUE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through it
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ beside the script
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture
def fake_campaign_cmd(tmp_path, monkeypatch) -> str:
    script = tmp_path / "fake_campaign.py"
    script.write_text(FAKE_CAMPAIGN)
    monkeypatch.setenv("FAKE_QUEUE_CALLS", str(tmp_path / "calls.jsonl"))
    return f"{sys.executable} {script}"


def _calls(tmp_path: Path) -> list[dict]:
    file = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []


def _done_records(queue_dir: Path) -> list[dict]:
    file = queue_dir / "QUEUE.done"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []


@needs_queue
def test_queue_runs_jobs_in_order_and_skips_finished_ones(tmp_path, fake_campaign_cmd, monkeypatch, capsys):
    queue_mod = _load_queue_module()
    qdir = tmp_path / "q"
    qdir.mkdir()
    queue = qdir / "QUEUE.txt"
    monkeypatch.setenv("FAKE_QUEUE_FILE", str(queue))
    queue.write_text(
        "# E1.0 cells\n"
        "data/campaigns/one --max-wall 3600\n"
        "\n"
        "data/campaigns/fail_two   # this one fails\n"
        "data/campaigns/one --max-wall 3600.0\n"  # the same job again
        "data/campaigns/bad --chunks zero\n"
        "data/campaigns/append_three --arms a,b --chunks 2\n"
    )
    argv = ["--queue", str(queue), "--campaign-cmd", fake_campaign_cmd]
    assert queue_mod.main(argv) == 0
    calls = _calls(tmp_path)
    assert [c["dir"] for c in calls] == ["one", "fail_two", "append_three", "appended_job"]
    assert calls[0]["options"] == ["--max-wall", "3600"] and calls[2]["options"] == ["--arms", "a,b", "--chunks", "2"]
    done = _done_records(qdir)
    assert [(d["job"], d["exit"], d["status"]) for d in done] == [
        ("data/campaigns/one --max-wall 3600", 0, "ok"),
        ("data/campaigns/fail_two", 2, "failed"),
        ("data/campaigns/append_three --arms a,b --chunks 2", 0, "ok"),
        ("data/campaigns/appended_job", 0, "ok"),
    ]
    assert done[0]["reason"] == "criteria met: max_shots" and done[0]["wall_seconds"] >= 0
    assert done[0]["started"] <= done[0]["ended"] and Path(done[1]["stderr"]).read_text().strip() == "campaign stopped"
    log = [json.loads(line) for line in (qdir / "queue.log").read_text().splitlines()]
    assert {"timestamp", "level", "context", "message"} <= set(log[0])
    assert any(e["message"] == "job failed" and e["stderr_tail"] == ["campaign stopped"] for e in log)
    assert any(e["message"] == "invalid queue line skipped" and e["line_number"] == 6 for e in log)
    assert log[-1]["message"] == "queue drained"

    # Everything is done: a second runner starts nothing; --list shows it.
    assert queue_mod.main(argv) == 0
    assert len(_calls(tmp_path)) == 4
    assert queue_mod.main([*argv, "--list"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(e["job"], e["done"]) for e in listed] == [(d["job"], True) for d in done]
    # A missing queue file is an error.
    assert queue_mod.main(["--queue", str(qdir / "missing.txt"), "--campaign-cmd", fake_campaign_cmd]) == 1


@needs_queue
@pytest.mark.parametrize("signals", [1, 2])
def test_queue_stop_signal_reaches_the_campaign(tmp_path, fake_campaign_cmd, signals):
    qdir = tmp_path / "q"
    qdir.mkdir()
    queue = qdir / "QUEUE.txt"
    queue.write_text("data/campaigns/slow_one\ndata/campaigns/two\n")
    proc = subprocess.Popen(
        [sys.executable, str(QUEUE_SCRIPT), "--queue", str(queue), "--campaign-cmd", fake_campaign_cmd],
        env=os.environ.copy(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while not _calls(tmp_path):
        assert time.monotonic() < deadline, "the first job did not start"
        time.sleep(0.05)
    for _ in range(signals):
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.1)
    assert proc.wait(timeout=30) == 130
    child = _calls(tmp_path)[0]["pid"]
    assert not cp._running(child, None)
    assert [c["dir"] for c in _calls(tmp_path)] == ["slow_one"]  # the next job was not started
    assert _done_records(qdir) == []  # the interrupted job stays pending
    log = [json.loads(line) for line in (qdir / "queue.log").read_text().splitlines()]
    interrupted = [e for e in log if e["message"] == "job interrupted by a stop request; it stays pending"]
    assert len(interrupted) == 1 and interrupted[0]["exit"] == (0 if signals == 1 else 130)
    # The next runner resumes the interrupted job and then the rest.
    env = {**os.environ, "FAKE_QUEUE_FAST": "1"}
    assert subprocess.run([sys.executable, str(QUEUE_SCRIPT), "--queue", str(queue), "--campaign-cmd", fake_campaign_cmd], env=env, capture_output=True).returncode == 0
    assert [d["job"] for d in _done_records(qdir)] == ["data/campaigns/slow_one", "data/campaigns/two"]


@needs_queue
def test_queue_runs_a_real_campaign(tmp_path, fake_binary):
    queue_mod = _load_queue_module()
    directory = _init(tmp_path, _config("queued", chunk_shots=50, max_shots=100, arms={"a": _fake_arm(tmp_path, "a", 2, gamma_source=UNIFORM)}, binary=fake_binary))
    qdir = tmp_path / "q"
    qdir.mkdir()
    (qdir / "QUEUE.txt").write_text(f"{directory} --chunks 1\n{directory}\n")
    assert queue_mod.main(["--queue", str(qdir / "QUEUE.txt"), "--campaign-cmd", f"{sys.executable} -m rtd.campaign"]) == 0
    done = _done_records(qdir)
    assert [(d["exit"], d["reason"]) for d in done] == [(0, "--chunks reached"), (0, "criteria met: max_shots")]
    assert records.completed_chunks(directory, "a") == [0, 1]
    assert records.load(directory, "a").gamma_seeds == {0: cp.chunk_gamma_seed(1, 0), 1: cp.chunk_gamma_seed(1, 1)}


# ---------------------------------------------------------------------------------------------
# Real decoder


needs_binary = pytest.mark.skipif(not REAL_BINARY.is_file(), reason=f"{REAL_BINARY} not built")


@needs_binary
@pytest.mark.parametrize("mode", ["xyz", "xz", "z_only"])
def test_real_decoder_small_campaign(tmp_path, mode):
    arm = {"spec": "configs/xz_relay5_f32.json", "workers": 2, "cpus": CPUS, "record_solutions": 0, "save_commits": False, "save_solution_supports": False, "extra_args": ["--warmup", "1"]}
    directory = _init(tmp_path, _config(f"real_{mode}", mode=mode, chunk_shots=150, max_shots=300, arms={"relay5": arm}))
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: max_shots"
    assert not list((directory / "chunks").glob("*/shots"))
    rec = records.load(directory, "relay5")
    assert rec.shots == 300 and rec.halves == records.MODES[mode]
    done_failures = sum(d["combined"]["failures"] for d in rec.done)
    assert int(rec["logical_failure"].sum()) == done_failures
    if mode == "xz":
        assert np.array_equal(rec["logical_failure"], rec.half("x")["logical_failure"] | rec.half("z")["logical_failure"])
    else:
        assert set(rec.half(rec.halves[0])) <= set(rec.combined)
        assert rec["predicted_observables"].shape[0] == 300
    for half in rec.halves:
        run = rec.runs[half][0]
        assert run["execution"]["workers"] == 2 and run["execution"]["cpus"] == CPUS
        assert run["execution"]["warmup_per_worker"] == 1
    summary = cp.summarize(directory, bootstrap=50)
    assert summary["arms"]["relay5"]["combined"]["shots"] == 300
    status = cp.status(directory)
    assert status["arms"]["relay5"]["shots"] == 300 and status["raw_shot_bytes"] == 0


# ---------------------------------------------------------------------------------------------
# Sliding-window arms

# Appended to the fake decoder: with "fake_windows": K in the spec it also writes the arrays of a
# sliding-window decode, derived from the shot's detector weight so that the summary can be
# checked against them.
FAKE_WINDOWS = r"""
K = spec.get("fake_windows")
if K:
    position = np.arange(K)[None, :]
    attempts = np.ones((n, K), np.uint8) + ((weight[:, None] + position) % 4 == 1)
    attempts[:, -1] *= (weight % 3 != 0).astype(np.uint8)  # an earlier final window decided it
    decoded = attempts > 0
    converged = (((weight[:, None] + position) % 5 != 0) | ~decoded).astype(np.uint8)
    np.save(out / "win_iterations.npy", np.where(decoded, weight[:, None] + 3 * position + 1, 0).astype(np.uint32))
    np.save(out / "win_legs.npy", decoded.astype(np.uint32))
    np.save(out / "win_attempts.npy", attempts)
    np.save(out / "win_converged.npy", converged)
    np.save(out / "win_cap_hit.npy", (decoded & (position == 0) & (weight[:, None] % 7 == 0)).astype(np.uint8))
    np.save(out / "win_weight.npy", np.where(converged == 1, 1.5, np.inf))
    np.save(out / "win_committed_weight.npy", np.ones((n, K)))
    np.save(out / "win_unexplained.npy", ((converged == 0) * 2).astype(np.uint32))
    np.save(out / "win_flagged.npy", (converged == 0).astype(np.uint8))
    np.save(out / "win_virtual.npy", np.zeros((n, K), np.uint32))
    np.save(out / "win_decode_ns.npy", np.full((n, K), 1000, np.uint64))
    np.save(out / "flagged.npy", (converged == 0).any(axis=1).astype(np.uint8))
    slots = int(opts.get("--record-solutions", 0))
    if slots:
        np.save(out / "sol_count.npy", np.where(decoded, converged * (weight[:, None] % 4), 0).astype(np.uint32))
        np.save(out / "returned_class.npy", np.zeros((n, K), np.uint64))
        for name, dtype in (("sol_leg", np.uint32), ("sol_iterations", np.uint32), ("sol_class", np.uint64), ("sol_hash", np.uint64), ("sol_size", np.uint32)):
            np.save(out / f"{name}.npy", np.zeros((n, K, slots), dtype))
        np.save(out / "sol_weight.npy", np.full((n, K, slots), np.inf))
"""


@pytest.fixture
def fake_windowed_binary(tmp_path) -> Path:
    path = tmp_path / "fake_rtd_decode_windows"
    path.write_text(FAKE_DECODER.format(python=sys.executable) + FAKE_WINDOWS)
    path.chmod(0o755)
    return path


def _expected_windows(arrays: dict) -> dict:
    attempts = arrays["win_attempts"]
    decoded = attempts > 0
    return {
        "positions": attempts.shape[1],
        "windows": int(decoded.sum()),
        "skipped_after_final": int((~decoded).sum()),
        "converged": int(arrays["win_converged"][decoded].sum()),
        "cap_hit": int(arrays["win_cap_hit"][decoded].sum()),
        "deferred": int((attempts > 1).sum()),
        "flagged_shots": int(arrays["flagged"].sum()),
        "iterations_mean": float(arrays["win_iterations"][decoded].mean()),
        "max_per_shot_max": float(arrays["win_iterations"].max()),
    }


def _check_window_block(block: dict, expected: dict) -> None:
    assert block["positions"] == expected["positions"]
    assert block["windows"] == expected["windows"]
    assert block["skipped_after_final"] == expected["skipped_after_final"]
    assert block["converged"]["count"] == expected["converged"] and block["converged"]["of"] == expected["windows"]
    lo, hi = block["converged"]["wilson95"]
    assert lo <= block["converged"]["fraction"] <= hi
    assert block["cap_hit"]["count"] == expected["cap_hit"]
    assert block["deferred"]["count"] == expected["deferred"]
    assert block["flagged_shots"]["count"] == expected["flagged_shots"]
    assert block["iterations_per_window"]["mean"] == pytest.approx(expected["iterations_mean"])
    assert "mean_bootstrap95" in block["iterations_per_window"]
    assert block["max_window_iterations_per_shot"]["max"] == expected["max_per_shot_max"]
    assert block["decode_time_us_per_window"]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", ["xz", "z_only"])
def test_windowed_arms_are_summarized_per_window(tmp_path, fake_windowed_binary, mode):
    arms = {
        "win": {**_fake_arm(tmp_path, "win", 4, fake_windows=3), "record_solutions": 2},
        "whole": _fake_arm(tmp_path, "whole", 4),
    }
    directory = _init(tmp_path, _config(f"windowed_{mode}", mode=mode, chunk_shots=60, max_shots=120, arms=arms, binary=fake_windowed_binary))
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    rec = records.load(directory, "win")
    for half in rec.halves:
        assert rec.windows(half) == 3 and rec.solution_slots(half) == 2
        assert rec.half(half)["sol_leg"].shape == (120, 3, 2)
    summary = cp.summarize(directory, bootstrap=30)
    win = summary["arms"]["win"]
    assert win["combined"]["flagged"]["count"] == int(rec["flagged"].sum())
    if mode == "z_only":
        _check_window_block(win["combined"]["windows"], _expected_windows(rec.half("z")))
    for half in rec.halves:
        block = win["halves"][half]["windows"]
        _check_window_block(block, _expected_windows(rec.half(half)))
        assert block["solutions_found_per_window"]["max"] <= 3
    if mode == "xz":
        over = win["windows_over_halves"]
        x, z = rec.half("x")["win_iterations"].astype(np.uint64), rec.half("z")["win_iterations"].astype(np.uint64)
        decoded = (rec.half("x")["win_attempts"] > 0) | (rec.half("z")["win_attempts"] > 0)
        assert over["windows"] == int(decoded.sum())
        assert over["iterations_max_per_window"]["max"] == float(np.maximum(x, z)[decoded].max())
        assert over["iterations_sum_per_window"]["mean"] == pytest.approx(float((x + z)[decoded].mean()))
    else:
        assert "windows_over_halves" not in win
    # A whole-shot arm has no per-window statistics.
    whole = summary["arms"]["whole"]
    assert "windows" not in whole["combined"] and "flagged" not in whole["combined"]
    assert all("windows" not in h for h in whole.get("halves", {}).values())
    written = json.loads((directory / "summary.json").read_text())
    assert written["arms"]["win"]["halves"][rec.halves[0]]["windows"]["positions"] == 3


SLIDING_BINARY = os.environ.get("RTD_SLIDING_DECODER")


@pytest.mark.skipif(not SLIDING_BINARY, reason="set RTD_SLIDING_DECODER to an rtd_decode that decodes sliding windows")
def test_real_sliding_campaign(tmp_path):
    spec = json.loads((cp.REPO_ROOT / "configs" / "xz_relay5_f32.json").read_text())
    # bb18 over 3 rounds: Rt = 4, so (W, C) = (3, 1) gives two windows, the second one final.
    spec["window"] = {"mode": "sliding", "width": 3, "commit": 1, "converge_rounds": 3, "boundary": "exact",
                      "on_failure": "flag", "max_deferrals": 0, "iteration_cap": None}  # fmt: skip
    spec_file = tmp_path / "sliding.json"
    spec_file.write_text(json.dumps(spec))
    arm = {"spec": str(spec_file), "workers": 2, "cpus": CPUS, "record_solutions": 2, "save_commits": True, "save_solution_supports": False, "extra_args": []}
    directory = _init(tmp_path, _config("real_sliding", mode="xz", chunk_shots=100, max_shots=200, arms={"sliding": arm}, binary=Path(SLIDING_BINARY)))
    assert cp.run_campaign(directory)["reason"] == "criteria met: max_shots"
    rec = records.load(directory, "sliding")
    assert rec.shots == 200
    for half in ("x", "z"):
        arrays = rec.half(half)
        assert rec.windows(half) == 2 and rec.solution_slots(half) == 2
        assert arrays["commit_ptr"].size == 200 * 2 + 1 and int(arrays["commit_ptr"][-1]) == arrays["commit_faults"].size
        assert rec.runs[half][0]["window"]["mode"] == "sliding"
        assert rec.runs[half][0]["plan"]["positions"] == 2
    summary = cp.summarize(directory, bootstrap=30)
    for half in ("x", "z"):
        block = summary["arms"]["sliding"]["halves"][half]["windows"]
        assert block["positions"] == 2 and block["windows"] == int((rec.half(half)["win_attempts"] > 0).sum())
    assert "windows_over_halves" in summary["arms"]["sliding"]


# ---------------------------------------------------------------------------------------------
# Detector order of the readout layer and the uniform window boundary


def test_readout_in_round_order_config():
    plain = cp.parse_config(_config("plain"))
    assert plain.circuit.readout_in_round_order is False and plain.circuit.artifact_name == "bb18_choi_p0.004_r3"
    raw = _config("ordered")
    raw["circuit"]["readout_in_round_order"] = True
    ordered = cp.parse_config(raw)
    assert ordered.circuit.readout_in_round_order is True and ordered.circuit.artifact_name == "bb18_choi_p0.004_r3_rorder"
    raw["circuit"]["relay_bp_compat"] = True
    assert cp.parse_config(raw).circuit.artifact_name == "bb18_choi_p0.004_r3_compat_rorder"
    for bad, match in ((1, "readout_in_round_order"), ("yes", "readout_in_round_order")):
        raw["circuit"]["readout_in_round_order"] = bad
        with pytest.raises(cp.CampaignError, match=match):
            cp.parse_config(raw)
    raw["circuit"].update(readout_in_round_order=True, experiment="z")
    with pytest.raises(cp.CampaignError, match="choi"):
        cp.parse_config(raw)


def _metadata_dir(path: Path, readout_in_round_order: bool) -> Path:
    from rtd.bb_code import get_code
    from rtd.circuit import build_memory_circuit
    from rtd.noise import NoiseModel

    info = build_memory_circuit(get_code("bb18"), 3, "choi", NoiseModel.uniform(3e-3),
                                readout_in_round_order=readout_in_round_order).detectors
    path.mkdir(parents=True)
    np.save(path / "det_round.npy", info.round)
    np.save(path / "det_type.npy", info.kind)
    np.save(path / "det_check.npy", info.check)
    return path


def test_uniform_row_order_check(tmp_path):
    native = _metadata_dir(tmp_path / "native", False)
    ordered = _metadata_dir(tmp_path / "ordered", True)
    assert cp.uniform_row_mismatch(ordered) is None
    reason = cp.uniform_row_mismatch(native)
    assert reason is not None and "round 4" in reason and "1 of 4 rounds differ" in reason
    assert "unreadable" in cp.uniform_row_mismatch(tmp_path / "missing")
    arm = cp.ArmConfig("u", Path("spec.json"), 1, None, 0, False, False, ())
    uniform = {"u": {"window": {"mode": "sliding", "boundary": "uniform"}}}
    exact = {"u": {"window": {"mode": "sliding", "boundary": "exact"}}}
    with pytest.raises(cp.CampaignError, match="readout_in_round_order"):
        cp.check_uniform_rows([arm], uniform, {"xyz": native}, ("xyz",))
    cp.check_uniform_rows([arm], uniform, {"xyz": ordered}, ("xyz",))
    cp.check_uniform_rows([arm], exact, {"xyz": native}, ("xyz",))
    cp.check_uniform_rows([arm], {"u": {"window": {"mode": "whole_shot"}}}, {"xyz": native}, ("xyz",))


@pytest.mark.parametrize("ordered", [False, True])
def test_init_refuses_uniform_boundary_on_misordered_rounds(tmp_path, fake_binary, ordered):
    arm = _fake_arm(tmp_path, "u", 4, window={"mode": "sliding", "width": 2, "commit": 1, "boundary": "uniform"})
    config = _config(f"uniform_{ordered}", arms={"u": arm}, binary=fake_binary)
    config["circuit"]["readout_in_round_order"] = ordered
    if ordered:
        directory = _init(tmp_path, config)
        init = json.loads((directory / "init.json").read_text())
        assert init["artifacts"]["xyz"]["path"].endswith("bb18_choi_p0.004_r3_rorder")
        assert "u" in init["specs"]
    else:
        with pytest.raises(cp.CampaignError, match="round 4 lists its detectors"):
            _init(tmp_path, config)
    # Each XZ half holds one detector type, so its rounds agree whatever the readout order.
    xz = _config(f"uniform_xz_{ordered}", mode="xz", arms={"u": arm}, binary=fake_binary)
    xz["circuit"]["readout_in_round_order"] = ordered
    _init(tmp_path, xz, file_name="xz.json")
