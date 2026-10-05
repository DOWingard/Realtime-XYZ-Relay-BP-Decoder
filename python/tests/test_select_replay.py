"""rtd_select_replay against the decoder itself.

A run of rtd_decode records the first 5 solutions (converged legs) of every decode with their
supports. Replaying the first S of them through a stopping rule must say exactly what a decode
with that rule and a relay rule of S converged legs does: the iterations and legs it runs, the
number of solutions it holds, whether the rule ended it before the relay rule, and the class it
returns. With the fixed rule every confidence value of the replay must equal the one the decoder
recorded, bit for bit. Sliding-window runs exercise the window plan the tool rebuilds from the
run's spec. The tool's refusals are checked too.

Needs a build holding both rtd_decode and rtd_select_replay: RTD_BIN_DIR, else the first
build/*release*/src or build/*debug*/src that has both; skipped otherwise.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
R9 = REPO / "test" / "fixtures" / "bb18_choi_r9"
SHOTS = 40
RECORDED = 5

# The confidence values the replay writes as rep_<name> and the decoder as conf_<name>.
VALUES = ("seen", "weight", "best_class", "gap", "second_class", "distinct", "agreement", "first_legs",
          "first_iterations", "class_sum_class", "class_sum_top", "agreement_class", "q_supp", "q_sum_sq",
          "q_total", "components")
# (rule, replay flags, spec "stop" object, smallest S the rule accepts)
STOPS = (
    ("fixed", [], {"rule": "fixed"}, 1),
    ("agree", ["--count", "2"], {"rule": "agree", "count": 2}, 1),
    ("agree_distinct", ["--count", "2"], {"rule": "agree_distinct", "count": 2}, 1),
    ("gap", ["--threshold", "1.0"], {"rule": "gap", "threshold": 1.0}, 1),
    ("gap_extend", ["--count", "2", "--threshold", "3.0"], {"rule": "gap_extend", "count": 2, "threshold": 3.0}, 2),
)
SLIDING = {"mode": "sliding", "width": 4, "commit": 2, "converge_rounds": 4, "boundary": "exact",
           "on_failure": "commit_anyway", "max_deferrals": 0, "iteration_cap": None}


def _tools() -> Path | None:
    if "RTD_BIN_DIR" in os.environ:
        candidates = [Path(os.environ["RTD_BIN_DIR"])]
    else:
        candidates = [d for preset in ("release", "debug") for d in sorted((REPO / "build").glob(f"*{preset}*/src"))]
    for d in candidates:
        if (d / "rtd_decode").is_file() and (d / "rtd_select_replay").is_file():
            return d
    return None


BIN = _tools()
pytestmark = pytest.mark.skipif(BIN is None, reason="no build with rtd_decode and rtd_select_replay")


def spec(count: int, stop: dict | None = None, window: dict | None = None) -> dict:
    """A small Relay-BP spec with the default selection policy (or a stopping rule) whose rules
    see all `count` solutions of the relay rule."""
    return {
        "version": 3, "policy": "f32", "backend": "cpu", "layout": "row_major", "column_order": "wavefront",
        "block_rows": 64, "executor": {"type": "serial"}, "alpha": {"rule": "constant", "value": 1.0},
        "gamma0": 0.125, "pre_iter": 20, "set_max_iter": 15, "num_sets": 40,
        "stopping": {"rule": "after_n_converged", "count": count},
        "gamma_source": {"type": "uniform", "seed": 7, "low": -0.24, "high": 0.66},
        "window": window or {"mode": "whole_shot"},
        "selection": {"rule": "lowest_weight", "stop": stop or {"rule": "fixed"}, "capacity": max(count, RECORDED),
                      "confidence": None, "history": None},
    }


def decode(tmp: Path, name: str, body: dict, record: bool) -> Path:
    config = tmp / f"{name}.json"
    config.write_text(json.dumps(body))
    out = tmp / name
    cmd = [str(BIN / "rtd_decode"), "--artifact", str(R9 / "artifact"), "--shots", str(R9 / "shots"),
           "--config", str(config), "--out", str(out), "--count", str(SHOTS), "--workers", "2", "--no-verify"]
    if record:
        cmd += ["--record-solutions", str(RECORDED), "--save-solution-supports"]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return out


def replay(tmp: Path, run: Path, slots: int, flags: list[str], name: str, verify: bool = False) -> tuple[int, Path]:
    out = tmp / name
    cmd = [str(BIN / "rtd_select_replay"), "--artifact", str(R9 / "artifact"), "--run", str(run),
           "--slots", str(slots), *flags, "--out", str(out)]
    if not verify:
        cmd.append("--no-verify")
    done = subprocess.run(cmd, capture_output=True, text=True)
    return done.returncode, out


def load(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    return {p.stem[len(prefix):]: np.load(p) for p in directory.glob(f"{prefix}*.npy")}


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        return a.shape == b.shape and bool(np.all(a.view(np.uint64) == b.view(np.uint64)))
    return bool(np.array_equal(a, b))


@pytest.fixture(scope="module")
def recordings(tmp_path_factory) -> dict[str, Path]:
    tmp = tmp_path_factory.mktemp("select_replay")
    return {"whole": decode(tmp, "whole_rec", spec(RECORDED), True),
            "sliding": decode(tmp, "sliding_rec", spec(RECORDED, window=SLIDING), True),
            "tmp": tmp}


def check_stop_outcome(rep: dict[str, np.ndarray], run: Path, slots: int, cells: np.ndarray, where: str) -> int:
    """The replayed stop outcome against a direct decode's arrays over the given [S, K] cells."""
    conf = load(run, "conf_")
    sliding = (run / "win_iterations.npy").is_file()
    iterations = np.load(run / ("win_iterations.npy" if sliding else "iterations.npy")).reshape(rep["stop_iterations"].shape)
    legs = np.load(run / ("win_legs.npy" if sliding else "legs.npy")).reshape(rep["stop_legs"].shape)
    # Under the lowest-weight rule the decided class is the class of the returned correction
    # (leg 0's final estimate when nothing converged); returned_class.npy exists only in recordings.
    returned = conf["decided_class"]
    early = rep["stop_by_rule"].astype(bool) & (rep["stop_solutions"] < slots)
    assert np.array_equal(rep["stop_iterations"][cells], iterations[cells]), where
    assert np.array_equal(rep["stop_legs"][cells], legs[cells]), where
    assert np.array_equal(rep["stop_solutions"][cells], conf["found"][cells]), where
    assert np.array_equal(rep["stop_decision"][cells], returned[cells]), where
    assert np.array_equal(early[cells], conf["stopped_early"][cells].astype(bool)), where
    return int(early[cells].sum())


def test_whole_shot_replays_match_direct_decodes(recordings):
    tmp = recordings["tmp"]
    early = 0
    for slots in range(1, RECORDED + 1):
        for rule, flags, stop, smallest in STOPS:
            if slots < smallest:
                continue
            where = f"S {slots} {rule}"
            code, out = replay(tmp, recordings["whole"], slots, ["--stop", rule, *flags], f"whole_s{slots}_{rule}")
            assert code == 0, where
            rep = load(out, "rep_")
            direct = decode(tmp, f"whole_direct_s{slots}_{rule}", spec(slots, stop), False)
            cells = np.ones(rep["stop_legs"].shape, dtype=bool)
            early += check_stop_outcome(rep, direct, slots, cells, where)
            if rule == "fixed":
                conf = load(direct, "conf_")
                assert np.array_equal(rep["state"], conf["gap_state"]), where
                for name in VALUES:
                    assert same_bits(rep[name], conf[name]), f"{where} {name}"
    assert early > 0  # some rule stopped some decode before its relay rule


def test_sliding_replays_rebuild_the_window_plan(recordings):
    tmp = recordings["tmp"]
    run = recordings["sliding"]
    attempts = np.load(run / "win_attempts.npy")
    assert attempts.shape[1] > 1
    # S = 5 with the fixed rule is the recording run itself: every decoded window, every value.
    code, out = replay(tmp, run, RECORDED, ["--stop", "fixed"], "sliding_s5_fixed", verify=True)
    assert code == 0
    rep = load(out, "rep_")
    conf = load(run, "conf_")
    decoded = attempts > 0
    check_stop_outcome(rep, run, RECORDED, decoded, "sliding S 5 fixed")
    assert np.array_equal(rep["state"][decoded], conf["gap_state"][decoded])
    for name in VALUES:
        assert same_bits(rep[name][decoded], conf[name][decoded]), name
    # Any other S or rule changes what the first window commits, so later windows decode other
    # syndromes; the first window's decode is the same problem in both runs.
    first = np.zeros(attempts.shape, dtype=bool)
    first[:, 0] = True
    for slots, (rule, flags, stop, _) in ((2, STOPS[0]), (5, STOPS[3]), (4, STOPS[4])):
        code, out = replay(tmp, run, slots, ["--stop", rule, *flags], f"sliding_s{slots}_{rule}")
        assert code == 0
        direct = decode(tmp, f"sliding_direct_s{slots}_{rule}", spec(slots, stop, SLIDING), False)
        check_stop_outcome(load(out, "rep_"), direct, slots, first, f"sliding S {slots} {rule}")
    json.loads((out / "replay.json").read_text())  # the record is written and parses


def test_the_tool_refuses_what_a_recording_cannot_answer(recordings):
    tmp = recordings["tmp"]
    whole = recordings["whole"]
    # More slots than were recorded.
    assert replay(tmp, whole, RECORDED + 1, [], "refuse_slots")[0] == 2
    # A run that stopped after 3 converged legs cannot say what a fourth would have found.
    short = decode(tmp, "short_rec", spec(3), True)
    assert replay(tmp, short, 3, [], "short_ok")[0] == 0
    assert replay(tmp, short, 4, [], "refuse_stop_count")[0] == 2
    # gap_extend's n0 must lie within the replayed slots; agree needs its count.
    assert replay(tmp, whole, 2, ["--stop", "gap_extend", "--count", "3", "--threshold", "1"], "refuse_n0")[0] == 2
    assert replay(tmp, whole, 2, ["--stop", "agree"], "refuse_count")[0] == 2
    assert replay(tmp, whole, 2, ["--stop", "median"], "refuse_rule")[0] == 2
    # A directory that holds no recording.
    assert replay(tmp, tmp / "missing", 2, [], "refuse_missing")[0] == 3
    # An output directory that is not empty is kept.
    code, out = replay(tmp, whole, 2, [], "kept")
    assert code == 0
    assert replay(tmp, whole, 2, [], "kept")[0] == 4
