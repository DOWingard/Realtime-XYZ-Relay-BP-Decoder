"""The paired stopping criterion of rtd-campaign: one pair, a list of pairs that must all hold,
and null.

Most tests build a campaign directory by hand (campaign.json, chunk and done markers, per-shot
arrays) with failure patterns from a fixed integer hash, so the expected counts do not depend on
stim, numpy's random streams or a decoder. GOLDEN holds what the driver returned for the
single-pair and null forms before the list form existed; those forms must keep returning exactly
that. The tests that run the driver end to end use the stand-in decoder of test_campaign.py.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rtd import campaign as cp
from rtd import records

GOLDEN = Path(__file__).with_name("data") / "campaign_paired_stop_golden.json"

# ---------------------------------------------------------------------------------------------
# A hand-built campaign

CHUNKS = 4
CHUNK_SHOTS = 2000
ARMS = ("ref", "close", "far", "never", "same", "partial", "late")
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(salt: int, n: int) -> np.ndarray:
    """splitmix64 of (salt << 32) + i for i < n: a fixed pseudo-random integer per shot."""
    with np.errstate(over="ignore"):
        z = (np.uint64(salt) << np.uint64(32)) + np.arange(n, dtype=np.uint64)
        z = (z + np.uint64(0x9E3779B97F4A7C15)) & _M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        return z ^ (z >> np.uint64(31))


def failure_patterns(n: int = CHUNKS * CHUNK_SHOTS) -> dict[str, np.ndarray]:
    """Failure flags per arm over all shots: 'close' fails mostly with the reference, 'far'
    independently and more often, 'never' never, 'same' exactly with the reference."""
    ref = _mix(1, n) % np.uint64(40) == 0
    close = (ref & (_mix(2, n) % np.uint64(8) != 0)) | (_mix(3, n) % np.uint64(300) == 0)
    far = _mix(4, n) % np.uint64(30) == 0
    return {"ref": ref, "close": close, "far": far, "never": np.zeros(n, bool), "same": ref.copy(),
            "partial": close.copy(), "late": far.copy()}


def _arm(spec: str = "configs/xz_relay5_f32.json") -> dict:
    return {"spec": spec, "workers": 1, "cpus": None, "record_solutions": 0, "save_commits": False,
            "save_solution_supports": False, "extra_args": []}


def build_campaign(root: Path, name: str, stop: dict, pairs: list | None = None) -> Path:
    """A finished-looking xyz campaign: CHUNKS chunks of CHUNK_SHOTS shots, every arm done on
    every chunk except 'partial' (chunk 2 missing) and 'late' (no chunk)."""
    directory = root / name
    config = {"name": name, "circuit": {"code": "bb18", "experiment": "choi", "p": 0.004, "rounds": 3, "relay_bp_compat": False},
              "mode": "xyz", "seed": 5, "chunk_shots": CHUNK_SHOTS, "arms": {a: _arm() for a in ARMS},
              "stop": {"max_shots": None, "max_wall_seconds": None, "min_failures": None, "failure_arm": None, "paired": None, **stop},
              "keep_raw": False}
    if pairs is not None:
        config["pairs"] = pairs
    (directory / "chunks").mkdir(parents=True)
    (directory / "campaign.json").write_text(json.dumps(config, indent=1))
    (directory / "init.json").write_text("{}")
    fails = failure_patterns()
    for c in range(CHUNKS):
        cdir = directory / "chunks" / str(c)
        cdir.mkdir()
        (cdir / "chunk.json").write_text(json.dumps({"chunk": c, "shots": CHUNK_SHOTS, "sample_seconds": 1.5 + c}))
        rows = slice(c * CHUNK_SHOTS, (c + 1) * CHUNK_SHOTS)
        for arm in ARMS:
            if arm == "late" or (arm == "partial" and c == 2):
                continue
            out = cdir / arm
            out.mkdir()
            f = fails[arm][rows].astype(np.uint8)
            weight = (_mix(9, CHUNKS * CHUNK_SHOTS)[rows] % np.uint64(17)).astype(np.float64)
            np.save(out / "logical_failure.npy", f)
            np.save(out / "success.npy", np.ones(CHUNK_SHOTS, np.uint8))
            np.save(out / "iterations.npy", (weight + 1).astype(np.uint32))
            np.save(out / "legs.npy", np.ones(CHUNK_SHOTS, np.uint32))
            np.save(out / "decode_ns.npy", (weight * 1000 + 1).astype(np.uint64))
            np.save(out / "weight.npy", weight)
            (out / "run.json").write_text(json.dumps({"summary": {"shots": CHUNK_SHOTS}}))
            done = {"combined": {"shots": CHUNK_SHOTS, "failures": int(f.sum()), "converged": CHUNK_SHOTS},
                    "elapsed_seconds": 2.0 + 0.25 * c}
            (out / "done.json").write_text(json.dumps(done))
    return directory


def _pair(reference: str, test: str, width: float) -> dict:
    return {"reference": reference, "test": test, "rel_halfwidth": width}


# The single-pair and null configurations recorded in GOLDEN.
SINGLE_CASES = {
    "null": {"stop": {"max_shots": 100_000, "min_failures": 150, "failure_arm": "ref"}},
    "null_wall": {"stop": {"max_shots": None, "max_wall_seconds": 30.0, "min_failures": 400, "failure_arm": "ref"}},
    "close_met": {"stop": {"max_shots": 100_000, "min_failures": 1000, "failure_arm": "ref", "paired": _pair("ref", "close", 0.25)}},
    "close_unmet": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "close", 0.05)}},
    "far": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "far", 0.25)}},
    "far_reversed": {"stop": {"max_shots": 100_000, "paired": _pair("far", "ref", 0.5)}},
    "never": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "never", 0.25)}},
    "never_reference": {"stop": {"max_shots": 100_000, "paired": _pair("never", "ref", 0.25)}},
    "same": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "same", 0.25)}},
    "partial": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "partial", 0.3)}},
    "late": {"stop": {"max_shots": 100_000, "paired": _pair("ref", "late", 0.25)}},
    "with_pairs": {"stop": {"max_shots": 8000, "paired": _pair("ref", "close", 0.25)},
                   "pairs": [{"reference": "ref", "test": "far"}, {"reference": "ref", "test": "close"}]},
    "pairs_only": {"stop": {"max_shots": 8000}, "pairs": [{"reference": "far", "test": "close"}]},
}


def observe(directory: Path) -> dict:
    """What the driver reports about the stopping criteria and pairs of a campaign, as JSON."""
    campaign = cp.load_campaign(directory)
    summary = cp.summarize(directory, bootstrap=20)
    paired = [{k: v for k, v in p.items() if k != "bootstrap"} for p in summary["paired"]]
    out = {"evaluate_stop": cp.evaluate_stop(campaign), "status_stop": cp.status(directory)["stop"],
           "pairs": [list(p) for p in campaign.config.pairs], "summary_paired": paired}
    return json.loads(json.dumps(cp.strict_json(out), default=cp._json_default))


def golden_observations(root: Path) -> dict:
    return {name: observe(build_campaign(root, name, case["stop"], case.get("pairs"))) for name, case in SINGLE_CASES.items()}


# ---------------------------------------------------------------------------------------------
# Single pair and null: unchanged


def test_fixture_counts_are_the_designed_ones():
    f = failure_patterns()
    ref = f["ref"]
    assert 150 < ref.sum() < 260 and (ref & f["close"]).sum() > 100 and (f["close"] & ~ref).sum() > 10
    assert not f["never"].any() and np.array_equal(f["same"], ref)


def test_single_pair_and_null_match_the_golden_record(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(SINGLE_CASES)
    assert golden_observations(tmp_path) == golden


def test_golden_record_covers_every_branch():
    golden = json.loads(GOLDEN.read_text())
    reasons = {g["evaluate_stop"]["criteria"]["paired"].get("reason") for g in golden.values() if "paired" in g["evaluate_stop"]["criteria"]}
    # (The guard "V is 0 or unbounded" cannot fire once both arms have failed: V and its upper
    # limit are then finite and positive for every count pattern of paired_ratio.)
    assert reasons == {None, "no chunk decoded by both arms", "V is undefined until both arms have failed"}
    met = {name: g["evaluate_stop"]["met"] for name, g in golden.items()}
    assert met["close_met"] == ["max_shots", "paired"] or met["close_met"] == ["paired"]
    assert "paired" not in met["close_unmet"] and met["null"] == ["min_failures"]
    assert golden["same"]["evaluate_stop"]["criteria"]["paired"]["method"] == "poisson-zero-discordant"


# ---------------------------------------------------------------------------------------------
# List of pairs


def _parsed(paired, pairs=None):
    config = {"name": "p", "circuit": {"code": "bb18", "experiment": "choi", "p": 0.004, "rounds": 3, "relay_bp_compat": False},
              "mode": "xyz", "seed": 1, "chunk_shots": 10, "arms": {a: _arm() for a in ARMS},
              "stop": {"max_shots": 100, "max_wall_seconds": None, "min_failures": 10, "failure_arm": "ref", "paired": paired},
              "keep_raw": False}
    if pairs is not None:
        config["pairs"] = pairs
    return cp.parse_config(config)


def test_list_parsing():
    cfg = _parsed([_pair("ref", "close", 0.25), _pair("ref", "far", 1)], pairs=[{"reference": "ref", "test": "far"}, {"reference": "far", "test": "close"}])
    assert cfg.stop.paired == (cp.PairedStop("ref", "close", 0.25), cp.PairedStop("ref", "far", 1.0))
    assert isinstance(cfg.stop.paired[1].rel_halfwidth, float)
    # Listed pairs are summarized: the ones not named in "pairs" come first, in list order.
    assert cfg.pairs == (("ref", "close"), ("ref", "far"), ("far", "close"))
    assert cfg.stop.failure_arm == "ref" and cfg.stop.min_failures == 10
    one = _parsed([_pair("ref", "close", 0.25)])
    assert one.stop.paired == (cp.PairedStop("ref", "close", 0.25),) and one.pairs == (("ref", "close"),)
    # The object form and null are unchanged.
    assert _parsed(_pair("ref", "close", 0.25)).stop.paired == cp.PairedStop("ref", "close", 0.25)
    assert _parsed(None).stop.paired is None and _parsed(None).pairs == ()


@pytest.mark.parametrize(
    ("paired", "match"),
    [
        ([], "empty list"),
        ([_pair("ref", "ref", 0.25)], r"stop.paired\[0\] must name two different arms"),
        ([_pair("ref", "close", 0.25), _pair("ref", "nope", 0.25)], r"stop.paired\[1\] must name two different arms"),
        ([_pair("ref", "close", 0.25), _pair("ref", "close", 0.5)], r"stop.paired\[1\] lists the pair of stop.paired\[0\] twice"),
        ([_pair("ref", "close", 0.25), _pair("close", "ref", 0.25)], "again reversed"),
        ([_pair("ref", "close", 0)], "finite number > 0"),
        ([_pair("ref", "close", -0.1)], "finite number > 0"),
        ([_pair("ref", "close", float("inf"))], "finite number > 0"),
        ([_pair("ref", "close", float("nan"))], "finite number > 0"),
        ([_pair("ref", "close", True)], "rel_halfwidth must be"),
        ([_pair("ref", "close", "0.25")], "rel_halfwidth must be"),
        ([{"reference": "ref", "test": "close"}], "lacks required fields"),
        ([{**_pair("ref", "close", 0.25), "extra": 1}], "unknown fields"),
        (["ref/close"], "must be a JSON object"),
        ([{"reference": 1, "test": "close", "rel_halfwidth": 0.25}], "reference must be"),
    ],
)
def test_list_validation(paired, match):
    with pytest.raises(cp.CampaignError, match=match):
        _parsed(paired)


def test_list_holds_only_when_every_pair_holds(tmp_path):
    singles = {}
    for name, pair in (("s_close", _pair("ref", "close", 0.25)), ("s_far", _pair("ref", "far", 0.15)),
                       ("s_far_wide", _pair("ref", "far", 0.25))):
        singles[name] = cp.evaluate_stop(cp.load_campaign(build_campaign(tmp_path, name, {"max_shots": 100_000, "paired": pair})))
    assert singles["s_close"]["met"] == ["paired"] and singles["s_far"]["met"] == [] and singles["s_far_wide"]["met"] == ["paired"]

    # One pair unmet: the list is unmet, and each entry is exactly the single-pair record.
    d = build_campaign(tmp_path, "l_unmet", {"max_shots": 100_000, "paired": [_pair("ref", "close", 0.25), _pair("ref", "far", 0.15)]})
    ev = cp.evaluate_stop(cp.load_campaign(d))
    crit = ev["criteria"]["paired"]
    assert ev["met"] == [] and crit["met"] is False and crit["pairs_met"] == 1 and crit["unmet"] == ["ref/far"]
    assert crit["pairs"] == [singles["s_close"]["criteria"]["paired"], singles["s_far"]["criteria"]["paired"]]

    # Every pair within its own width: the list holds.
    d = build_campaign(tmp_path, "l_met", {"max_shots": 100_000, "paired": [_pair("ref", "close", 0.25), _pair("ref", "far", 0.25)]})
    ev = cp.evaluate_stop(cp.load_campaign(d))
    assert ev["met"] == ["paired"] and ev["criteria"]["paired"]["unmet"] == [] and ev["criteria"]["paired"]["pairs_met"] == 2
    assert ev["criteria"]["paired"]["pairs"][1] == singles["s_far_wide"]["criteria"]["paired"]

    # Guards of the single pair apply per entry: no common chunk, an arm without failures.
    d = build_campaign(tmp_path, "l_guards", {"max_shots": 100_000, "paired": [_pair("ref", "close", 0.25), _pair("ref", "late", 0.25),
                                                                               _pair("ref", "never", 0.25), _pair("ref", "partial", 0.3)]})
    crit = cp.evaluate_stop(cp.load_campaign(d))["criteria"]["paired"]
    assert crit["met"] is False and crit["unmet"] == ["ref/late", "ref/never"]
    assert crit["pairs"][1]["reason"] == "no chunk decoded by both arms" and crit["pairs"][1]["shots"] == 0
    assert crit["pairs"][2]["reason"] == "V is undefined until both arms have failed"
    assert crit["pairs"][3]["shots"] == 3 * CHUNK_SHOTS  # common chunks only
    status = cp.status(d)
    json.dumps(cp.strict_json(status), allow_nan=False, default=cp._json_default)


def test_list_and_min_failures_whichever_comes_first(tmp_path):
    n_ref = int(failure_patterns()["ref"].sum())
    both = [_pair("ref", "close", 0.25), _pair("ref", "far", 0.25)]
    cases = {
        "paired_first": ({"min_failures": n_ref + 1, "failure_arm": "ref", "paired": both}, ["paired"]),
        "failures_first": ({"min_failures": n_ref, "failure_arm": "ref", "paired": [_pair("ref", "far", 0.15), _pair("ref", "close", 0.25)]},
                           ["min_failures"]),
        "both": ({"min_failures": 10, "failure_arm": "ref", "paired": both}, ["min_failures", "paired"]),
        "neither": ({"min_failures": n_ref + 1, "failure_arm": "ref", "paired": [_pair("ref", "far", 0.15)]}, []),
    }
    for name, (stop, met) in cases.items():
        ev = cp.evaluate_stop(cp.load_campaign(build_campaign(tmp_path, name, {"max_shots": 100_000, **stop})))
        assert ev["met"] == met, name
        assert ev["criteria"]["min_failures"] == {"arm": "ref", "value": n_ref, "threshold": stop["min_failures"], "met": "min_failures" in met}


def test_summary_reports_the_listed_pairs(tmp_path):
    d = build_campaign(tmp_path, "l_summary", {"max_shots": 8000, "paired": [_pair("ref", "close", 0.25), _pair("far", "close", 0.5)]},
                       pairs=[{"reference": "ref", "test": "far"}])
    summary = cp.summarize(d, bootstrap=20)
    assert [(p["reference"], p["test"]) for p in summary["paired"]] == [("ref", "close"), ("far", "close"), ("ref", "far")]
    assert summary["config"]["stop"]["paired"][1] == _pair("far", "close", 0.5)
    written = json.loads((d / "summary.json").read_text())
    assert written["paired"][0]["n11"] == summary["paired"][0]["n11"]


# ---------------------------------------------------------------------------------------------
# End to end with the stand-in decoder (init exports artifacts: needs the reference group)


def _driver_helpers():
    pytest.importorskip("relay_bp", reason="campaign init exports artifacts; needs `uv sync --group reference`")
    import test_campaign as tc

    return tc


@pytest.fixture
def fake_binary(tmp_path) -> Path:
    tc = _driver_helpers()
    path = tmp_path / "fake_rtd_decode"
    path.write_text(tc.FAKE_DECODER.format(python=__import__("sys").executable))
    path.chmod(0o755)
    return path


def _all_within(directory: Path, pairs: list[dict], chunks: list[int]) -> bool:
    """The list criterion recomputed from the arm records on the given chunks only."""
    from rtd import stats

    for p in pairs:
        r = records.load(directory, p["reference"], names=("logical_failure",), chunks=chunks)["logical_failure"]
        t = records.load(directory, p["test"], names=("logical_failure",), chunks=chunks)["logical_failure"]
        n11, n10, n01, _ = stats.paired_counts(r, t)
        if n11 + n10 == 0 or n11 + n01 == 0:
            return False
        v = stats.paired_ratio(n11, n10, n01, cp.WILSON_Z)
        if not (np.isfinite(v.value) and v.value > 0 and np.isfinite(v.hi)):
            return False
        if max(v.hi / v.value - 1, 1 - v.lo / v.value) > p["rel_halfwidth"]:
            return False
    return True


def test_run_stops_at_the_first_chunk_where_every_pair_holds(tmp_path, fake_binary):
    tc = _driver_helpers()
    arms = {"ref": tc._fake_arm(tmp_path, "ref", 3), "t2": tc._fake_arm(tmp_path, "t2", 2), "t3": tc._fake_arm(tmp_path, "t3", 3, 1)}
    listed = [_pair("ref", "t2", 0.2), _pair("ref", "t3", 0.3)]
    config = tc._config("stop_list", chunk_shots=100, max_shots=40_000, arms=arms, binary=fake_binary, min_failures=100_000, failure_arm="ref",
                        paired=listed)
    directory = tc._init(tmp_path, config)
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: paired"
    chunks = records.completed_chunks(directory, "ref")
    assert _all_within(directory, listed, chunks) and not _all_within(directory, listed, chunks[:-1])
    crit = result["criteria"]["paired"]
    assert crit["met"] and [(e["reference"], e["test"]) for e in crit["pairs"]] == [("ref", "t2"), ("ref", "t3")]
    assert all(e["rel_halfwidth"] <= e["threshold"] for e in crit["pairs"])
    evaluated = tc._log_events(directory, "stopping criteria evaluated")
    assert [e["evaluation"]["criteria"]["paired"]["met"] for e in evaluated][-2:] == [False, True]


def test_run_with_a_list_stops_at_min_failures_when_that_comes_first(tmp_path, fake_binary):
    tc = _driver_helpers()
    arms = {"ref": tc._fake_arm(tmp_path, "ref", 3), "t2": tc._fake_arm(tmp_path, "t2", 2)}
    config = tc._config("stop_list_fail", chunk_shots=100, max_shots=40_000, arms=arms, binary=fake_binary, min_failures=30, failure_arm="ref",
                        paired=[_pair("ref", "t2", 0.01)])
    result = cp.run_campaign(tc._init(tmp_path, config))
    assert result["reason"] == "criteria met: min_failures" and result["criteria"]["paired"]["met"] is False


def test_old_style_campaign_resumes_and_can_switch_to_a_list(tmp_path, fake_binary):
    tc = _driver_helpers()
    arms = {"ref": tc._fake_arm(tmp_path, "ref", 3), "t2": tc._fake_arm(tmp_path, "t2", 2), "t3": tc._fake_arm(tmp_path, "t3", 3, 1)}
    # The config form every campaign initialized so far has: one pair object.
    old = tc._config("old_style", chunk_shots=100, max_shots=40_000, arms=arms, binary=fake_binary, min_failures=100_000, failure_arm="ref",
                     paired=_pair("ref", "t2", 0.2))
    directory = tc._init(tmp_path, old)
    written = (directory / "campaign.json").read_bytes()
    assert cp.run_campaign(directory, max_chunks=2)["reason"] == "--chunks reached"
    ev = cp.evaluate_stop(cp.load_campaign(directory))
    assert set(ev["criteria"]["paired"]) >= {"reference", "test", "threshold", "shots", "met"} and "pairs" not in ev["criteria"]["paired"]
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: paired" and result["criteria"]["paired"]["test"] == "t2"
    assert (directory / "campaign.json").read_bytes() == written
    stopped_at = records.completed_chunks(directory, "ref")

    # Changing the stop to a list is a compatible update; t3 is resolved later than t2, so the
    # run continues, and a list already met when a later run starts stops it at once.
    new = copy.deepcopy(old)
    new["stop"]["paired"] = [_pair("ref", "t2", 0.2), _pair("ref", "t3", 0.15)]
    tc._init(tmp_path, new, "new.json")
    assert json.loads((directory / "campaign.json").read_text())["stop"]["paired"] == new["stop"]["paired"]
    assert len(list(directory.glob("campaign.*.json"))) == 1
    result = cp.run_campaign(directory)
    assert result["reason"] == "criteria met: paired"
    after = records.completed_chunks(directory, "ref")
    assert len(after) > len(stopped_at) and _all_within(directory, new["stop"]["paired"], after)
    assert not _all_within(directory, new["stop"]["paired"], after[:-1])
    assert cp.run_campaign(directory)["chunks_processed"] == 0


def test_list_criterion_stops_a_catch_up_only_when_newly_met(tmp_path, fake_binary):
    tc = _driver_helpers()
    arms = {"ref": tc._fake_arm(tmp_path, "ref", 3), "t2": tc._fake_arm(tmp_path, "t2", 2), "t3": tc._fake_arm(tmp_path, "t3", 3, 1)}
    listed = [_pair("ref", "t2", 0.2), _pair("ref", "t3", 0.3)]
    config = tc._config("catch_up", chunk_shots=300, max_shots=9000, arms=arms, binary=fake_binary, paired=listed)
    directory = tc._init(tmp_path, config)
    # Sample every chunk with the reference and t2 only: the list cannot hold without t3.
    assert cp.run_campaign(directory, arms=["ref", "t2"])["reason"] == "criteria met: max_shots"
    total = records.completed_chunks(directory, "ref")
    assert len(total) == 30 and records.completed_chunks(directory, "t3") == []
    # Catching t3 up, the list becomes met after some chunk: that stops the run there.
    result = cp.run_campaign(directory, arms=["t3"])
    assert result["reason"] == "criteria met: paired"
    done_t3 = records.completed_chunks(directory, "t3")
    assert 1 <= len(done_t3) < 30 and _all_within(directory, listed, done_t3)
    assert not _all_within(directory, listed, done_t3[:-1])
    assert tc._log_events(directory, "stopping criteria evaluated after a resumed chunk")[-1]["newly_met"] == ["paired"]
    # Already met at the start, it no longer stops the catch-up of an added arm.
    grown = copy.deepcopy(config)
    grown["arms"]["t5"] = tc._fake_arm(tmp_path, "t5", 5)
    tc._init(tmp_path, grown, "grown.json")
    assert cp.run_campaign(directory, arms=["t5"])["reason"] == "criteria met: max_shots, paired"
    assert records.completed_chunks(directory, "t5") == total
