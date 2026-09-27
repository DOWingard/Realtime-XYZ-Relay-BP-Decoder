import hashlib
import json
import math

import numpy as np
import pytest

pytest.importorskip("relay_bp", reason="needs `uv sync --group reference`")

from rtd.golden import (  # noqa: E402
    RelayLegs,
    _parse_args,
    check_relay_legs,
    main,
    manifest_path,
    parse_relay_log,
    validate_args,
)

N_BITS = 15
CIRCUIT_SHA = "0" * 64

MIN_SUM = ["--decoder", "min_sum", "--alpha", "none", "--alpha-scaling", "1.0", "--gamma0", "none", "--max-iter", "2"]
RELAY = [
    "--decoder", "relay", "--alpha", "none", "--alpha-scaling", "1.0", "--gamma0", "0.125",
    "--pre-iter", "2", "--num-sets", "4", "--set-max-iter", "3", "--stopping", "nconv", "--stop-nconv", "2",
    "--gamma-rows", "3", "--gamma-seed", "11", "--gamma-interval", "-0.24", "0.66",
]  # fmt: skip


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repetition_h() -> np.ndarray:
    """Checks i = e_i + e_{i+1}: a chain Tanner graph with N_BITS - 1 detectors."""
    h = np.zeros((N_BITS - 1, N_BITS), dtype=np.uint8)
    for i in range(N_BITS - 1):
        h[i, i] = h[i, i + 1] = 1
    return h


@pytest.fixture
def problem(tmp_path):
    h = _repetition_h()
    priors = np.random.default_rng(3).uniform(0.02, 0.2, size=N_BITS)
    art = tmp_path / "artifact"
    art.mkdir()
    np.save(art / "H_indptr.npy", np.concatenate([[0], np.cumsum(h.sum(axis=1))]).astype(np.uint32))
    np.save(art / "H_indices.npy", np.concatenate([np.flatnonzero(row) for row in h]).astype(np.uint32))
    np.save(art / "priors.npy", priors)
    names = ("H_indptr.npy", "H_indices.npy", "priors.npy")
    (art / "manifest.json").write_text(
        json.dumps(
            {
                "num_detectors": h.shape[0],
                "num_columns": N_BITS,
                "source_circuit": {"sha256": CIRCUIT_SHA},
                "sha256": {name: _sha(art / name) for name in names},
            }
        )
    )

    # Random sparse errors, plus a long error string that BP cannot resolve in 2 iterations.
    errors = (np.random.default_rng(4).random((10, N_BITS)) < priors).astype(np.uint8)
    errors[0] = 0
    errors[1, 3:11] = 1
    shots = tmp_path / "shots"
    shots.mkdir()
    np.save(shots / "detectors.npy", ((errors.astype(np.int64) @ h.T.astype(np.int64)) % 2).astype(np.uint8))
    np.save(shots / "observables.npy", np.zeros((errors.shape[0], 1), dtype=np.uint8))
    (shots / "manifest.json").write_text(
        json.dumps({"sha256": {"detectors.npy": _sha(shots / "detectors.npy"), "circuit.stim": CIRCUIT_SHA}})
    )
    return {"artifact": art, "shots": shots, "h": h, "priors": priors, "tmp": tmp_path}


def _run(problem, name: str, algorithm: list[str], source: list[str], *extra: str) -> tuple[dict, dict]:
    out = problem["tmp"] / name
    argv = ["--artifact", str(problem["artifact"]), "--out", str(out), "--float", "f32", *algorithm, *source, *extra]
    assert main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    arrays = {p.name: np.load(p) for p in out.glob("*.npy")}
    for file_name, digest in manifest["sha256"].items():
        assert _sha(out / file_name) == digest
    assert set(manifest["sha256"]) == set(arrays)
    return arrays, manifest


def _check_common(problem, arrays: dict, manifest: dict, posterior_shots: int) -> None:
    h, priors = problem["h"], problem["priors"]
    s, m = arrays["detectors.npy"].shape
    assert m == h.shape[0] and manifest["num_shots"] == s
    expected = {
        "detectors.npy": (np.uint8, (s, m)),
        "decoding.npy": (np.uint8, (s, N_BITS)),
        "success.npy": (np.uint8, (s,)),
        "iterations.npy": (np.int64, (s,)),
        "weight.npy": (np.float64, (s,)),
    }
    if posterior_shots:
        expected["posterior.npy"] = (np.float64, (posterior_shots, N_BITS))
    for name, (dtype, shape) in expected.items():
        assert arrays[name].dtype == dtype and arrays[name].shape == shape, name

    decoding, success = arrays["decoding.npy"], arrays["success.npy"].astype(bool)
    reproduced = (decoding.astype(np.int64) @ h.T.astype(np.int64)) % 2
    assert np.array_equal(np.all(reproduced == arrays["detectors.npy"], axis=1), success)

    lam = [math.log((1.0 - p) / p) for p in priors.tolist()]
    for row, ok, w in zip(decoding, success, arrays["weight.npy"]):
        if ok:
            total = 0.0
            for j in range(N_BITS):
                if row[j]:
                    total += lam[j]
            assert w == total
        else:
            assert w == math.inf
    assert manifest["summary"]["converged"] == int(success.sum())


def test_parse_relay_log():
    text = (
        "# pre_iter: 20: sets: 40 set_max_iter: 15\n"
        "# gamma_distribution: (-0.24, 0.66) # set_idx, num_iter, converged, unique_best_solution\n"
        "-1, 20, 0, 0\n"
        "0, 15, 0, 0\n"
        "1, 7, 1, 1\n"
    )
    legs = parse_relay_log(text)
    assert legs.iterations.tolist() == [20, 15, 7] and legs.iterations.dtype == np.int64
    assert legs.converged.tolist() == [0, 0, 1] and legs.converged.dtype == np.uint8
    assert legs.unique_best.tolist() == [0, 0, 1]

    header = text.split("-1,")[0]
    with pytest.raises(ValueError, match="second decode"):
        parse_relay_log(text + "-1, 3, 1, 1\n")
    with pytest.raises(ValueError, match="expected -1"):
        parse_relay_log(header + "0, 15, 0, 0\n")
    with pytest.raises(ValueError, match="expected 'set_idx"):
        parse_relay_log(header + "-1, 20, 2, 0\n")
    with pytest.raises(ValueError, match="no '#' header"):
        parse_relay_log("-1, 20, 1, 1\n")
    with pytest.raises(ValueError, match="no leg records"):
        parse_relay_log(header)


def test_check_relay_legs_rejects_inconsistent_records():
    params = {"pre_iter": 20, "num_sets": 40, "set_max_iter": 15, "stopping_criterion": "nconv", "stop_nconv": 1}
    legs = RelayLegs(np.array([20, 15, 7]), np.array([0, 0, 1], np.uint8), np.array([0, 0, 1], np.uint8))
    check_relay_legs(legs, True, 42, params)
    with pytest.raises(RuntimeError, match="sum to 42"):
        check_relay_legs(legs, True, 41, params)
    with pytest.raises(RuntimeError, match="success is False"):
        check_relay_legs(legs, False, 42, params)
    with pytest.raises(RuntimeError, match="stopping 'all'"):
        check_relay_legs(legs, True, 42, {**params, "stopping_criterion": "all"})


def test_min_sum_golden(problem):
    source = ["--shots", str(problem["shots"]), "--first", "0", "--count", "10"]
    arrays, manifest = _run(problem, "min_sum", MIN_SUM, source, "--posterior-shots", "4")
    _check_common(problem, arrays, manifest, posterior_shots=4)
    success = arrays["success.npy"]
    assert success[0] == 1 and arrays["iterations.npy"][0] == 1 and arrays["weight.npy"][0] == 0.0
    assert success[1] == 0 and arrays["iterations.npy"][1] == 2
    assert not any(name.startswith("leg") or name == "gammas.npy" for name in arrays)
    assert manifest["relay_bp_class"] == "MinSumBPDecoderF32"
    assert manifest["config"]["max_iter"] == 2 and manifest["config"]["alpha"] is None
    assert manifest["syndromes"]["type"] == "shots" and manifest["syndromes"]["count"] == 10


def test_relay_golden(problem):
    source = ["--shots", str(problem["shots"]), "--first", "1", "--count", "8"]
    arrays, manifest = _run(problem, "relay", RELAY, source, "--posterior-shots", "8")
    _check_common(problem, arrays, manifest, posterior_shots=8)

    gammas = np.random.default_rng(11).uniform(-0.24, 0.66, size=(3, N_BITS))
    assert arrays["gammas.npy"].dtype == np.float64 and np.array_equal(arrays["gammas.npy"], gammas)
    ptr = arrays["legs_ptr.npy"]
    assert ptr.dtype == np.int64 and ptr.shape == (9,) and ptr[0] == 0 and np.all(np.diff(ptr) >= 1)
    total_legs = int(ptr[-1])
    for name, dtype in (("leg_iterations.npy", np.int64), ("leg_converged.npy", np.uint8), ("leg_unique_best.npy", np.uint8)):
        assert arrays[name].dtype == dtype and arrays[name].shape == (total_legs,)
    for s in range(8):
        legs = slice(ptr[s], ptr[s + 1])
        assert arrays["leg_iterations.npy"][legs].sum() == arrays["iterations.npy"][s]
        assert arrays["leg_converged.npy"][legs].any() == arrays["success.npy"][s]
    # The long error string (shot 1, the first decoded here) needs relay legs beyond leg 0.
    assert ptr[1] > 1
    assert manifest["config"]["stopping_criterion"] == "nconv" and manifest["config"]["logging"] is True
    table = manifest["config"]["gamma_table"]
    assert (table["rows"], table["seed"], table["interval"]) == (3, 11, [-0.24, 0.66])

    parallel, _ = _run(problem, "relay_parallel", RELAY, source, "--posterior-shots", "8", "--workers", "2")
    assert parallel.keys() == arrays.keys()
    for name in arrays:
        assert np.array_equal(parallel[name], arrays[name]), name


def test_single_columns_golden(problem):
    arrays, manifest = _run(problem, "single", MIN_SUM, ["--single-columns", "6", "--seed", "5"])
    _check_common(problem, arrays, manifest, posterior_shots=0)
    columns = np.random.default_rng(5).choice(N_BITS, size=6, replace=False)
    assert arrays["columns.npy"].dtype == np.int64 and np.array_equal(arrays["columns.npy"], columns)
    detectors = arrays["detectors.npy"]
    assert detectors.shape[0] == 7 and not detectors[0].any()
    assert np.array_equal(detectors[1:], problem["h"][:, columns].T)
    assert manifest["syndromes"] == {"type": "single_columns", "count": 6, "seed": 5, "columns": columns.tolist()}


def test_manifest_paths_are_relative_or_bare_names(problem, monkeypatch):
    inside = problem["tmp"] / "work"
    (inside / "sub").mkdir(parents=True)
    monkeypatch.chdir(inside)
    assert manifest_path(inside / "sub") == "sub"
    assert manifest_path(inside / "sub" / ".." / "sub") == "sub"
    assert manifest_path(problem["artifact"]) == "artifact"

    monkeypatch.chdir(problem["tmp"])
    source = ["--shots", "shots", "--first", "0", "--count", "2"]
    argv = ["--artifact", "artifact", "--out", "rel", "--float", "f32", *MIN_SUM, *source]
    assert main(argv) == 0
    manifest = json.loads((problem["tmp"] / "rel" / "manifest.json").read_text())
    assert manifest["artifact"]["path"] == "artifact" and manifest["syndromes"]["path"] == "shots"
    assert str(problem["tmp"]) not in json.dumps(manifest)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--decoder", "min_sum", "--alpha", "none", "--alpha-scaling", "1", "--gamma0", "none"], "--max-iter required"),
        (["--decoder", "relay", "--alpha", "none", "--alpha-scaling", "1", "--gamma0", "0.1"], "--pre-iter, --num-sets"),
        ([*MIN_SUM, "--pre-iter", "3"], "--pre-iter only valid with --decoder relay"),
        ([*RELAY[:-3], "--gamma-interval", "0.5", "0.5"], "LO < HI"),
    ],
)
def test_validation_names_the_offending_flag(argv, message):
    args = _parse_args(["--artifact", "a", "--out", "o", "--float", "f32", "--single-columns", "3", "--seed", "1", *argv])
    with pytest.raises(ValueError, match=message):
        validate_args(args)


def test_shots_source_requires_first_and_count():
    args = _parse_args(["--artifact", "a", "--out", "o", "--float", "f64", *MIN_SUM, "--shots", "s", "--count", "3"])
    with pytest.raises(ValueError, match="--first required with --shots"):
        validate_args(args)
