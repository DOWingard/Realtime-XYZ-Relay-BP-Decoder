import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rtd.golden import load_artifact as golden_load_artifact
from rtd.xz import (
    HalfIndex,
    SplitError,
    infer_observable_type,
    merged_prior,
    split_artifact,
    split_problem,
    split_shots,
    split_shots_dir,
)

REPO = Path(__file__).resolve().parents[2]
GROSS_ARTIFACT = REPO / "data" / "artifacts" / "gross_choi_p0.003"
GROSS_SHOTS = REPO / "data" / "shots" / "gross_choi_p0.003"

# Rows alternate X-type (0) and Z-type (1) detectors. Columns (faults):
#   0: {0}        X-row only, flips X observable 0
#   1: {1}        Z-row only, flips Z observable 1
#   2: {0, 1}     both types (a Y-like fault), flips both observables
#   3: {2, 4}     X rows
#   4: {3}        Z row
#   5: {2, 3, 4}  both types
#   6: {5}        Z row
#   7: {2, 4}     X rows, same X support as 3 and 5
DET_TYPE = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
OBS_TYPE = np.array([0, 1], dtype=np.uint8)
COLUMNS = [[0], [1], [0, 1], [2, 4], [3], [2, 3, 4], [5], [2, 4]]
OBS_COLUMNS = [[0], [1], [0, 1], [], [], [], [], []]
PRIORS = np.array([0.01, 0.02, 0.003, 0.004, 0.005, 0.006, 0.007, 0.0081])


def _matrix(columns: list[list[int]], rows: int) -> sp.csr_matrix:
    r = [i for col in columns for i in col]
    c = [j for j, col in enumerate(columns) for _ in col]
    return sp.csr_matrix((np.ones(len(r), dtype=np.uint8), (r, c)), shape=(rows, len(columns)))


def _problem(columns=COLUMNS, obs_columns=OBS_COLUMNS, priors=PRIORS):
    return _matrix(columns, DET_TYPE.size), _matrix(obs_columns, OBS_TYPE.size), np.asarray(priors, dtype=np.float64)


def _dense(indptr: np.ndarray, indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = sp.csr_matrix((np.ones(indices.size, dtype=np.uint8), indices, indptr), shape=shape)
    return m.toarray()


def test_split_groups_by_restricted_support_and_orders_by_representative():
    h, a, p = _problem()
    x = split_problem(h, a, p, DET_TYPE, OBS_TYPE, "x")
    assert x.rows.tolist() == [0, 2, 4]
    assert x.observables.tolist() == [0]
    # X supports: {0} <- 0, 2; {2,4} (local rows 1, 2) <- 3, 5, 7. Z-only columns 1, 4, 6 dropped.
    assert x.members_ptr.tolist() == [0, 2, 5]
    assert x.members.tolist() == [0, 2, 3, 5, 7]
    assert x.representatives.tolist() == [0, 3]
    assert x.dropped_columns == 3 and x.merged_groups == 2 and x.merged_columns == 5
    assert _dense(x.h_indptr, x.h_indices, (3, 2)).tolist() == [[1, 0], [0, 1], [0, 1]]
    assert _dense(x.a_indptr, x.a_indices, (1, 2)).tolist() == [[1, 0]]
    for arr in (x.h_indptr, x.h_indices, x.a_indptr, x.a_indices, x.members_ptr, x.members, x.rows, x.observables):
        assert arr.dtype == np.uint32

    z = split_problem(h, a, p, DET_TYPE, OBS_TYPE, "z")
    assert z.rows.tolist() == [1, 3, 5]
    assert z.observables.tolist() == [1]
    # Z supports: {1} <- 1, 2; {3} <- 4, 5; {5} <- 6.
    assert z.members.tolist() == [1, 2, 4, 5, 6]
    assert z.representatives.tolist() == [1, 4, 6]
    assert z.merged_groups == 2 and z.merged_columns == 4 and z.dropped_columns == 3
    assert _dense(z.h_indptr, z.h_indices, (3, 3)).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_merged_prior_is_the_left_fold_in_ascending_index():
    h, a, p = _problem()
    x = split_problem(h, a, p, DET_TYPE, OBS_TYPE, "x")
    q3, q5, q7 = (float(p[j]) for j in (3, 5, 7))
    acc = q3 * (1.0 - q5) + q5 * (1.0 - q3)
    acc = acc * (1.0 - q7) + q7 * (1.0 - acc)
    assert x.priors[1] == acc  # bitwise: the same operations in the same order
    assert x.priors[0] == float(p[0]) * (1.0 - float(p[2])) + float(p[2]) * (1.0 - float(p[0]))
    assert merged_prior([0.25]) == 0.25
    # Probability of an odd number of events, checked against the product formula.
    qs = [0.1, 0.2, 0.3]
    assert merged_prior(qs) == pytest.approx((1 - np.prod([1 - 2 * q for q in qs])) / 2, rel=1e-15)


def test_pairing_violation_is_an_error():
    # Column 3 flips the X observable but has only Z-type rows: the X half could not see it.
    columns = [[0], [1], [3], [2, 4]]
    obs = [[0], [1], [0], []]
    h, a, p = _problem(columns, obs, [0.01, 0.02, 0.03, 0.04])
    with pytest.raises(SplitError, match="flip a paired observable but have no detector"):
        split_problem(h, a, p, DET_TYPE, OBS_TYPE, "x")
    split_problem(h, a, p, DET_TYPE, OBS_TYPE, "z")  # the Z half is unaffected


def test_merged_members_must_flip_the_same_paired_observables():
    # Columns 0 and 1 share X support {0}, but only column 0 flips the X observable.
    columns = [[0], [0, 1], [3]]
    obs = [[0], [], [1]]
    h, a, p = _problem(columns, obs, [0.01, 0.02, 0.03])
    with pytest.raises(SplitError, match="flip different paired observables"):
        split_problem(h, a, p, DET_TYPE, OBS_TYPE, "x")


def test_bad_inputs():
    h, a, p = _problem()
    with pytest.raises(SplitError, match="unknown half"):
        split_problem(h, a, p, DET_TYPE, OBS_TYPE, "y")
    with pytest.raises(SplitError, match="inconsistent shapes"):
        split_problem(h, a, p[:-1], DET_TYPE, OBS_TYPE, "x")


def test_observable_type_inference():
    h, a, _ = _problem()
    assert infer_observable_type(h, a, DET_TYPE).tolist() == [0, 1]
    # An observable flipped only by faults with rows of both types fits both halves.
    h2, a2, _ = _problem([[0, 1], [2]], [[0], [1]], [0.1, 0.1])
    with pytest.raises(SplitError, match="cannot be inferred"):
        infer_observable_type(h2, a2, DET_TYPE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_xyz_artifact(directory: Path, circuit_sha: str = "c" * 64) -> None:
    h, a, p = _problem()
    directory.mkdir(parents=True)
    hc, ac = sp.csr_matrix(h), sp.csr_matrix(a)
    files = {
        "H_indptr.npy": hc.indptr.astype(np.uint32),
        "H_indices.npy": hc.indices.astype(np.uint32),
        "A_indptr.npy": ac.indptr.astype(np.uint32),
        "A_indices.npy": ac.indices.astype(np.uint32),
        "priors.npy": p,
        "col_to_dem.npy": np.arange(10, 10 + p.size, dtype=np.int64),
        "det_check.npy": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "det_round.npy": np.array([1, 1, 2, 2, 3, 3], dtype=np.int32),
        "det_type.npy": DET_TYPE,
    }
    for name, arr in files.items():
        np.save(directory / name, arr)
    manifest = {
        "format_version": 1,
        "source_circuit": {"path": "circuit.stim", "sha256": circuit_sha},
        "num_detectors": 6,
        "num_columns": p.size,
        "num_observables": 2,
        "nnz_H": int(hc.nnz),
        "nnz_A": int(ac.nnz),
        "versions": {"rtd": "test"},
        "sha256": {name: _sha(directory / name) for name in files},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_split_artifact_writes_loadable_halves(tmp_path):
    parent = tmp_path / "xyz"
    _write_xyz_artifact(parent)
    out = tmp_path / "xyz_xz"
    manifests = split_artifact(parent, out, observable_type=OBS_TYPE)
    assert set(manifests) == {"x", "z"}
    x = manifests["x"]
    assert (x["num_detectors"], x["num_columns"], x["num_observables"], x["nnz_H"]) == (3, 2, 1, 3)
    assert x["source_circuit"]["sha256"] == "c" * 64
    assert x["xz"]["half"] == "x" and x["xz"]["merged_groups"] == 2 and x["xz"]["merged_columns"] == 5
    assert x["xz"]["parent_manifest_sha256"] == _sha(parent / "manifest.json")
    for name, digest in x["sha256"].items():
        assert _sha(out / "x" / name) == digest
    assert np.load(out / "x" / "col_to_dem.npy").tolist() == [10, 13]
    assert np.load(out / "x" / "det_round.npy").tolist() == [1, 2, 3]
    assert np.load(out / "z" / "det_global.npy").tolist() == [1, 3, 5]
    golden = golden_load_artifact(out / "z")  # the reference-decoder tool reads it as an artifact
    assert (golden.num_detectors, golden.num_columns) == (3, 3)
    with pytest.raises(FileExistsError):
        split_artifact(parent, out, observable_type=OBS_TYPE)
    with pytest.raises(SplitError, match="already one half"):
        split_artifact(out / "x", tmp_path / "again", observable_type=OBS_TYPE[:1])


def test_split_artifact_rejects_a_corrupted_parent(tmp_path):
    parent = tmp_path / "xyz"
    _write_xyz_artifact(parent)
    np.save(parent / "priors.npy", PRIORS * 2)
    with pytest.raises(SplitError, match="does not match its manifest checksum"):
        split_artifact(parent, tmp_path / "out", observable_type=OBS_TYPE)
    assert not (tmp_path / "out").exists()


def test_split_shots_and_shots_dir(tmp_path):
    parent = tmp_path / "xyz"
    _write_xyz_artifact(parent)
    split_artifact(parent, tmp_path / "xz", observable_type=OBS_TYPE)
    rng = np.random.default_rng(1)
    dets = rng.integers(0, 2, size=(7, 6), dtype=np.uint8)
    obs = rng.integers(0, 2, size=(7, 2), dtype=np.uint8)
    index = HalfIndex.from_artifact(tmp_path / "xz" / "z")
    d, o = split_shots(dets, obs, index)
    assert np.array_equal(d, dets[:, [1, 3, 5]]) and np.array_equal(o, obs[:, [1]])

    shots = tmp_path / "shots"
    shots.mkdir()
    np.save(shots / "detectors.npy", dets)
    np.save(shots / "observables.npy", obs)
    (shots / "manifest.json").write_text(
        json.dumps(
            {
                "rounds": 2,
                "code": {"k": 1},
                "sha256": {
                    "circuit.stim": "c" * 64,
                    "detectors.npy": _sha(shots / "detectors.npy"),
                    "observables.npy": _sha(shots / "observables.npy"),
                },
            }
        )
    )
    written = split_shots_dir(shots, tmp_path / "xz", tmp_path / "shots_xz", first=2, count=4)
    for half, rows, cols in (("x", [0, 2, 4], [0]), ("z", [1, 3, 5], [1])):
        directory = tmp_path / "shots_xz" / half
        assert np.array_equal(np.load(directory / "detectors.npy"), dets[2:6][:, rows])
        assert np.array_equal(np.load(directory / "observables.npy"), obs[2:6][:, cols])
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest == written[half]
        assert manifest["rounds"] == 2 and manifest["shots"] == 4
        assert manifest["sha256"]["circuit.stim"] == "c" * 64
        assert manifest["sha256"]["detectors.npy"] == _sha(directory / "detectors.npy")
        assert manifest["xz"]["first"] == 2 and manifest["xz"]["count"] == 4

    other = tmp_path / "other"
    _write_xyz_artifact(other, circuit_sha="d" * 64)
    split_artifact(other, tmp_path / "other_xz", observable_type=OBS_TYPE)
    with pytest.raises(SplitError, match="comes from circuit"):
        split_shots_dir(shots, tmp_path / "other_xz", tmp_path / "bad")


def test_real_circuit_halves_reproduce_sampled_syndromes(tmp_path):
    """H_T e_T = σ_T and A_T e_T = observable flips for faults sampled from the error model,
    where e_T[g] is the parity of the faults merged into column g."""
    pytest.importorskip("relay_bp", reason="needs `uv sync --group reference`")
    from rtd.bb_code import get_code
    from rtd.circuit import build_memory_circuit
    from rtd.export import export_artifact
    from rtd.noise import NoiseModel

    code = get_code("bb18")
    mc = build_memory_circuit(code, 3, "choi", NoiseModel.uniform(5e-3), relay_bp_compat=True)
    mc.circuit.to_file(tmp_path / "circuit.stim")
    export_artifact(tmp_path / "circuit.stim", tmp_path / "xyz")
    split_artifact(tmp_path / "xyz", tmp_path / "xz", observable_type=mc.observable_kind)
    inferred = split_artifact(tmp_path / "xyz", tmp_path / "xz_inferred")
    col_to_dem = np.load(tmp_path / "xyz" / "col_to_dem.npy")
    dem = mc.circuit.detector_error_model(decompose_errors=False)
    dets, obs, errs = dem.compile_sampler(seed=3).sample(400, return_errors=True)
    e = errs[:, col_to_dem].astype(np.uint8)
    for half in ("x", "z"):
        d = tmp_path / "xz" / half
        assert json.loads((d / "manifest.json").read_text())["sha256"] == {
            k: v for k, v in inferred[half]["sha256"].items()
        }
        members_ptr, members = np.load(d / "col_members_ptr.npy"), np.load(d / "col_members.npy")
        e_half = np.stack(
            [np.bitwise_xor.reduce(e[:, members[members_ptr[g] : members_ptr[g + 1]]], axis=1) for g in range(members_ptr.size - 1)],
            axis=1,
        )
        m, k = int(np.load(d / "H_indptr.npy").size - 1), int(np.load(d / "A_indptr.npy").size - 1)
        h_half = _dense(np.load(d / "H_indptr.npy"), np.load(d / "H_indices.npy"), (m, e_half.shape[1]))
        a_half = _dense(np.load(d / "A_indptr.npy"), np.load(d / "A_indices.npy"), (k, e_half.shape[1]))
        index = HalfIndex.from_artifact(d)
        d_half, o_half = split_shots(dets.astype(np.uint8), obs.astype(np.uint8), index)
        assert np.array_equal((e_half.astype(np.int64) @ h_half.T) % 2, d_half)
        assert np.array_equal((e_half.astype(np.int64) @ a_half.T) % 2, o_half)


@pytest.mark.skipif(not GROSS_ARTIFACT.is_dir() or not GROSS_SHOTS.is_dir(), reason="gross R = 12 artifact not present")
def test_gross_halves_have_the_measured_size(tmp_path):
    manifests = split_artifact(GROSS_ARTIFACT, tmp_path / "xz", shots=GROSS_SHOTS)
    for half in ("x", "z"):
        m = manifests[half]
        assert (m["num_detectors"], m["num_columns"], m["nnz_H"], m["num_observables"]) == (936, 8784, 30672, 12)
        assert m["xz"]["dropped_columns"] == 8784 and m["xz"]["merged_columns"] == 62496
        assert np.all(np.diff(np.load(tmp_path / "xz" / half / "det_round.npy")) >= 0)
