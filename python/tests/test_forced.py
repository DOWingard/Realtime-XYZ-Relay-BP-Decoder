"""Forced decoding problems: [H; A] stacking, extended syndromes, and the forced artifact on the toy
repetition-code fixture."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rtd import forced

REPO = Path(__file__).resolve().parents[2]
TOY = REPO / "test" / "fixtures" / "toy_rep3"


def test_stack_rows_and_forced_shots():
    H = sp.random(5, 9, density=0.4, random_state=1, format="csr")
    A = sp.random(2, 9, density=0.5, random_state=2, format="csr")
    indptr, indices = forced.stack_rows(H.indptr, H.indices, A.indptr, A.indices)
    stacked = sp.csr_matrix((np.ones(indices.size), indices, indptr), shape=(7, 9))
    want = sp.vstack([H, A]).tocsr()
    want.data[:] = 1
    assert (stacked != want).nnz == 0
    det, obs = forced.forced_shots(np.array([[1, 0, 1]], np.uint8), np.array([[0, 1]], np.uint8))
    assert det.tolist() == [[1, 0, 1, 0, 1]] and obs.tolist() == [[0, 1]]
    with pytest.raises(forced.ForcedError):
        forced.forced_shots(np.zeros((2, 3), np.uint8), np.zeros((1, 1), np.uint8))
    with pytest.raises(forced.ForcedError):
        forced.stack_rows(np.array([1, 2]), np.array([0]), A.indptr, A.indices)


def test_forced_artifact_on_the_toy(tmp_path: Path):
    out = tmp_path / "toy_forced"
    manifest = forced.forced_artifact(TOY, out)
    m, n, k = 14, 30, 1
    assert manifest["num_detectors"] == m + k and manifest["num_columns"] == n
    H = sp.csr_matrix((np.ones(np.load(out / "H_indices.npy").size), np.load(out / "H_indices.npy"),
                       np.load(out / "H_indptr.npy")), shape=(m + k, n))
    A = sp.csr_matrix((np.ones(np.load(TOY / "A_indices.npy").size), np.load(TOY / "A_indices.npy"),
                       np.load(TOY / "A_indptr.npy")), shape=(k, n))
    # Every e: H' e = [H e; A e], so solving H' e = [sigma; L] fixes the class to L.
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.integers(0, 2, n)
        top = (H @ e) % 2
        assert np.array_equal(top[m:], (A @ e) % 2)
    assert np.load(out / "det_type.npy")[-1] == forced.OBSERVABLE_ROW_TYPE
    assert np.load(out / "det_round.npy")[-1] == np.load(TOY / "det_round.npy").max() + 1
    assert forced.forced_artifact(TOY, out)["forced"]["source_manifest_sha256"] == manifest["forced"]["source_manifest_sha256"]
    other = tmp_path / "other"
    shutil.copytree(TOY, other)
    manifest_file = other / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["created"] = "changed"
    manifest_file.write_text(json.dumps(data))
    with pytest.raises(forced.ForcedError):
        forced.forced_artifact(other, out)
    assert forced.forced_artifact(other, out, overwrite=True)["forced"]["source"] == str(other)


def test_forced_artifact_rejects_unreadable_sources(tmp_path: Path):
    with pytest.raises(forced.ForcedError):
        forced.forced_artifact(tmp_path / "missing", tmp_path / "out")
    assert forced.main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 1
    assert forced.main([str(TOY), str(tmp_path / "cli")]) == 0
    assert (tmp_path / "cli" / "manifest.json").is_file()
