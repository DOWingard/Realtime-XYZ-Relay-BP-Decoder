import json

import numpy as np
import pytest

pytest.importorskip("relay_bp", reason="needs `uv sync --group reference`")

from rtd.bb_code import get_code  # noqa: E402
from rtd.circuit import EXPERIMENTS, build_memory_circuit  # noqa: E402
from rtd.export import export_artifact  # noqa: E402
from rtd.noise import NoiseModel  # noqa: E402


def _csr_matvec_mod2(indptr: np.ndarray, indices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Row parities of M @ v mod 2 for each row vector v in `vectors` (shape [shots, columns])."""
    out = np.zeros((vectors.shape[0], len(indptr) - 1), dtype=np.uint8)
    for r in range(len(indptr) - 1):
        cols = indices[indptr[r] : indptr[r + 1]]
        out[:, r] = np.bitwise_xor.reduce(vectors[:, cols], axis=1) if cols.size else 0
    return out


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("name", ["bb18", "bb72"])
def test_artifact_reproduces_sampled_syndromes(tmp_path, name, experiment):
    code = get_code(name)
    mc = build_memory_circuit(code, code.distance, experiment, NoiseModel.uniform(3e-3))
    circuit_path = tmp_path / "circuit.stim"
    mc.circuit.to_file(circuit_path)
    art = tmp_path / "artifact"
    manifest = export_artifact(circuit_path, art)

    h_indptr, h_indices = np.load(art / "H_indptr.npy"), np.load(art / "H_indices.npy")
    a_indptr, a_indices = np.load(art / "A_indptr.npy"), np.load(art / "A_indices.npy")
    priors, col_to_dem = np.load(art / "priors.npy"), np.load(art / "col_to_dem.npy")
    m, n = manifest["num_detectors"], manifest["num_columns"]

    assert h_indptr.dtype == h_indices.dtype == np.uint32
    assert (h_indptr.size, h_indices.size, priors.size) == (m + 1, manifest["nnz_H"], n)
    for r in range(m):
        row = h_indices[h_indptr[r] : h_indptr[r + 1]]
        assert np.all(np.diff(row.astype(np.int64)) > 0)
    assert np.all((priors > 0) & (priors < 0.5))
    assert np.array_equal(np.load(art / "det_round.npy"), mc.detectors.round)
    assert np.array_equal(np.load(art / "det_type.npy"), mc.detectors.kind)

    # A syndrome is H e and an observable flip is A e for the fault vector e that produced them.
    dem = mc.circuit.detector_error_model(decompose_errors=False)
    dets, obs, errs = dem.compile_sampler(seed=7).sample(500, return_errors=True)
    e = errs[:, col_to_dem].astype(np.uint8)
    assert np.array_equal(_csr_matvec_mod2(h_indptr, h_indices, e), dets.astype(np.uint8))
    assert np.array_equal(_csr_matvec_mod2(a_indptr, a_indices, e), obs.astype(np.uint8))

    written = json.loads((art / "manifest.json").read_text())
    assert written["versions"]["relay_bp"]["commit"] is not None
