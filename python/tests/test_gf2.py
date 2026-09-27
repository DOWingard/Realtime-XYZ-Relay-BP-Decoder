import numpy as np
import pytest

from rtd import gf2


def test_rank_and_nullspace():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 2, size=(7, 12), dtype=np.uint8)
    kernel = gf2.nullspace(m)
    assert kernel.shape == (12 - gf2.rank(m), 12)
    assert not gf2.matmul(m, kernel.T).any()
    assert gf2.rank(kernel) == kernel.shape[0]


def test_inverse_round_trip():
    rng = np.random.default_rng(2)
    while True:
        m = rng.integers(0, 2, size=(6, 6), dtype=np.uint8)
        if gf2.rank(m) == 6:
            break
    assert np.array_equal(gf2.matmul(m, gf2.inverse(m)), np.eye(6, dtype=np.uint8))


def test_inverse_rejects_singular():
    with pytest.raises(ValueError):
        gf2.inverse(np.array([[1, 1], [1, 1]], dtype=np.uint8))
