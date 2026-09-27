import numpy as np
import pytest

from rtd import gf2
from rtd.bb_code import CODE_PARAMS, Monomial, get_code, parse_polynomial

EXPECTED_NK = {"bb18": (18, 4), "bb72": (72, 12), "gross": (144, 12), "two_gross": (288, 12)}


def test_parse_polynomial_keeps_term_order():
    assert parse_polynomial("x^3+y+y^2") == (Monomial(3, 0), Monomial(0, 1), Monomial(0, 2))
    assert parse_polynomial("x+1+xy^2") == (Monomial(1, 0), Monomial(0, 0), Monomial(1, 2))


def test_parse_polynomial_rejects_garbage():
    with pytest.raises(ValueError):
        parse_polynomial("x^3+z")


@pytest.mark.parametrize("name", sorted(CODE_PARAMS))
def test_code_parameters_and_logicals(name):
    code = get_code(name)
    assert (code.n, code.k) == EXPECTED_NK[name]
    assert not gf2.matmul(code.hx, code.hz.T).any()
    # Z logicals commute with X checks and vice versa; the pairs are conjugate.
    assert not gf2.matmul(code.hx, code.lz.T).any()
    assert not gf2.matmul(code.hz, code.lx.T).any()
    assert np.array_equal(gf2.matmul(code.lx, code.lz.T), np.eye(code.k, dtype=np.uint8))
    # Logicals are not stabilizers.
    assert gf2.rank(np.vstack([code.hz, code.lz])) == gf2.rank(code.hz) + code.k


def test_check_neighbors_match_parity_check_rows():
    code = get_code("gross")
    for i in range(code.half):
        x_support = sorted(code.x_check_neighbor(i, d) for d in range(6))
        z_support = sorted(code.z_check_neighbor(i, d) for d in range(6))
        assert x_support == np.flatnonzero(code.hx[i]).tolist()
        assert z_support == np.flatnonzero(code.hz[i]).tolist()
