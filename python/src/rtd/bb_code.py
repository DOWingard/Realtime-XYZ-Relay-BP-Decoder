"""Bivariate bicycle (BB) codes.

A BB code is defined by two cyclic groups Z_l and Z_m and two polynomials
A = A1 + A2 + A3 and B = B1 + B2 + B3, each term a monomial x^a y^b with
x = S_l (x) I_m and y = I_l (x) S_m, where S_n is the n x n cyclic shift.
Every term is a permutation matrix, so each check touches exactly one qubit
per term. The parity-check matrices are H_X = [A | B] and H_Z = [B^T | A^T]
over n = 2 l m data qubits (left block, then right block).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from rtd import gf2

_TERM = re.compile(r"^(?:x(?:\^(\d+))?)?(?:y(?:\^(\d+))?)?$")


@dataclass(frozen=True)
class Monomial:
    x: int
    y: int


def parse_polynomial(text: str) -> tuple[Monomial, ...]:
    """Parse e.g. "x^3+y+y^2" or "x+1+xy^2" into its terms, keeping their order.

    Term order matters: the syndrome-extraction schedule refers to A1, A2, A3 by position.
    """
    terms = []
    for raw in text.replace(" ", "").split("+"):
        if raw == "1":
            terms.append(Monomial(0, 0))
            continue
        match = _TERM.match(raw)
        if not raw or match is None:
            raise ValueError(f"cannot parse monomial {raw!r} in polynomial {text!r}")
        has_x, has_y = "x" in raw, "y" in raw
        terms.append(
            Monomial(
                x=int(match.group(1) or 1) if has_x else 0,
                y=int(match.group(2) or 1) if has_y else 0,
            )
        )
    return tuple(terms)


def _shift(n: int, power: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=np.uint8), power % n, axis=1)


@dataclass(frozen=True)
class BBCode:
    name: str
    l: int
    m: int
    a_poly: str
    b_poly: str
    distance: int
    a_terms: tuple[np.ndarray, ...] = field(init=False, repr=False)
    b_terms: tuple[np.ndarray, ...] = field(init=False, repr=False)
    hx: np.ndarray = field(init=False, repr=False)
    hz: np.ndarray = field(init=False, repr=False)
    lx: np.ndarray = field(init=False, repr=False)
    lz: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        def monomial_matrix(t: Monomial) -> np.ndarray:
            return np.kron(_shift(self.l, t.x), _shift(self.m, t.y))

        a_terms = tuple(monomial_matrix(t) for t in parse_polynomial(self.a_poly))
        b_terms = tuple(monomial_matrix(t) for t in parse_polynomial(self.b_poly))
        if len(a_terms) != 3 or len(b_terms) != 3:
            raise ValueError(f"{self.name}: BB schedule needs exactly 3 terms in A and in B")
        a = np.bitwise_xor.reduce(a_terms)
        b = np.bitwise_xor.reduce(b_terms)
        hx = np.hstack([a, b])
        hz = np.hstack([b.T, a.T])
        if gf2.matmul(hx, hz.T).any():
            raise ValueError(f"{self.name}: H_X and H_Z do not commute")
        lx, lz = _paired_logicals(hx, hz)

        object.__setattr__(self, "a_terms", a_terms)
        object.__setattr__(self, "b_terms", b_terms)
        object.__setattr__(self, "hx", hx)
        object.__setattr__(self, "hz", hz)
        object.__setattr__(self, "lx", lx)
        object.__setattr__(self, "lz", lz)

    @property
    def half(self) -> int:
        """Qubits per data block, and checks per type (l * m)."""
        return self.l * self.m

    @property
    def n(self) -> int:
        return 2 * self.half

    @property
    def k(self) -> int:
        return self.lx.shape[0]

    def x_check_neighbor(self, check: int, direction: int) -> int:
        """Data qubit that X check `check` touches through term `direction`.

        Directions 0-2 are A1..A3 into the left block, 3-5 are B1..B3 into the right block.
        """
        if direction < 3:
            return int(np.flatnonzero(self.a_terms[direction][check])[0])
        return self.half + int(np.flatnonzero(self.b_terms[direction - 3][check])[0])

    def z_check_neighbor(self, check: int, direction: int) -> int:
        """Data qubit that Z check `check` touches through term `direction`.

        Directions 0-2 are B1^T..B3^T into the left block, 3-5 are A1^T..A3^T into the right block.
        """
        if direction < 3:
            return int(np.flatnonzero(self.b_terms[direction].T[check])[0])
        return self.half + int(np.flatnonzero(self.a_terms[direction - 3].T[check])[0])


def _independent_logicals(kernel_of: np.ndarray, stabilizers: np.ndarray) -> np.ndarray:
    """Basis of ker(kernel_of) modulo rowspace(stabilizers), chosen greedily in kernel order."""
    chosen: list[np.ndarray] = []
    current_rank = gf2.rank(stabilizers)
    for v in gf2.nullspace(kernel_of):
        trial = np.vstack([stabilizers, *chosen, v])
        r = gf2.rank(trial)
        if r > current_rank:
            chosen.append(v)
            current_rank = r
    return np.array(chosen, dtype=np.uint8)


def _paired_logicals(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logical X and Z operators with X_i anticommuting with Z_j exactly when i == j.

    Z logicals commute with all X checks (kernel of H_X) and are not Z stabilizers;
    X logicals likewise with the roles swapped. The pairing lx @ lz.T = I makes
    (lx[i], lz[i]) the Pauli X and Z of logical qubit i.
    """
    lz = _independent_logicals(hx, hz)
    lx = _independent_logicals(hz, hx)
    if lx.shape[0] != lz.shape[0]:
        raise ValueError(f"logical counts disagree: {lx.shape[0]} X vs {lz.shape[0]} Z")
    lx = gf2.matmul(gf2.inverse(gf2.matmul(lx, lz.T)), lx)
    return lx, lz


CODE_PARAMS: dict[str, dict] = {
    "bb18": dict(l=3, m=3, a_poly="x+1+y", b_poly="x+1+xy^2", distance=3),
    "bb72": dict(l=6, m=6, a_poly="x^3+y+y^2", b_poly="y^3+x+x^2", distance=6),
    "gross": dict(l=12, m=6, a_poly="x^3+y+y^2", b_poly="y^3+x+x^2", distance=12),
    "two_gross": dict(l=12, m=12, a_poly="x^3+y^2+y^7", b_poly="y^3+x+x^2", distance=18),
}


@cache
def get_code(name: str) -> BBCode:
    if name not in CODE_PARAMS:
        raise KeyError(f"unknown code {name!r}; known: {sorted(CODE_PARAMS)}")
    return BBCode(name, **CODE_PARAMS[name])
