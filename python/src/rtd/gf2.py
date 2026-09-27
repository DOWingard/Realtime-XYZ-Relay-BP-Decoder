"""Dense linear algebra over GF(2) on uint8 numpy arrays."""

from __future__ import annotations

import numpy as np


def row_reduce(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the nonzero rows of the reduced row echelon form and the pivot columns."""
    m = (np.asarray(mat, dtype=np.uint8) & 1).copy()
    rows, cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        candidates = np.flatnonzero(m[r:, c])
        if candidates.size == 0:
            continue
        p = r + int(candidates[0])
        if p != r:
            m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        others = others[others != r]
        m[others] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def rank(mat: np.ndarray) -> int:
    return len(row_reduce(mat)[1])


def nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis of {v : mat @ v = 0 mod 2}, one vector per row."""
    rref, pivots = row_reduce(mat)
    n = np.asarray(mat).shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, pc in enumerate(pivots):
            basis[i, pc] = rref[r, f]
    return basis


def inverse(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.uint8) & 1
    n = m.shape[0]
    if m.shape != (n, n):
        raise ValueError(f"inverse needs a square matrix, got shape {m.shape}")
    rref, pivots = row_reduce(np.hstack([m, np.eye(n, dtype=np.uint8)]))
    if pivots[:n] != list(range(n)) or len(rref) < n:
        raise ValueError("matrix is singular over GF(2)")
    return rref[:n, n:].copy()


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) @ np.asarray(b, dtype=np.int64) % 2).astype(np.uint8)
