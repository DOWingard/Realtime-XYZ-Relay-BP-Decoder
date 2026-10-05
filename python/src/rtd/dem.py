"""Stim detector error model -> the decoding problem (H, A, priors, time structure).

A detector error model lists independent fault mechanisms; mechanism j occurs with probability
p_j and flips a set of detectors and a set of logical observables. Column j of the check matrix
H in F_2^{m x n} marks the detectors it flips (m detectors), column j of A in F_2^{k x n} the
observables it flips (k observables). A shot's syndrome is sigma = H e and its logical flip is
l = A e for the fault vector e in F_2^n; a decoder estimates e from sigma and predicts A e.

Every `error(p) targets` instruction of the flattened model (repeat blocks unrolled, detector
shifts applied) becomes one column, in model order. A model built with decompose_errors=True
separates the components of a composite mechanism with `^`; the mechanism flips the symmetric
difference of its components' targets, which is what is recorded. The models this project
decodes are built with decompose_errors=False, where stim merges mechanisms with identical
targets, so every column is distinct.

Mechanisms with p <= t or p >= 1 - t (t = prune_threshold, 0 by default) are decided and are
removed: a p = 0 fault never happens; a p = 1 fault always happens, so its detector and
observable flips are folded into a syndrome bias b_sigma and an observable bias b_l. The pruned
problem then decodes sigma XOR b_sigma and predicts (A e) XOR b_l. This is the rule of relay_bp's
CheckMatrices.prune_decided_errors, so the column map is identical to the one it produces.

The time structure is the round of every detector, rho(i), read from the detector coordinates
with `round_of`: an int selects a coordinate (1 for this project's circuits, whose coordinates
are (check, round, type)), a callable maps the coordinate tuple to the round. Sliding-window
decoding needs rounds 1 ... Rt with the same number of detectors in each round, in order.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

logger = logging.getLogger("rtd.dem")

RoundOf = int | Callable[[Sequence[float]], int]


@dataclass(frozen=True)
class DecodingProblem:
    """H and A in CSR form (uint32), the priors (float64), and what the conversion recorded.

    col_to_dem[j] is the index among the model's error mechanisms of column j. det_round is None
    unless a round function was given. The biases are None unless a certain (p = 1) mechanism
    was pruned.
    """

    num_detectors: int
    num_columns: int
    num_observables: int
    h_indptr: np.ndarray
    h_indices: np.ndarray
    a_indptr: np.ndarray
    a_indices: np.ndarray
    priors: np.ndarray
    col_to_dem: np.ndarray
    det_round: np.ndarray | None
    syndrome_bias: np.ndarray | None
    observables_bias: np.ndarray | None
    columns_in_model: int
    pruned_p0: int
    pruned_p1: int
    prune_threshold: float
    decomposed: bool

    def h_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix(
            (np.ones(self.h_indices.size, dtype=np.uint8), self.h_indices, self.h_indptr),
            shape=(self.num_detectors, self.num_columns),
        )

    def a_matrix(self) -> sp.csr_matrix:
        return sp.csr_matrix(
            (np.ones(self.a_indices.size, dtype=np.uint8), self.a_indices, self.a_indptr),
            shape=(self.num_observables, self.num_columns),
        )

    def summary(self) -> dict[str, object]:
        """Sizes and conversion facts, for logs and manifests."""
        return {
            "detectors": self.num_detectors,
            "columns": self.num_columns,
            "observables": self.num_observables,
            "nnz_H": int(self.h_indices.size),
            "nnz_A": int(self.a_indices.size),
            "columns_in_model": self.columns_in_model,
            "pruned_p0": self.pruned_p0,
            "pruned_p1": self.pruned_p1,
            "prune_threshold": self.prune_threshold,
            "decomposed": self.decomposed,
            "rounds": None if self.det_round is None else int(self.det_round.max(initial=0)),
        }


def parse_dem(dem: stim.DetectorErrorModel) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray, bool]:
    """H, A and the priors straight from the model, one column per error mechanism in model order.

    Returns (H, A, priors, decomposed), where decomposed says whether any mechanism had `^`
    separated components (whose targets were combined by symmetric difference).
    """
    det_rows: list[int] = []
    det_cols: list[int] = []
    obs_rows: list[int] = []
    obs_cols: list[int] = []
    priors: list[float] = []
    decomposed = False
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        j = len(priors)
        dets: list[int] = []
        obs: list[int] = []
        separated = False
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
            elif t.is_separator():
                separated = True
        if separated or len(set(dets)) != len(dets) or len(set(obs)) != len(obs):
            decomposed = decomposed or separated
            dets = _odd(dets)
            obs = _odd(obs)
        det_rows.extend(dets)
        det_cols.extend([j] * len(dets))
        obs_rows.extend(obs)
        obs_cols.extend([j] * len(obs))
        priors.append(inst.args_copy()[0])
    n = len(priors)
    h = sp.csc_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (det_rows, det_cols)), shape=(dem.num_detectors, n)
    )
    a = sp.csc_matrix(
        (np.ones(len(obs_rows), dtype=np.uint8), (obs_rows, obs_cols)), shape=(dem.num_observables, n)
    )
    return h, a, np.array(priors, dtype=np.float64), decomposed


def _odd(values: list[int]) -> list[int]:
    """The values that occur an odd number of times: the symmetric difference of the components."""
    seen: set[int] = set()
    for v in values:
        seen ^= {v}
    return sorted(seen)


def to_csr(m: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """CSR (indptr, indices) of a 0/1 matrix as uint32, indices strictly increasing in each row."""
    csr = sp.csr_matrix(m, dtype=np.uint8)
    csr.eliminate_zeros()
    csr.sum_duplicates()
    csr.sort_indices()
    if csr.nnz and not np.all(csr.data == 1):
        raise ValueError("matrix has entries other than 0/1")
    if csr.nnz >= 2**32 or max(csr.shape) >= 2**32:
        raise ValueError(f"matrix too large for uint32 indices: shape {csr.shape}, nnz {csr.nnz}")
    return csr.indptr.astype(np.uint32), csr.indices.astype(np.uint32)


def detector_rounds(dem: stim.DetectorErrorModel, round_of: RoundOf) -> np.ndarray:
    """rho(i) for every detector, from its coordinates, as int32."""
    coords = dem.get_detector_coordinates()
    m = dem.num_detectors
    rounds = np.empty(m, dtype=np.int32)
    for i in range(m):
        c = coords.get(i)
        if c is None or len(c) == 0:
            raise ValueError(f"detector {i} has no coordinates, so its round cannot be read")
        if isinstance(round_of, int):
            if not -len(c) <= round_of < len(c):
                raise ValueError(f"detector {i} has coordinates {list(c)}; there is no coordinate {round_of}")
            value = c[round_of]
        else:
            value = round_of(c)
        if value != int(value):
            raise ValueError(f"detector {i}: round {value} (from coordinates {list(c)}) is not an integer")
        rounds[i] = int(value)
    return rounds


def check_time_structure(det_round: np.ndarray) -> tuple[int, int]:
    """(Rt, M) when the detectors come in rounds 1 ... Rt of M detectors each, in order.

    This is what sliding-window decoding requires; a ValueError says what is wrong otherwise.
    """
    m = det_round.size
    if m == 0:
        raise ValueError("the model has no detectors")
    if det_round[0] != 1:
        raise ValueError(
            f"the first detector's round is {int(det_round[0])}; rounds must be numbered from 1 "
            "(pass a round_of that adds the offset)"
        )
    if np.any(np.diff(det_round) < 0):
        first = int(np.flatnonzero(np.diff(det_round) < 0)[0]) + 1
        raise ValueError(f"detectors are not grouped by round: detector {first} goes back to round {int(det_round[first])}")
    rounds_total = int(det_round[-1])
    counts = np.bincount(det_round, minlength=rounds_total + 1)[1:]
    if np.any(counts == 0) or np.any(counts != counts[0]):
        raise ValueError(
            f"rounds must hold equal numbers of detectors; found counts {counts[:8].tolist()}"
            f"{' ...' if counts.size > 8 else ''} over {rounds_total} rounds"
        )
    return rounds_total, int(counts[0])


def problem_from_dem(
    dem: stim.DetectorErrorModel,
    *,
    round_of: RoundOf | None = None,
    prune_threshold: float = 0.0,
) -> DecodingProblem:
    """The decoding problem of a detector error model (see the module docstring)."""
    if not 0.0 <= prune_threshold < 0.5:
        raise ValueError(f"prune_threshold must be in [0, 0.5), got {prune_threshold}")
    t0 = time.perf_counter()
    h, a, p, decomposed = parse_dem(dem)
    parse_seconds = time.perf_counter() - t0
    if decomposed:
        logger.warning(
            "the detector error model has decomposed mechanisms (built with decompose_errors=True); "
            "their components are combined, but stim merges mechanisms per decomposition, so the "
            "columns differ from the undecomposed model (decompose_errors=False)",
            extra={"errors": dem.num_errors},
        )
    to_zero = p <= 0.0 + prune_threshold
    to_one = p >= 1.0 - prune_threshold
    keep = np.flatnonzero(~(to_zero | to_one))
    syndrome_bias = observables_bias = None
    if to_one.any():
        certain = np.flatnonzero(to_one)
        syndrome_bias = (np.asarray(h[:, certain].sum(axis=1)).reshape(-1) % 2).astype(np.uint8)
        observables_bias = (np.asarray(a[:, certain].sum(axis=1)).reshape(-1) % 2).astype(np.uint8)
    if to_zero.any() or to_one.any():
        logger.warning(
            "decided error mechanisms pruned",
            extra={"pruned_p0": int(to_zero.sum()), "pruned_p1": int(to_one.sum()), "threshold": prune_threshold},
        )
    h_indptr, h_indices = to_csr(h[:, keep])
    a_indptr, a_indices = to_csr(a[:, keep])
    det_round = detector_rounds(dem, round_of) if round_of is not None else None
    problem = DecodingProblem(
        num_detectors=dem.num_detectors,
        num_columns=int(keep.size),
        num_observables=dem.num_observables,
        h_indptr=h_indptr,
        h_indices=h_indices,
        a_indptr=a_indptr,
        a_indices=a_indices,
        priors=np.ascontiguousarray(p[keep], dtype=np.float64),
        col_to_dem=keep.astype(np.int64),
        det_round=det_round,
        syndrome_bias=syndrome_bias,
        observables_bias=observables_bias,
        columns_in_model=int(p.size),
        pruned_p0=int(to_zero.sum()),
        pruned_p1=int(to_one.sum()),
        prune_threshold=prune_threshold,
        decomposed=decomposed,
    )
    empty_columns = int(np.sum(np.diff(sp.csc_matrix(h[:, keep]).indptr) == 0))
    if empty_columns:
        logger.warning(
            "error mechanisms that flip no detector: no syndrome can reveal them",
            extra={"columns": empty_columns},
        )
    logger.info(
        "detector error model converted",
        extra={**problem.summary(), "parse_seconds": parse_seconds, "seconds": time.perf_counter() - t0},
    )
    return problem


def problem_from_artifact(directory: str | Path, *, rounds: bool = True) -> DecodingProblem:
    """The decoding problem stored in an artifact directory written by rtd-export."""
    d = Path(directory)
    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{d} is not an artifact directory (no manifest.json)")
    manifest = json.loads(manifest_path.read_text())

    def load(name: str, dtype: type) -> np.ndarray:
        array = np.load(d / name)
        if array.dtype != dtype:
            raise ValueError(f"{d / name}: dtype {array.dtype}, expected {np.dtype(dtype)}")
        return np.ascontiguousarray(array)

    def optional(name: str) -> np.ndarray | None:
        return load(name, np.uint8) if (d / name).is_file() else None

    pruning = manifest.get("pruning", {})
    col_to_dem = np.load(d / "col_to_dem.npy") if (d / "col_to_dem.npy").is_file() else np.arange(manifest["num_columns"])
    problem = DecodingProblem(
        num_detectors=int(manifest["num_detectors"]),
        num_columns=int(manifest["num_columns"]),
        num_observables=int(manifest["num_observables"]),
        h_indptr=load("H_indptr.npy", np.uint32),
        h_indices=load("H_indices.npy", np.uint32),
        a_indptr=load("A_indptr.npy", np.uint32),
        a_indices=load("A_indices.npy", np.uint32),
        priors=load("priors.npy", np.float64),
        col_to_dem=col_to_dem.astype(np.int64),
        det_round=load("det_round.npy", np.int32) if rounds else None,
        syndrome_bias=optional("syndrome_bias.npy"),
        observables_bias=optional("observables_bias.npy"),
        columns_in_model=int(pruning.get("columns_in_model", manifest["num_columns"])),
        pruned_p0=0,
        pruned_p1=0,
        prune_threshold=float(pruning.get("threshold", 0.0)),
        decomposed=False,
    )
    logger.debug("artifact loaded", extra={"artifact": str(d), **problem.summary()})
    return problem
