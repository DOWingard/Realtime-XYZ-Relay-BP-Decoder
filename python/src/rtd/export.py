"""Export the decoding problem (H, A, priors) of a circuit as an artifact for the C++ decoder.

The matrices come from relay_bp's CheckMatrices.from_dem on the circuit's
undecomposed detector error model, so the C++ decoder sees exactly the problem
relay_bp decodes. from_dem is called with its default decomposed_hyperedges=None:
that tries a decomposed (graphlike) conversion and falls back to the
undecomposed one when the model has hyperedges. Passing False explicitly makes
it return the decomposed matrices instead. Because of that silent fallback,
the result is always checked against a direct parse of the error model:
one column per error mechanism, in model order.

Artifact directory contents (rows = detectors, columns = error mechanisms):
    H_indptr.npy, H_indices.npy   uint32  CSR of H, column indices ascending within each row
    A_indptr.npy, A_indices.npy   uint32  CSR of the observable matrix A
    priors.npy                    float64 error probability of each column
    col_to_dem.npy                int64   index of each column among the model's error mechanisms
    syndrome_bias.npy             uint8   present only if pruning removed p = 1 columns
    observables_bias.npy          uint8   likewise
    det_check.npy, det_round.npy, det_type.npy   detector metadata from the circuit's coordinates
    manifest.json                 sizes, degree statistics, versions, source circuit checksum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

from rtd import log

logger = logging.getLogger("rtd.export")

ARTIFACT_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dem(dem: stim.DetectorErrorModel) -> tuple[sp.csc_matrix, sp.csc_matrix, np.ndarray]:
    """H, A and priors straight from the model: one column per error mechanism, in model order."""
    det_rows, det_cols, obs_rows, obs_cols, priors = [], [], [], [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        j = len(priors)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det_rows.append(t.val)
                det_cols.append(j)
            elif t.is_logical_observable_id():
                obs_rows.append(t.val)
                obs_cols.append(j)
        priors.append(inst.args_copy()[0])
    n = len(priors)
    h = sp.csc_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (det_rows, det_cols)), shape=(dem.num_detectors, n)
    )
    a = sp.csc_matrix(
        (np.ones(len(obs_rows), dtype=np.uint8), (obs_rows, obs_cols)), shape=(dem.num_observables, n)
    )
    return h, a, np.array(priors, dtype=np.float64)


def _same_binary(x: sp.spmatrix, y: sp.spmatrix) -> bool:
    return x.shape == y.shape and (sp.csc_matrix(x, dtype=np.int16) != sp.csc_matrix(y, dtype=np.int16)).nnz == 0


def _to_csr(m: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    csr = sp.csr_matrix(m, dtype=np.uint8)
    csr.eliminate_zeros()
    csr.sum_duplicates()
    csr.sort_indices()
    if csr.nnz and not np.all(csr.data == 1):
        raise ValueError("matrix has entries other than 0/1")
    if csr.nnz >= 2**32 or max(csr.shape) >= 2**32:
        raise ValueError(f"matrix too large for uint32 indices: shape {csr.shape}, nnz {csr.nnz}")
    return csr.indptr.astype(np.uint32), csr.indices.astype(np.uint32)


def _degree_stats(degrees: np.ndarray) -> dict[str, float]:
    return {
        "min": int(degrees.min()),
        "max": int(degrees.max()),
        "mean": float(degrees.mean()),
        "zero": int(np.sum(degrees == 0)),
    }


def _relay_bp_source() -> dict[str, str | None]:
    dist = distribution("relay-bp")
    commit = None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return {"version": dist.version, "commit": commit}


def export_artifact(circuit_path: Path, out: Path, prune_threshold: float = 0.0) -> dict:
    from relay_bp.stim.sinter.check_matrices import CheckMatrices

    circuit = stim.Circuit.from_file(circuit_path)
    circuit_sha = _sha256(circuit_path)
    logger.info(
        "circuit loaded",
        extra={
            "circuit": str(circuit_path),
            "sha256": circuit_sha,
            "detectors": circuit.num_detectors,
            "observables": circuit.num_observables,
        },
    )

    t0 = time.perf_counter()
    dem = circuit.detector_error_model(decompose_errors=False)
    logger.info("detector error model built", extra={"errors": dem.num_errors, "seconds": time.perf_counter() - t0})

    t0 = time.perf_counter()
    raw = CheckMatrices.from_dem(dem, prune_decided_errors=False)
    logger.info("from_dem finished", extra={"columns": raw.check_matrix.shape[1], "seconds": time.perf_counter() - t0})

    h_direct, a_direct, p_direct = parse_dem(dem)
    if not (
        _same_binary(raw.check_matrix, h_direct)
        and _same_binary(raw.observables_matrix, a_direct)
        and np.array_equal(raw.error_priors, p_direct)
    ):
        raise RuntimeError(
            "from_dem output is not the undecomposed model in model order "
            f"(got H {raw.check_matrix.shape}, expected {h_direct.shape}); "
            "it may have taken the decomposed (graphlike) branch"
        )
    logger.info("from_dem output verified against direct parse of the error model")

    # Same selection rule as CheckMatrices.prune_decided_errors, so the column map is exact.
    p_raw = raw.error_priors
    decided = (p_raw <= 0.0 + prune_threshold) | (p_raw >= 1.0 - prune_threshold)
    col_to_dem = np.flatnonzero(~decided).astype(np.int64)
    cm = raw.prune_decided_errors(threshold=prune_threshold)
    if not (
        _same_binary(cm.check_matrix, raw.check_matrix[:, col_to_dem])
        and np.array_equal(cm.error_priors, p_raw[col_to_dem])
    ):
        raise RuntimeError("pruned matrices do not match the expected kept columns")
    if decided.any():
        logger.warning(
            "decided error mechanisms pruned",
            extra={
                "pruned_p0": int(np.sum(p_raw <= 0.0 + prune_threshold)),
                "pruned_p1": int(np.sum(p_raw >= 1.0 - prune_threshold)),
                "threshold": prune_threshold,
            },
        )

    h_indptr, h_indices = _to_csr(cm.check_matrix)
    a_indptr, a_indices = _to_csr(cm.observables_matrix)
    m, n = cm.check_matrix.shape
    row_degree = np.diff(h_indptr.astype(np.int64))
    col_degree = np.diff(sp.csc_matrix(cm.check_matrix).indptr)
    if row_degree.min() == 0 or col_degree.min() == 0:
        logger.warning(
            "empty rows or columns in H",
            extra={"empty_rows": int(np.sum(row_degree == 0)), "empty_columns": int(np.sum(col_degree == 0))},
        )

    coords = circuit.get_detector_coordinates()
    if len(coords) != m or any(len(coords[i]) != 3 for i in range(m)):
        raise ValueError("detector coordinates must be (check, round, type) for every detector")
    coord_array = np.array([coords[i] for i in range(m)])

    out.mkdir(parents=True, exist_ok=True)
    files = {
        "H_indptr.npy": h_indptr,
        "H_indices.npy": h_indices,
        "A_indptr.npy": a_indptr,
        "A_indices.npy": a_indices,
        "priors.npy": np.ascontiguousarray(cm.error_priors, dtype=np.float64),
        "col_to_dem.npy": col_to_dem,
        "det_check.npy": coord_array[:, 0].astype(np.int32),
        "det_round.npy": coord_array[:, 1].astype(np.int32),
        "det_type.npy": coord_array[:, 2].astype(np.uint8),
    }
    if cm.syndrome_bias is not None:
        files["syndrome_bias.npy"] = np.asarray(cm.syndrome_bias, dtype=np.uint8).reshape(-1)
    if cm.observables_bias is not None:
        files["observables_bias.npy"] = np.asarray(cm.observables_bias, dtype=np.uint8).reshape(-1)
    for name, array in files.items():
        np.save(out / name, array)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "source_circuit": {"path": str(circuit_path), "sha256": circuit_sha},
        "num_detectors": m,
        "num_columns": n,
        "num_observables": int(cm.observables_matrix.shape[0]),
        "nnz_H": int(h_indices.size),
        "nnz_A": int(a_indices.size),
        "row_degree": _degree_stats(row_degree),
        "column_degree": _degree_stats(col_degree),
        "prior_range": [float(cm.error_priors.min()), float(cm.error_priors.max())],
        "pruning": {
            "threshold": prune_threshold,
            "columns_in_model": int(p_raw.size),
            "columns_pruned": int(decided.sum()),
        },
        "from_dem": {"decomposed_hyperedges": None, "prune_decided_errors": True},
        "versions": {
            "relay_bp": _relay_bp_source(),
            "beliefmatching": version("beliefmatching"),
            "stim": stim.__version__,
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "rtd": version("rtd"),
        },
        "sha256": {name: _sha256(out / name) for name in sorted(files)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("artifact written", extra={"out": str(out), "rows": m, "columns": n, "nnz_H": manifest["nnz_H"]})
    return manifest


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", type=Path, required=True, help="circuit.stim written by rtd-sample")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help="prune columns with p <= t or p >= 1 - t (relay_bp's rule; 0 prunes only p = 0 and p = 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("run started", extra={"cli_args": cli_args})
    try:
        if not args.circuit.is_file():
            raise FileNotFoundError(f"circuit file not found: {args.circuit}")
        if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{args.out} is not empty; pass --overwrite to replace it")
        if not 0.0 <= args.prune_threshold < 0.5:
            raise ValueError(f"--prune-threshold must be in [0, 0.5), got {args.prune_threshold}")
        export_artifact(args.circuit, args.out, args.prune_threshold)
    except ImportError:
        logger.exception(
            "relay_bp is not installed; run `uv sync --group reference`", extra={"cli_args": cli_args}
        )
        return 1
    except Exception:
        logger.exception("run failed", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
