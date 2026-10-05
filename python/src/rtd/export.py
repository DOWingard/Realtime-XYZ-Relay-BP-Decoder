"""Export the decoding problem (H, A, priors) of a circuit as an artifact for the C++ decoder.

The matrices come from rtd.dem's direct conversion of the circuit's undecomposed detector error
model: one column per error mechanism, in model order, with decided (p = 0 or p = 1) mechanisms
pruned by relay_bp's rule. When IBM's relay_bp is installed, its CheckMatrices.from_dem is run on
the same model as a cross-check and must agree exactly (it is the conversion relay_bp decodes
with). from_dem is called with its default decomposed_hyperedges=None: that tries a decomposed
(graphlike) conversion and falls back to the undecomposed one when the model has hyperedges;
the comparison would catch the decomposed branch.

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
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import stim

from rtd import dem as dem_conversion
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
    h, a, p, _ = dem_conversion.parse_dem(dem)
    return h, a, p


def _same_binary(x: sp.spmatrix, y: sp.spmatrix) -> bool:
    return x.shape == y.shape and (sp.csc_matrix(x, dtype=np.int16) != sp.csc_matrix(y, dtype=np.int16)).nnz == 0


def _cross_check_relay_bp(dem: stim.DetectorErrorModel, problem: dem_conversion.DecodingProblem) -> bool:
    """Compares the conversion with relay_bp's from_dem; False when relay_bp is not installed."""
    try:
        from relay_bp.stim.sinter.check_matrices import CheckMatrices
    except ImportError:
        logger.warning(
            "relay_bp is not installed; the conversion is not cross-checked against its from_dem "
            "(install the 'reference' dependency group to enable the check)"
        )
        return False
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
    cm = raw.prune_decided_errors(threshold=problem.prune_threshold)
    ref_h = dem_conversion.to_csr(cm.check_matrix)
    ref_a = dem_conversion.to_csr(cm.observables_matrix)
    same = (
        all(np.array_equal(x, y) for x, y in zip(ref_h, (problem.h_indptr, problem.h_indices), strict=True))
        and all(np.array_equal(x, y) for x, y in zip(ref_a, (problem.a_indptr, problem.a_indices), strict=True))
        and np.array_equal(np.asarray(cm.error_priors, dtype=np.float64), problem.priors)
    )
    for ours, theirs in ((problem.syndrome_bias, cm.syndrome_bias), (problem.observables_bias, cm.observables_bias)):
        if (ours is None) != (theirs is None) or (
            ours is not None and not np.array_equal(ours, np.asarray(theirs, dtype=np.uint8).reshape(-1))
        ):
            same = False
    if not same:
        raise RuntimeError("the pruned conversion differs from relay_bp's prune_decided_errors")
    logger.info("conversion verified against relay_bp's from_dem and prune_decided_errors")
    return True


def _degree_stats(degrees: np.ndarray) -> dict[str, float]:
    return {
        "min": int(degrees.min()),
        "max": int(degrees.max()),
        "mean": float(degrees.mean()),
        "zero": int(np.sum(degrees == 0)),
    }


def _optional_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _relay_bp_source() -> dict[str, str | None]:
    dist = distribution("relay-bp")
    commit = None
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return {"version": dist.version, "commit": commit}


def export_artifact(circuit_path: Path, out: Path, prune_threshold: float = 0.0) -> dict:
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

    problem = dem_conversion.problem_from_dem(dem, prune_threshold=prune_threshold)
    cross_checked = _cross_check_relay_bp(dem, problem)

    h_indptr, h_indices = problem.h_indptr, problem.h_indices
    a_indptr, a_indices = problem.a_indptr, problem.a_indices
    m, n = problem.num_detectors, problem.num_columns
    row_degree = np.diff(h_indptr.astype(np.int64))
    col_degree = np.bincount(h_indices, minlength=n)
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
        "priors.npy": problem.priors,
        "col_to_dem.npy": problem.col_to_dem,
        "det_check.npy": coord_array[:, 0].astype(np.int32),
        "det_round.npy": coord_array[:, 1].astype(np.int32),
        "det_type.npy": coord_array[:, 2].astype(np.uint8),
    }
    if problem.syndrome_bias is not None:
        files["syndrome_bias.npy"] = problem.syndrome_bias
    if problem.observables_bias is not None:
        files["observables_bias.npy"] = problem.observables_bias
    for name, array in files.items():
        np.save(out / name, array)

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "source_circuit": {"path": str(circuit_path), "sha256": circuit_sha},
        "num_detectors": m,
        "num_columns": n,
        "num_observables": problem.num_observables,
        "nnz_H": int(h_indices.size),
        "nnz_A": int(a_indices.size),
        "row_degree": _degree_stats(row_degree),
        "column_degree": _degree_stats(col_degree),
        "prior_range": [float(problem.priors.min()), float(problem.priors.max())],
        "pruning": {
            "threshold": prune_threshold,
            "columns_in_model": problem.columns_in_model,
            "columns_pruned": problem.pruned_p0 + problem.pruned_p1,
        },
        "conversion": {"parser": "rtd.dem.problem_from_dem", "relay_bp_cross_check": cross_checked},
        "from_dem": {"decomposed_hyperedges": None, "prune_decided_errors": True} if cross_checked else None,
        "versions": {
            "relay_bp": _relay_bp_source() if cross_checked else None,
            "beliefmatching": _optional_version("beliefmatching"),
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
    except Exception:
        logger.exception("run failed", extra={"cli_args": cli_args})
        return 1
    logger.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
