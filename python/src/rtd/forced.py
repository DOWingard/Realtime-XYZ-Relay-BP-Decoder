"""Forced decoding problems: the observables appended to the checks, so that every solution lies in a
given logical class.

A correction e explains the syndrome when H e = sigma (mod 2), and its logical class is A e. Stacking
H' = [H; A] and sigma' = [sigma; L] makes H' e = sigma' mean "explains sigma AND has class L", so
any solution a decoder finds for (H', sigma') is a correction of class L. For the true class L of a
sampled shot this asks whether the decoder, pointed at the right class, finds a correction lighter
than the one it chose on its own. Each appended row touches every fault that flips that observable
(hundreds to thousands of columns, against tens for a detector), which can slow or disturb belief
propagation; a control on shots the decoder got right tells whether it does.

The forced artifact is an ordinary exported artifact (same columns and priors, k more rows), so the
unchanged rtd_decode decodes it; the shots are the sampled ones with the class bits appended to the
detector bits (`forced_shots`).

    python -m rtd.forced SRC_ARTIFACT OUT_DIR [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rtd import log
from rtd.xz import sha256_file, write_file_atomic

logger = logging.getLogger("rtd.forced")

# det_type of an appended observable row: neither detector type, so no code mistakes it for one.
OBSERVABLE_ROW_TYPE = 255

_ARRAYS = ("H_indptr.npy", "H_indices.npy", "A_indptr.npy", "A_indices.npy", "priors.npy", "det_round.npy",
           "det_type.npy")


class ForcedError(RuntimeError):
    """The source artifact or the shots cannot be extended."""


def stack_rows(h_indptr: np.ndarray, h_indices: np.ndarray, a_indptr: np.ndarray, a_indices: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """CSR of [H; A]: A's rows appended after H's, column indices unchanged."""
    h_indptr = np.asarray(h_indptr)
    a_indptr = np.asarray(a_indptr)
    if h_indptr.size == 0 or a_indptr.size == 0 or h_indptr[0] != 0 or a_indptr[0] != 0:
        raise ForcedError("CSR pointer arrays must be non-empty and start at 0")
    indptr = np.concatenate([h_indptr, a_indptr[1:] + h_indptr[-1]]).astype(h_indptr.dtype)
    indices = np.concatenate([np.asarray(h_indices), np.asarray(a_indices)]).astype(np.asarray(h_indices).dtype)
    return indptr, indices


def forced_shots(detectors: np.ndarray, observables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sigma' = [sigma, L] per shot (the true class bits appended to the syndrome); observables
    unchanged, so a converged forced decode predicts exactly L."""
    detectors = np.asarray(detectors, dtype=np.uint8)
    observables = np.asarray(observables, dtype=np.uint8)
    if detectors.ndim != 2 or observables.ndim != 2 or detectors.shape[0] != observables.shape[0]:
        logger.error("forced shots: incompatible shapes", extra={"detectors": list(detectors.shape),
                                                                 "observables": list(observables.shape)})
        raise ForcedError(f"shapes {detectors.shape} and {observables.shape} are not [S, m] and [S, k]")
    return np.ascontiguousarray(np.concatenate([detectors, observables], axis=1)), np.ascontiguousarray(observables)


def forced_artifact(src: Path, out: Path, overwrite: bool = False) -> dict[str, Any]:
    """Writes the artifact of (H' = [H; A], A, priors): det_round and det_type extended by the k
    appended rows (round = last round + 1, type OBSERVABLE_ROW_TYPE). The source circuit checksum is
    kept, so shots sampled from that circuit (with extended syndromes) are accepted. An existing
    output made from the same source manifest is reused; one made from another source is an error
    unless overwrite."""
    t0 = time.perf_counter()
    src, out = Path(src), Path(out)
    try:
        manifest = json.loads((src / "manifest.json").read_text())
        m, n, k = int(manifest["num_detectors"]), int(manifest["num_columns"]), int(manifest["num_observables"])
        arrays = {name: np.load(src / name, allow_pickle=False) for name in _ARRAYS}
    except (OSError, ValueError, KeyError) as e:
        logger.error("source artifact unreadable", exc_info=True, extra={"src": str(src), "recovery": "none"})
        raise ForcedError(f"{src} is not a readable artifact ({e!r})") from e
    if arrays["H_indptr.npy"].size != m + 1 or arrays["A_indptr.npy"].size != k + 1:
        raise ForcedError(f"{src}: pointer sizes do not match m = {m}, k = {k}")
    if (src / "syndrome_bias.npy").exists():
        raise ForcedError(f"{src} has a syndrome bias; the forced rows would need one too")
    indptr, indices = stack_rows(arrays["H_indptr.npy"], arrays["H_indices.npy"], arrays["A_indptr.npy"],
                                 arrays["A_indices.npy"])
    rounds = arrays["det_round.npy"]
    files = {
        "H_indptr.npy": indptr, "H_indices.npy": indices,
        "A_indptr.npy": arrays["A_indptr.npy"], "A_indices.npy": arrays["A_indices.npy"],
        "priors.npy": arrays["priors.npy"],
        "det_round.npy": np.concatenate([rounds, np.full(k, int(rounds.max()) + 1 if rounds.size else 0, rounds.dtype)]),
        "det_type.npy": np.concatenate([arrays["det_type.npy"],
                                        np.full(k, OBSERVABLE_ROW_TYPE, arrays["det_type.npy"].dtype)]),
    }
    row_degree = np.diff(indptr.astype(np.int64))
    new_manifest = {
        **{key: manifest[key] for key in ("format_version", "source_circuit", "pruning", "from_dem", "versions")
           if key in manifest},
        "created": datetime.now(timezone.utc).isoformat(),
        "num_detectors": m + k, "num_columns": n, "num_observables": k,
        "nnz_H": int(indices.size), "nnz_A": int(arrays["A_indices.npy"].size),
        "row_degree": {"min": int(row_degree.min()), "max": int(row_degree.max()), "mean": float(row_degree.mean()),
                       "zero": int(np.count_nonzero(row_degree == 0))},
        "forced": {"source": str(src), "source_manifest_sha256": sha256_file(src / "manifest.json"),
                   "appended_rows": k, "rule": "H' = [H; A], sigma' = [sigma; L]: every solution has class L",
                   "appended_row_degree": np.diff(arrays["A_indptr.npy"].astype(np.int64)).tolist()},
    }
    if out.exists():
        if not overwrite:
            existing = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").is_file() else {}
            if (existing.get("forced") or {}).get("source_manifest_sha256") == new_manifest["forced"]["source_manifest_sha256"]:
                logger.info("forced artifact reused", extra={"out": str(out)})
                return existing
            logger.error("forced artifact exists from another source", extra={"out": str(out), "src": str(src),
                                                                               "recovery": "none; pass overwrite"})
            raise ForcedError(f"{out} exists and was made from another source; pass overwrite")
        logger.warning("forced artifact overwritten", extra={"out": str(out)})
        shutil.rmtree(out)
    tmp = out.with_name(f".{out.name}.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for name, array in files.items():
            np.save(tmp / name, array)
        new_manifest["sha256"] = {name: sha256_file(tmp / name) for name in sorted(files)}
        write_file_atomic(tmp / "manifest.json", (json.dumps(new_manifest, indent=2) + "\n").encode())
        tmp.rename(out)
    except BaseException:
        logger.error("writing the forced artifact failed", exc_info=True,
                     extra={"out": str(out), "recovery": "partial directory removed"})
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info("forced artifact written", extra={"src": str(src), "out": str(out), "rows": m + k, "columns": n,
                                                  "appended_row_degree_max": int(row_degree[m:].max()) if k else 0,
                                                  "seconds": round(time.perf_counter() - t0, 3)})
    return new_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="exported artifact directory (manifest.json, H, A, priors)")
    parser.add_argument("out", type=Path, help="output directory of the forced artifact")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    log.configure(logging.INFO)
    try:
        manifest = forced_artifact(args.src, args.out, args.overwrite)
    except (ForcedError, OSError, ValueError, KeyError):
        logger.exception("forced artifact failed", extra={"src": str(args.src), "recovery": "none"})
        return 1
    print(json.dumps({"out": str(args.out), "num_detectors": manifest["num_detectors"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
