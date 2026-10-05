"""Windowed goldens: the reference's per-window records with relay_bp as the inner decoder.

A golden directory holds everything the C++ windowed decoder needs to reproduce the decode and
the arrays it must match, named as rtd_decode writes them:
    manifest.json      artifact path and manifest checksum, shots source, first, count, the
                       window object, decoder and gamma-table settings, relay_bp version,
                       summary, and sha256 of every file (plus the source circuit's, so the
                       directory is also accepted as an rtd_decode --shots directory)
    spec.json          the equivalent rtd_decode version-2 spec (gamma_source explicit_shapes)
    gammas/shape_<i>.npy   float64 [T, n_i] = default_rng([seed, i]).uniform(low, high, (T, n_i))
    detectors.npy, observables.npy                   uint8 [S, m], [S, k]
    win_iterations, win_legs, win_unexplained, win_virtual   uint32 [S, K]
    win_attempts, win_converged, win_flagged                 uint8 [S, K]
    win_weight, win_committed_weight                         float64 [S, K]
    commit_ptr uint64 [S K + 1], commit_faults uint32        committed global columns per window
    predicted_observables uint8 [S, k], logical_failure uint8 [S], flagged uint8 [S]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from rtd.window_ref import (
    GlobalProblem,
    PlanError,
    WindowPlan,
    _manifest_path,
    _prepare_out,
    _sha256_file,
    build_plan,
    check_outcomes,
    decode_shots,
    load_problem,
    load_shots_dir,
    outcome_arrays,
    relay_from_args,
    spec_from_args,
)
from rtd.window_ref_inner import InnerSpec, RelayParams, SeededShapeGammas

logger = logging.getLogger("rtd.window_ref")

GOLDEN_FORMAT_VERSION = 1
GOLDEN_ARRAYS = (
    "win_iterations",
    "win_legs",
    "win_attempts",
    "win_converged",
    "win_weight",
    "win_committed_weight",
    "win_unexplained",
    "win_flagged",
    "win_virtual",
    "commit_ptr",
    "commit_faults",
    "predicted_observables",
    "logical_failure",
    "flagged",
)


def write_arrays(out: Path, arrays: dict[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array)
        back = np.load(path, allow_pickle=False)
        if back.dtype != array.dtype or back.shape != array.shape or back.tobytes() != array.tobytes():
            raise RuntimeError(f"{path} does not read back as written")


def commit_sets_partition(plan: WindowPlan) -> bool:
    """Exact boundary: the commit sets of the attempt-0 placements partition the columns."""
    seen = np.zeros(plan.num_columns, dtype=np.int64)
    for p in plan.placements:
        if p.attempt != 0:
            continue
        shape = plan.shapes[p.shape]
        seen[p.columns[shape.commit == 1].astype(np.int64)] += 1
    return bool(np.all(seen == 1))


def rtd_decode_spec(params: RelayParams, window: dict[str, Any]) -> dict[str, Any]:
    """The rtd_decode version-2 spec equivalent to the relay_bp settings (relay_bp's alpha None is
    a constant scale of 1)."""
    if params.alpha is None:
        alpha: dict[str, Any] = {"rule": "constant", "value": 1.0}
    elif params.alpha == 0:
        alpha = {"rule": "adaptive", "scaling": params.alpha_scaling}
    else:
        alpha = {"rule": "constant", "value": params.alpha}
    if params.stopping == "nconv":
        stopping: dict[str, Any] = {"rule": "after_n_converged", "count": params.stop_nconv}
    elif params.stopping == "pre_iter":
        stopping = {"rule": "after_leg0"}
    else:
        stopping = {"rule": "all_legs"}
    return {
        "version": 2,
        "policy": params.float_type,
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "serial"},
        "alpha": alpha,
        "gamma0": params.gamma0,
        "pre_iter": params.pre_iter,
        "set_max_iter": params.set_max_iter,
        "num_sets": params.num_sets,
        "stopping": stopping,
        "gamma_source": {"type": "explicit_shapes", "directory": "gammas"},
        "window": window,
    }


def _relay_config(params: RelayParams, gammas: SeededShapeGammas) -> dict[str, Any]:
    """relay_bp constructor arguments in rtd-golden's manifest form, readable by the C++ golden
    loader."""
    config: dict[str, Any] = {"check_matrix": "window shape H as scipy.sparse.csr_matrix, uint8", "error_priors": "window shape priors"}
    for key, value in params.kwargs(np.zeros((0, 0))).items():
        config[key] = "gammas/shape_<i>.npy" if key == "explicit_gammas" else (list(value) if isinstance(value, tuple) else value)
    config["gamma_table"] = {
        "rows": gammas.rows,
        "seed": gammas.seed,
        "interval": [gammas.low, gammas.high],
        "generator": "numpy.random.default_rng([seed, shape]).uniform(interval[0], interval[1], size=(rows, shape_columns))",
        "leg_row": "relay leg r >= 1 uses row r mod rows",
    }
    return config


def run_golden(args: argparse.Namespace) -> dict[str, Any]:
    from rtd.export import _relay_bp_source

    t_start = time.perf_counter()
    out: Path = args.out
    params, gammas = relay_from_args(args)
    spec = spec_from_args(args)
    if spec.converge_rounds != spec.width or spec.iteration_cap is not None:
        raise PlanError("golden_scope", "relay_bp checks every row and has no iteration budget: goldens need converge = width and no cap")
    _prepare_out(out, args.overwrite)
    problem = load_problem(args.artifact)
    plan = build_plan(problem, spec)
    if spec.boundary == "exact" and not commit_sets_partition(plan):
        raise RuntimeError("the attempt-0 commit sets do not partition the columns")
    syndromes, observables, source = load_shots_dir(args.shots, args.first, args.count, problem)
    inner_spec = InnerSpec("relay", relay=params, gammas=gammas)

    t0 = time.perf_counter()
    outcomes = decode_shots(plan, syndromes, inner_spec, args.first, args.workers)
    decode_seconds = time.perf_counter() - t0
    check = check_outcomes(problem, plan, syndromes, outcomes)
    all_arrays = outcome_arrays(outcomes, plan, problem.k, observables)
    arrays = {name: all_arrays[name] for name in GOLDEN_ARRAYS}
    arrays["detectors"] = syndromes
    arrays["observables"] = observables
    logger.info("golden self-checks passed", extra={"out": str(out), **check})

    write_arrays(out, arrays)
    (out / "gammas").mkdir(exist_ok=True)
    for shape in plan.shapes:
        write_arrays(out / "gammas", {f"shape_{shape.index}": gammas(shape.index, shape.columns)})
    (out / "spec.json").write_text(json.dumps(rtd_decode_spec(params, spec.window_json()), indent=2) + "\n")

    win_conv = all_arrays["win_converged"][all_arrays["win_attempts"] > 0]
    summary = {
        "shots": len(outcomes),
        "windows_decoded": int((all_arrays["win_attempts"] > 0).sum()),
        "windows_not_converged": int((win_conv == 0).sum()),
        "windows_deferred": int((all_arrays["win_attempts"] > 1).sum()),
        "windows_skipped_after_final": int((all_arrays["win_attempts"] == 0).sum()),
        "shots_all_windows_converged": check["all_windows_converged"],
        "flagged_shots": int(arrays["flagged"].sum()),
        "logical_failures": int(arrays["logical_failure"].sum()),
        "virtual_commits": int(arrays["win_virtual"].sum()),
        "mean_iterations_per_shot": float(all_arrays["iterations"].mean()),
        "max_window_iterations": int(arrays["win_iterations"].max()),
    }
    artifact_manifest = problem.manifest
    sha = {"circuit.stim": artifact_manifest.get("source_circuit", {}).get("sha256")}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            sha[path.relative_to(out).as_posix()] = _sha256_file(path)
    manifest = {
        "format_version": GOLDEN_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": _manifest_path(args.artifact), "manifest_sha256": problem.manifest_sha256},
        "shots": source,
        "first": args.first,
        "count": args.count,
        "rounds": plan.rounds_total - 1,
        "window": spec.window_json(),
        "decoder": {
            "float": params.float_type,
            "gamma0": params.gamma0,
            "pre_iter": params.pre_iter,
            "set_max_iter": params.set_max_iter,
            "num_sets": params.num_sets,
            "stopping": rtd_decode_spec(params, {})["stopping"],
            "alpha": rtd_decode_spec(params, {})["alpha"],
            "inner": "relay_bp, one freshly constructed decoder per window decode",
        },
        "gamma": {"rows": gammas.rows, "seed": gammas.seed, "low": gammas.low, "high": gammas.high},
        "config": _relay_config(params, gammas),
        "num_shapes": len(plan.shapes),
        "num_positions": plan.num_positions,
        "plan": plan.json(),
        "weights": "math.log((1 - p) / p) per local column, summed in ascending local index",
        "skipped_windows": "after a final placement is committed the remaining positions are not decoded; "
        "their records are those of an empty window (attempts 0, converged 1, all counts 0)",
        "shots_dir_compat": "sha256['circuit.stim'] is the artifact's source-circuit checksum (the circuit is not copied), "
        "so this directory is accepted as rtd_decode --shots",
        "relay_bp": _relay_bp_source(),
        "versions": {"numpy": np.__version__, "scipy": version("scipy"), "rtd": version("rtd")},
        "timing": {"decode_seconds": decode_seconds, "total_seconds": time.perf_counter() - t_start, "workers": args.workers},
        "summary": summary,
        "sha256": sha,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("window golden written", extra={"out": str(out), **summary})
    return manifest


def load_golden(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((path / "manifest.json").read_text())
    arrays = {}
    for name, expected in manifest["sha256"].items():
        if name == "circuit.stim":
            continue
        if _sha256_file(path / name) != expected:
            raise ValueError(f"{path / name} does not match the golden manifest")
        if name.endswith(".npy") and "/" not in name:
            arrays[name[:-4]] = np.load(path / name, allow_pickle=False)
    return manifest, arrays


def golden_problem(manifest: dict[str, Any], repo_root: Path) -> GlobalProblem:
    problem = load_problem(repo_root / manifest["artifact"]["path"])
    if problem.manifest_sha256 != manifest["artifact"]["manifest_sha256"]:
        raise ValueError("the golden's artifact manifest checksum does not match the artifact on disk")
    return problem
