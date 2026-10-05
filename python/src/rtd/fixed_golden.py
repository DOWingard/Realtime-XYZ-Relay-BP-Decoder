"""Goldens and test vectors of the fixed-point (intN.S.M) windowed Relay-BP decoder.

Golden directory (the C++ fixed-point golden test and rtd_decode read it; array names are
rtd_decode's):
    manifest.json      artifact path and manifest checksum, shots source, first, count, the window
                       object, the decoder (arithmetic, relay schedule, alpha rule), gamma-table
                       settings, the plan, a summary, and sha256 of every file (plus the source
                       circuit's, so the directory is also an rtd_decode --shots directory)
    spec.json          the equivalent rtd_decode version-2 spec ("arithmetic": "intN.S.M",
                       gamma_source explicit_shapes)
    gammas/shape_<i>.npy   float64 [T, n_i] = default_rng([seed, i]).uniform(low, high, (T, n_i));
                       relay leg r >= 1 of a window of shape i uses row r mod T
    detectors.npy, observables.npy                   uint8 [S, m], [S, k]
    win_iterations, win_legs, win_unexplained, win_virtual   uint32 [S, K]
    win_attempts, win_converged, win_cap_hit, win_flagged    uint8 [S, K]
    win_weight, win_committed_weight                         float64 [S, K]
    commit_ptr uint64 [S K + 1], commit_faults uint32        committed global columns per window
    predicted_observables uint8 [S, k], logical_failure, flagged, success uint8 [S],
    iterations, legs uint32 [S], weight float64 [S]

Test vectors (vectors.npz + vectors.json next to the golden arrays): the same decode in a form a
hardware team can replay without this repository. vectors.json documents every array; see
VECTOR_ARRAYS below and write_vectors().
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from rtd import log
from rtd.fixed_ref import FORMATS, AlphaRule, FixedFormat, FixedInner, FixedInnerSpec, RelayParams
from rtd.window_ref import (
    InnerContext,
    PlanError,
    WindowSpec,
    _manifest_path,
    _prepare_out,
    _sha256_file,
    build_plan,
    check_outcomes,
    decode_shot,
    decode_shots,
    load_problem,
    load_shots_dir,
    outcome_arrays,
)
from rtd.window_ref_golden import commit_sets_partition, write_arrays
from rtd.window_ref_inner import InnerResult, SeededShapeGammas

logger = logging.getLogger("rtd.fixed_golden")

GOLDEN_FORMAT_VERSION = 1
VECTORS_FORMAT_VERSION = 1
GOLDEN_ARRAYS = (
    "win_iterations",
    "win_legs",
    "win_attempts",
    "win_converged",
    "win_cap_hit",
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
    "success",
    "iterations",
    "legs",
    "weight",
)

CHOICES = {
    "rounding": "round half away from zero, for S*lambda and for (1 - gamma)*M",
    "saturation": "bias and stored marginal clip to +-(2^N - 1); nu clips to the same range",
    "hard_decision": "e_j = 1 when the unsaturated sigma_j <= 0",
    "accumulator": "exact (no overflow): sigma and every partial sum",
    "alpha": "alpha*x = x - (x >> k); adaptive: k = t + 1, t = iteration index within the leg",
    "memory_strength": "beta_int = round((1 - gamma) M) clipped to [0, 2M]; gamma*x = x - beta(x)x",
    "multiplier": "beta(x)x = sign(x) sum_{bit b of |x|} floor(beta_int 2^b / M)",
    "first_iteration": "the first check pass of every leg reads lambda_int (messages restart from the priors)",
    "marginal_init": "M = lambda_int at the start of a decode; kept from leg to leg",
    "weight": "sum of lambda_int over the support, divided by S once",
}


def rtd_decode_spec(fmt: FixedFormat, params: RelayParams, window: dict[str, Any]) -> dict[str, Any]:
    """The rtd_decode version-2 spec of this decoder."""
    return {
        "version": 2,
        "arithmetic": fmt.name,
        "backend": "cpu",
        "layout": "row_major",
        "column_order": "wavefront",
        "block_rows": 64,
        "executor": {"type": "serial"},
        "alpha": params.alpha.json(),
        "gamma0": params.gamma0,
        "pre_iter": params.pre_iter,
        "set_max_iter": params.set_max_iter,
        "num_sets": params.num_sets,
        "stopping": params.stopping_json(),
        "gamma_source": {"type": "explicit_shapes", "directory": "gammas"},
        "window": window,
    }


def _decoder_json(fmt: FixedFormat, params: RelayParams) -> dict[str, Any]:
    return {
        "arithmetic": fmt.name,
        "gamma0": params.gamma0,
        "pre_iter": params.pre_iter,
        "set_max_iter": params.set_max_iter,
        "num_sets": params.num_sets,
        "stopping": params.stopping_json(),
        "alpha": params.alpha.json(),
        "inner": "rtd.fixed_ref integer emulator",
    }


def params_from_args(args: argparse.Namespace) -> tuple[FixedFormat, RelayParams, SeededShapeGammas]:
    fmt = FixedFormat.parse(args.format)
    alpha = AlphaRule("adaptive", scaling=args.alpha_scaling) if args.alpha == "adaptive" else AlphaRule("constant", value=float(args.alpha))
    params = RelayParams(
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        set_max_iter=args.set_max_iter,
        num_sets=args.num_sets,
        stopping=args.stopping,
        stop_count=args.stop_count,
        alpha=alpha,
    )
    params.validate()
    lo, hi = args.gamma_interval
    if not lo < hi:
        raise ValueError(f"--gamma-interval needs LO < HI (got {lo}, {hi})")
    return fmt, params, SeededShapeGammas(args.gamma_rows, args.gamma_seed, lo, hi)


def run_golden(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    out: Path = args.out
    fmt, params, gammas = params_from_args(args)
    spec = WindowSpec(args.width, args.commit, args.converge, args.boundary, args.on_failure, args.max_deferrals, args.iteration_cap)
    spec.validate()
    _prepare_out(out, args.overwrite)
    problem = load_problem(args.artifact)
    plan = build_plan(problem, spec)
    if spec.boundary == "exact" and not commit_sets_partition(plan):
        raise RuntimeError("the attempt-0 commit sets do not partition the columns")
    syndromes, observables, source = load_shots_dir(args.shots, args.first, args.count, problem)
    inner_spec = FixedInnerSpec(fmt, params, gammas)

    t0 = time.perf_counter()
    outcomes = decode_shots(plan, syndromes, inner_spec, args.first, args.workers)
    decode_seconds = time.perf_counter() - t0
    check = check_outcomes(problem, plan, syndromes, outcomes)
    all_arrays = outcome_arrays(outcomes, plan, problem.k, observables)
    arrays = {name: all_arrays[name] for name in GOLDEN_ARRAYS}
    arrays["detectors"] = syndromes
    arrays["observables"] = observables
    logger.info("fixed-point golden self-checks passed", extra={"out": str(out), **check})

    write_arrays(out, arrays)
    (out / "gammas").mkdir(exist_ok=True)
    for shape in plan.shapes:
        write_arrays(out / "gammas", {f"shape_{shape.index}": gammas(shape.index, shape.columns)})
    (out / "spec.json").write_text(json.dumps(rtd_decode_spec(fmt, params, spec.window_json()), indent=2) + "\n")

    decoded = all_arrays["win_attempts"] > 0
    summary = {
        "shots": len(outcomes),
        "windows_decoded": int(decoded.sum()),
        "windows_not_converged": int((all_arrays["win_converged"][decoded] == 0).sum()),
        "windows_cap_hit": int(all_arrays["win_cap_hit"].sum()),
        "windows_deferred": int((all_arrays["win_attempts"] > 1).sum()),
        "windows_skipped_after_final": int((all_arrays["win_attempts"] == 0).sum()),
        "shots_all_windows_converged": check["all_windows_converged"],
        "flagged_shots": int(arrays["flagged"].sum()),
        "logical_failures": int(arrays["logical_failure"].sum()),
        "virtual_commits": int(arrays["win_virtual"].sum()),
        "mean_iterations_per_shot": float(all_arrays["iterations"].mean()),
        "max_window_iterations": int(arrays["win_iterations"].max()),
    }
    sha = {"circuit.stim": problem.manifest.get("source_circuit", {}).get("sha256")}
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
        "decoder": _decoder_json(fmt, params),
        "fixed_point_choices": CHOICES,
        "gamma": {"rows": gammas.rows, "seed": gammas.seed, "low": gammas.low, "high": gammas.high},
        "gamma_table": {
            "generator": "numpy.random.default_rng([seed, shape]).uniform(low, high, size=(rows, shape_columns))",
            "leg_row": "relay leg r >= 1 uses row r mod rows",
        },
        "num_shapes": len(plan.shapes),
        "num_positions": plan.num_positions,
        "plan": plan.json(),
        "weights": "win_weight: sum of lambda_int over the returned support / S; win_committed_weight: math.log((1 - p) / p) "
        "per committed local column, summed in ascending local index (the window layer's rule, in double)",
        "skipped_windows": "after a final placement is committed the remaining positions are not decoded; "
        "their records are those of an empty window (attempts 0, converged 1, all counts 0)",
        "shots_dir_compat": "sha256['circuit.stim'] is the artifact's source-circuit checksum, so this directory is accepted as rtd_decode --shots",
        "versions": {"numpy": np.__version__, "scipy": version("scipy"), "rtd": version("rtd")},
        "timing": {"decode_seconds": decode_seconds, "total_seconds": time.perf_counter() - t_start, "workers": args.workers},
        "summary": summary,
        "sha256": sha,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("fixed-point golden written", extra={"out": str(out), **summary})
    return manifest


def load_golden(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The manifest and the top-level arrays of a golden directory, checksums verified."""
    manifest = json.loads((path / "manifest.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    for name, expected in manifest["sha256"].items():
        if name == "circuit.stim" or name.startswith("vectors."):
            continue
        if _sha256_file(path / name) != expected:
            raise ValueError(f"{path / name} does not match the golden manifest")
        if name.endswith(".npy") and "/" not in name:
            arrays[name[:-4]] = np.load(path / name, allow_pickle=False)
    return manifest, arrays


def _golden_settings(manifest: dict[str, Any]) -> tuple[FixedFormat, RelayParams, SeededShapeGammas, WindowSpec]:
    d = manifest["decoder"]
    a = d["alpha"]
    alpha = AlphaRule("constant", value=a["value"]) if a["rule"] == "constant" else AlphaRule("adaptive", scaling=a["scaling"])
    stopping = d["stopping"]
    params = RelayParams(d["gamma0"], d["pre_iter"], d["set_max_iter"], d["num_sets"], stopping["rule"], stopping.get("count", 1), alpha)
    g = manifest["gamma"]
    w = manifest["window"]
    spec = WindowSpec(w["width"], w["commit"], w["converge_rounds"], w["boundary"], w["on_failure"], w["max_deferrals"], w["iteration_cap"])
    return FixedFormat.parse(d["arithmetic"]), params, SeededShapeGammas(g["rows"], g["seed"], g["low"], g["high"]), spec


# ---------------------------------------------------------------------------------------------
# Test vectors


class _RecordingInner:
    """Keeps the local support each window decode returned, in call order."""

    name = "fixed-recording"

    def __init__(self, inner: FixedInner):
        self.inner = inner
        self.calls: list[tuple[InnerContext, np.ndarray]] = []

    def decode(self, h: Any, priors: np.ndarray, syndrome: np.ndarray, ctx: InnerContext) -> InnerResult:
        res = self.inner.decode(h, priors, syndrome, ctx)
        self.calls.append((ctx, np.asarray(res.support, dtype=np.int64)))
        return res

    def close(self) -> None:
        self.inner.close()


VECTOR_ARRAYS: dict[str, str] = {
    "detectors": "uint8 [S, Rt*M]: the detector stream of each shot, round after round (row (r - 1)*M + i is check i of round r)",
    "observables": "uint8 [S, k]: the true logical observable flips",
    "shape_rows, shape_columns": "uint32 [num_shapes]: local window size of each shape",
    "shape<i>_H_indptr, shape<i>_H_indices": "uint32: the window check matrix H~ of shape i (local CSR, rows in round order)",
    "shape<i>_lambda": "int8 [n_i]: quantised priors lambda_int = sat(round(S ln((1 - p)/p)))",
    "shape<i>_commit": "uint8 [n_i]: commit mask m_com (1 = the column is decided by this window)",
    "shape<i>_converge": "uint8 [rows_i]: convergence mask (1 = H~ e = s must hold on this row)",
    "shape<i>_frame": "uint64 [n_i]: frame matrix A~ by column: bit o = logical observable o flipped (0 for uncommitted columns)",
    "shape<i>_beta": "uint8 [T, n_i]: beta_int = round((1 - gamma) M) of relay leg r >= 1 (row r mod T); leg 0 uses beta(gamma0)",
    "placement_window, placement_attempt, placement_shape, placement_first_round, placement_final": "uint32/uint8 [P]: the plan",
    "placement<k>_<a>_columns": "uint32 [n]: global column of each local column (0xFFFFFFFF = virtual, uniform boundary)",
    "win_attempts": "uint8 [S, K]: attempts decoded at window position k (0 = skipped after a final placement)",
    "win_iterations, win_legs": "uint32 [S, K]: BP iterations and relay legs, summed over the attempts",
    "win_converged, win_cap_hit, win_flagged": "uint8 [S, K]: outcome of the committed attempt",
    "win_carry": "uint8 [S, K, M]: the window's first-round detectors after the carry of earlier commits (Algorithm 2's d~[0:M] xor u)",
    "win_frame": "uint64 [S, K]: accumulated logical frame after the window (bit o = observable o)",
    "solution_ptr, solution_local": "uint64 [S K + 1], uint32: the local support of the returned e of each window (committed attempt)",
    "commit_local_ptr, commit_local": "uint64 [S K + 1], uint32: local committed columns (e AND m_com) of each window",
    "commit_ptr, commit_faults": "uint64 [S K + 1], uint32: committed global columns of each window (virtual ones omitted)",
    "predicted_observables": "uint8 [S, k]: the final frame",
    "logical_failure": "uint8 [S]: predicted_observables != observables anywhere",
}


def write_vectors(golden: Path, repo_root: Path) -> dict[str, Any]:
    """Re-decodes every shot of a golden in one process, checks it reproduces the golden arrays
    exactly, and writes vectors.npz and vectors.json into the golden directory."""
    t0 = time.perf_counter()
    manifest, arrays = load_golden(golden)
    problem = load_problem(repo_root / manifest["artifact"]["path"])
    if problem.manifest_sha256 != manifest["artifact"]["manifest_sha256"]:
        raise ValueError("the golden's artifact manifest checksum does not match the artifact on disk")
    fmt, params, gammas, spec = _golden_settings(manifest)
    plan = build_plan(problem, spec)
    tables = {shape.index: np.load(golden / "gammas" / f"shape_{shape.index}.npy") for shape in plan.shapes}
    for shape in plan.shapes:
        if not np.array_equal(tables[shape.index], gammas(shape.index, shape.columns)):
            raise ValueError(f"gamma table of shape {shape.index} differs from its generator")
    inner = _RecordingInner(FixedInner(fmt, params, _TableByShape(tables)))
    syndromes, observables = arrays["detectors"], arrays["observables"]
    S, K, M = syndromes.shape[0], plan.num_positions, plan.per_round
    first = int(manifest["first"])
    outcomes = []
    solution: list[np.ndarray] = []
    commit_local: list[np.ndarray] = []
    for s in range(S):
        inner.calls.clear()
        o = decode_shot(plan, syndromes[s], first + s, inner)
        outcomes.append(o)
        last_call = {}
        for ctx, support in inner.calls:
            last_call[ctx.window] = (ctx, support)
        for k in range(K):
            if o.windows[k].attempts == 0:
                solution.append(np.zeros(0, dtype=np.uint32))
                commit_local.append(np.zeros(0, dtype=np.uint32))
                continue
            ctx, support = last_call[k]
            shape = plan.shapes[ctx.shape]
            solution.append(support.astype(np.uint32))
            commit_local.append(support[shape.commit[support] == 1].astype(np.uint32))
    again = outcome_arrays(outcomes, plan, problem.k, observables)
    for name in GOLDEN_ARRAYS:
        if not _same(again[name], arrays[name]):
            raise RuntimeError(f"re-decoding {golden} does not reproduce {name}")

    out: dict[str, np.ndarray] = {"detectors": syndromes, "observables": observables}
    out["shape_rows"] = np.array([sh.rows for sh in plan.shapes], dtype=np.uint32)
    out["shape_columns"] = np.array([sh.columns for sh in plan.shapes], dtype=np.uint32)
    for sh in plan.shapes:
        i = sh.index
        out[f"shape{i}_H_indptr"] = sh.indptr()
        out[f"shape{i}_H_indices"] = sh.indices()
        lam = np.array([math.log((1.0 - p) / p) if p > 0 else math.inf for p in sh.priors.tolist()])
        out[f"shape{i}_lambda"] = fmt.quantise_llr(lam).astype(np.int8)
        out[f"shape{i}_commit"] = sh.commit.astype(np.uint8)
        out[f"shape{i}_converge"] = sh.converge.astype(np.uint8)
        out[f"shape{i}_frame"] = sh.cls.astype(np.uint64)
        out[f"shape{i}_beta"] = fmt.quantise_gamma(tables[i]).astype(np.uint8)
    out["placement_window"] = np.array([p.window for p in plan.placements], dtype=np.uint32)
    out["placement_attempt"] = np.array([p.attempt for p in plan.placements], dtype=np.uint32)
    out["placement_shape"] = np.array([p.shape for p in plan.placements], dtype=np.uint32)
    out["placement_first_round"] = np.array([p.first_round for p in plan.placements], dtype=np.uint32)
    out["placement_final"] = np.array([p.final for p in plan.placements], dtype=np.uint8)
    for p in plan.placements:
        out[f"placement{p.window}_{p.attempt}_columns"] = p.columns.astype(np.uint32)
    for name in ("win_attempts", "win_iterations", "win_legs", "win_converged", "win_cap_hit", "win_flagged", "commit_ptr", "commit_faults", "predicted_observables", "logical_failure"):
        out[name] = again[name]
    carry = np.zeros((S, K, M), dtype=np.uint8)
    frame = np.zeros((S, K), dtype=np.uint64)
    for s, o in enumerate(outcomes):
        for k, w in enumerate(o.windows):
            if w.residual_first_round is not None:
                carry[s, k] = w.residual_first_round
            frame[s, k] = w.frame_after
    out["win_carry"] = carry
    out["win_frame"] = frame
    for ptr_name, name, lists in (("solution_ptr", "solution_local", solution), ("commit_local_ptr", "commit_local", commit_local)):
        ptr = np.zeros(S * K + 1, dtype=np.uint64)
        ptr[1:] = np.cumsum([x.size for x in lists])
        out[ptr_name] = ptr
        out[name] = np.concatenate(lists).astype(np.uint32) if lists else np.zeros(0, dtype=np.uint32)
    np.savez_compressed(golden / "vectors.npz", **out)
    description = {
        "format_version": VECTORS_FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "golden": golden.name,
        "arithmetic": fmt.name,
        "format": {"magnitude_bits": fmt.bits, "llr_scale": fmt.scale, "memory_scale": fmt.memory},
        "decoder": manifest["decoder"],
        "fixed_point_choices": CHOICES,
        "window": manifest["window"],
        "rounds_total": plan.rounds_total,
        "detectors_per_round": M,
        "num_positions": K,
        "num_shapes": len(plan.shapes),
        "shots": S,
        "first_shot": first,
        "observables": problem.k,
        "gamma_rows": int(manifest["gamma"]["rows"]),
        "arrays": VECTOR_ARRAYS,
        "stream": "window k starts at round t_k = 1 + k C; its syndrome is the residual of rounds [t_k, t_k + rows/M) "
        "(rounds past Rt are zero); after decoding, e AND m_com is committed: its rows flip the residual and A~ e_com is "
        "XORed into the frame",
        "reproduced": "every array was reproduced by a second, single-process decode before writing",
        "sha256_npz": None,
    }
    (golden / "vectors.json").write_text(json.dumps(description, indent=2) + "\n")
    description["sha256_npz"] = _sha256_file(golden / "vectors.npz")
    (golden / "vectors.json").write_text(json.dumps(description, indent=2) + "\n")
    logger.info("test vectors written", extra={"golden": str(golden), "shots": S, "seconds": time.perf_counter() - t0})
    return description


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Identical dtype, shape and bytes (so doubles compare bit for bit)."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class _TableByShape:
    """Explicit per-shape tables, called like SeededShapeGammas."""

    def __init__(self, tables: dict[int, np.ndarray]):
        self._tables = tables

    def __call__(self, shape: int, n: int) -> np.ndarray:
        table = self._tables[shape]
        if table.shape[1] != n:
            raise ValueError(f"gamma table of shape {shape} has {table.shape[1]} columns, the shape {n}")
        return table


def load_vectors(golden: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """vectors.json and the arrays of vectors.npz, the npz checksum verified."""
    description = json.loads((golden / "vectors.json").read_text())
    if description["sha256_npz"] != _sha256_file(golden / "vectors.npz"):
        raise ValueError(f"{golden / 'vectors.npz'} does not match vectors.json")
    with np.load(golden / "vectors.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return description, arrays


# ---------------------------------------------------------------------------------------------
# Command line


def _add_decoder_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=FORMATS, required=True)
    p.add_argument("--alpha", required=True, help="'adaptive' (alpha = 1 - 2^-(t+1)) or a constant 1 - 2^-k")
    p.add_argument("--alpha-scaling", type=float, default=1.0)
    p.add_argument("--gamma0", type=float, required=True)
    p.add_argument("--pre-iter", type=int, required=True)
    p.add_argument("--set-max-iter", type=int, required=True)
    p.add_argument("--num-sets", type=int, required=True)
    p.add_argument("--stopping", choices=("after_n_converged", "after_leg0", "all_legs"), required=True)
    p.add_argument("--stop-count", type=int, default=1)
    p.add_argument("--gamma-rows", type=int, required=True)
    p.add_argument("--gamma-seed", type=int, required=True)
    p.add_argument("--gamma-interval", type=float, nargs=2, required=True, metavar=("LO", "HI"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m rtd.fixed_golden", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("golden", help="decode shots with the integer emulator and write a golden directory")
    g.add_argument("--artifact", type=Path, required=True)
    g.add_argument("--shots", type=Path, required=True)
    g.add_argument("--first", type=int, default=0)
    g.add_argument("--count", type=int, required=True)
    g.add_argument("--width", type=int, required=True)
    g.add_argument("--commit", type=int, required=True)
    g.add_argument("--converge", type=int, required=True)
    g.add_argument("--boundary", choices=("exact", "uniform"), required=True)
    g.add_argument("--on-failure", choices=("commit_anyway", "defer", "flag"), required=True)
    g.add_argument("--max-deferrals", type=int, default=0)
    g.add_argument("--iteration-cap", type=int, default=None)
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--overwrite", action="store_true")
    _add_decoder_flags(g)
    v = sub.add_parser("vectors", help="write vectors.npz / vectors.json for a golden directory")
    v.add_argument("golden", type=Path)
    v.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(getattr(logging, args.log_level.upper(), logging.INFO))
    try:
        if args.command == "golden":
            run_golden(args)
        else:
            write_vectors(args.golden, args.repo_root)
    except (PlanError, ValueError, FileExistsError, RuntimeError, OSError) as e:
        logger.exception("fixed-point golden command failed", extra={"command": args.command, "error_type": type(e).__name__})
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
