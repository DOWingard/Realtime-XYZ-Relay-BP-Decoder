"""Build a BB memory-experiment circuit and sample detector/observable shots to disk.

Output directory contents:
    circuit.stim          the noisy circuit
    detectors.npy         uint8 [shots, num_detectors], 1 = detector fired
    observables.npy       uint8 [shots, num_observables], 1 = logical observable flipped
    det_check.npy         int32 [num_detectors], stabilizer index within its type
    det_round.npy         int32 [num_detectors], 1..rounds, rounds + 1 = final readout layer
    det_type.npy          uint8 [num_detectors], 0 = X-type, 1 = Z-type
    observable_type.npy   uint8 [num_observables], 0 = logical X, 1 = logical Z
    manifest.json         parameters, versions, counts and checksums needed to reproduce the run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import stim

from rtd import log
from rtd.bb_code import CODE_PARAMS, get_code
from rtd.circuit import EXPERIMENTS, MemoryCircuit, build_memory_circuit
from rtd.noise import NoiseModel

logger = logging.getLogger("rtd.sample")

MANIFEST_VERSION = 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", required=True, choices=sorted(CODE_PARAMS))
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--p", type=float, required=True, help="uniform circuit-noise strength")
    parser.add_argument("--rounds", type=int, default=None, help="noisy syndrome cycles (default: code distance)")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100_000, help="shots sampled per call")
    parser.add_argument(
        "--relay-bp-compat",
        action="store_true",
        help="reproduce the final-cycle idle-noise placement of relay_bp's reference circuits",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def check_deterministic(circuit: stim.Circuit, shots: int = 64) -> None:
    """Every detector and observable must be 0 on every shot of the noiseless circuit."""
    dets, obs = circuit.without_noise().compile_detector_sampler(seed=0).sample(shots, separate_observables=True)
    bad_dets = np.flatnonzero(dets.any(axis=0))
    bad_obs = np.flatnonzero(obs.any(axis=0))
    if bad_dets.size or bad_obs.size:
        raise RuntimeError(
            f"noiseless circuit has non-deterministic detectors {bad_dets[:10].tolist()} "
            f"(total {bad_dets.size}) or observables {bad_obs.tolist()}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_to_disk(mc: MemoryCircuit, shots: int, seed: int, batch: int, out: Path) -> dict[str, float]:
    circuit = mc.circuit
    detectors = np.lib.format.open_memmap(
        out / "detectors.npy", mode="w+", dtype=np.uint8, shape=(shots, circuit.num_detectors)
    )
    observables = np.lib.format.open_memmap(
        out / "observables.npy", mode="w+", dtype=np.uint8, shape=(shots, circuit.num_observables)
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    fired = 0
    flipped = 0
    start = time.perf_counter()
    for lo in range(0, shots, batch):
        hi = min(lo + batch, shots)
        dets, obs = sampler.sample(hi - lo, separate_observables=True)
        detectors[lo:hi] = dets
        observables[lo:hi] = obs
        fired += int(dets.sum())
        flipped += int(obs.any(axis=1).sum())
        logger.debug("batch sampled", extra={"first_shot": lo, "last_shot": hi - 1})
    elapsed = time.perf_counter() - start
    detectors.flush()
    observables.flush()
    return {
        "seconds": elapsed,
        "shots_per_second": shots / elapsed if elapsed > 0 else float("inf"),
        "mean_detector_fire_rate": fired / (shots * circuit.num_detectors),
        "fraction_shots_with_raw_observable_flip": flipped / shots,
    }


def run(args: argparse.Namespace) -> Path:
    if args.shots < 1 or args.batch < 1:
        raise ValueError(f"--shots and --batch must be positive (got {args.shots}, {args.batch})")
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is not empty; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    code = get_code(args.code)
    logger.info(
        "code built",
        extra={"code": code.name, "n": code.n, "k": code.k, "seconds": time.perf_counter() - t0},
    )

    rounds = args.rounds if args.rounds is not None else code.distance
    noise = NoiseModel.uniform(args.p)
    t0 = time.perf_counter()
    mc = build_memory_circuit(code, rounds, args.experiment, noise, relay_bp_compat=args.relay_bp_compat)
    circuit = mc.circuit
    logger.info(
        "circuit built",
        extra={
            "experiment": args.experiment,
            "rounds": rounds,
            "qubits": circuit.num_qubits,
            "detectors": circuit.num_detectors,
            "observables": circuit.num_observables,
            "measurements": circuit.num_measurements,
            "seconds": time.perf_counter() - t0,
        },
    )

    check_deterministic(circuit)
    logger.info("noiseless circuit is deterministic")

    circuit_path = out / "circuit.stim"
    circuit.to_file(circuit_path)
    np.save(out / "det_check.npy", mc.detectors.check)
    np.save(out / "det_round.npy", mc.detectors.round)
    np.save(out / "det_type.npy", mc.detectors.kind)
    np.save(out / "observable_type.npy", mc.observable_kind)

    logger.info("sampling started", extra={"shots": args.shots, "seed": args.seed, "batch": args.batch})
    stats = sample_to_disk(mc, args.shots, args.seed, args.batch, out)
    logger.info("sampling finished", extra=stats)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "code": {"name": code.name, **CODE_PARAMS[code.name], "n": code.n, "k": code.k},
        "experiment": args.experiment,
        "rounds": rounds,
        "noise": noise.to_dict(),
        "relay_bp_compat": args.relay_bp_compat,
        "shots": args.shots,
        "seed": args.seed,
        "batch": args.batch,
        "counts": {
            "qubits": circuit.num_qubits,
            "detectors": circuit.num_detectors,
            "observables": circuit.num_observables,
            "measurements": circuit.num_measurements,
        },
        "versions": {"stim": stim.__version__, "numpy": np.__version__, "rtd": version("rtd")},
        "sha256": {
            name: _sha256(out / name)
            for name in ("circuit.stim", "detectors.npy", "observables.npy")
        },
        "sampling": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("outputs written", extra={"out": str(out)})
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("run started", extra={"cli_args": {k: str(v) for k, v in vars(args).items()}})
    try:
        run(args)
    except Exception:
        logger.exception("run failed", extra={"cli_args": {k: str(v) for k, v in vars(args).items()}})
        return 1
    logger.info("run completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
