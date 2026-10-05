"""Chunked, resumable decoding campaigns.

A campaign samples shots in chunks from recorded seeds, decodes every chunk with a pinned copy of
rtd_decode under one or more decoder configurations ("arms"), keeps the per-shot records and
deletes the raw shots, which the seed regenerates bit for bit. All arms decode the same shots, so
their failures can be compared shot by shot (paired statistics).

    rtd-campaign init CONFIG.json      create data/campaigns/<name>/ and pin binary, specs, artifacts
    rtd-campaign run DIR               decode chunks until a stopping criterion holds (resumable)
    rtd-campaign status DIR            progress and stopping criteria as JSON
    rtd-campaign summarize DIR         summary.json: rates with intervals, costs, paired ratios;
                                       per-window statistics for sliding-window arms

Config (every field required; null where allowed):

    {"name": "a3_xz_p0.003",
     "circuit": {"code": "gross", "experiment": "choi", "p": 0.003, "rounds": 12, "relay_bp_compat": true},
     "mode": "xyz" | "xz" | "z_only" | "x_only",
     "seed": 1, "chunk_shots": 20000,
     "arms": {"relay5": {"spec": "configs/xz_relay5_f32.json", "workers": 6, "cpus": "0-5",
                         "record_solutions": 0, "save_commits": false,
                         "save_solution_supports": false, "extra_args": []}},
     "stop": {"max_shots": 2000000, "max_wall_seconds": null, "min_failures": 100,
              "failure_arm": "relay5",
              "paired": {"reference": "a", "test": "b", "rel_halfwidth": 0.25}
                        | [{"reference": "a", "test": "b", "rel_halfwidth": 0.25}, ...] | null},
     "keep_raw": false}

Optional fields: circuit "readout_in_round_order" (choi only, default false: list the final layer's
detectors in the noisy rounds' order, see rtd.circuit; the artifact name gets "_rorder"),
"rtd_decode" (binary to pin; default build/release/src/rtd_decode), "pairs"
(extra [{"reference", "test"}] pairs to summarize), "description", "gamma_seed_per_chunk"
(default true, see below), and per arm "decode_timeout_seconds" (wall seconds allowed for one
rtd_decode invocation, i.e. one half of one chunk; null or absent = no limit). Relative paths are
resolved against the repository root.

Modes: xyz decodes the whole detector error model; xz splits it by detector type (rtd.xz) and
decodes both halves, a shot failing when either half fails; z_only / x_only decode one half.

Chunk c samples with stim seed (seed * 1000003 + c) mod 2^63 in fixed batches, and its
chunks/<c>/chunk.json records the SHA-256 of the sampled detector and observable bytes, so a
regenerated chunk is verified before it is decoded again. Directory layout:

    campaign.json            the config, byte for byte
    init.json                pinned inputs: binary, specs and the files they reference, circuit,
                             artifacts (paths and SHA-256)
    circuit.stim             the sampled circuit
    bin/rtd_decode           pinned binary
    specs/<arm>/             the arm's pinned spec (same file name) and the files it references
    campaign.log             JSON-lines log of every run (also on stderr)
    chunks/<c>/chunk.json    seed, shots, batch, checksums, sampling time
    chunks/<c>/shots/<half>/ raw shots while the chunk is being decoded (half = xyz, x or z)
    chunks/<c>/<arm>/        rtd_decode output (xyz) or <arm>/x/, <arm>/z/; done.json when complete
    chunks/<c>/logs/<arm>.<half>.log   rtd_decode's JSON-lines log (.pid while it runs)
    summary.json             written by `summarize`

Every JSON file is replaced atomically (temporary file, fsync, rename), so a full disk or a crash
never leaves a truncated marker. An arm's done.json is written only after every file rtd_decode
wrote for it has been synced to disk, so after a power loss a done.json never names outputs that
were still in the page cache. A marker that is unreadable anyway (e.g. written by an older
driver) counts as absent: it is renamed to <name>.corrupt-<time> and its work is redone.

Stopping criteria (checked after every chunk; any one stops the run): max_shots sampled;
max_wall_seconds of campaign compute time (sampling plus decoding, summed over all runs);
min_failures of failure_arm (the combined outcome for split modes); the paired interval on
V = P_fail(test) / P_fail(reference) narrower than +-rel_halfwidth. At least one of max_shots and
max_wall_seconds must be set. Chunks sampled earlier that still lack a selected arm are decoded
first; max_wall_seconds, min_failures and paired are also checked after each of those, and one
that becomes true during the run stops it. A criterion that already held when the run started
does not stop that catch-up, so an arm added to a finished campaign still decodes its chunks.

stop.paired is one pair object, a non-empty list of them, or null. A list holds when every listed
pair's interval is within its own rel_halfwidth (each pair is judged exactly like a single pair:
same counts, interval and guards). It lets a campaign with several test arms stop when the
reference has min_failures failures or every comparison is resolved, whichever comes first.
Every listed pair names two different arms, no pair (or its reverse) is listed twice, and every
rel_halfwidth is a finite number > 0. In status and run results the criterion keeps the name
"paired"; for a list its record holds one entry per pair (field "pairs", with "unmet" and
"pairs_met"). A single object keeps its one-pair record. Listed pairs are also summarized.

The interval on V (rtd.stats.paired_ratio, z for 95%) is the delta-method interval on ln V,
    Var(ln V) = 1/(n11 + n01) + 1/(n11 + n10) - 2 n11 / ((n11 + n01)(n11 + n10)),
n11 both arms fail, n10 only the reference, n01 only the test arm, on the shots of the chunks both
arms have completed; degenerate counts use the exact rules documented there. Its relative half
width is max(hi / V - 1, 1 - lo / V). The only guards against deciding on too few events: a pair
is unmet until both arms have failed at least once and while V is 0 or unbounded (V or the upper
limit infinite). There is no minimum count beyond that, so strongly correlated arms (few
discordant shots) can meet a +-25% width early: n11 = 10, n10 = 0, n01 = 1 already gives
-17%/+21% after 10 reference failures.

Signals: the first SIGINT/SIGTERM/SIGHUP lets the current chunk finish and then stops; a second
one abandons it (rtd_decode is terminated, its partial output removed; a later run redoes it).
SIGHUP is left alone when it is ignored at start (nohup). rtd_decode gets SIGTERM from the kernel
if the driver dies without cleaning up, and a run first terminates any decoder that an earlier,
killed run left behind (recorded in chunks/<c>/logs/*.pid), deletes raw shots left behind by
chunks whose arms are all done, and removes temporary files of interrupted runs.
A failed chunk is logged and retried once, unless a stop was requested; a second failure stops
the run with an error. A decode that exceeds its arm's decode_timeout_seconds has its process
group killed and counts as a failure. Raw shots kept from an earlier attempt are checked against
the checksums in chunk.json before they are decoded again, and regenerated if they differ.

Relay gamma draws: rtd_decode keys a shot's draws on (spec seed, row of the shot in its shots
file). Chunk files restart at row 0, so with the spec's seed alone row i of every chunk would draw
the same values. With gamma_seed_per_chunk (default), an arm whose spec has a "uniform"
gamma_source decodes chunk c with a copy of its pinned spec whose gamma_source.seed is

    seed_c = splitmix64(spec_seed XOR splitmix64(c + 1))     (64-bit arithmetic)

and nothing else changed. seed_c depends only on the spec seed and c, so every arm with the same
spec seed, and both halves of a split shot, see the same draws for the same shot, which keeps
paired comparisons and replays aligned. The copy lives next to the pinned spec while the arm is
decoded (so relative references resolve as before); done.json records seed_c. Explicit gamma
tables are used as pinned.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import stim

from rtd import log, records
from rtd.bb_code import CODE_PARAMS, get_code
from rtd.circuit import EXPERIMENTS, MemoryCircuit
from rtd.noise import NoiseModel
from rtd.records import MODES, arm_output_dir, read_json_marker
from rtd.sample import build_circuit, check_deterministic, code_record, sample_batches
from rtd.xz import HalfIndex, sha256_file, split_artifact, split_shots, write_file_atomic, write_shots_dir

logger = logging.getLogger("rtd.campaign")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "build" / "release" / "src" / "rtd_decode"
DEFAULT_CAMPAIGNS = REPO_ROOT / "data" / "campaigns"
DEFAULT_ARTIFACTS = REPO_ROOT / "data" / "artifacts"

SEED_MULTIPLIER = 1_000_003
SEED_MODULUS = 2**63
SAMPLE_BATCH = 100_000  # shots per stim sample call; part of what a chunk's seed reproduces
RAW_SHOTS_LIMIT_BYTES = 2 * 1024**3
WILSON_Z = 1.959963984540054
_PR_SET_PDEATHSIG = 1  # <linux/prctl.h>
_KILL_WAIT_SECONDS = 10.0
_MASK64 = (1 << 64) - 1
GAMMA_SEED_RULE = "splitmix64(spec_seed ^ splitmix64(chunk + 1))"
_CHUNK_SPEC_PREFIX = ".chunk-"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESERVED_ARM_NAMES = {"shots", "logs", "chunk.json"}
_DRIVER_FLAGS = {"--artifact", "--shots", "--config", "--out", "--first", "--count", "--workers", "--cpus", "--help", "-h"}
_OPTIONAL_TOP = {"rtd_decode", "pairs", "description", "gamma_seed_per_chunk"}
_ARM_FIELDS = ("spec", "workers", "cpus", "record_solutions", "save_commits", "save_solution_supports", "extra_args")
# Arm fields that only change how a decode is supervised, not what it computes.
_ARM_OPERATIONAL = {"decode_timeout_seconds"}
# Criteria that decoding already-sampled chunks can change (sampling adds shots only on new chunks).
_RESUME_CRITERIA = ("max_wall_seconds", "min_failures", "paired")


class CampaignError(RuntimeError):
    """A configuration, pinned input or chunk problem that stops the campaign."""


class DecodeTimeout(CampaignError):
    """rtd_decode ran longer than its arm's decode_timeout_seconds and was killed."""


class Abandoned(RuntimeError):
    """A second stop signal arrived while a chunk was being decoded."""


# ---------------------------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class CircuitConfig:
    code: str
    experiment: str
    p: float
    rounds: int
    relay_bp_compat: bool
    readout_in_round_order: bool = False

    @property
    def artifact_name(self) -> str:
        suffix = ("_compat" if self.relay_bp_compat else "") + ("_rorder" if self.readout_in_round_order else "")
        return f"{self.code}_{self.experiment}_p{self.p!r}_r{self.rounds}{suffix}"


@dataclass(frozen=True)
class ArmConfig:
    name: str
    spec: Path
    workers: int
    cpus: str | None
    record_solutions: int
    save_commits: bool
    save_solution_supports: bool
    extra_args: tuple[str, ...]
    decode_timeout_seconds: float | None = None

    def decode_flags(self) -> list[str]:
        flags = ["--workers", str(self.workers)]
        if self.cpus:
            flags += ["--cpus", self.cpus]
        if self.record_solutions:
            flags += ["--record-solutions", str(self.record_solutions)]
        if self.save_commits:
            flags.append("--save-commits")
        if self.save_solution_supports:
            flags.append("--save-solution-supports")
        return flags + list(self.extra_args)


@dataclass(frozen=True)
class PairedStop:
    reference: str
    test: str
    rel_halfwidth: float


@dataclass(frozen=True)
class StopConfig:
    max_shots: int | None
    max_wall_seconds: float | None
    min_failures: int | None
    failure_arm: str | None
    # One pair (a config object) or a tuple of pairs (a config list), which all must hold.
    paired: PairedStop | tuple[PairedStop, ...] | None


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    circuit: CircuitConfig
    mode: str
    seed: int
    chunk_shots: int
    arms: dict[str, ArmConfig]
    stop: StopConfig
    keep_raw: bool
    rtd_decode: Path
    pairs: tuple[tuple[str, str], ...]
    raw: dict[str, Any]
    gamma_seed_per_chunk: bool = True

    @property
    def halves(self) -> tuple[str, ...]:
        return MODES[self.mode]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _require(obj: dict[str, Any], keys: tuple[str, ...], where: str, optional: set[str] | None = None) -> None:
    if not isinstance(obj, dict):
        raise CampaignError(f"{where} must be a JSON object")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise CampaignError(f"{where} lacks required fields {missing} (every field is required; use null where allowed)")
    unknown = sorted(set(obj) - set(keys) - (optional or set()))
    if unknown:
        raise CampaignError(f"{where} has unknown fields {unknown}")


def _typed(value: Any, kind: type | tuple[type, ...], where: str, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    # bool is an int subclass; a flag must not pass as a count or the other way round.
    if isinstance(value, bool) and kind is not bool:
        raise CampaignError(f"{where} must be {kind}, got {value!r}")
    if not isinstance(value, kind):
        raise CampaignError(f"{where} must be {kind}{' or null' if allow_none else ''}, got {value!r}")
    return value


def parse_config(raw: dict[str, Any]) -> CampaignConfig:
    """Validates a campaign config (module docstring) and resolves its paths."""
    _require(raw, ("name", "circuit", "mode", "seed", "chunk_shots", "arms", "stop", "keep_raw"), "config", _OPTIONAL_TOP)
    name = _typed(raw["name"], str, "name")
    if not _NAME.match(name):
        raise CampaignError(f"name {name!r} must match {_NAME.pattern}")

    c = raw["circuit"]
    _require(c, ("code", "experiment", "p", "rounds", "relay_bp_compat"), "circuit", {"readout_in_round_order"})
    if c["code"] not in CODE_PARAMS:
        raise CampaignError(f"circuit.code {c['code']!r} is not one of {sorted(CODE_PARAMS)}")
    if c["experiment"] not in EXPERIMENTS:
        raise CampaignError(f"circuit.experiment {c['experiment']!r} is not one of {EXPERIMENTS}")
    p = float(_typed(c["p"], (int, float), "circuit.p"))
    if not 0.0 < p < 1.0:
        raise CampaignError(f"circuit.p must lie in (0, 1), got {p}")
    rounds = _typed(c["rounds"], int, "circuit.rounds")
    if rounds < 1:
        raise CampaignError(f"circuit.rounds must be at least 1, got {rounds}")
    readout_in_round_order = _typed(c.get("readout_in_round_order", False), bool, "circuit.readout_in_round_order")
    if readout_in_round_order and c["experiment"] != "choi":
        raise CampaignError("circuit.readout_in_round_order applies to the choi experiment only")
    circuit = CircuitConfig(c["code"], c["experiment"], p, rounds, _typed(c["relay_bp_compat"], bool, "circuit.relay_bp_compat"),
                            readout_in_round_order)

    mode = raw["mode"]
    if mode not in MODES:
        raise CampaignError(f"mode {mode!r} is not one of {sorted(MODES)}")
    seed = _typed(raw["seed"], int, "seed")
    if seed < 0:
        raise CampaignError(f"seed must be non-negative, got {seed}")
    chunk_shots = _typed(raw["chunk_shots"], int, "chunk_shots")
    if chunk_shots < 1:
        raise CampaignError(f"chunk_shots must be at least 1, got {chunk_shots}")

    arms_raw = raw["arms"]
    if not isinstance(arms_raw, dict) or not arms_raw:
        raise CampaignError("arms must be a non-empty object {name: arm}")
    arms = {}
    for arm_name, a in arms_raw.items():
        where = f"arms.{arm_name}"
        if not _NAME.match(arm_name) or arm_name in _RESERVED_ARM_NAMES:
            raise CampaignError(f"arm name {arm_name!r} must match {_NAME.pattern} and not be one of {sorted(_RESERVED_ARM_NAMES)}")
        _require(a, _ARM_FIELDS, where, _ARM_OPERATIONAL)
        workers = _typed(a["workers"], int, f"{where}.workers")
        if workers < 1:
            raise CampaignError(f"{where}.workers must be at least 1")
        record = _typed(a["record_solutions"], int, f"{where}.record_solutions")
        if not 0 <= record <= 20:
            raise CampaignError(f"{where}.record_solutions must lie in 0..20, got {record}")
        extra = a["extra_args"]
        if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
            raise CampaignError(f"{where}.extra_args must be a list of strings")
        owned = sorted(set(extra) & _DRIVER_FLAGS)
        if owned:
            raise CampaignError(f"{where}.extra_args may not set {owned}; the driver sets them")
        timeout = _typed(a.get("decode_timeout_seconds"), (int, float), f"{where}.decode_timeout_seconds", allow_none=True)
        if timeout is not None and not (math.isfinite(timeout) and timeout > 0):
            raise CampaignError(f"{where}.decode_timeout_seconds must be a positive number of seconds or null, got {timeout!r}")
        arms[arm_name] = ArmConfig(
            name=arm_name,
            spec=_resolve(_typed(a["spec"], str, f"{where}.spec")),
            workers=workers,
            cpus=_typed(a["cpus"], str, f"{where}.cpus", allow_none=True),
            record_solutions=record,
            save_commits=_typed(a["save_commits"], bool, f"{where}.save_commits"),
            save_solution_supports=_typed(a["save_solution_supports"], bool, f"{where}.save_solution_supports"),
            extra_args=tuple(extra),
            decode_timeout_seconds=None if timeout is None else float(timeout),
        )

    s = raw["stop"]
    _require(s, ("max_shots", "max_wall_seconds", "min_failures", "failure_arm", "paired"), "stop")
    max_shots = _typed(s["max_shots"], int, "stop.max_shots", allow_none=True)
    max_wall = _typed(s["max_wall_seconds"], (int, float), "stop.max_wall_seconds", allow_none=True)
    min_failures = _typed(s["min_failures"], int, "stop.min_failures", allow_none=True)
    failure_arm = _typed(s["failure_arm"], str, "stop.failure_arm", allow_none=True)
    if max_shots is None and max_wall is None:
        raise CampaignError("stop needs max_shots or max_wall_seconds, so the campaign is bounded")
    if max_shots is not None and max_shots < 1:
        raise CampaignError("stop.max_shots must be at least 1")
    if min_failures is not None and (failure_arm not in arms):
        raise CampaignError(f"stop.min_failures needs stop.failure_arm to name an arm, got {failure_arm!r}")
    paired: PairedStop | tuple[PairedStop, ...] | None = None
    if isinstance(s["paired"], list):
        paired = _parse_paired_list(s["paired"], arms)
    elif s["paired"] is not None:
        _require(s["paired"], ("reference", "test", "rel_halfwidth"), "stop.paired")
        paired = PairedStop(s["paired"]["reference"], s["paired"]["test"], float(s["paired"]["rel_halfwidth"]))
        if paired.reference not in arms or paired.test not in arms or paired.reference == paired.test:
            raise CampaignError("stop.paired must name two different arms")
        if not paired.rel_halfwidth > 0:
            raise CampaignError("stop.paired.rel_halfwidth must be positive")
    stop = StopConfig(max_shots, None if max_wall is None else float(max_wall), min_failures, failure_arm, paired)

    pairs = []
    for i, pr in enumerate(raw.get("pairs", [])):
        _require(pr, ("reference", "test"), f"pairs[{i}]")
        if pr["reference"] not in arms or pr["test"] not in arms:
            raise CampaignError(f"pairs[{i}] names an unknown arm")
        pairs.append((pr["reference"], pr["test"]))
    if isinstance(paired, PairedStop):
        if (paired.reference, paired.test) not in pairs:
            pairs.insert(0, (paired.reference, paired.test))
    elif paired is not None:
        listed = [(pr.reference, pr.test) for pr in paired if (pr.reference, pr.test) not in pairs]
        pairs[:0] = listed

    return CampaignConfig(
        name=name,
        circuit=circuit,
        mode=mode,
        seed=seed,
        chunk_shots=chunk_shots,
        arms=arms,
        stop=stop,
        keep_raw=_typed(raw["keep_raw"], bool, "keep_raw"),
        rtd_decode=_resolve(raw.get("rtd_decode") or DEFAULT_BINARY),
        pairs=tuple(pairs),
        raw=raw,
        gamma_seed_per_chunk=_typed(raw.get("gamma_seed_per_chunk", True), bool, "gamma_seed_per_chunk"),
    )


def _parse_paired_list(items: list[Any], arms: dict[str, ArmConfig]) -> tuple[PairedStop, ...]:
    """stop.paired given as a list: every pair must hold for the criterion to hold."""
    if not items:
        raise CampaignError("stop.paired is an empty list; use null for no paired criterion")
    out: list[PairedStop] = []
    seen: dict[tuple[str, str], int] = {}
    for i, item in enumerate(items):
        where = f"stop.paired[{i}]"
        _require(item, ("reference", "test", "rel_halfwidth"), where)
        reference = _typed(item["reference"], str, f"{where}.reference")
        test = _typed(item["test"], str, f"{where}.test")
        if reference not in arms or test not in arms or reference == test:
            raise CampaignError(f"{where} must name two different arms of the campaign, got {reference!r} and {test!r}")
        width = _typed(item["rel_halfwidth"], (int, float), f"{where}.rel_halfwidth")
        if not (math.isfinite(width) and width > 0):
            raise CampaignError(f"{where}.rel_halfwidth must be a finite number > 0, got {width!r}")
        for key, kind in (((reference, test), "twice"), ((test, reference), "again reversed")):
            if key in seen:
                raise CampaignError(f"{where} lists the pair of stop.paired[{seen[key]}] {kind} ({reference!r}, {test!r})")
        seen[(reference, test)] = i
        out.append(PairedStop(reference, test, float(width)))
    return tuple(out)


def chunk_seed(seed: int, chunk: int) -> int:
    return (seed * SEED_MULTIPLIER + chunk) % SEED_MODULUS


def splitmix64(x: int) -> int:
    """SplitMix64's output function applied to x + 0x9E3779B97F4A7C15, all modulo 2^64: a
    bijection of 64-bit integers that scatters nearby inputs across the whole range."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def chunk_gamma_seed(spec_seed: int, chunk: int) -> int:
    """The gamma seed chunk `chunk` is decoded with. Both maps are bijections, so different chunks
    of one spec seed get different seeds, and chunk c + 1 is not a neighbour of seed_c."""
    if not 0 <= spec_seed <= _MASK64 or chunk < 0:
        raise ValueError(f"spec seed {spec_seed} must be a 64-bit unsigned integer and chunk {chunk} non-negative")
    return splitmix64(spec_seed ^ splitmix64(chunk + 1))


@dataclass(frozen=True)
class GammaPlan:
    """How one arm's relay gamma draws are seeded per chunk."""

    source: str  # gamma_source.type of the pinned spec ("absent" when it has none)
    spec_seed: int | None  # set for a "uniform" source
    per_chunk: bool  # a per-chunk seed replaces spec_seed

    def seed(self, chunk: int) -> int | None:
        if self.spec_seed is None:
            return None
        return chunk_gamma_seed(self.spec_seed, chunk) if self.per_chunk else self.spec_seed

    def record(self, chunk: int) -> dict[str, Any]:
        entry: dict[str, Any] = {"source": self.source, "per_chunk": self.per_chunk, "spec_seed": self.spec_seed, "seed": self.seed(chunk)}
        if self.per_chunk:
            entry["rule"] = GAMMA_SEED_RULE
        return entry


def gamma_plan(spec: dict[str, Any], per_chunk_enabled: bool, where: str) -> GammaPlan:
    """The seeding of a parsed spec (version 1, or version 2 with a window object): a "uniform"
    gamma_source gets a per-chunk seed when enabled; any other source is left as it is."""
    source = spec.get("gamma_source") if isinstance(spec, dict) else None
    if not isinstance(source, dict):
        return GammaPlan("absent", None, False)
    kind = str(source.get("type"))
    if kind != "uniform":
        return GammaPlan(kind, None, False)
    seed = source.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MASK64:
        raise CampaignError(f"{where}: gamma_source.seed must be an unsigned 64-bit integer, got {seed!r}")
    return GammaPlan(kind, seed, per_chunk_enabled)


def chunk_spec_path(pinned: Path, chunk: int) -> Path:
    """Where chunk `chunk`'s copy of a pinned spec lives: beside it, so that references relative
    to the spec's directory resolve to the pinned files."""
    return pinned.with_name(f"{_CHUNK_SPEC_PREFIX}{chunk}.{pinned.name}")


def write_chunk_spec(pinned: Path, chunk: int, seed: int) -> Path:
    """A copy of the pinned spec with gamma_source.seed = `seed` and every other value unchanged."""
    spec = json.loads(pinned.read_text())
    spec["gamma_source"]["seed"] = seed
    target = chunk_spec_path(pinned, chunk)
    write_file_atomic(target, _json_bytes(spec))
    return target


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


# ---------------------------------------------------------------------------------------------
# Pinned inputs


@contextlib.contextmanager
def _locked(lock_file: Path) -> Iterator[None]:
    """Exclusive advisory lock, so concurrent inits do not build the same artifact twice."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("w") as f:
        t0 = time.perf_counter()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.perf_counter() - t0
        if waited > 1.0:
            logger.info("waited for lock", extra={"lock": str(lock_file), "seconds": waited})
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _verify_artifact_files(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest.get("sha256", {}).items():
        actual = sha256_file(path / name)
        if actual != expected:
            raise CampaignError(f"{path / name} does not match its manifest checksum ({actual} != {expected})")
    return manifest


def ensure_artifact(circuit_file: Path, circuit_sha: str, name: str, root: Path) -> Path:
    """The XYZ artifact of this circuit under `root`: the canonical directory, any existing artifact
    exported from the same circuit (matched by SHA-256), or a fresh export."""
    root.mkdir(parents=True, exist_ok=True)
    with _locked(root / f".{name}.lock"):
        canonical = root / name
        if (canonical / "manifest.json").is_file():
            manifest = _verify_artifact_files(canonical)
            found = (manifest.get("source_circuit") or {}).get("sha256")
            if found != circuit_sha:
                raise CampaignError(
                    f"{canonical} was exported from circuit {found}, not {circuit_sha}; "
                    "move it away or fix the circuit parameters"
                )
            logger.info("artifact reused", extra={"artifact": str(canonical)})
            return canonical
        for candidate in sorted(root.iterdir()):
            manifest_file = candidate / "manifest.json"
            if candidate.name.startswith(".") or not manifest_file.is_file():
                continue
            try:
                manifest = json.loads(manifest_file.read_text())
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "unreadable artifact manifest skipped while searching for a match",
                    exc_info=True,
                    extra={"manifest": str(manifest_file)},
                )
                continue
            if (
                "xz" not in manifest
                and manifest.get("format_version") == 1
                and (manifest.get("source_circuit") or {}).get("sha256") == circuit_sha
                and (manifest.get("pruning") or {}).get("threshold") == 0.0
            ):
                _verify_artifact_files(candidate)
                logger.info(
                    "existing artifact mapped by circuit checksum",
                    extra={"artifact": str(candidate), "canonical_name": name, "circuit_sha256": circuit_sha},
                )
                return candidate

        from rtd.export import export_artifact

        partial = root / f".{name}.partial-{os.getpid()}"
        shutil.rmtree(partial, ignore_errors=True)
        t0 = time.perf_counter()
        logger.info("exporting artifact", extra={"artifact": str(canonical), "circuit": str(circuit_file)})
        try:
            # The cached artifact carries its circuit, and its manifest names the final location.
            partial.mkdir(parents=True)
            shutil.copyfile(circuit_file, partial / "circuit.stim")
            manifest = export_artifact(partial / "circuit.stim", partial)
            manifest["source_circuit"]["path"] = str(canonical / "circuit.stim")
            (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            partial.rename(canonical)
        except BaseException:
            logger.exception("artifact export failed; partial output removed", extra={"artifact": str(canonical)})
            shutil.rmtree(partial, ignore_errors=True)
            raise
        logger.info("artifact exported", extra={"artifact": str(canonical), "seconds": time.perf_counter() - t0})
        return canonical


def ensure_xz(parent: Path, halves: tuple[str, ...], observable_type: np.ndarray) -> dict[str, Path]:
    """The XZ halves of `parent`, cached next to it as <parent>_xz/<half>/."""
    xz_dir = parent.with_name(parent.name + "_xz")
    parent_sha = sha256_file(parent / "manifest.json")
    with _locked(parent.parent / f".{xz_dir.name}.lock"):
        missing = []
        for half in halves:
            manifest_file = xz_dir / half / "manifest.json"
            if not manifest_file.is_file():
                missing.append(half)
                continue
            manifest = _verify_artifact_files(xz_dir / half)
            if manifest.get("xz", {}).get("parent_manifest_sha256") != parent_sha:
                raise CampaignError(f"{xz_dir / half} was split from a different {parent / 'manifest.json'}; move it away")
            logger.info("XZ half reused", extra={"half": half, "artifact": str(xz_dir / half)})
        if missing:
            t0 = time.perf_counter()
            split_artifact(parent, xz_dir, observable_type=observable_type, halves=tuple(missing))
            logger.info("XZ halves split", extra={"halves": missing, "out": str(xz_dir), "seconds": time.perf_counter() - t0})
    return {half: xz_dir / half for half in halves}


def binary_flags(binary: Path) -> set[str]:
    """Flags listed in the binary's --help text."""
    try:
        result = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.error("cannot run the decoder binary", exc_info=True, extra={"binary": str(binary)})
        raise CampaignError(f"cannot run {binary} --help: {e}") from e
    if result.returncode != 0:
        raise CampaignError(f"{binary} --help exited with {result.returncode}: {result.stderr[-500:]}")
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", result.stdout))


# A decoder that predates windowed decoding lacks this flag, and it would read a version-2 spec's
# window object as unknown and silently decode whole shots.
_WINDOW_FLAG = "--save-commits"


def check_binary_supports(binary: Path, arms: list[ArmConfig], specs: dict[str, dict[str, Any]] | None = None) -> None:
    """Every flag an arm passes is listed in the binary's --help; with `specs` (parsed, by arm),
    a spec asking for sliding windows needs a binary that decodes them."""
    supported = binary_flags(binary)
    for arm in arms:
        flags = {f for f in arm.decode_flags() if f.startswith("--")} | {"--artifact", "--shots", "--config", "--out"}
        unsupported = sorted(flags - supported)
        if unsupported:
            raise CampaignError(
                f"arm {arm.name}: the pinned rtd_decode ({binary}) does not support {unsupported}; "
                "build a binary that does and start a new campaign with it (config field rtd_decode)"
            )
        window = (specs or {}).get(arm.name, {}).get("window")
        mode = window.get("mode") if isinstance(window, dict) else None
        if mode not in (None, "whole_shot") and _WINDOW_FLAG not in supported:
            raise CampaignError(
                f"arm {arm.name}: the spec asks for window mode {mode!r}, but the pinned rtd_decode ({binary}) predates "
                f"windowed decoding (no {_WINDOW_FLAG}) and would decode whole shots; set rtd_decode to a newer build"
            )


def uniform_row_mismatch(artifact: Path) -> str | None:
    """Why the uniform window boundary cannot decode this artifact, or None when it can.

    That boundary decodes every window position with one window shape cut from the bulk rounds and
    reads each round's syndrome by row position within the round, so every round must list the
    same detectors, identified by (det_type, det_check), in the same order."""
    try:
        rounds = np.load(artifact / "det_round.npy").astype(np.int64)
        kinds = np.load(artifact / "det_type.npy").astype(np.int64)
        checks = np.load(artifact / "det_check.npy").astype(np.int64)
    except (OSError, ValueError) as e:
        return f"detector metadata unreadable ({e!r}), so the row order cannot be checked"
    if not rounds.size or rounds.shape != kinds.shape or rounds.shape != checks.shape:
        return f"detector metadata of inconsistent sizes {rounds.shape}, {kinds.shape}, {checks.shape}"
    sizes = np.bincount(rounds)[1:]
    if np.any(np.diff(rounds) < 0) or rounds[0] != 1 or sizes.size == 0 or np.any(sizes != sizes[0]):
        return "detectors are not grouped into rounds 1, 2, ... of equal size"
    key = (kinds * (int(checks.max()) + 1) + checks).reshape(sizes.size, int(sizes[0]))
    differs = np.flatnonzero((key != key[0]).any(axis=1))
    if differs.size:
        r = int(differs[0])
        row = int(np.flatnonzero(key[r] != key[0])[0])
        return (f"round {r + 1} lists its detectors in another (type, check) order than round 1 (position {row}: "
                f"type {int(kinds[r * sizes[0] + row])} check {int(checks[r * sizes[0] + row])} against type "
                f"{int(kinds[row])} check {int(checks[row])}); {differs.size} of {sizes.size} rounds differ")
    return None


def check_uniform_rows(arms: list[ArmConfig], specs: dict[str, dict[str, Any]], artifacts: dict[str, Path],
                       halves: tuple[str, ...]) -> None:
    """Refuses an arm whose spec asks for the uniform window boundary on an artifact whose rounds do
    not list their detectors in one order (uniform_row_mismatch): it would decode the misordered
    rounds with the syndrome bits of other detectors."""
    for arm in arms:
        window = specs.get(arm.name, {}).get("window")
        if not isinstance(window, dict) or window.get("mode") != "sliding" or window.get("boundary") != "uniform":
            continue
        for half in halves:
            reason = uniform_row_mismatch(artifacts[half])
            if reason is not None:
                logger.error("uniform boundary refused for this artifact", extra={
                    "arm": arm.name, "half": half, "artifact": str(artifacts[half]), "reason": reason,
                    "recovery": "none; build the circuit with circuit.readout_in_round_order or use the exact boundary"})
                raise CampaignError(
                    f"arm {arm.name}: the uniform window boundary needs every round of {artifacts[half]} to list its "
                    f"detectors in the same order, but {reason}; set circuit.readout_in_round_order (choi) or use the "
                    "exact boundary")
            logger.debug("uniform boundary row order checked", extra={"arm": arm.name, "half": half, "artifact": str(artifacts[half])})


def _spec_references(spec: Any) -> list[str]:
    """String values of "path" / "directory" keys anywhere in a spec: files it loads."""
    found = []
    if isinstance(spec, dict):
        for key, value in spec.items():
            if key in ("path", "directory") and isinstance(value, str):
                found.append(value)
            else:
                found += _spec_references(value)
    elif isinstance(spec, list):
        for value in spec:
            found += _spec_references(value)
    return found


def pin_spec(spec: Path, arm: str, campaign_dir: Path) -> dict[str, Any]:
    """Copies the spec byte for byte to specs/<arm>/<its file name>, with the files it references
    by relative path, and records the SHA-256 of every copied file.

    rtd_decode resolves a spec's relative references against the spec's own directory, so each
    arm gets a directory of its own: two arms whose specs use the same relative name (say
    "gammas") keep their own tables.
    """
    if not spec.is_file():
        raise CampaignError(f"spec {spec} of arm {arm} not found")
    try:
        parsed = json.loads(spec.read_text())
    except json.JSONDecodeError as e:
        raise CampaignError(f"spec {spec} of arm {arm} is not valid JSON: {e}") from e
    arm_dir = campaign_dir / "specs" / arm
    if arm_dir.exists():
        logger.warning(
            "replacing an unfinished earlier pin of this arm",
            extra={"arm": arm, "dir": str(arm_dir), "recovery": "pinning the spec again"},
        )
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True)
    target = arm_dir / spec.name
    shutil.copyfile(spec, target)
    copied = []
    for ref in _spec_references(parsed):
        ref_path = Path(ref)
        if ref_path.is_absolute():
            logger.warning(
                "spec references an absolute path, which is not pinned",
                extra={"arm": arm, "spec": str(spec), "reference": ref, "recovery": "used in place; changes to it go undetected"},
            )
            continue
        if ".." in ref_path.parts:
            raise CampaignError(f"spec {spec} references {ref} outside its directory; use an absolute path")
        source = spec.parent / ref_path
        destination = arm_dir / ref_path
        if destination == target:
            continue  # the spec naming itself
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        else:
            raise CampaignError(f"spec {spec} references {ref}, which does not exist")
        copied.append(ref)
    reference_sha = {
        f.relative_to(arm_dir).as_posix(): sha256_file(f) for f in sorted(arm_dir.rglob("*")) if f.is_file() and f != target
    }
    return {
        "source": str(spec),
        "path": str(target),
        "relative_path": target.relative_to(campaign_dir).as_posix(),
        "sha256": sha256_file(target),
        "copied_references": copied,
        "reference_sha256": reference_sha,
    }


# ---------------------------------------------------------------------------------------------
# Campaign state


@dataclass
class Campaign:
    directory: Path
    config: CampaignConfig
    init: dict[str, Any]

    @property
    def binary(self) -> Path:
        return self.directory / "bin" / "rtd_decode"

    def spec(self, arm: str) -> Path:
        # Campaigns pinned before specs got per-arm directories keep specs/<arm>.json.
        return self.directory / self.init["specs"][arm].get("relative_path", f"specs/{arm}.json")

    def artifact(self, half: str) -> Path:
        return Path(self.init["artifacts"][half]["path"])

    @property
    def gamma_seed_per_chunk(self) -> bool:
        """Whether chunks get their own gamma seeds. Campaigns initialized before the switch
        existed recorded nothing and keep the spec seeds, so their chunks stay consistent."""
        return bool(self.init.get("gamma_seed_per_chunk", False))

    def gamma_plan(self, arm: str) -> GammaPlan:
        return gamma_plan(json.loads(self.spec(arm).read_text()), self.gamma_seed_per_chunk, f"pinned spec of arm {arm}")

    def chunk_dir(self, chunk: int) -> Path:
        return self.directory / "chunks" / str(chunk)

    def chunk_records(self) -> dict[int, dict[str, Any]]:
        """chunk.json of every chunk whose sampling completed, by chunk, ascending."""
        base = self.directory / "chunks"
        if not base.is_dir():
            return {}
        found = {}
        for entry in sorted((e for e in base.iterdir() if e.name.isdigit()), key=lambda e: int(e.name)):
            record = read_json_marker(entry / "chunk.json", quarantine=True)
            if record is not None:
                found[int(entry.name)] = record
        return found

    def chunks(self) -> list[int]:
        """Chunks whose sampling completed (readable chunk.json), ascending."""
        return list(self.chunk_records())

    def chunk_record(self, chunk: int) -> dict[str, Any] | None:
        return read_json_marker(self.chunk_dir(chunk) / "chunk.json", quarantine=True)

    def done_record(self, chunk: int, arm: str) -> dict[str, Any] | None:
        return read_json_marker(self.chunk_dir(chunk) / arm / "done.json", quarantine=True)


def load_campaign(directory: Path) -> Campaign:
    config_file = directory / "campaign.json"
    init_file = directory / "init.json"
    if not config_file.is_file() or not init_file.is_file():
        raise CampaignError(f"{directory} is not an initialized campaign (campaign.json / init.json missing)")
    try:
        raw, init = json.loads(config_file.read_text()), json.loads(init_file.read_text())
    except (OSError, ValueError) as e:
        logger.error("campaign files unreadable", exc_info=True, extra={"campaign": str(directory), "recovery": "none; stopping"})
        raise CampaignError(f"{directory}: campaign.json or init.json is unreadable ({e!r}); restore it or start a new campaign") from e
    return Campaign(directory, parse_config(raw), init)


def _circuit_for(config: CampaignConfig) -> MemoryCircuit:
    c = config.circuit
    _, _, mc = build_circuit(c.code, c.experiment, c.p, c.rounds, c.relay_bp_compat, c.readout_in_round_order)
    return mc


def _compatible_update(old: CampaignConfig, new: CampaignConfig) -> list[str]:
    """Arms added to an existing campaign; anything else that changed (except the stopping
    criteria, keep_raw and pairs) would change the meaning of recorded chunks and is refused."""
    for key in ("name", "circuit", "mode", "seed", "chunk_shots"):
        if old.raw[key] != new.raw[key]:
            raise CampaignError(f"config field {key!r} differs from the initialized campaign; start a new campaign")
    if (old.raw.get("rtd_decode") or None) != (new.raw.get("rtd_decode") or None):
        raise CampaignError("config field 'rtd_decode' differs from the initialized campaign; start a new campaign")
    if old.gamma_seed_per_chunk != new.gamma_seed_per_chunk:
        raise CampaignError("config field 'gamma_seed_per_chunk' differs from the initialized campaign; start a new campaign")

    def semantic(arm_cfg: Any) -> Any:
        return {k: v for k, v in arm_cfg.items() if k not in _ARM_OPERATIONAL} if isinstance(arm_cfg, dict) else arm_cfg

    for arm, cfg in old.raw["arms"].items():
        if arm not in new.raw["arms"] or semantic(new.raw["arms"][arm]) != semantic(cfg):
            raise CampaignError(f"arm {arm!r} was removed or changed; add a new arm instead")
    return sorted(set(new.arms) - set(old.arms))


def init_campaign(
    config_file: Path,
    campaigns_root: Path = DEFAULT_CAMPAIGNS,
    artifacts_root: Path = DEFAULT_ARTIFACTS,
) -> Path:
    """Creates (or extends with new arms) data/campaigns/<name>/ and pins its inputs."""
    t_start = time.perf_counter()
    config_bytes = config_file.read_bytes()
    raw = json.loads(config_bytes)
    config = parse_config(raw)
    directory = campaigns_root / config.name
    logger.info("campaign init started", extra={"campaign": str(directory), "config": str(config_file)})

    existing: Campaign | None = None
    if (directory / "campaign.json").is_file():
        existing = load_campaign(directory)
        added = _compatible_update(existing.config, config)
        if not added and config_bytes == (directory / "campaign.json").read_bytes():
            logger.info("campaign already initialized with this config", extra={"campaign": str(directory)})
            return directory
        logger.info("extending campaign", extra={"campaign": str(directory), "added_arms": added})
        if "gamma_seed_per_chunk" not in existing.init:
            logger.warning(
                "campaign was initialized before per-chunk gamma seeds; it keeps the spec seeds",
                extra={"campaign": str(directory), "recovery": "every chunk and arm keeps the seed of its pinned spec"},
            )
    elif directory.exists() and any(directory.iterdir()):
        chunks_dir = directory / "chunks"
        if chunks_dir.is_dir() and any(chunks_dir.iterdir()):
            raise CampaignError(f"{directory} holds chunks but no campaign.json; remove it or rename the campaign")
        logger.warning(
            "completing an earlier init that did not finish",
            extra={"campaign": str(directory), "recovery": "pinning everything again"},
        )

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks").mkdir(exist_ok=True)
    init = dict(existing.init) if existing else {}

    if existing is None:
        if not config.rtd_decode.is_file():
            raise CampaignError(f"decoder binary {config.rtd_decode} not found; build it or set rtd_decode in the config")
        bin_dir = directory / "bin"
        bin_dir.mkdir(exist_ok=True)
        pinned = bin_dir / "rtd_decode"
        shutil.copyfile(config.rtd_decode, pinned)
        pinned.chmod(pinned.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        init["binary"] = {"source": str(config.rtd_decode), "path": str(pinned), "sha256": sha256_file(pinned)}
        logger.info("binary pinned", extra=init["binary"])

        mc = _circuit_for(config)
        check_deterministic(mc.circuit)
        circuit_file = directory / "circuit.stim"
        mc.circuit.to_file(circuit_file)
        circuit_sha = sha256_file(circuit_file)
        init["circuit"] = {
            "path": str(circuit_file),
            "sha256": circuit_sha,
            "detectors": mc.circuit.num_detectors,
            "observables": mc.circuit.num_observables,
            "rounds": mc.rounds,
            "observable_type": mc.observable_kind.tolist(),
        }
        raw_bytes = config.chunk_shots * mc.circuit.num_detectors
        if raw_bytes > RAW_SHOTS_LIMIT_BYTES:
            raise CampaignError(
                f"one chunk of {config.chunk_shots} shots holds {raw_bytes / 1e9:.2f} GB of raw detectors; "
                f"the limit is {RAW_SHOTS_LIMIT_BYTES / 1e9:.2f} GB, lower chunk_shots"
            )

        xyz = ensure_artifact(circuit_file, circuit_sha, config.circuit.artifact_name, artifacts_root)
        artifacts = {"xyz": xyz}
        if config.mode != "xyz":
            artifacts.update(ensure_xz(xyz, config.halves, mc.observable_kind))
        init["artifacts"] = {
            half: {"path": str(path), "manifest_sha256": sha256_file(path / "manifest.json")}
            for half, path in artifacts.items()
        }
        for half in config.halves:
            if half != "xyz":
                k_half = json.loads((artifacts[half] / "manifest.json").read_text())["num_observables"]
                if k_half == 0:
                    logger.warning("half has no paired observable; it can never fail", extra={"half": half})
        init["created"] = datetime.now(timezone.utc).isoformat()
        init["versions"] = {"stim": stim.__version__, "numpy": np.__version__, "rtd": version("rtd")}
        init["sample_batch"] = SAMPLE_BATCH
        init["gamma_seed_per_chunk"] = config.gamma_seed_per_chunk
        if config.gamma_seed_per_chunk:
            init["gamma_seed_rule"] = GAMMA_SEED_RULE
        init["specs"] = {}

    new_arms = [config.arms[a] for a in config.arms if a not in init.get("specs", {})]
    parsed_specs = {}
    for arm in new_arms:
        try:
            parsed_specs[arm.name] = json.loads(arm.spec.read_text())
        except (OSError, ValueError):
            logger.debug("spec not parsed for the binary check; pinning reports it", exc_info=True, extra={"arm": arm.name})
    check_binary_supports(directory / "bin" / "rtd_decode", new_arms, parsed_specs)
    check_uniform_rows(new_arms, parsed_specs, {h: Path(e["path"]) for h, e in init["artifacts"].items()}, config.halves)
    per_chunk = bool(init.get("gamma_seed_per_chunk", False))
    for arm in new_arms:
        pinned = pin_spec(arm.spec, arm.name, directory)
        plan = gamma_plan(json.loads(Path(pinned["path"]).read_text()), per_chunk, f"spec of arm {arm.name}")
        pinned["gamma"] = {"source": plan.source, "spec_seed": plan.spec_seed, "per_chunk": plan.per_chunk}
        init["specs"][arm.name] = pinned
        logger.info("spec pinned", extra={"arm": arm.name, **pinned})

    write_file_atomic(directory / "init.json", _json_bytes(init))
    if existing is not None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copyfile(directory / "campaign.json", directory / f"campaign.{stamp}.json")
    # Written last: its presence marks a complete init.
    write_file_atomic(directory / "campaign.json", config_bytes)
    logger.info("campaign initialized", extra={"campaign": str(directory), "seconds": time.perf_counter() - t_start})
    return directory


def verify_pins(campaign: Campaign) -> None:
    """The binary, specs, the files the specs reference and the artifact manifests are still the
    ones recorded at init."""
    if sha256_file(campaign.binary) != campaign.init["binary"]["sha256"]:
        raise CampaignError(f"{campaign.binary} changed since init")
    for arm in campaign.config.arms:
        pinned = campaign.init["specs"].get(arm)
        if pinned is None:
            raise CampaignError(f"arm {arm} is not pinned; run `rtd-campaign init` with the config again")
        spec = campaign.spec(arm)
        if sha256_file(spec) != pinned["sha256"]:
            raise CampaignError(f"{spec} changed since init")
        if pinned.get("copied_references") and "reference_sha256" not in pinned:
            logger.warning(
                "the files this arm's spec references were pinned without checksums; not verified",
                extra={"arm": arm, "references": pinned["copied_references"], "recovery": "decoding continues"},
            )
        for relative, expected in pinned.get("reference_sha256", {}).items():
            file = spec.parent / relative
            if not file.is_file() or sha256_file(file) != expected:
                raise CampaignError(f"{file}, referenced by the pinned spec of arm {arm}, changed since init")
    for half, entry in campaign.init["artifacts"].items():
        if sha256_file(Path(entry["path"]) / "manifest.json") != entry["manifest_sha256"]:
            raise CampaignError(f"artifact {entry['path']} ({half}) changed since init")


# ---------------------------------------------------------------------------------------------
# Chunks


def sample_chunk(mc: MemoryCircuit, seed: int, shots: int, batch: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Detectors and observables of one chunk, uint8 [shots, m] and [shots, k], and the time taken."""
    t0 = time.perf_counter()
    dets = np.empty((shots, mc.circuit.num_detectors), dtype=np.uint8)
    obs = np.empty((shots, mc.circuit.num_observables), dtype=np.uint8)
    for lo, hi, d, o in sample_batches(mc.circuit, shots, seed, batch):
        dets[lo:hi] = d
        obs[lo:hi] = o
    return dets, obs, time.perf_counter() - t0


def _bytes_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class _Stop:
    """Stop requests from signals: the first finishes the chunk, the second abandons it."""

    def __init__(self) -> None:
        self.count = 0
        self.signal_name = ""

    @property
    def requested(self) -> bool:
        return self.count >= 1

    @property
    def abandon(self) -> bool:
        return self.count >= 2

    def handler(self, signum: int, _frame: Any) -> None:
        self.count += 1
        self.signal_name = signal.Signals(signum).name
        if self.count == 1:
            logger.warning(
                "stop requested; finishing the current chunk (send the signal again to abandon it)",
                extra={"signal": self.signal_name},
            )
        else:
            logger.warning("abandoning the current chunk", extra={"signal": self.signal_name})


def _parent_death_hook() -> Callable[[], None] | None:
    """A hook run in the forked child before exec: the kernel sends the decoder SIGTERM when the
    driver dies, also when the driver cannot clean up (SIGKILL, the OOM killer)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        logger.warning(
            "prctl unavailable; a decoder can outlive a killed driver",
            exc_info=True,
            extra={"recovery": "the next run terminates it through its pid file"},
        )
        return None
    prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    parent, sigterm = os.getpid(), int(signal.SIGTERM)

    def hook() -> None:
        prctl(_PR_SET_PDEATHSIG, sigterm, 0, 0, 0)
        # The driver may have died between fork and prctl, in which case no signal would come.
        if os.getppid() != parent:
            os.kill(os.getpid(), sigterm)

    return hook


def _proc_stat(pid: int) -> tuple[str, str] | None:
    """(state, start time in clock ticks since boot) of process `pid`, or None if it does not exist.
    The start time tells a process apart from a later one that reuses its pid."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2 :].split()  # the command name may contain spaces and ')'
    return fields[0], fields[19]


def _running(pid: int, start_ticks: str | None) -> bool:
    stat_ = _proc_stat(pid)
    return stat_ is not None and stat_[0] != "Z" and (start_ticks is None or stat_[1] == start_ticks)


def _kill_group(pid: int, start_ticks: str | None) -> None:
    """SIGTERM, then SIGKILL, to the process group led by `pid`, waiting for the leader to exit."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError as e:
            logger.error("cannot signal a stale decoder", exc_info=True, extra={"pid": pid, "recovery": "none; stopping"})
            raise CampaignError(f"cannot stop decoder {pid} left by an earlier run: {e}") from e
        deadline = time.monotonic() + _KILL_WAIT_SECONDS
        while time.monotonic() < deadline:
            if not _running(pid, start_ticks):
                return
            time.sleep(0.05)
        logger.warning("stale decoder did not exit", extra={"pid": pid, "signal": sig.name, "seconds": _KILL_WAIT_SECONDS})
    raise CampaignError(f"decoder {pid} left by an earlier run survived SIGKILL")


def reap_stale_decoders(directory: Path) -> int:
    """Terminates decoders that an earlier run started and never stopped, found through their pid
    files. Called with the run lock held, so no live driver owns them. Returns how many."""
    reaped = 0
    for pid_file in sorted((directory / "chunks").glob("*/logs/*.pid")):
        info = read_json_marker(pid_file)
        if info is not None and _running(int(info["pid"]), info.get("start_ticks")):
            if info.get("start_ticks") is None:
                logger.warning(
                    "pid file without a start time; the process is left alone",
                    extra={"pid_file": str(pid_file), "pid": info["pid"], "recovery": "none"},
                )
            else:
                logger.warning(
                    "a decoder started by an earlier run is still running; terminating it",
                    extra={"pid": info["pid"], "pid_file": str(pid_file), "recovery": "its chunk is decoded again"},
                )
                _kill_group(int(info["pid"]), info["start_ticks"])
                reaped += 1
        pid_file.unlink(missing_ok=True)
    return reaped


def sync_tree(directory: Path) -> int:
    """fsyncs every file under `directory` and every directory in it (directory entries too).
    Returns the number of files synced."""
    files = 0
    dirs = [directory]
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            dirs.append(path)
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        files += 1
    for d in dirs:
        fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return files


def _tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def clean_leftovers(campaign: Campaign) -> dict[str, int]:
    """Removes what interrupted runs left behind: raw shots of chunks whose every arm is done
    (unless keep_raw), partially written shot directories, and per-chunk spec copies. Called with
    the run lock held, so nothing else is writing them."""
    removed = {"raw_shot_dirs": 0, "partial_dirs": 0, "chunk_specs": 0, "bytes": 0}
    base = campaign.directory / "chunks"
    for chunk_dir in sorted(base.iterdir(), key=lambda p: p.name) if base.is_dir() else []:
        for partial in chunk_dir.glob(".shots.partial-*"):
            removed["bytes"] += _tree_bytes(partial)
            shutil.rmtree(partial, ignore_errors=True)
            removed["partial_dirs"] += 1
            logger.warning("removed partially written shots of an interrupted run", extra={"dir": str(partial), "recovery": "none needed"})
        shots = chunk_dir / "shots"
        if campaign.config.keep_raw or not shots.is_dir() or not chunk_dir.name.isdigit():
            continue
        chunk = int(chunk_dir.name)
        if campaign.chunk_record(chunk) is not None and all(campaign.done_record(chunk, arm) is not None for arm in campaign.config.arms):
            size = _tree_bytes(shots)
            shutil.rmtree(shots, ignore_errors=True)
            removed["raw_shot_dirs"] += 1
            removed["bytes"] += size
            logger.info("deleted leftover raw shots of a finished chunk", extra={"chunk": chunk, "bytes": size})
    for stale in (campaign.directory / "specs").glob(f"*/{_CHUNK_SPEC_PREFIX}*"):
        stale.unlink(missing_ok=True)
        removed["chunk_specs"] += 1
        logger.warning("removed a per-chunk spec copy left by an interrupted run", extra={"path": str(stale), "recovery": "rewritten when needed"})
    return removed


class _Runner:
    def __init__(self, campaign: Campaign, stop: _Stop) -> None:
        self.c = campaign
        self.stop = stop
        self.mc: MemoryCircuit | None = None
        self.halves = {h: HalfIndex.from_artifact(campaign.artifact(h)) for h in campaign.config.halves if h != "xyz"}
        self.child_hook = _parent_death_hook()
        self.gamma = {arm: campaign.gamma_plan(arm) for arm in campaign.config.arms if arm in campaign.init.get("specs", {})}

    def check_abandon(self, chunk: int, phase: str) -> None:
        """Acts on a second stop signal between the phases of a chunk (sampling, each decode)."""
        if self.stop.abandon:
            raise Abandoned(f"chunk {chunk} abandoned before {phase}")

    def circuit(self) -> MemoryCircuit:
        if self.mc is None:
            mc = _circuit_for(self.c.config)
            scratch = self.c.directory / f".circuit_check-{os.getpid()}.stim"
            try:
                mc.circuit.to_file(scratch)
                sha = sha256_file(scratch)
            finally:
                scratch.unlink(missing_ok=True)
            if sha != self.c.init["circuit"]["sha256"]:
                raise CampaignError(
                    f"the rebuilt circuit (sha256 {sha}) differs from the one recorded at init "
                    f"({self.c.init['circuit']['sha256']}); stim or the circuit builder changed"
                )
            self.mc = mc
        return self.mc

    # -- sampling ------------------------------------------------------------------------------

    def _shots_manifest(self, chunk: int, seed: int, batch: int, half: str, m: int, k: int) -> dict[str, Any]:
        cfg = self.c.config
        return {
            "manifest_version": 1,
            "created": datetime.now(timezone.utc).isoformat(),
            "code": code_record(get_code(cfg.circuit.code)),
            "experiment": cfg.circuit.experiment,
            "rounds": cfg.circuit.rounds,
            "noise": NoiseModel.uniform(cfg.circuit.p).to_dict(),
            "relay_bp_compat": cfg.circuit.relay_bp_compat,
            "seed": seed,
            "batch": batch,
            "counts": {"detectors": m, "observables": k},
            "versions": {"stim": stim.__version__, "numpy": np.__version__, "rtd": version("rtd")},
            "campaign": {"name": cfg.name, "chunk": chunk, "half": half},
        }

    def write_shots(self, chunk: int, record: dict[str, Any]) -> float:
        """Samples chunk `chunk` (verifying against `record` when it has checksums) and writes its
        shots directory per half. Returns the sampling time."""
        mc = self.circuit()
        dets, obs, seconds = sample_chunk(mc, record["seed"], record["shots"], record["batch"])
        det_sha, obs_sha = _bytes_sha(dets), _bytes_sha(obs)
        regenerated = "detectors_sha256" in record
        if regenerated:
            if det_sha != record["detectors_sha256"] or obs_sha != record["observables_sha256"]:
                raise CampaignError(
                    f"chunk {chunk} regenerated from seed {record['seed']} differs from the recorded one "
                    "(stim version or CPU architecture changed?)"
                )
            logger.info("chunk regenerated and verified", extra={"chunk": chunk, "seconds": seconds})
        else:
            record["detectors_sha256"] = det_sha
            record["observables_sha256"] = obs_sha
        per_half = {half: self._half_shots(half, dets, obs) for half in self.c.config.halves}
        half_sha = {half: {"detectors": _bytes_sha(d), "observables": _bytes_sha(o)} for half, (d, o) in per_half.items()}
        if regenerated and "shots_sha256" in record and record["shots_sha256"] != half_sha:
            raise CampaignError(f"chunk {chunk}: the regenerated halves differ from the checksums recorded in chunk.json")
        record.setdefault("shots_sha256", half_sha)
        # Written next to the final place and renamed when every half is complete, so "shots/
        # exists" means complete shots even after a crash or a full disk.
        shots_dir = self.c.chunk_dir(chunk) / "shots"
        partial = self.c.chunk_dir(chunk) / f".shots.partial-{os.getpid()}"
        shutil.rmtree(shots_dir, ignore_errors=True)
        shutil.rmtree(partial, ignore_errors=True)
        circuit_sha = self.c.init["circuit"]["sha256"]
        try:
            for half, (d, o) in per_half.items():
                if half == "xyz":
                    metadata = {
                        "det_check.npy": mc.detectors.check,
                        "det_round.npy": mc.detectors.round,
                        "det_type.npy": mc.detectors.kind,
                        "observable_type.npy": mc.observable_kind,
                    }
                else:
                    index = self.halves[half]
                    metadata = {
                        "det_check.npy": mc.detectors.check[index.det_global],
                        "det_round.npy": mc.detectors.round[index.det_global],
                        "det_type.npy": mc.detectors.kind[index.det_global],
                        "observable_type.npy": mc.observable_kind[index.obs_global],
                    }
                manifest = self._shots_manifest(chunk, record["seed"], record["batch"], half, d.shape[1], o.shape[1])
                write_shots_dir(partial / half, d, o, manifest, circuit_sha, metadata)
            partial.rename(shots_dir)
        except BaseException:
            logger.error(
                "writing the chunk's shots failed",
                exc_info=True,
                extra={"chunk": chunk, "partial": str(partial), "recovery": "partial shots removed; they are written again on the next attempt"},
            )
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return seconds

    def _half_shots(self, half: str, dets: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (dets, obs) if half == "xyz" else split_shots(dets, obs, self.halves[half])

    def verify_shots(self, chunk: int, record: dict[str, Any]) -> bool:
        """Whether the raw shots on disk are exactly the ones chunk.json records, so that a retry
        or a resumed run may decode them instead of sampling again."""
        t0 = time.perf_counter()
        base = self.c.chunk_dir(chunk) / "shots"
        expected = record.get("shots_sha256")
        if expected is None and self.c.config.halves == ("xyz",) and "detectors_sha256" in record:
            # Written before per-half checksums were recorded: the one half is the sampled array.
            expected = {"xyz": {"detectors": record["detectors_sha256"], "observables": record["observables_sha256"]}}
        if expected is None:
            logger.warning(
                "raw shots cannot be verified (chunk.json records no per-half checksums)",
                extra={"chunk": chunk, "recovery": "regenerating them from the seed"},
            )
            return False
        try:
            for half in self.c.config.halves:
                for name in ("detectors", "observables"):
                    actual = _bytes_sha(np.load(base / half / f"{name}.npy", allow_pickle=False))
                    if actual != expected[half][name]:
                        logger.warning(
                            "raw shots on disk do not match the checksums in chunk.json",
                            extra={
                                "chunk": chunk, "half": half, "array": name, "expected": expected[half][name],
                                "actual": actual, "recovery": "regenerating them from the seed",
                            },
                        )  # fmt: skip
                        return False
        except (OSError, ValueError, KeyError):
            logger.warning(
                "raw shots on disk are unreadable",
                exc_info=True,
                extra={"chunk": chunk, "dir": str(base), "recovery": "regenerating them from the seed"},
            )
            return False
        logger.info("raw shots reused after verification", extra={"chunk": chunk, "seconds": time.perf_counter() - t0})
        return True

    def create_chunk(self, chunk: int, shots: int) -> dict[str, Any]:
        chunk_dir = self.c.chunk_dir(chunk)
        if chunk_dir.exists():
            logger.warning("removing an incomplete chunk directory", extra={"chunk": chunk, "dir": str(chunk_dir)})
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True)
        record: dict[str, Any] = {
            "chunk": chunk,
            "seed": chunk_seed(self.c.config.seed, chunk),
            "shots": shots,
            "batch": min(SAMPLE_BATCH, shots),
        }
        self._check_disk(shots)
        seconds = self.write_shots(chunk, record)
        record["sample_seconds"] = seconds
        record["created"] = datetime.now(timezone.utc).isoformat()
        write_file_atomic(chunk_dir / "chunk.json", _json_bytes(record))
        logger.info("chunk sampled", extra={"chunk": chunk, "seed": record["seed"], "shots": shots, "seconds": seconds})
        return record

    def _check_disk(self, shots: int) -> None:
        need = shots * self.c.init["circuit"]["detectors"] * 2 + (1 << 30)
        free = shutil.disk_usage(self.c.directory).free
        if free < need:
            raise CampaignError(f"only {free / 1e9:.1f} GB free on the campaign's disk; a chunk needs about {need / 1e9:.1f} GB")

    def shots_present(self, chunk: int) -> bool:
        base = self.c.chunk_dir(chunk) / "shots"
        return all((base / h / "manifest.json").is_file() for h in self.c.config.halves)

    def delete_shots(self, chunk: int) -> None:
        for partial in self.c.chunk_dir(chunk).glob(".shots.partial-*"):
            shutil.rmtree(partial, ignore_errors=True)
        if self.c.config.keep_raw:
            return
        shutil.rmtree(self.c.chunk_dir(chunk) / "shots", ignore_errors=True)
        logger.debug("raw shots deleted", extra={"chunk": chunk})

    # -- decoding ------------------------------------------------------------------------------

    def _decode(self, chunk: int, arm: ArmConfig, half: str, config: Path, gamma_seed: int | None) -> dict[str, Any]:
        self.check_abandon(chunk, f"decoding arm {arm.name}, half {half}")
        out = arm_output_dir(self.c.directory, chunk, arm.name, half)
        logs = self.c.chunk_dir(chunk) / "logs"
        logs.mkdir(exist_ok=True)
        log_file = logs / f"{arm.name}.{half}.log"
        pid_file = logs / f"{arm.name}.{half}.pid"
        cmd = [
            str(self.c.binary),
            "--artifact", str(self.c.artifact(half)),
            "--shots", str(self.c.chunk_dir(chunk) / "shots" / half),
            "--config", str(config),
            "--out", str(out),
            *arm.decode_flags(),
        ]  # fmt: skip
        context = {"chunk": chunk, "arm": arm.name, "half": half}
        logger.debug("rtd_decode started", extra={**context, "cmd": cmd, "gamma_seed": gamma_seed})
        t0 = time.perf_counter()
        timeout = arm.decode_timeout_seconds
        with log_file.open("w") as err:
            # Own session: a terminal's Ctrl-C or hangup reaches only this driver, which decides.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True, preexec_fn=self.child_hook
            )
            try:
                stat_ = _proc_stat(proc.pid)
                pid_info = {"pid": proc.pid, "start_ticks": stat_[1] if stat_ else None, "binary": str(self.c.binary)}
                write_file_atomic(pid_file, _json_bytes(pid_info))
                while True:
                    try:
                        code = proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop.abandon:
                            self._terminate(proc)
                            raise Abandoned(f"chunk {chunk}, arm {arm.name}, half {half} abandoned") from None
                        elapsed = time.perf_counter() - t0
                        if timeout is not None and elapsed > timeout:
                            logger.error(
                                "rtd_decode exceeded decode_timeout_seconds; killing its process group",
                                extra={
                                    **context, "pid": proc.pid, "timeout_seconds": timeout, "elapsed_seconds": elapsed,
                                    "log": str(log_file), "shots": str(self.c.chunk_dir(chunk) / "shots" / half),
                                    "recovery": "the chunk counts as failed: retried once, then the run stops",
                                },
                            )  # fmt: skip
                            self._terminate(proc)
                            tail = log_file.read_text(errors="replace").splitlines()[-15:]
                            raise DecodeTimeout(
                                f"rtd_decode on chunk {chunk}, arm {arm.name}, half {half} ran {elapsed:.0f} s, longer than "
                                f"decode_timeout_seconds = {timeout:g}, and was killed; log {log_file}; last lines:\n" + "\n".join(tail)
                            ) from None
            finally:
                if proc.poll() is None:
                    logger.warning(
                        "stopping rtd_decode after a driver error",
                        extra={"chunk": chunk, "arm": arm.name, "half": half, "pid": proc.pid, "recovery": "SIGTERM, then SIGKILL"},
                    )
                    self._terminate(proc)
                pid_file.unlink(missing_ok=True)
        seconds = time.perf_counter() - t0
        run_file = out / "run.json"
        if code != 0 or not run_file.is_file():
            tail = log_file.read_text(errors="replace").splitlines()[-15:]
            raise CampaignError(
                f"rtd_decode failed on chunk {chunk}, arm {arm.name}, half {half} (exit {code}); "
                f"log {log_file}; last lines:\n" + "\n".join(tail)
            )
        run = json.loads(run_file.read_text())
        used = ((run.get("decoder") or {}).get("gamma_source") or {}).get("seed")
        if gamma_seed is not None and used is not None and used != gamma_seed:
            raise CampaignError(f"rtd_decode on chunk {chunk}, arm {arm.name}, half {half} reports gamma seed {used}, not {gamma_seed}")
        summary = run.get("summary", {})
        result = {
            "shots": summary.get("shots"),
            "failures": (summary.get("block_error") or {}).get("count"),
            "not_converged": (summary.get("not_converged") or {}).get("count"),
            "iterations_mean": (summary.get("iterations") or {}).get("mean"),
            "decode_wall_seconds": summary.get("wall_seconds"),
            "run_id": run.get("run_id"),
            "seconds": seconds,
        }
        logger.info("half decoded", extra={"chunk": chunk, "arm": arm.name, "half": half, **result})
        return result

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("rtd_decode ignored SIGTERM; killing it", extra={"pid": proc.pid})
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def decode_arm(self, chunk: int, arm: ArmConfig) -> None:
        arm_dir = self.c.chunk_dir(chunk) / arm.name
        shutil.rmtree(arm_dir, ignore_errors=True)
        t0 = time.perf_counter()
        pinned = self.c.spec(arm.name)
        plan = self.gamma[arm.name]
        seed = plan.seed(chunk)
        config = pinned
        try:
            if plan.per_chunk and seed is not None:
                config = write_chunk_spec(pinned, chunk, seed)
                logger.debug("chunk spec written", extra={"chunk": chunk, "arm": arm.name, "spec": str(config), "gamma_seed": seed})
            config_sha = sha256_file(config)
            halves = {half: self._decode(chunk, arm, half, config, seed) for half in self.c.config.halves}
            arrays = _load_outputs(self.c, chunk, arm.name, ("logical_failure", "success"))
        except BaseException:
            logger.warning("removing the arm's partial output", extra={"chunk": chunk, "arm": arm.name, "dir": str(arm_dir)})
            shutil.rmtree(arm_dir, ignore_errors=True)
            raise
        finally:
            # The copy is derived from the pinned spec and seed_c, both recorded; run.json holds its contents.
            if config != pinned:
                config.unlink(missing_ok=True)
        combined = records.combine_halves(arrays)
        shots = int(combined["logical_failure"].size)
        done = {
            "chunk": chunk,
            "arm": arm.name,
            "halves": halves,
            "combined": {
                "shots": shots,
                "failures": int(combined["logical_failure"].sum()),
                "converged": int(combined["success"].sum()),
            },
            "binary_sha256": self.c.init["binary"]["sha256"],
            "spec_sha256": self.c.init["specs"][arm.name]["sha256"],
            "config_sha256": config_sha,
            "gamma": plan.record(chunk),
            "elapsed_seconds": time.perf_counter() - t0,
            "finished": datetime.now(timezone.utc).isoformat(),
        }
        # The outputs reach the disk before the marker that says they are complete.
        t_sync = time.perf_counter()
        synced = sync_tree(arm_dir)
        logger.debug("arm outputs synced", extra={"chunk": chunk, "arm": arm.name, "files": synced,
                                                  "seconds": time.perf_counter() - t_sync})
        write_file_atomic(arm_dir / "done.json", _json_bytes(done))
        logger.info(
            "arm decoded",
            extra={"chunk": chunk, "arm": arm.name, **done["combined"], "gamma_seed": seed, "seconds": done["elapsed_seconds"]},
        )

    def process_chunk(self, chunk: int, arms: list[ArmConfig], new_shots: int | None) -> None:
        """Samples (new_shots set) or regenerates chunk `chunk` as needed and decodes `arms` on it."""
        t0 = time.perf_counter()
        record = self.c.chunk_record(chunk)
        fresh = record is None
        if record is None:
            if new_shots is None:
                raise CampaignError(f"chunk {chunk} has no readable chunk.json and no shot count to sample it with")
            record = self.create_chunk(chunk, new_shots)
        pending = [a for a in arms if self.c.done_record(chunk, a.name) is None]
        # Shots just sampled were hashed while being written; older ones are checked before reuse.
        if pending and not (self.shots_present(chunk) and (fresh or self.verify_shots(chunk, record))):
            self.check_abandon(chunk, "regenerating its shots")
            self._check_disk(record["shots"])
            self.write_shots(chunk, dict(record))
        for arm in pending:
            self.decode_arm(chunk, arm)
        self.delete_shots(chunk)
        logger.info("chunk finished", extra={"chunk": chunk, "arms": [a.name for a in pending], "seconds": time.perf_counter() - t0})


def _load_outputs(campaign: Campaign, chunk: int, arm: str, names: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    arrays = {}
    for half in campaign.config.halves:
        directory = arm_output_dir(campaign.directory, chunk, arm, half)
        arrays[half] = {n: np.load(directory / f"{n}.npy", allow_pickle=False) for n in names}
    return arrays


# ---------------------------------------------------------------------------------------------
# Stopping criteria


def compute_seconds(campaign: Campaign) -> float:
    """Sampling plus decoding time recorded over all chunks and arms."""
    total = 0.0
    for chunk, record in campaign.chunk_records().items():
        total += float(record.get("sample_seconds", 0.0))
        for arm in campaign.config.arms:
            done = campaign.done_record(chunk, arm)
            if done is not None:
                total += float(done.get("elapsed_seconds", 0.0))
    return total


def paired_failures(campaign: Campaign, reference: str, test: str) -> tuple[np.ndarray, np.ndarray]:
    """Combined failure flags of two arms on the chunks both have completed."""
    common = sorted(set(records.completed_chunks(campaign.directory, reference)) & set(records.completed_chunks(campaign.directory, test)))
    if not common:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
    ref = records.load(campaign.directory, reference, names=("logical_failure",), chunks=common)
    tst = records.load(campaign.directory, test, names=("logical_failure",), chunks=common)
    return ref["logical_failure"], tst["logical_failure"]


def evaluate_stop(campaign: Campaign) -> dict[str, Any]:
    """Every configured criterion with its current value and whether it holds."""
    cfg = campaign.config.stop
    records_by_chunk = campaign.chunk_records()
    chunks = list(records_by_chunk)
    sampled = sum(int(r["shots"]) for r in records_by_chunk.values())
    criteria: dict[str, Any] = {}
    if cfg.max_shots is not None:
        criteria["max_shots"] = {"value": sampled, "threshold": cfg.max_shots, "met": sampled >= cfg.max_shots}
    if cfg.max_wall_seconds is not None:
        seconds = compute_seconds(campaign)
        criteria["max_wall_seconds"] = {"value": seconds, "threshold": cfg.max_wall_seconds, "met": seconds >= cfg.max_wall_seconds}
    if cfg.min_failures is not None and cfg.failure_arm is not None:
        failures = 0
        for c in chunks:
            done = campaign.done_record(c, cfg.failure_arm)
            if done is not None:
                failures += int(done["combined"]["failures"])
        criteria["min_failures"] = {"arm": cfg.failure_arm, "value": failures, "threshold": cfg.min_failures, "met": failures >= cfg.min_failures}
    if isinstance(cfg.paired, PairedStop):
        criteria["paired"] = _paired_criterion(campaign, cfg.paired)
    elif cfg.paired is not None:
        criteria["paired"] = _paired_list_criterion(campaign, cfg.paired)
    return {"shots_sampled": sampled, "chunks": len(chunks), "criteria": criteria, "met": [k for k, v in criteria.items() if v["met"]]}


def _paired_criterion(campaign: Campaign, paired: PairedStop) -> dict[str, Any]:
    from rtd import stats

    ref, test = paired_failures(campaign, paired.reference, paired.test)
    entry: dict[str, Any] = {"reference": paired.reference, "test": paired.test, "threshold": paired.rel_halfwidth, "shots": int(ref.size)}
    if ref.size == 0:
        return {**entry, "met": False, "reason": "no chunk decoded by both arms"}
    n11, n10, n01, _ = stats.paired_counts(ref, test)
    entry.update({"n11": int(n11), "n10": int(n10), "n01": int(n01)})
    if n11 + n10 == 0 or n11 + n01 == 0:
        return {**entry, "met": False, "reason": "V is undefined until both arms have failed"}
    ratio = stats.paired_ratio(n11, n10, n01, WILSON_Z)
    entry.update({"ratio": ratio.value, "interval": [ratio.lo, ratio.hi], "method": ratio.method})
    if not (math.isfinite(ratio.value) and ratio.value > 0 and math.isfinite(ratio.hi)):
        return {**entry, "met": False, "reason": "V is 0 or unbounded; the relative width is undefined"}
    half_width = max(ratio.hi / ratio.value - 1.0, 1.0 - ratio.lo / ratio.value)
    return {**entry, "rel_halfwidth": half_width, "met": half_width <= paired.rel_halfwidth}


def _paired_list_criterion(campaign: Campaign, pairs: tuple[PairedStop, ...]) -> dict[str, Any]:
    """All listed pairs, each judged exactly like a single paired criterion; holds when all do."""
    t0 = time.perf_counter()
    entries = [_paired_criterion(campaign, pr) for pr in pairs]
    unmet = [f"{e['reference']}/{e['test']}" for e in entries if not e["met"]]
    record = {"rule": "every pair within its rel_halfwidth", "pairs": entries,
              "pairs_met": len(entries) - len(unmet), "unmet": unmet, "met": not unmet}  # fmt: skip
    logger.debug(
        "paired list criterion evaluated",
        extra={"pairs": len(entries), "unmet": unmet,
               "reasons": {f"{e['reference']}/{e['test']}": e["reason"] for e in entries if "reason" in e},
               "seconds": time.perf_counter() - t0},
    )  # fmt: skip
    return record


# ---------------------------------------------------------------------------------------------
# Run


@contextlib.contextmanager
def _exclusive_run(directory: Path) -> Iterator[None]:
    """Only one run per campaign at a time: two would decode the same chunk into one directory."""
    with (directory / ".run.lock").open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            logger.error("another run holds the campaign lock", extra={"campaign": str(directory)})
            raise CampaignError(f"another `rtd-campaign run` is active on {directory}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class _CampaignLog:
    """Adds the campaign's log file to the root logger for the duration of a command.

    The file always receives info and above (debug too when the root logger is at debug), also
    when the driver is used as a library with logging left at Python's WARNING default.
    """

    def __init__(self, path: Path) -> None:
        self.handler = logging.FileHandler(path)
        self.handler.setFormatter(log.JsonFormatter())
        self.previous_level = logging.WARNING

    def __enter__(self) -> None:
        root = logging.getLogger()
        self.previous_level = root.level
        level = min(root.level if root.level != logging.NOTSET else logging.WARNING, logging.INFO)
        root.setLevel(level)
        self.handler.setLevel(level)
        root.addHandler(self.handler)

    def __exit__(self, *exc: object) -> None:
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self.previous_level)
        self.handler.close()


def run_campaign(
    directory: Path,
    max_wall: float | None = None,
    arms: list[str] | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    """Decodes chunks until a stopping criterion holds; returns the final stop evaluation.

    max_wall bounds this invocation (checked between chunks); max_chunks bounds the chunks it
    processes; arms restricts decoding to those arms (the others stay pending on new chunks).
    """
    campaign = load_campaign(directory)
    with _CampaignLog(directory / "campaign.log"), _exclusive_run(directory):
        stop = _Stop()
        handled = [signal.SIGINT, signal.SIGTERM]
        # A hangup (closed terminal) stops the run like SIGTERM, unless the run was started under
        # nohup, which ignores it on purpose.
        if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
            handled.append(signal.SIGHUP)
        previous = {s: signal.signal(s, stop.handler) for s in handled}
        try:
            return _run(campaign, stop, max_wall, arms, max_chunks)
        finally:
            for s, h in previous.items():
                signal.signal(s, h)


def _run(campaign: Campaign, stop: _Stop, max_wall: float | None, arm_names: list[str] | None, max_chunks: int | None) -> dict[str, Any]:
    t_start = time.perf_counter()
    cfg = campaign.config
    selected = arm_names if arm_names else list(cfg.arms)
    unknown = sorted(set(selected) - set(cfg.arms))
    if unknown:
        raise CampaignError(f"unknown arms {unknown}; the campaign has {sorted(cfg.arms)}")
    arms = [cfg.arms[a] for a in selected]
    verify_pins(campaign)
    check_binary_supports(campaign.binary, arms)
    reaped = reap_stale_decoders(campaign.directory)
    if reaped:
        logger.warning("terminated decoders left by an earlier run", extra={"count": reaped, "recovery": "their chunks are decoded again"})
    cleaned = clean_leftovers(campaign)
    if any(cleaned.values()):
        logger.info("leftovers of earlier runs removed", extra=cleaned)
    runner = _Runner(campaign, stop)
    if not campaign.gamma_seed_per_chunk and any(p.spec_seed is not None for p in runner.gamma.values()):
        logger.warning(
            "gamma draws repeat across chunks: every chunk uses its spec's seed",
            extra={
                "campaign": str(campaign.directory),
                "reason": "gamma_seed_per_chunk is false" if "gamma_seed_per_chunk" in campaign.init else "initialized before per-chunk seeds",
                "recovery": "none; summary.json states the effect on the intervals",
            },
        )
    logger.info(
        "campaign run started",
        extra={
            "campaign": str(campaign.directory), "arms": selected, "max_wall": max_wall, "max_chunks": max_chunks,
            "mode": cfg.mode, "gamma_seed_per_chunk": campaign.gamma_seed_per_chunk,
        },
    )  # fmt: skip

    processed = 0
    reason = ""

    def limits_reached() -> str:
        if stop.requested:
            return f"signal {stop.signal_name}"
        if max_wall is not None and time.perf_counter() - t_start >= max_wall:
            return "--max-wall reached"
        if max_chunks is not None and processed >= max_chunks:
            return "--chunks reached"
        return ""

    def process(chunk: int, new_shots: int | None) -> None:
        for attempt in (1, 2):
            try:
                runner.process_chunk(chunk, arms, new_shots)
                return
            except Abandoned:
                raise
            except Exception as e:
                # A retry can cost a whole chunk; after a stop request the next run redoes it instead.
                retry = attempt == 1 and not stop.requested
                if retry:
                    recovery = "retrying once"
                elif attempt == 1:
                    recovery = "not retrying because a stop was requested; the next run redoes the chunk"
                else:
                    recovery = "stopping the run"
                logger.error("chunk failed", exc_info=True, extra={"chunk": chunk, "attempt": attempt, "recovery": recovery})
                if not retry:
                    if attempt == 1:
                        raise CampaignError(
                            f"chunk {chunk} failed after a stop was requested ({stop.signal_name}); not retried, "
                            f"the next run redoes it. Error: {e}"
                        ) from e
                    raise CampaignError(f"chunk {chunk} failed twice; the run stopped. Last error: {e}") from e

    # Chunks sampled earlier that some selected arm has not decoded (resume, or an added arm).
    pending_chunks = [c for c in campaign.chunks() if any(campaign.done_record(c, a.name) is None for a in arms)]
    already_met = set(evaluate_stop(campaign)["met"]) if pending_chunks else set()
    if already_met & set(_RESUME_CRITERIA):
        logger.info(
            "criteria already met at the start do not stop decoding the chunks sampled earlier",
            extra={"met": sorted(already_met), "pending_chunks": pending_chunks},
        )
    for chunk in pending_chunks:
        reason = limits_reached()
        if reason:
            break
        process(chunk, None)
        processed += 1
        evaluation = evaluate_stop(campaign)
        newly = [k for k in _RESUME_CRITERIA if k in evaluation["met"] and k not in already_met]
        logger.info("stopping criteria evaluated after a resumed chunk", extra={"chunk": chunk, "evaluation": evaluation, "newly_met": newly})
        if newly:
            reason = "criteria met: " + ", ".join(newly)
            break

    evaluation = evaluate_stop(campaign)
    while not reason:
        reason = limits_reached()
        if reason:
            break
        evaluation = evaluate_stop(campaign)
        logger.info("stopping criteria evaluated", extra={"evaluation": evaluation})
        if evaluation["met"]:
            reason = "criteria met: " + ", ".join(evaluation["met"])
            break
        chunk = (max(campaign.chunks()) + 1) if campaign.chunks() else 0
        shots = cfg.chunk_shots
        if cfg.stop.max_shots is not None:
            shots = min(shots, cfg.stop.max_shots - evaluation["shots_sampled"])
        process(chunk, shots)
        processed += 1

    evaluation = evaluate_stop(campaign)
    evaluation["reason"] = reason
    evaluation["chunks_processed"] = processed
    evaluation["seconds"] = time.perf_counter() - t_start
    logger.info("campaign run finished", extra={"campaign": str(campaign.directory), **evaluation})
    return evaluation


# ---------------------------------------------------------------------------------------------
# Status and summaries


def status(directory: Path) -> dict[str, Any]:
    campaign = load_campaign(directory)
    arms = {}
    for arm in campaign.config.arms:
        done = [campaign.done_record(c, arm) for c in campaign.chunks()]
        done = [d for d in done if d is not None]
        arms[arm] = {
            "chunks_done": len(done),
            "shots": sum(d["combined"]["shots"] for d in done),
            "failures": sum(d["combined"]["failures"] for d in done),
            "converged": sum(d["combined"]["converged"] for d in done),
            "decode_seconds": sum(d["elapsed_seconds"] for d in done),
        }
    raw_bytes = sum(f.stat().st_size for f in (directory / "chunks").glob("*/shots/*/*.npy"))
    return {
        "campaign": campaign.config.name,
        "mode": campaign.config.mode,
        "gamma_seed_per_chunk": campaign.gamma_seed_per_chunk,
        "arms": arms,
        "stop": evaluate_stop(campaign),
        "compute_seconds": compute_seconds(campaign),
        "raw_shot_bytes": raw_bytes,
    }


def _rate_block(stats: Any, failures: int, shots: int, rounds: int, k: int | None) -> dict[str, Any]:
    block = stats.wilson(failures, shots, WILSON_Z)
    out: dict[str, Any] = {
        "p_block": {"count": failures, "of": shots, "rate": block.value, "wilson95": [block.lo, block.hi]},
        "block_per_cycle": {
            "rate": stats.block_per_cycle(block.value, rounds),
            "wilson95": [stats.block_per_cycle(block.lo, rounds), stats.block_per_cycle(block.hi, rounds)],
            "rounds": rounds,
            "formula": "p_L = 1 - (1 - P_block)^(1/R)",
        },
    }
    if k:
        try:
            out["per_qubit_per_cycle"] = {
                "rate": stats.per_qubit_per_cycle(block.value, rounds, k),
                "wilson95": [stats.per_qubit_per_cycle(block.lo, rounds, k), stats.per_qubit_per_cycle(block.hi, rounds, k)],
                "rounds": rounds,
                "logical_qubits": k,
                "formula": "p_L = [1 - (2(1 - P_block)^(1/k) - 1)^(1/R)] / 2",
            }
        except ValueError:
            logger.warning(
                "per-qubit rate undefined at this P_block; omitted",
                exc_info=True,
                extra={"failures": failures, "shots": shots, "k": k},
            )
    return out


_QUANTILES = (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999))


def _cost_block(stats: Any, values: np.ndarray, bootstrap: int | None, seed: int, scale: float = 1.0) -> dict[str, Any]:
    """Mean, extremes and nearest-rank quantiles; with bootstrap, percentile intervals on the mean
    and on every quantile (all from one set of resamples)."""
    x = np.asarray(values, dtype=np.float64) * scale
    if x.size == 0:
        return {}
    out: dict[str, Any] = {"mean": float(np.mean(x)), "min": float(x.min()), "max": float(x.max())}
    if bootstrap:
        mean = stats.bootstrap_mean(x, bootstrap, seed)
        out["mean_bootstrap95"] = [mean.lo, mean.hi]
        estimates = stats.bootstrap_quantiles(x, [q for _, q in _QUANTILES], bootstrap, seed)
        out["quantiles"] = {
            label: {"value": est.value, "bootstrap95": [est.lo, est.hi]} for (label, _), est in zip(_QUANTILES, estimates, strict=True)
        }
    else:
        out["quantiles"] = {label: {"value": stats.nearest_rank(x, q)} for label, q in _QUANTILES}
    return out


def _outcome_summary(
    stats: Any, arrays: dict[str, np.ndarray], rounds: int, k: int | None, bootstrap: int, seed: int, costs: dict[str, str]
) -> dict[str, Any]:
    fail = arrays["logical_failure"] != 0
    shots = int(fail.size)
    out: dict[str, Any] = {"shots": shots, "failures": int(fail.sum())}
    out.update(_rate_block(stats, int(fail.sum()), shots, rounds, k))
    if "success" in arrays:
        conv = stats.wilson(int(np.count_nonzero(arrays["success"])), shots, WILSON_Z)
        out["converged"] = {"count": int(np.count_nonzero(arrays["success"])), "of": shots, "fraction": conv.value, "wilson95": [conv.lo, conv.hi]}
    for label, name in costs.items():
        if name in arrays:
            is_iterations = name.startswith("iterations")
            scale = 1e-3 if name.startswith("decode_ns") else 1.0
            out[label] = _cost_block(stats, arrays[name], bootstrap if is_iterations else None, seed, scale)
    return out


def _fraction(stats: Any, hits: int, trials: int) -> dict[str, Any]:
    estimate = stats.wilson(hits, trials, WILSON_Z)
    return {"count": hits, "of": trials, "fraction": estimate.value, "wilson95": [estimate.lo, estimate.hi]}


def _window_summary(stats: Any, arrays: dict[str, np.ndarray], bootstrap: int, seed: int) -> dict[str, Any]:
    """Per-window statistics of one half of a sliding-window arm, over the window positions that
    ran a decode (win_attempts > 0; the others were decided by an earlier final window). The
    iteration intervals resample windows as if independent, although the windows of one shot
    share its syndrome."""
    attempts = arrays["win_attempts"]
    decoded = attempts > 0
    windows = int(decoded.sum())
    iterations = arrays["win_iterations"]
    out: dict[str, Any] = {
        "positions": int(attempts.shape[1]),
        "windows": windows,
        "skipped_after_final": int(decoded.size - windows),
        "iterations_per_window": _cost_block(stats, iterations[decoded], bootstrap, seed),
        # The largest window of a shot sets its latency when windows are decoded as they come.
        "max_window_iterations_per_shot": _cost_block(stats, np.where(decoded, iterations, 0).max(axis=1), bootstrap, seed),
        "legs_per_window": _cost_block(stats, arrays["win_legs"][decoded], None, seed),
        "converged": _fraction(stats, int(np.count_nonzero(arrays["win_converged"][decoded])), windows),
        "cap_hit": _fraction(stats, int(np.count_nonzero(arrays["win_cap_hit"][decoded])), windows),
        "deferred": _fraction(stats, int(np.count_nonzero(attempts[decoded] > 1)), windows),
        "deferral_attempts": int((attempts[decoded].astype(np.int64) - 1).sum()),
        "flagged_windows": int(np.count_nonzero(arrays["win_flagged"])),
        "windows_with_unexplained": int(np.count_nonzero(arrays["win_unexplained"])),
        "unexplained_total": int(arrays["win_unexplained"].astype(np.uint64).sum()),
        "virtual_commits_total": int(arrays["win_virtual"].astype(np.uint64).sum()),
    }
    if "flagged" in arrays:
        out["flagged_shots"] = _fraction(stats, int(np.count_nonzero(arrays["flagged"])), int(arrays["flagged"].size))
    if "win_decode_ns" in arrays:
        out["decode_time_us_per_window"] = _cost_block(stats, arrays["win_decode_ns"][decoded], None, seed, 1e-3)
    if "sol_count" in arrays:
        out["solutions_found_per_window"] = _cost_block(stats, arrays["sol_count"][decoded], None, seed)
    return out


def _windows_over_halves(stats: Any, per_half: dict[str, dict[str, np.ndarray]], seed: int) -> dict[str, Any] | None:
    """Window by window over the halves of a split problem decoded with the same K: iterations
    summed (one decoder doing both halves) and the maximum (one decoder per half, in parallel)."""
    halves = list(per_half)
    if len(halves) < 2 or not all("win_iterations" in per_half[h] for h in halves):
        return None
    shapes = {per_half[h]["win_iterations"].shape for h in halves}
    if len(shapes) != 1:
        logger.warning("halves have different window counts; no per-window combination", extra={"shapes": sorted(shapes)})
        return None
    stacked = np.stack([per_half[h]["win_iterations"].astype(np.uint64) for h in halves])
    decoded = np.logical_or.reduce([per_half[h]["win_attempts"] > 0 for h in halves])
    converged = np.logical_and.reduce([(per_half[h]["win_converged"] != 0) | (per_half[h]["win_attempts"] == 0) for h in halves])
    return {
        "positions": int(stacked.shape[2]),
        "windows": int(decoded.sum()),
        "iterations_sum_per_window": _cost_block(stats, stacked.sum(axis=0)[decoded], None, seed),
        "iterations_max_per_window": _cost_block(stats, stacked.max(axis=0)[decoded], None, seed),
        "converged_every_half": _fraction(stats, int(np.count_nonzero(converged[decoded])), int(decoded.sum())),
    }


def summarize(directory: Path, bootstrap: int = 1000, seed: int = 0) -> dict[str, Any]:
    """summary.json per arm (and per half), plus paired statistics for the configured pairs.
    Sliding-window arms also get per-window statistics ("windows") per half, and for split
    problems their combination over the halves ("windows_over_halves")."""
    from rtd import stats

    t0 = time.perf_counter()
    campaign = load_campaign(directory)
    cfg = campaign.config
    rounds = cfg.circuit.rounds
    logical_qubits = get_code(cfg.circuit.code).k
    names = ("logical_failure", "success", "iterations", "legs", "decode_ns", "weight")
    # Present only for sliding-window arms (and sol_count only when solutions were recorded).
    windowed = ("flagged", *records.WINDOW_ARRAYS, "sol_count")
    summary: dict[str, Any] = {
        "campaign": cfg.name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": cfg.raw,
        "bootstrap": {"resamples": bootstrap, "seed": seed},
        "conventions": {
            "p_block": "fraction of shots in which any decoded observable is wrong (split modes: any half)",
            "block_per_cycle": "1 - (1 - P_block)^(1/R)",
        },
        "gamma_seed_per_chunk": campaign.gamma_seed_per_chunk,
        "caveats": {"gamma_streams": _gamma_caveat(campaign)},
        "arms": {},
        "paired": [],
    }
    for arm in cfg.arms:
        rec = records.load(directory, arm, names=names, optional=windowed)
        if rec.shots == 0:
            summary["arms"][arm] = {"shots": 0}
            continue
        entry: dict[str, Any] = {
            "chunks": len(rec.chunks),
            "combined": _outcome_summary(
                stats,
                rec.combined,
                rounds,
                logical_qubits,
                bootstrap,
                seed,
                {"iterations": "iterations", "iterations_sum": "iterations_sum", "iterations_max": "iterations_max", "legs_sum": "legs_sum", "decode_time_us_sum": "decode_ns_sum"}
                if len(rec.halves) > 1
                else {"iterations": "iterations", "legs": "legs", "decode_time_us": "decode_ns"},
            ),
        }
        if len(rec.halves) > 1 or rec.halves[0] != "xyz":
            entry["halves"] = {
                half: _outcome_summary(
                    stats, rec.per_half[half], rounds, None, bootstrap, seed,
                    {"iterations": "iterations", "legs": "legs", "decode_time_us": "decode_ns"},
                )  # fmt: skip
                for half in rec.halves
            }
        if rec.windows(rec.halves[0]) is not None:
            if "flagged" in rec.combined:
                flagged = rec.combined["flagged"]
                entry["combined"]["flagged"] = _fraction(stats, int(np.count_nonzero(flagged)), int(flagged.size))
            if len(rec.halves) == 1:
                entry["combined"]["windows"] = _window_summary(stats, rec.per_half[rec.halves[0]], bootstrap, seed)
            for half in entry.get("halves", {}):
                entry["halves"][half]["windows"] = _window_summary(stats, rec.per_half[half], bootstrap, seed)
            over_halves = _windows_over_halves(stats, rec.per_half, seed)
            if over_halves is not None:
                entry["windows_over_halves"] = over_halves
            logger.info(
                "windowed arm summarized",
                extra={"arm": arm, "windows_per_shot": {h: rec.windows(h) for h in rec.halves}, "shots": rec.shots},
            )
        summary["arms"][arm] = entry
        logger.info("arm summarized", extra={"arm": arm, "shots": rec.shots, "failures": entry["combined"]["failures"]})

    for reference, test in cfg.pairs:
        ref, tst = paired_failures(campaign, reference, test)
        pair: dict[str, Any] = {"reference": reference, "test": test, "shots": int(ref.size)}
        if ref.size:
            n11, n10, n01, _ = stats.paired_counts(ref, tst)
            pair.update({"n11": int(n11), "n10": int(n10), "n01": int(n01)})
            ratio = stats.paired_ratio(n11, n10, n01, WILSON_Z)
            pair["ratio"] = ratio._asdict()
            if n11 + n10 > 0 and n11 + n01 > 0:
                pair["bootstrap"] = stats.paired_bootstrap_ratio(ref, tst, bootstrap, seed)._asdict()
        summary["paired"].append(pair)

    summary["seconds"] = time.perf_counter() - t0
    out = directory / "summary.json"
    write_file_atomic(out, (json.dumps(strict_json(summary), indent=2, default=_json_default, allow_nan=False) + "\n").encode())
    logger.info("summary written", extra={"summary": str(out), "seconds": summary["seconds"]})
    return summary


def _gamma_caveat(campaign: Campaign) -> str:
    if campaign.gamma_seed_per_chunk:
        return (
            f"Chunk c is decoded with gamma seed {GAMMA_SEED_RULE} (64-bit), and rtd_decode keys a shot's draws on "
            "(seed, row of the shot in its chunk), so no two shots of the campaign share draws. Both halves of a split "
            "shot use the same (seed, row) stream, so their draws are correlated within a shot; the combined outcome "
            "treats the halves as decoded independently. Arms with the same spec seed see the same draws on the same "
            "shot, as paired comparisons want. Explicit gamma tables are shared by every shot by construction."
        )
    return (
        "Every chunk uses its spec's gamma seed and rtd_decode keys a shot's draws on its row within the chunk file, "
        "so row i of every chunk, and both halves of a shot in split modes, use the same draws. The syndromes are "
        "independent, so the rates are unbiased, but failures of shots that share a row are weakly positively "
        "correlated through the shared draws. The Wilson, bootstrap and paired intervals treat shots as independent "
        "and are therefore slightly too narrow."
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def strict_json(value: Any) -> Any:
    """The value with non-finite floats written as the strings "inf", "-inf" and "nan", so the
    output is standard JSON (json.dumps would emit the non-standard Infinity and NaN)."""
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


# ---------------------------------------------------------------------------------------------
# Command line


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="create a campaign directory from a config")
    p_init.add_argument("config", type=Path)
    p_init.add_argument("--campaigns-root", type=Path, default=DEFAULT_CAMPAIGNS)
    p_init.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    p_run = sub.add_parser("run", help="decode chunks until a stopping criterion holds")
    p_run.add_argument("directory", type=Path)
    p_run.add_argument("--max-wall", type=float, default=None, help="seconds for this invocation (checked between chunks)")
    p_run.add_argument("--arms", default=None, help="comma-separated arms to decode (default all)")
    p_run.add_argument("--chunks", type=int, default=None, help="process at most this many chunks")
    p_status = sub.add_parser("status", help="progress and stopping criteria as JSON")
    p_status.add_argument("directory", type=Path)
    p_sum = sub.add_parser("summarize", help="write summary.json")
    p_sum.add_argument("directory", type=Path)
    p_sum.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples")
    p_sum.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.configure(logging.DEBUG if args.verbose else logging.INFO)
    cli_args = {k: str(v) for k, v in vars(args).items()}
    logger.info("command started", extra={"cli_args": cli_args})
    try:
        if args.command == "init":
            directory = init_campaign(args.config, args.campaigns_root, args.artifacts_root)
            print(directory)
        elif args.command == "run":
            arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
            result = run_campaign(args.directory, args.max_wall, arms, args.chunks)
            print(json.dumps(strict_json(result), indent=2, default=_json_default))
        elif args.command == "status":
            print(json.dumps(strict_json(status(args.directory)), indent=2, default=_json_default))
        elif args.command == "summarize":
            summarize(args.directory, args.bootstrap, args.seed)
    except Abandoned as e:
        logger.warning("run abandoned; the interrupted chunk will be redone by the next run", extra={"detail": str(e)})
        return 130
    except CampaignError as e:
        logger.error("campaign stopped", exc_info=True, extra={"error_detail": str(e), "cli_args": cli_args})
        return 2
    except Exception:
        logger.exception("command failed", extra={"cli_args": cli_args})
        return 1
    logger.info("command completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
