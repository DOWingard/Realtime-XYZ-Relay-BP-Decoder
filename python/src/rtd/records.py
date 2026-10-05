"""Per-shot, per-window and per-solution records of one campaign arm, concatenated over chunks.

A campaign decodes chunk c's shots into chunks/<c>/<arm>/ (one rtd_decode output directory) or,
when the problem was split by detector type, into chunks/<c>/<arm>/<half>/ for each half. A
chunk counts for an arm once chunks/<c>/chunk.json (the sampling record) and
chunks/<c>/<arm>/done.json both exist and are readable, the same rule the campaign driver uses
when it counts shots and failures.

load() concatenates every .npy array of every completed chunk along the shot axis, in ascending
chunk order, and adds two index arrays: `chunk` (the chunk of each shot) and `shot_in_chunk`
(its row within that chunk, which is also its row in the chunk's regenerated shots). Ragged
arrays stored as CSR (commit_ptr/commit_faults, solsup_ptr/solsup_idx) are concatenated with
their offsets shifted, so the pointer array still indexes the concatenated data.

Outcome of a shot when the problem was split into halves: it fails when any half predicts a
wrong observable, and it converged only when every half converged.

`combined` holds, in every mode (for the arrays that were loaded):
- the shot outcome: logical_failure (any half), flagged (any half), success (every half);
- per-shot sums and maxima of the costs over the halves: iterations_sum, iterations_max,
  legs_sum, legs_max, decode_ns_sum, decode_ns_max, and weight_sum.
With one half (the unsplit xyz problem, z_only, x_only) the sums and maxima equal the half's own
values, and `combined` also holds every array of that half under its own name (iterations,
predicted_observables, win_*, ...): the same array objects as in `half(h)`, except the outcome
arrays above, which are recomputed as uint8 with the same values. With two halves those
per-half arrays are only in `half(h)`: their columns or meaning differ between the halves.
Analysis that must work in every mode therefore uses the names in the list above.

Sliding-window arms: rtd_decode writes one record per window position k of every shot, [S, K]
arrays named in WINDOW_ARRAYS (plus `flagged` [S]), and with recorded solutions the arrays in
SOLUTION_ARRAYS ([S, K] and [S, K, N], N slots per window; K = 1 for a whole-shot arm). They are
concatenated along the shot axis like every other array, keep their K and N axes (chunks decoded
with another K or N are rejected), and stay per half: `windows(half)` and `solution_slots(half)`
give K and N. A position with win_attempts == 0 ran no decode (an earlier final window decided its
faults). `load(..., optional=...)` loads such arrays where the arm has them, so one call serves
windowed and whole-shot arms alike.

Relay gamma draws: `gamma_seeds` maps each chunk to the gamma seed the arm decoded it with (None
when its spec has no seeded source or the driver did not record one), and rtd_decode keys a shot's
draws on (that seed, shot_in_chunk). `gamma_seed` gives the seed per shot.

A marker that cannot be parsed (a crash or a full disk while an older driver wrote it) counts as
absent: the chunk is not complete for that arm, and a warning is logged.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("rtd.records")

# Mode -> the halves decoded per chunk. "xyz" is the unsplit problem.
MODES: dict[str, tuple[str, ...]] = {"xyz": ("xyz",), "xz": ("x", "z"), "z_only": ("z",), "x_only": ("x",)}

# CSR pointer array -> the data array it indexes.
PTR_PAIRS: dict[str, str] = {"commit_ptr": "commit_faults", "solsup_ptr": "solsup_idx"}
_RAGGED = set(PTR_PAIRS) | set(PTR_PAIRS.values())

# Arrays of a sliding-window decode, one entry per (shot, window position): [S, K].
WINDOW_ARRAYS: tuple[str, ...] = (
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
    "win_decode_ns",
)
# Recorded solutions: sol_count and returned_class [S, K], the rest [S, K, N].
SOLUTION_ARRAYS: tuple[str, ...] = (
    "sol_count",
    "returned_class",
    "sol_leg",
    "sol_iterations",
    "sol_weight",
    "sol_class",
    "sol_hash",
    "sol_size",
)

# Per-shot arrays combined across halves: name -> how.
_COMBINE_ANY = ("logical_failure", "flagged")
_COMBINE_ALL = ("success",)
_COMBINE_SUM_MAX = ("iterations", "legs", "decode_ns")


class RecordsError(ValueError):
    """A campaign directory or chunk is missing or inconsistent."""


def arm_output_dir(campaign: Path, chunk: int, arm: str, half: str) -> Path:
    base = campaign / "chunks" / str(chunk) / arm
    return base if half == "xyz" else base / half


def read_json_marker(path: Path, quarantine: bool = False) -> dict[str, Any] | None:
    """The JSON object in a completion marker, or None when it is missing or unreadable.

    An unreadable marker is treated as absent, so the work it marks is redone rather than every
    later command failing. With `quarantine`, it is renamed to <name>.corrupt-<UTC time> so that
    it is reported once and kept for inspection.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(
            "completion marker unreadable; treated as absent",
            exc_info=True,
            extra={"marker": str(path), "recovery": "the work it marks counts as not done"},
        )
        return None
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object, got {type(value).__name__}")
        return value
    except ValueError:  # json.JSONDecodeError is a ValueError
        recovery = "the work it marks counts as not done"
        if quarantine:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = path.with_name(f"{path.name}.corrupt-{stamp}")
            try:
                path.rename(target)
                recovery += f"; renamed to {target.name}"
            except OSError as e:
                recovery += f"; renaming it failed ({e!r})"
        logger.warning(
            "completion marker is not valid JSON; treated as absent",
            exc_info=True,
            extra={"marker": str(path), "bytes": len(text), "recovery": recovery},
        )
        return None


def completed_chunks(campaign: Path, arm: str) -> list[int]:
    """Chunks with a readable chunk.json and a readable done.json for this arm, ascending."""
    chunks_dir = campaign / "chunks"
    if not chunks_dir.is_dir():
        return []
    done = []
    for entry in chunks_dir.iterdir():
        if (
            entry.name.isdigit()
            and (entry / arm / "done.json").is_file()
            and read_json_marker(entry / arm / "done.json") is not None
            and read_json_marker(entry / "chunk.json") is not None
        ):
            done.append(int(entry.name))
    return sorted(done)


def _campaign_mode(campaign: Path) -> str:
    config_file = campaign / "campaign.json"
    if not config_file.is_file():
        raise RecordsError(f"{campaign} is not a campaign directory (no campaign.json)")
    mode = json.loads(config_file.read_text()).get("mode")
    if mode not in MODES:
        raise RecordsError(f"{config_file} has mode {mode!r}; expected one of {sorted(MODES)}")
    return mode


def load_half_dir(
    directory: Path, names: Iterable[str] | None = None, optional: Iterable[str] = ()
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every .npy array of one rtd_decode output directory (or the requested ones, which must
    exist, plus the `optional` ones that do) and its run.json."""
    run_file = directory / "run.json"
    if not run_file.is_file():
        raise RecordsError(f"{directory} has no run.json; the decode did not finish")
    run = json.loads(run_file.read_text())
    required = None if names is None else set(names)
    wanted = None if required is None else required | set(optional)
    arrays = {}
    for file in sorted(directory.glob("*.npy")):
        if wanted is None or file.stem in wanted:
            arrays[file.stem] = np.load(file, allow_pickle=False)
    if required is not None:
        missing = required - set(arrays)
        if missing:
            raise RecordsError(f"{directory} has no arrays {sorted(missing)}")
    return arrays, run


def _concatenate(parts: list[dict[str, np.ndarray]], shots: list[int], where: str) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    names = set(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if set(part) != names:
            raise RecordsError(
                f"{where}: chunk {i} has arrays {sorted(set(part) ^ names)} that another chunk lacks; "
                "the arm was decoded with different options"
            )
    out: dict[str, np.ndarray] = {}
    for name in sorted(names):
        if name in PTR_PAIRS:
            pointers = []
            offset = 0
            for i, part in enumerate(parts):
                ptr = part[name].astype(np.uint64)
                if ptr.size == 0 or ptr[0] != 0:
                    raise RecordsError(f"{where}: {name} of chunk {i} does not start at 0")
                pointers.append(ptr[1:] + np.uint64(offset) if i else ptr + np.uint64(offset))
                offset += int(ptr[-1])
            out[name] = np.concatenate(pointers)
            continue
        if name not in _RAGGED:
            for i, part in enumerate(parts):
                if part[name].ndim == 0 or part[name].shape[0] != shots[i]:
                    logger.warning(
                        "array is not indexed by shot; concatenated along its first axis anyway",
                        extra={"where": where, "array": name, "shape": list(part[name].shape), "shots": shots[i]},
                    )
                    break
            # Per-window and per-solution arrays keep their K (and N) axes; chunks decoded with
            # another window plan or number of solution slots cannot be joined.
            trailing = {part[name].shape[1:] for part in parts if part[name].ndim > 0}
            if len(trailing) > 1:
                raise RecordsError(
                    f"{where}: {name} has per-shot shapes {sorted(trailing)} in different chunks; the chunks "
                    "were decoded with different window plans or solution slots"
                )
        out[name] = np.concatenate([part[name] for part in parts], axis=0)
    for ptr, data in PTR_PAIRS.items():
        if ptr in out and data in out and int(out[ptr][-1]) != out[data].shape[0]:
            raise RecordsError(f"{where}: {ptr} ends at {int(out[ptr][-1])} but {data} has {out[data].shape[0]} entries")
    return out


def combine_halves(per_half: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The shot outcome over all halves (module docstring) and the per-half cost sums and maxima;
    with a single half, also every array of that half."""
    halves = list(per_half)
    first = per_half[halves[0]]
    combined: dict[str, np.ndarray] = dict(first) if len(halves) == 1 else {}
    for name in _COMBINE_ANY:
        if all(name in per_half[h] for h in halves):
            combined[name] = np.logical_or.reduce([per_half[h][name] != 0 for h in halves]).astype(np.uint8)
    for name in _COMBINE_ALL:
        if all(name in per_half[h] for h in halves):
            combined[name] = np.logical_and.reduce([per_half[h][name] != 0 for h in halves]).astype(np.uint8)
    for name in _COMBINE_SUM_MAX:
        if all(name in per_half[h] for h in halves):
            stacked = np.stack([per_half[h][name].astype(np.uint64) for h in halves])
            combined[f"{name}_sum"] = stacked.sum(axis=0)
            combined[f"{name}_max"] = stacked.max(axis=0)
    if all("weight" in per_half[h] for h in halves):
        combined["weight_sum"] = np.sum([per_half[h]["weight"] for h in halves], axis=0)
    return combined


@dataclass
class ArmRecords:
    """One arm's records over its completed chunks; `combined[name]` also via `records[name]`."""

    campaign: Path
    arm: str
    mode: str
    halves: tuple[str, ...]
    chunks: list[int]
    chunk: np.ndarray  # int32 [S]
    shot_in_chunk: np.ndarray  # int64 [S]
    per_half: dict[str, dict[str, np.ndarray]]
    combined: dict[str, np.ndarray]
    runs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # half -> run.json per chunk
    done: list[dict[str, Any]] = field(default_factory=list)  # done.json per chunk
    gamma_seeds: dict[int, int | None] = field(default_factory=dict)  # chunk -> gamma seed used

    @property
    def shots(self) -> int:
        return int(self.chunk.size)

    @property
    def gamma_seed(self) -> np.ndarray:
        """uint64 [S]: the gamma seed each shot was decoded with (its draws are keyed on this seed
        and shot_in_chunk). Raises when a chunk has no recorded seed."""
        unknown = sorted(c for c in self.chunks if self.gamma_seeds.get(c) is None)
        if unknown:
            raise RecordsError(f"arm {self.arm}: chunks {unknown} have no recorded gamma seed")
        per_chunk = np.array([self.gamma_seeds[c] for c in self.chunks], dtype=np.uint64)
        return per_chunk[np.searchsorted(np.asarray(self.chunks, dtype=np.int64), self.chunk)]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.combined[name]

    def half(self, name: str) -> dict[str, np.ndarray]:
        return self.per_half[name]

    def _half_arrays(self, half: str | None) -> dict[str, np.ndarray]:
        if half is None:
            if len(self.halves) != 1:
                raise RecordsError(f"arm {self.arm} has halves {self.halves}; name one")
            half = self.halves[0]
        return self.per_half[half]

    def windows(self, half: str | None = None) -> int | None:
        """K, the window positions per shot of a sliding-window arm (from its win_* arrays), or
        None when none were loaded (a whole-shot arm). `half` may be omitted with one half."""
        arrays = self._half_arrays(half)
        for name in WINDOW_ARRAYS:
            if name in arrays:
                return int(arrays[name].shape[1])
        return None

    def solution_slots(self, half: str | None = None) -> int | None:
        """N, the solution slots per window when solutions were recorded, or None."""
        arrays = self._half_arrays(half)
        for name in SOLUTION_ARRAYS:
            if name in arrays and arrays[name].ndim == 3:
                return int(arrays[name].shape[2])
        return None


def load_chunk(
    campaign: Path,
    arm: str,
    chunk: int,
    names: Iterable[str] | None = None,
    halves: Iterable[str] | None = None,
    optional: Iterable[str] = (),
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]], dict[str, Any]]:
    """(arrays per half, run.json per half, done.json) of one completed chunk of an arm."""
    mode = _campaign_mode(campaign)
    halves = tuple(halves) if halves is not None else MODES[mode]
    done_file = campaign / "chunks" / str(chunk) / arm / "done.json"
    done = read_json_marker(done_file)
    if done is None:
        raise RecordsError(f"chunk {chunk} of arm {arm} is not complete ({done_file} missing or unreadable)")
    arrays, runs = {}, {}
    for half in halves:
        if half not in MODES[mode]:
            raise RecordsError(f"half {half!r} is not decoded in mode {mode} (halves {MODES[mode]})")
        arrays[half], runs[half] = load_half_dir(arm_output_dir(campaign, chunk, arm, half), names, optional)
    return arrays, runs, done


def load(
    campaign_dir: Path | str,
    arm: str,
    names: Iterable[str] | None = None,
    halves: Iterable[str] | None = None,
    chunks: Iterable[int] | None = None,
    optional: Iterable[str] = (),
) -> ArmRecords:
    """Concatenated records of `arm` over its completed chunks (or the given ones), in chunk order.

    names: array names (file stems) to load, e.g. ("logical_failure", "iterations"); default all.
    Pass names when the arm saved large arrays (decodings, solution supports) that are not needed.
    optional: with `names`, further arrays loaded where the arm wrote them (e.g. WINDOW_ARRAYS,
    which only sliding-window arms have); every chunk must then have the same ones.
    """
    campaign = Path(campaign_dir)
    t0 = time.perf_counter()
    mode = _campaign_mode(campaign)
    halves = tuple(halves) if halves is not None else MODES[mode]
    wanted_names = None if names is None else tuple(names)
    optional_names = tuple(optional)
    available = completed_chunks(campaign, arm)
    if chunks is None:
        selected = available
    else:
        selected = sorted(set(chunks))
        missing = sorted(set(selected) - set(available))
        if missing:
            raise RecordsError(f"arm {arm} has not completed chunks {missing}")

    parts: dict[str, list[dict[str, np.ndarray]]] = {h: [] for h in halves}
    runs: dict[str, list[dict[str, Any]]] = {h: [] for h in halves}
    dones: list[dict[str, Any]] = []
    shots: list[int] = []
    for c in selected:
        arrays, run, done = load_chunk(campaign, arm, c, wanted_names, halves, optional_names)
        counts = {h: int(run[h]["summary"]["shots"]) for h in halves}
        if len(set(counts.values())) != 1:
            raise RecordsError(f"chunk {c} of arm {arm}: halves decoded different shot counts {counts}")
        shots.append(next(iter(counts.values())))
        for h in halves:
            parts[h].append(arrays[h])
            runs[h].append(run[h])
        dones.append(done)

    per_half = {h: _concatenate(parts[h], shots, f"{campaign.name}/{arm}/{h}") for h in halves}
    chunk_index = np.repeat(np.asarray(selected, dtype=np.int32), shots) if shots else np.zeros(0, np.int32)
    shot_in_chunk = (
        np.concatenate([np.arange(s, dtype=np.int64) for s in shots]) if shots else np.zeros(0, np.int64)
    )
    combined = combine_halves(per_half) if selected else {}
    logger.debug(
        "records loaded",
        extra={
            "campaign": str(campaign),
            "arm": arm,
            "chunks": len(selected),
            "shots": int(chunk_index.size),
            "seconds": time.perf_counter() - t0,
        },
    )
    return ArmRecords(
        campaign=campaign,
        arm=arm,
        mode=mode,
        halves=halves,
        chunks=selected,
        chunk=chunk_index,
        shot_in_chunk=shot_in_chunk,
        per_half=per_half,
        combined=combined,
        runs=runs,
        done=dones,
        gamma_seeds={c: (d.get("gamma") or {}).get("seed") for c, d in zip(selected, dones, strict=True)},
    )
