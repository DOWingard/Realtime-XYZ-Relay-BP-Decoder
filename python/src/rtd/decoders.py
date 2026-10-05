"""Relay-BP decoders for Python: whole shot, sliding windows, and streams of rounds.

The decoders are the C++ ones rtd_decode runs (through the extension module rtd._native), built
from arrays in memory instead of artifact files. With the same decoder spec and the same shots
they return the same predictions as rtd_decode, bit for bit: shot s of a batch draws its relay
gamma values from stream stream_offset + s, as shot s of an rtd_decode run does.

    import rtd, stim
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = rtd.RelayDecoder.from_detector_error_model(dem, config="relay_bp5")
    predictions = decoder.decode_batch(detectors)                  # [shots, k] uint8

    windowed = rtd.WindowedDecoder.from_detector_error_model(
        dem, width=12, commit=8, config="relay_bp5", round_of=1)   # coordinate 1 is the round
    predictions, records = windowed.decode_batch(detectors, records=True)

    stream = windowed.stream()
    for bits in rounds[:-1]:
        stream.push_round(bits)
        while stream.window_ready():
            commit = stream.decode_next()
    stream.push_final(rounds[-1])
    while stream.window_ready():
        commit = stream.decode_next()
    stream.predicted()                                             # [k] uint8

A config is a decoder spec in rtd_decode's JSON format (every field required), given as a path, a
dict, or the name of a preset in PRESETS. Its "window" object is replaced by the window arguments
for a WindowedDecoder and by {"mode": "whole_shot"} for a RelayDecoder. Everything rtd_decode
decodes on the CPU is accepted: the number format ("policy": f32 | f64, or "arithmetic": f32 | f64
| int4.2.8 | int5.2.8 | int6.2.8, the fixed-point formats needing a shift-form alpha) and, in a
version-3 spec, the "selection" policy (which solution a decode returns, when it stops, its
confidence and the low-confidence action of a window). With a policy the records hold its
confidence per decode (conf_* [shots, K]) and, for windows with a "history", the per-window
signal over the last L windows (hist_<signal> [shots, K, L], hist_state), under rtd_decode's
names.

Arrays: detection events are uint8 (or bool) arrays of shape [shots, m], or bit-packed
[shots, ceil(m / 8)] with bitorder="little" (stim's b8 layout); C-contiguous uint8 input is read
in place, anything else is converted once (logged). Every returned array is a numpy array that
owns the buffer the decoder wrote; nothing is copied on the way out.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stim

from rtd.dem import DecodingProblem, RoundOf, check_time_structure, problem_from_artifact, problem_from_dem

logger = logging.getLogger("rtd.decoders")

try:
    from rtd import _native
except ImportError as _import_error:  # pragma: no cover - exercised only without the extension
    _native = None
    _NATIVE_IMPORT_ERROR: ImportError | None = _import_error
else:
    _NATIVE_IMPORT_ERROR = None

# Relay-BP-5 of Mueller et al. (arXiv:2506.01779): leg 0 of 80 iterations with memory strength
# 0.125, then up to 600 relay legs of 60 iterations with memory strengths drawn uniformly from
# [-0.24, 0.66), stopping after 5 converged legs; plain min-sum (alpha = 1) in single precision.
RELAY_BP5: dict[str, Any] = {
    "version": 2,
    "policy": "f32",
    "backend": "cpu",
    "layout": "row_major",
    "column_order": "wavefront",
    "block_rows": 64,
    "executor": {"type": "serial"},
    "alpha": {"rule": "constant", "value": 1.0},
    "gamma0": 0.125,
    "pre_iter": 80,
    "set_max_iter": 60,
    "num_sets": 600,
    "stopping": {"rule": "after_n_converged", "count": 5},
    "gamma_source": {"type": "uniform", "seed": 1, "low": -0.24, "high": 0.66},
    "window": {"mode": "whole_shot"},
}

# Plain min-sum belief propagation, up to 10,000 iterations, no relay legs.
MIN_SUM: dict[str, Any] = {
    **RELAY_BP5,
    "gamma0": None,
    "pre_iter": 10000,
    "set_max_iter": 0,
    "num_sets": 0,
    "stopping": {"rule": "after_leg0"},
    "gamma_source": {"type": "none"},
}

PRESETS: dict[str, dict[str, Any]] = {"relay_bp5": RELAY_BP5, "min_sum": MIN_SUM}

# Warnings and errors about a batch (many shots not converged, windows at the iteration cap, flagged
# shots) are logged for at most this many batches per decoder and kind; later batches are counted
# in the decoder's totals only.
_WARNED_BATCHES = 10


def _require_native() -> Any:
    if _native is None:
        raise ImportError(
            "the extension module rtd._native is not built: configure CMake with "
            "-DRTD_ENABLE_PYTHON=ON -DRTD_PYTHON_OUTPUT_DIR=<repo>/python/src/rtd and build the "
            "rtd_python target (see python/README.md)"
        ) from _NATIVE_IMPORT_ERROR
    return _native


def load_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """The spec dict and the directory its relative gamma paths resolve against."""
    if isinstance(config, dict):
        return copy.deepcopy(config), Path.cwd()
    if isinstance(config, str) and config in PRESETS:
        return copy.deepcopy(PRESETS[config]), Path.cwd()
    path = Path(config)
    if not path.is_file():
        raise ValueError(
            f"config {config!r} is neither a decoder spec file nor a preset ({', '.join(sorted(PRESETS))})"
        )
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        logger.error("decoder spec is not valid JSON; nothing is built", exc_info=True, extra={"path": str(path)})
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(spec, dict):
        logger.error("decoder spec is not a JSON object; nothing is built", extra={"path": str(path), "type": type(spec).__name__})
        raise ValueError(f"{path} does not hold a JSON object")
    return spec, path.resolve().parent


def sliding_window(
    width: int,
    commit: int,
    *,
    converge_rounds: int | None = None,
    boundary: str = "exact",
    on_failure: str = "commit_anyway",
    max_deferrals: int = 0,
    iteration_cap: int | None = None,
) -> dict[str, Any]:
    """The spec's window object for a sliding window of W = width rounds that commits C = commit.

    converge_rounds (C', default W) are the rounds whose detectors must be explained for a
    window's decode to count as converged.
    """
    return {
        "mode": "sliding",
        "width": width,
        "commit": commit,
        "converge_rounds": width if converge_rounds is None else converge_rounds,
        "boundary": boundary,
        "on_failure": on_failure,
        "max_deferrals": max_deferrals,
        "iteration_cap": iteration_cap,
    }


def _parse_spec(spec: dict[str, Any], base: Path) -> Any:
    native = _require_native()
    try:
        return native.parse_spec(json.dumps(spec), str(base))
    except ValueError:
        logger.error("decoder spec rejected", exc_info=True, extra={"spec": spec, "base_dir": str(base)})
        raise


def _as_detection_events(shots: Any, width: int, bit_packed: bool, what: str) -> np.ndarray:
    """A C-contiguous uint8 view of the input, [shots, width]; a copy only when unavoidable."""
    array = np.asarray(shots)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{what} must be a 2-D array [shots, {width}], got shape {array.shape}")
    if array.shape[1] != width:
        raise ValueError(
            f"{what} must have {width} {'bytes' if bit_packed else 'columns'} per shot"
            f"{' (bit-packed)' if bit_packed else ''}, got shape {array.shape}"
        )
    if array.dtype == np.bool_:
        array = array.view(np.uint8)
    elif array.dtype != np.uint8:
        if bit_packed:
            raise ValueError(f"bit-packed {what} must be uint8, got {array.dtype}")
        logger.debug("detection events converted to uint8", extra={"dtype": str(array.dtype), "shape": array.shape})
        array = array.astype(np.uint8)
    if not array.flags.c_contiguous:
        logger.debug("detection events copied to C order", extra={"shape": array.shape})
        array = np.ascontiguousarray(array)
    return array


def _problem_arrays(problem: DecodingProblem) -> dict[str, Any]:
    def u8(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else np.ascontiguousarray(a, dtype=np.uint8)

    return {
        "num_rows": problem.num_detectors,
        "num_columns": problem.num_columns,
        "num_observables": problem.num_observables,
        "h_indptr": np.ascontiguousarray(problem.h_indptr, dtype=np.uint32),
        "h_indices": np.ascontiguousarray(problem.h_indices, dtype=np.uint32),
        "priors": np.ascontiguousarray(problem.priors, dtype=np.float64),
        "a_indptr": np.ascontiguousarray(problem.a_indptr, dtype=np.uint32),
        "a_indices": np.ascontiguousarray(problem.a_indices, dtype=np.uint32),
        "detector_round": None if problem.det_round is None else np.ascontiguousarray(problem.det_round, dtype=np.int32),
        "syndrome_bias": u8(problem.syndrome_bias),
        "observables_bias": u8(problem.observables_bias),
    }


class _Decoder:
    """What the whole-shot and windowed decoders share: spec, problem, batches and their logs."""

    _sliding: bool

    def __init__(
        self,
        problem: DecodingProblem,
        config: str | Path | dict[str, Any],
        window: dict[str, Any],
        *,
        workers: int,
        record_solutions: int,
    ) -> None:
        native = _require_native()
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        t0 = time.perf_counter()
        spec, base = load_config(config)
        spec["window"] = window
        self.spec: dict[str, Any] = spec
        self.problem = problem
        self._native_spec = _parse_spec(spec, base)
        try:
            self._native_problem = native.Problem(self._native_spec, **_problem_arrays(problem))
            self._native = self._build(native, workers, record_solutions)
        except (ValueError, RuntimeError):
            logger.error(
                "decoder construction failed",
                exc_info=True,
                extra={"window": window, "workers": workers, "record_solutions": record_solutions, **problem.summary()},
            )
            raise
        self.workers = workers
        self.record_solutions = record_solutions
        self._batches = 0
        self._shots = 0
        self._not_converged = 0
        self._windows_cap_hit = 0
        self._flagged = 0
        self._low_confidence = 0
        self._warned: dict[str, int] = {"not_converged": 0, "cap_hit": 0, "flagged": 0}
        logger.info(
            "decoder built",
            extra={
                "kind": type(self).__name__,
                "window": window,
                "number_format": spec.get("arithmetic", spec.get("policy")),
                "selection": spec.get("selection"),
                "workers": workers,
                "record_solutions": record_solutions,
                "detectors": problem.num_detectors,
                "columns": problem.num_columns,
                "observables": problem.num_observables,
                "seconds": time.perf_counter() - t0,
            },
        )

    def _build(self, native: Any, workers: int, record_solutions: int) -> Any:
        raise NotImplementedError

    @property
    def num_detectors(self) -> int:
        return self.problem.num_detectors

    @property
    def num_observables(self) -> int:
        return self.problem.num_observables

    @property
    def num_columns(self) -> int:
        return self.problem.num_columns

    @property
    def totals(self) -> dict[str, int]:
        """Batches, shots, non-converged shots, windows at the iteration cap, flagged shots and
        low-confidence decodes (under a selection policy) so far."""
        return {
            "batches": self._batches,
            "shots": self._shots,
            "not_converged": self._not_converged,
            "windows_cap_hit": self._windows_cap_hit,
            "flagged": self._flagged,
            "low_confidence": self._low_confidence,
        }

    def _limited(self, kind: str) -> bool:
        """Whether a batch event of this kind is still logged (at most _WARNED_BATCHES per decoder)."""
        if self._warned[kind] >= _WARNED_BATCHES:
            return False
        self._warned[kind] += 1
        return True

    def _run_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool,
        bit_packed_predictions: bool,
        stream_offset: int,
        workers: int | None,
        decodings: bool,
        commits: bool,
    ) -> dict[str, np.ndarray]:
        width = (self.num_detectors + 7) // 8 if bit_packed_shots else self.num_detectors
        events = _as_detection_events(shots, width, bit_packed_shots, "shots")
        used = self.workers if workers is None else workers
        if stream_offset < 0:
            raise ValueError(f"stream_offset must be non-negative, got {stream_offset}")
        logger.debug(
            "decoding batch",
            extra={"shots": events.shape[0], "bit_packed": bit_packed_shots, "stream_offset": stream_offset, "workers": used},
        )
        t0 = time.perf_counter()
        try:
            if self._sliding:
                arrays = self._native.decode_batch(
                    events, bit_packed_shots, stream_offset, used, bit_packed_predictions, decodings, commits
                )
            else:
                arrays = self._native.decode_batch(
                    events, bit_packed_shots, stream_offset, used, bit_packed_predictions, decodings
                )
        except (ValueError, RuntimeError):
            logger.error(
                "batch decode failed",
                exc_info=True,
                extra={"shots": events.shape[0], "stream_offset": stream_offset, "workers": used, "kind": type(self).__name__},
            )
            raise
        seconds = time.perf_counter() - t0
        shots_decoded = events.shape[0]
        not_converged = int(shots_decoded - np.count_nonzero(arrays["success"]))
        self._batches += 1
        self._shots += shots_decoded
        self._not_converged += not_converged
        fields: dict[str, Any] = {
            "shots": shots_decoded,
            "seconds": seconds,
            "not_converged": not_converged,
            "mean_iterations": float(arrays["iterations"].mean()),
        }
        if "conf_low" in arrays:
            fields["low_confidence"] = int(np.count_nonzero(arrays["conf_low"]))
            fields["stopped_early"] = int(np.count_nonzero(arrays["conf_stopped_early"]))
            fields["extra_legs"] = int(arrays["conf_extra_legs"].sum())
            self._low_confidence += fields["low_confidence"]
        if self._sliding:
            attempted = arrays["win_attempts"] > 0  # positions decided by an earlier final window ran no decode
            fields["windows_not_converged"] = int(np.count_nonzero(attempted & (arrays["win_converged"] == 0)))
            fields["windows_cap_hit"] = int(np.count_nonzero(arrays["win_cap_hit"]))
            fields["windows_flagged"] = int(np.count_nonzero(arrays["win_flagged"]))
            fields["flagged"] = int(np.count_nonzero(arrays["flagged"]))
            if "conf_low_deferrals" in arrays:
                fields["low_confidence_deferrals"] = int(arrays["conf_low_deferrals"].sum())
            self._windows_cap_hit += fields["windows_cap_hit"]
            self._flagged += fields["flagged"]
        logger.debug("batch decoded", extra=fields)
        further = {"further": "logged for at most %d batches per decoder; totals in .totals" % _WARNED_BATCHES}
        # A few non-converged shots are part of normal operation; a batch where many fail points
        # to a bad operating point or a mismatched problem.
        if not_converged > max(1, shots_decoded // 100) and self._limited("not_converged"):
            logger.warning("many shots did not converge in this batch", extra={**fields, **further})
        if self._sliding:
            window = self.spec["window"]
            confidence = (self.spec.get("selection") or {}).get("confidence") or {}
            policy = {
                "on_failure": window["on_failure"],
                "max_deferrals": window["max_deferrals"],
                "iteration_cap": window["iteration_cap"],
                "on_low": confidence.get("on_low", "none"),
            }
            # A window at the cap committed an attempt whose iteration budget ran out.
            if fields["windows_cap_hit"] > 0 and self._limited("cap_hit"):
                logger.warning("windows hit the iteration cap in this batch", extra={**fields, **policy, **further})
            # A flagged shot committed a window that did not converge (flag, or defer out of
            # attempts) or, under a selection policy, one of low confidence: its prediction is
            # the decoder's best guess and should not be trusted as a converged decode.
            if fields["flagged"] > 0 and self._limited("flagged"):
                logger.error("shots were flagged in this batch", extra={**fields, **policy, **further})
        return arrays

    @staticmethod
    def _split(arrays: dict[str, np.ndarray], records: bool) -> Any:
        predictions = arrays.pop("predicted_observables")
        return (predictions, arrays) if records else predictions

    @staticmethod
    def _check_record_options(records: bool, **asked: bool) -> None:
        """decodings / commits are records: asking for them without records=True is an error, so
        that the return type depends on records alone."""
        wanted = [name for name, value in asked.items() if value]
        if wanted and not records:
            logger.error("record arrays asked for without records=True", extra={"asked": wanted})
            raise ValueError(f"{' and '.join(wanted)} are returned with the records: pass records=True as well")


class RelayDecoder(_Decoder):
    """Relay-BP (or whichever relay configuration the spec gives) over whole shots."""

    _sliding = False

    def __init__(
        self,
        problem: DecodingProblem,
        config: str | Path | dict[str, Any],
        *,
        workers: int = 1,
        record_solutions: int = 0,
    ) -> None:
        super().__init__(problem, config, {"mode": "whole_shot"}, workers=workers, record_solutions=record_solutions)

    def _build(self, native: Any, workers: int, record_solutions: int) -> Any:
        return native.WholeShotDecoder(self._native_problem, self._native_spec, workers, record_solutions)

    @classmethod
    def from_detector_error_model(
        cls,
        dem: stim.DetectorErrorModel,
        config: str | Path | dict[str, Any],
        *,
        prune_threshold: float = 0.0,
        workers: int = 1,
        record_solutions: int = 0,
    ) -> RelayDecoder:
        """A decoder for a detector error model (build it with decompose_errors=False)."""
        return cls(problem_from_dem(dem, prune_threshold=prune_threshold), config, workers=workers, record_solutions=record_solutions)

    @classmethod
    def from_artifact(
        cls, directory: str | Path, config: str | Path | dict[str, Any], *, workers: int = 1, record_solutions: int = 0
    ) -> RelayDecoder:
        """A decoder for an artifact directory written by rtd-export."""
        return cls(problem_from_artifact(directory, rounds=False), config, workers=workers, record_solutions=record_solutions)

    def decode(self, syndrome: Any, *, stream: int = 0, return_record: bool = False) -> Any:
        """Predicted observable flips [k] of one shot; with return_record also its record.

        The record holds success, iterations, legs, best_leg, weight, decode_ns and the returned
        correction's support (fault indices).
        """
        predictions, arrays = self.decode_batch(
            np.asarray(syndrome).reshape(1, -1), stream_offset=stream, records=True, decodings=return_record
        )
        prediction = predictions[0]
        if not return_record:
            return prediction
        record: dict[str, Any] = {name: array[0] for name, array in arrays.items() if name != "decodings"}
        record["support"] = np.flatnonzero(arrays["decodings"][0]).astype(np.uint32)
        return prediction, record

    def decode_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        stream_offset: int = 0,
        workers: int | None = None,
        records: bool = False,
        decodings: bool = False,
    ) -> Any:
        """Predicted observable flips [shots, k] (bit-packed: [shots, ceil(k / 8)]).

        With records=True, returns (predictions, records): per-shot arrays under rtd_decode's
        names (success, iterations, legs, best_leg, weight, decode_ns; decodings [shots, n] with
        decodings=True; sol_* and returned_class [shots, 1, ...] when solutions are recorded;
        conf_* [shots, 1] under a selection policy). The return type depends on records only:
        decodings=True needs records=True.
        """
        self._check_record_options(records, decodings=decodings)
        arrays = self._run_batch(
            shots,
            bit_packed_shots=bit_packed_shots,
            bit_packed_predictions=bit_packed_predictions,
            stream_offset=stream_offset,
            workers=workers,
            decodings=decodings,
            commits=False,
        )
        return self._split(arrays, records)


@dataclass(frozen=True)
class WindowCommit:
    """What one decode_next of a stream decided.

    faults are global fault (column) indices, frame_delta the observable flips they imply; both
    are empty for a deferred window (deferred=True), which is decoded again, wider, once C more
    rounds have arrived. record holds the window's iterations, legs, attempts, convergence,
    weights and time; solutions the converged legs recorded (when the decoder records them).
    """

    window: int
    first_round: int
    rounds: int
    deferred: bool
    faults: np.ndarray
    frame_delta: np.ndarray
    record: dict[str, Any]
    solutions: dict[str, Any]


class Stream:
    """One stream of syndrome rounds decoded window by window (see WindowedDecoder.stream).

    Rounds 1 ... Rt - 1 go in with push_round and the readout round with push_final; each is the
    M detector bits of that round. decode_next decodes the oldest window whose rounds have all
    arrived. Pushing never decodes, so a caller chooses when decoding happens. The stream may be
    used from several threads; calls are serialised.
    """

    def __init__(self, native_stream: Any, detectors_per_round: int, rounds_total: int, num_observables: int) -> None:
        self._stream = native_stream
        self.detectors_per_round = detectors_per_round
        self.rounds_total = rounds_total
        self.num_observables = num_observables

    def reset(self, shot: int = 0) -> None:
        """Starts a new shot; `shot` keys its gamma streams, as the shot index does in a batch."""
        if shot < 0:
            raise ValueError(f"shot must be non-negative, got {shot}")
        self._stream.reset(shot)

    def _round(self, detectors: Any) -> np.ndarray:
        bits = np.asarray(detectors)
        if bits.dtype == np.bool_:
            bits = bits.view(np.uint8)
        elif bits.dtype != np.uint8:
            bits = bits.astype(np.uint8)
        return np.ascontiguousarray(bits.reshape(-1))

    def push_round(self, detectors: Any) -> None:
        try:
            self._stream.push_round(self._round(detectors))
        except (ValueError, RuntimeError):
            logger.error("round refused", exc_info=True, extra={"received": self._stream.rounds_received()})
            raise

    def push_final(self, detectors: Any) -> None:
        try:
            self._stream.push_final(self._round(detectors))
        except (ValueError, RuntimeError):
            logger.error("readout round refused", exc_info=True, extra={"received": self._stream.rounds_received()})
            raise

    def window_ready(self) -> bool:
        return bool(self._stream.window_ready())

    def decode_next(self) -> WindowCommit:
        try:
            d = self._stream.decode_next()
        except (ValueError, RuntimeError):
            logger.error("window decode failed", exc_info=True, extra={"received": self._stream.rounds_received()})
            raise
        record = d["record"]
        if not record["converged"] and record["attempts"] > 0:
            logger.debug("window did not converge", extra=record)
        return WindowCommit(
            window=d["window"],
            first_round=d["first_round"],
            rounds=d["rounds"],
            deferred=d["deferred"],
            faults=d["faults"],
            frame_delta=d["frame_delta"],
            record=record,
            solutions=d["solutions"],
        )

    def finished(self) -> bool:
        return bool(self._stream.finished())

    def closed(self) -> bool:
        return bool(self._stream.closed())

    def flagged(self) -> bool:
        return bool(self._stream.flagged())

    def rounds_received(self) -> int:
        return int(self._stream.rounds_received())

    def frame(self) -> np.ndarray:
        """The logical frame the commits so far imply, [k] uint8, before the observable bias."""
        return self._stream.frame()

    def predicted(self) -> np.ndarray:
        """The predicted observable flips so far, [k] uint8."""
        return self._stream.predicted()

    def summary(self) -> dict[str, Any]:
        return self._stream.summary()

    def records(self) -> list[dict[str, Any]]:
        """The records of this shot's window positions so far; under a selection policy each
        holds the committed attempt's confidence, low_confidence and low_confidence_deferrals."""
        return self._stream.records()["windows"]

    def history(self) -> dict[str, np.ndarray] | None:
        """Under a selection policy with a "history": the per-window signal over the last L
        windows after the last committed window ({signal: [L] float64, "state": [L] uint8,
        "lengths": [L]}); None otherwise."""
        return self._stream.history()


class WindowedDecoder(_Decoder):
    """Sliding-window Relay-BP: windows of W rounds, each committing the faults of its first C."""

    _sliding = True

    def __init__(
        self,
        problem: DecodingProblem,
        config: str | Path | dict[str, Any],
        *,
        width: int,
        commit: int,
        converge_rounds: int | None = None,
        boundary: str = "exact",
        on_failure: str = "commit_anyway",
        max_deferrals: int = 0,
        iteration_cap: int | None = None,
        workers: int = 1,
        record_solutions: int = 0,
    ) -> None:
        if problem.det_round is None:
            raise ValueError("a windowed decoder needs the round of every detector: give round_of")
        self.rounds_total, self.detectors_per_round = check_time_structure(problem.det_round)
        window = sliding_window(
            width,
            commit,
            converge_rounds=converge_rounds,
            boundary=boundary,
            on_failure=on_failure,
            max_deferrals=max_deferrals,
            iteration_cap=iteration_cap,
        )
        super().__init__(problem, config, window, workers=workers, record_solutions=record_solutions)
        self.plan: dict[str, Any] = self._native.plan()
        logger.info(
            "window plan built",
            extra={
                "window": window,
                "rounds_total": self.plan["rounds_total"],
                "detectors_per_round": self.plan["detectors_per_round"],
                "positions": self.plan["positions"],
                "shapes": [{k: s[k] for k in ("rows", "columns", "edges", "merged_columns")} for s in self.plan["shapes"]],
                "virtual_committed": self.plan["virtual_committed"],
            },
        )

    def _build(self, native: Any, workers: int, record_solutions: int) -> Any:
        return native.WindowedDecoder(self._native_problem, self._native_spec, workers, record_solutions)

    @classmethod
    def from_detector_error_model(
        cls,
        dem: stim.DetectorErrorModel,
        *,
        width: int,
        commit: int,
        config: str | Path | dict[str, Any],
        round_of: RoundOf,
        prune_threshold: float = 0.0,
        **kwargs: Any,
    ) -> WindowedDecoder:
        """A windowed decoder for a detector error model; round_of reads a detector's round from
        its coordinates (an index, or a function of the coordinate tuple)."""
        problem = problem_from_dem(dem, round_of=round_of, prune_threshold=prune_threshold)
        return cls(problem, config, width=width, commit=commit, **kwargs)

    @classmethod
    def from_artifact(
        cls, directory: str | Path, config: str | Path | dict[str, Any], *, width: int, commit: int, **kwargs: Any
    ) -> WindowedDecoder:
        """A windowed decoder for an artifact directory written by rtd-export."""
        return cls(problem_from_artifact(directory), config, width=width, commit=commit, **kwargs)

    @property
    def num_positions(self) -> int:
        """K, the window positions per shot."""
        return int(self._native.num_positions)

    def decode_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        stream_offset: int = 0,
        workers: int | None = None,
        records: bool = False,
        commits: bool = False,
        decodings: bool = False,
    ) -> Any:
        """Predicted observable flips [shots, k]; with records=True also the records.

        Records: per shot success (every window converged), iterations, legs, weight (sum of the
        committed faults' log-likelihood ratios), flagged, decode_ns; per window [shots, K]
        win_iterations, win_legs, win_attempts, win_converged, win_cap_hit, win_weight,
        win_committed_weight, win_unexplained, win_flagged, win_virtual, win_decode_ns; with
        commits=True, commit_ptr / commit_faults (the global faults window q = s*K + k committed
        are commit_faults[commit_ptr[q]:commit_ptr[q + 1]]); with decodings=True the committed
        correction [shots, n]; sol_* and returned_class when solutions are recorded; under a
        selection policy conf_* [shots, K] (conf_low_deferrals among them) and, with a history,
        hist_<signal> [shots, K, L] and hist_state. The return type depends on records only:
        commits=True and decodings=True need records=True.
        """
        self._check_record_options(records, commits=commits, decodings=decodings)
        arrays = self._run_batch(
            shots,
            bit_packed_shots=bit_packed_shots,
            bit_packed_predictions=bit_packed_predictions,
            stream_offset=stream_offset,
            workers=workers,
            decodings=decodings,
            commits=commits,
        )
        return self._split(arrays, records)

    def stream(self) -> Stream:
        """A new stream with its own decoders (it may outlive this decoder)."""
        try:
            native_stream = self._native.open_stream()
        except (ValueError, RuntimeError):
            logger.error("stream construction failed", exc_info=True)
            raise
        return Stream(native_stream, self.detectors_per_round, self.rounds_total, self.num_observables)
