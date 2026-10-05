"""Relay-BP decoders for sinter (sinter.Decoder / sinter.CompiledDecoder).

    import sinter, rtd
    tasks = [sinter.Task(circuit=circuit,
                         detector_error_model=circuit.detector_error_model(decompose_errors=False),
                         decoder="rtd_relay")]
    sinter.collect(num_workers=6, tasks=tasks, custom_decoders=rtd.sinter_decoders(),
                   max_errors=100)

Give each task its undecomposed detector error model: without one, sinter builds a decomposed
model (decompose_errors=True), whose mechanisms stim merges differently, so the decoder would
solve another problem than rtd_decode does on the same circuit.

A compiled decoder numbers the shots it decodes 0, 1, 2, ... across its batches and draws shot
s's relay gamma values from stream s, as rtd_decode does for shot s of a shots file. Every sinter
worker compiles its own decoder, so workers reuse the same gamma draws on different shots; the
draws are independent of the syndromes, so the estimates are unbiased.

Any spec rtd_decode runs on the CPU works here, fixed-point formats and version-3 selection
policies included. For a long collection, pass require_native_sha256 (the sha256 of the
extension module's file) so that every worker refuses to decode with another build of it.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import sinter
import stim

from rtd.decoders import RelayDecoder, WindowedDecoder, load_config
from rtd.dem import RoundOf, problem_from_dem

logger = logging.getLogger("rtd.sinter")


@functools.cache
def native_module_sha256() -> str:
    """The sha256 of the loaded extension module's file (rtd._native), computed once per process."""
    from rtd import decoders

    native = decoders._require_native()
    return hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()


class RtdCompiledDecoder(sinter.CompiledDecoder):
    """A decoder built for one detector error model; decodes sinter's bit-packed batches.

    With keep_records=True every batch's records (rtd_decode's per-shot and per-window arrays,
    and for windows the committed faults) are kept in `batches` as (first stream, records), so
    that a check can compare the decodes sinter's path made with another decoder's, shot by shot.
    """

    def __init__(self, decoder: RelayDecoder | WindowedDecoder, name: str, *, keep_records: bool = False) -> None:
        self.decoder = decoder
        self.name = name
        self.next_stream = 0
        self.keep_records = keep_records
        self.batches: list[tuple[int, dict[str, np.ndarray]]] = []

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        shots = bit_packed_detection_event_data.shape[0]
        first = self.next_stream
        options: dict[str, Any] = {"bit_packed_shots": True, "bit_packed_predictions": True, "stream_offset": first}
        if self.keep_records:
            if isinstance(self.decoder, WindowedDecoder):
                options["commits"] = True
            predictions, records = self.decoder.decode_batch(bit_packed_detection_event_data, records=True, **options)
            self.batches.append((first, records))
        else:
            predictions = self.decoder.decode_batch(bit_packed_detection_event_data, **options)
        self.next_stream += shots
        return predictions


class RtdSinterDecoder(sinter.Decoder):
    """A sinter decoder running rtd's Relay-BP, whole shot (window=None) or with sliding windows.

    Holds only plain data (the spec, its base directory and the window settings), so it pickles
    into sinter's worker processes; the decoder itself is built there, per detector error model.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any] = "relay_bp5",
        *,
        window: dict[str, Any] | None = None,
        round_of: RoundOf = 1,
        prune_threshold: float = 0.0,
        workers: int = 1,
        name: str = "rtd_relay",
        require_native_sha256: str | None = None,
    ) -> None:
        spec, base = load_config(config)
        self.spec = spec
        self.base = str(base)
        self.window = None if window is None else dict(window)
        self.round_of = round_of
        self.prune_threshold = prune_threshold
        self.workers = workers
        self.name = name
        self.require_native_sha256 = require_native_sha256

    def _config(self) -> dict[str, Any]:
        spec = dict(self.spec)
        source = dict(spec.get("gamma_source", {}))
        # Relative gamma table paths were written against the spec file's directory.
        for key in ("path", "directory"):
            if key in source and not Path(source[key]).is_absolute():
                source[key] = str(Path(self.base) / source[key])
        spec["gamma_source"] = source
        return spec

    def _check_native(self) -> None:
        if self.require_native_sha256 is None:
            return
        found = native_module_sha256()
        if found != self.require_native_sha256:
            logger.error(
                "the extension module is not the build this decoder was pinned to; refusing to decode",
                extra={"decoder": self.name, "expected_sha256": self.require_native_sha256, "found_sha256": found},
            )
            raise RuntimeError(
                f"rtd._native has sha256 {found}, but {self.name} was pinned to {self.require_native_sha256}"
            )

    def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        self._check_native()
        t0 = time.perf_counter()
        try:
            if self.window is None:
                problem = problem_from_dem(dem, prune_threshold=self.prune_threshold)
                decoder: RelayDecoder | WindowedDecoder = RelayDecoder(problem, self._config(), workers=self.workers)
            else:
                problem = problem_from_dem(dem, round_of=self.round_of, prune_threshold=self.prune_threshold)
                decoder = WindowedDecoder(problem, self._config(), workers=self.workers, **self.window)
        except Exception:
            logger.error(
                "cannot compile the decoder for this detector error model",
                exc_info=True,
                extra={"decoder": self.name, "detectors": dem.num_detectors, "errors": dem.num_errors, "window": self.window},
            )
            raise
        logger.info(
            "sinter decoder compiled",
            extra={"decoder": self.name, "detectors": dem.num_detectors, "window": self.window, "seconds": time.perf_counter() - t0},
        )
        return RtdCompiledDecoder(decoder, self.name)


def sinter_decoders(
    config: str | Path | dict[str, Any] = "relay_bp5",
    *,
    windows: Iterable[tuple[int, int]] = ((12, 8),),
    round_of: RoundOf = 1,
    converge_rounds: int | None = None,
    boundary: str = "exact",
    on_failure: str = "commit_anyway",
    max_deferrals: int = 0,
    iteration_cap: int | None = None,
    prune_threshold: float = 0.0,
    workers: int = 1,
    prefix: str = "rtd",
    require_native_sha256: str | None = None,
) -> dict[str, sinter.Decoder]:
    """Custom decoders for sinter.collect: "<prefix>_relay" (whole shot) and, for every (W, C) in
    `windows`, "<prefix>_window_<W>_<C>" (sliding windows of W rounds committing C). round_of
    reads a detector's round from its coordinates (index 1 for this project's circuits). With
    require_native_sha256, each decoder refuses to compile in a process whose extension module
    has another sha256 (see native_module_sha256)."""
    decoders: dict[str, sinter.Decoder] = {
        f"{prefix}_relay": RtdSinterDecoder(
            config,
            prune_threshold=prune_threshold,
            workers=workers,
            name=f"{prefix}_relay",
            require_native_sha256=require_native_sha256,
        )
    }
    for width, commit in windows:
        name = f"{prefix}_window_{width}_{commit}"
        decoders[name] = RtdSinterDecoder(
            config,
            window={
                "width": width,
                "commit": commit,
                "converge_rounds": converge_rounds,
                "boundary": boundary,
                "on_failure": on_failure,
                "max_deferrals": max_deferrals,
                "iteration_cap": iteration_cap,
            },
            round_of=round_of,
            prune_threshold=prune_threshold,
            workers=workers,
            name=name,
            require_native_sha256=require_native_sha256,
        )
    logger.debug("sinter decoders", extra={"names": sorted(decoders)})
    return decoders
