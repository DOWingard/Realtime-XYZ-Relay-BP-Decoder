"""Circuit generation, syndrome sampling and Relay-BP decoding for bivariate bicycle memory experiments.

The decoders (RelayDecoder, WindowedDecoder) and the sinter adapter (sinter_decoders) need the
compiled extension module rtd._native; they are imported on first use, so the rest of the
package works without it.
"""

from __future__ import annotations

from typing import Any

_LAZY = {
    "RelayDecoder": "rtd.decoders",
    "WindowedDecoder": "rtd.decoders",
    "Stream": "rtd.decoders",
    "WindowCommit": "rtd.decoders",
    "PRESETS": "rtd.decoders",
    "load_config": "rtd.decoders",
    "sliding_window": "rtd.decoders",
    "DecodingProblem": "rtd.dem",
    "problem_from_dem": "rtd.dem",
    "problem_from_artifact": "rtd.dem",
    "sinter_decoders": "rtd.sinter_adapter",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'rtd' has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
