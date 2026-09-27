"""Structured JSON logging shared by the command-line tools."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone

# Attributes every LogRecord carries; anything else passed via `extra=` is emitted as a field.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "context": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["error"] = repr(record.exc_info[1])
            entry["stack"] = "".join(traceback.format_exception(*record.exc_info))
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                entry[key] = value
        return json.dumps(entry, default=str)


def configure(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
