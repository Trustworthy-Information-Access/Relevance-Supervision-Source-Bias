"""Shared helpers for command implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.data import write_json


def emit(value: Any, output: str | Path | None = None) -> None:
    if output:
        write_json(value, output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def named_paths(items: list[str], *, option: str) -> dict[str, str]:
    """Parse repeated ``name=path`` command arguments."""
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Each {option} must have the form name=path")
        name, path = item.split("=", 1)
        if not name or not path or name in values:
            raise ValueError(f"Names must be non-empty and unique: {name!r}")
        values[name] = path
    return values
