"""
Memory helpers for persisting agent state between conversation turns.

Memory is stored as a flat JSON key-value file (memory.json) in the out
directory. It captures facts the user has stated (plan, billing cycle, etc.)
but never stores conversation history.
"""

from __future__ import annotations

import json
import os
from typing import Any

_FILENAME = "memory.json"


def _path(out_dir: str) -> str:
    return os.path.join(out_dir, _FILENAME)


def read_memory(out_dir: str) -> dict[str, Any]:
    """Load memory from *out_dir*/memory.json.

    Returns an empty dict if the file does not exist yet.
    """
    p = _path(out_dir)
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_memory(out_dir: str, memory: dict[str, Any]) -> None:
    """Overwrite *out_dir*/memory.json with *memory* (full replacement)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(_path(out_dir), "w", encoding="utf-8") as fh:
        json.dump(memory, fh, indent=2)


def update_memory(out_dir: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge *updates* into the existing memory and persist the result.

    Returns the updated memory dict.
    """
    memory = read_memory(out_dir)
    memory.update(updates)
    write_memory(out_dir, memory)
    return memory
