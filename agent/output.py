"""
Output helpers for writing agent results to the out directory.

Writes:
  - answer.json  — list of per-turn answers; each run appends a new entry
  - answer.md    — human-readable markdown; each turn is appended as a new section

Memory persistence is handled separately by agent.memory.write_memory.
"""

from __future__ import annotations

import json
import os

from agent.schema import Answer


def _answer_to_markdown(turn: int, query: str, ans: Answer) -> str:
    parts = [f"## Turn {turn}\n\n**Question:** {query}\n", f"**Answer:** {ans.final_answer}\n"]

    if ans.citations:
        parts.append("**Citations:**")
        for c in ans.citations:
            parts.append(f"- `{c.file}` lines {c.lines}")
        parts.append("")

    if ans.assumptions:
        parts.append("**Assumptions:**")
        for a in ans.assumptions:
            parts.append(f"- {a}")
        parts.append("")

    return "\n".join(parts)


def save_outputs(out_dir: str, turn: int, query: str, ans: Answer) -> None:
    """Append this turn's answer to answer.json and answer.md in *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)

    # answer.json — load existing list, append, write back
    json_path = os.path.join(out_dir, "answer.json")
    existing: list = []
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as fh:
            existing = json.load(fh)

    existing.append({"turn": turn, **ans.model_dump()})

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    # answer.md — append this turn's section
    with open(os.path.join(out_dir, "answer.md"), "a", encoding="utf-8") as fh:
        fh.write(_answer_to_markdown(turn, query, ans) + "\n---\n\n")
