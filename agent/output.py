"""
Output helpers for writing agent results to the out directory.

Writes:
  - answer.json  — structured Answer (Pydantic → JSON)
  - answer.md    — human-readable markdown

Memory persistence is handled separately by agent.memory.write_memory.
"""

from __future__ import annotations

import os

from agent.schema import Answer


def _answer_to_markdown(query: str, ans: Answer) -> str:
    parts = [f"## Question\n{query}\n", f"## Answer\n{ans.final_answer}\n"]

    if ans.citations:
        parts.append("## Citations")
        for c in ans.citations:
            parts.append(f"- `{c.file}` lines {c.lines}")
        parts.append("")

    if ans.assumptions:
        parts.append("## Assumptions")
        for a in ans.assumptions:
            parts.append(f"- {a}")
        parts.append("")

    return "\n".join(parts)


def save_outputs(out_dir: str, query: str, ans: Answer) -> None:
    """Write answer.json and answer.md to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "answer.json"), "w", encoding="utf-8") as fh:
        fh.write(ans.model_dump_json(indent=2))

    with open(os.path.join(out_dir, "answer.md"), "w", encoding="utf-8") as fh:
        fh.write(_answer_to_markdown(query, ans))
