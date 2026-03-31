from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Citation(BaseModel):
    file: str
    lines: str  # e.g. "12-18"


class MemoryUpdate(BaseModel):
    key: str
    value: Any


class ToolCall(BaseModel):
    tool: Literal["search_kb", "calc", "other"]
    input: str


class Answer(BaseModel):
    final_answer: str
    citations: list[Citation]
    assumptions: list[str]
    memory_updates: list[MemoryUpdate]
    tool_calls: list[ToolCall]
