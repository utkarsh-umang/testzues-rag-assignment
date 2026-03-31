"""
Keyword search retriever for the AcmeCloud KB.

Scoring:
  - Full phrase match  → score 2.0  (query found verbatim in a line)
  - Keyword match      → score 0.0–1.0  (fraction of meaningful keywords matched)

Each result carries the source file, 1-based line range (with ±2 lines of
context), the matched text, a score, a match_type, and the matched terms.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Stopwords – stripped before keyword extraction
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "i", "you", "we", "they",
        "it", "this", "that", "these", "those", "what", "how", "when", "where",
        "which", "who", "whom", "my", "your", "our", "their", "its", "me",
        "him", "her", "us", "them", "if", "not", "no", "so", "as", "up",
        "out", "about", "just", "also", "only",
    }
)

_CONTEXT_WINDOW = 2  # lines of context to include around each match


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    file: str            # relative (or absolute) path to the KB file
    lines: str           # 1-based range, e.g. "3-7"
    text: str            # matched text with context lines
    score: float         # higher = more relevant
    match_type: str      # "phrase" | "keyword"
    matched_terms: list[str] = field(default_factory=list)

    def citation(self) -> dict:
        """Return the citation dict expected by answer.json."""
        return {"file": self.file, "lines": self.lines}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class KeywordSearchRetriever:
    """
    Load all *.md files from *kb_dir* and provide keyword / phrase search.

    Usage::

        retriever = KeywordSearchRetriever("kb")
        results = retriever.search("what is the refund policy for annual plans")
        for r in results:
            print(r.score, r.file, r.lines, r.matched_terms)
    """

    def __init__(self, kb_dir: str) -> None:
        self.kb_dir = kb_dir
        # maps file path → list of raw lines (newlines preserved for joining)
        self._kb: dict[str, list[str]] = {}
        self._load_kb()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_kb(self) -> None:
        for fname in sorted(os.listdir(self.kb_dir)):
            if fname.endswith(".md"):
                fpath = os.path.join(self.kb_dir, fname)
                with open(fpath, encoding="utf-8") as fh:
                    self._kb[fpath] = fh.readlines()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_keywords(self, query: str) -> list[str]:
        """Tokenise *query* and return meaningful keywords (no stopwords)."""
        tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    def _context_range(self, line_idx: int, total: int) -> tuple[int, int]:
        start = max(0, line_idx - _CONTEXT_WINDOW)
        end = min(total - 1, line_idx + _CONTEXT_WINDOW)
        return start, end

    @staticmethod
    def _merge_ranges(
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Merge overlapping or adjacent [start, end] ranges."""
        if not ranges:
            return []
        sorted_r = sorted(ranges)
        merged: list[tuple[int, int]] = [sorted_r[0]]
        for start, end in sorted_r[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _merge_kw_ranges(
        ranges: list[tuple[int, int, set[str]]],
    ) -> list[tuple[int, int, set[str]]]:
        """Merge overlapping keyword ranges, accumulating the matched terms."""
        if not ranges:
            return []
        ranges = sorted(ranges, key=lambda x: x[0])
        merged: list[tuple[int, int, set[str]]] = [ranges[0]]
        for start, end, hits in ranges[1:]:
            prev_start, prev_end, prev_hits = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (
                    prev_start,
                    max(prev_end, end),
                    prev_hits | hits,
                )
            else:
                merged.append((start, end, hits))
        return merged

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str) -> list[SearchResult]:
        """
        Search the KB for *query*.

        Returns a list of :class:`SearchResult` sorted by score (descending).
        Phrase matches always outscore keyword-only matches.
        """
        results: list[SearchResult] = []
        phrase_covered: set[tuple[str, str]] = set()  # (fpath, lines_str)

        # ── 1. Full phrase search (score = 2.0) ────────────────────────
        phrase = query.strip().lower()
        for fpath, lines in self._kb.items():
            raw_ranges: list[tuple[int, int]] = []
            for i, line in enumerate(lines):
                if phrase in line.lower():
                    raw_ranges.append(self._context_range(i, len(lines)))

            for start, end in self._merge_ranges(raw_ranges):
                lines_str = f"{start + 1}-{end + 1}"
                text = "".join(lines[start : end + 1]).strip()
                results.append(
                    SearchResult(
                        file=fpath,
                        lines=lines_str,
                        text=text,
                        score=2.0,
                        match_type="phrase",
                        matched_terms=[query.strip()],
                    )
                )
                phrase_covered.add((fpath, lines_str))

        # ── 2. Individual keyword search (score = matched / total) ─────
        keywords = self._extract_keywords(query)
        if keywords:
            for fpath, lines in self._kb.items():
                # Collect per-line keyword hits
                line_hits: dict[int, set[str]] = {}
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    for kw in keywords:
                        if re.search(r"\b" + re.escape(kw) + r"\b", line_lower):
                            line_hits.setdefault(i, set()).add(kw)

                if not line_hits:
                    continue

                # Expand to context ranges and merge
                kw_ranges = [
                    (*self._context_range(i, len(lines)), hits)
                    for i, hits in line_hits.items()
                ]
                for start, end, hits in self._merge_kw_ranges(kw_ranges):
                    lines_str = f"{start + 1}-{end + 1}"
                    if (fpath, lines_str) in phrase_covered:
                        # Already represented as a higher-scored phrase match
                        continue
                    score = len(hits) / len(keywords)
                    text = "".join(lines[start : end + 1]).strip()
                    results.append(
                        SearchResult(
                            file=fpath,
                            lines=lines_str,
                            text=text,
                            score=score,
                            match_type="keyword",
                            matched_terms=sorted(hits),
                        )
                    )

        # Sort: highest score first; break ties by file then line
        results.sort(key=lambda r: (-r.score, r.file, r.lines))
        return results
