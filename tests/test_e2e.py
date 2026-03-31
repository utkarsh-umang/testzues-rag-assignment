"""
End-to-end tests for the AcmeCloud KB agent.

Coverage map:
  ✓ Run Fixture 1 and Fixture 2 through the CLI (agent.run.main)
  ✓ Validate answer.json schema — all required keys present and typed correctly
  ✓ Validate citations reference real KB files at valid line ranges
  ✓ Validate key expected facts for each fixture turn
  ✓ Validate memory persists across turns and appears in the turn-2 prompt

The LLM (ChatOpenAI) is mocked — no real API calls are made.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, call, patch

import pytest

from agent.run import main

KB_DIR = "kb"
FIXTURE_1 = "fixtures/conversation_1.json"
FIXTURE_2 = "fixtures/conversation_2.json"

# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------

def _llm_msg(payload: dict) -> MagicMock:
    m = MagicMock()
    m.content = json.dumps(payload)
    return m


def _ans(final_answer: str, citations: list[dict], memory_updates: list[dict] | None = None) -> dict:
    return {
        "final_answer": final_answer,
        "citations": citations,
        "assumptions": [],
        "memory_updates": memory_updates or [],
        "tool_calls": [{"tool": "search_kb", "input": "query"}],
    }


# Canned LLM responses — realistic answers grounded in the KB

FIXTURE1_RESPONSES = [
    _ans(
        "You are not eligible for a refund. Annual subscriptions are refundable only within 14 days "
        "of purchase and only if usage is below 10%. You purchased 20 days ago, so the refund window has closed.",
        [{"file": "kb/refunds_cancellation.md", "lines": "4-6"}],
        [{"key": "plan", "value": "Pro"}, {"key": "billing_cycle", "value": "annual"}],
    ),
    _ans(
        "You should be on the Business plan. SSO (SAML 2.0) and EU data residency are both exclusive "
        "to the Business tier.",
        [{"file": "kb/security_compliance.md", "lines": "1-3"}],
        [{"key": "region", "value": "EU"}],
    ),
    _ans(
        "HIPAA support is not offered on any AcmeCloud plan.",
        [{"file": "kb/security_compliance.md", "lines": "5-5"}],
    ),
]

FIXTURE2_RESPONSES = [
    _ans(
        "You need the Business plan at $249/month. Pro supports only 10 seats and 100k MAU; "
        "your 12 seats and 120k MAU exceed both limits.",
        [{"file": "kb/plans_pricing.md", "lines": "5-8"}],
        [{"key": "plan", "value": "Business"}, {"key": "seat_count", "value": 12}, {"key": "mau", "value": 120000}],
    ),
    _ans(
        "With the 15% annual discount, Business plan costs $249 × 12 × 0.85 = $2,539.80/year. "
        "Seat limits are soft — adding 2 seats mid-month will not block access; overage seats are "
        "billed at the start of the next cycle, not prorated.",
        [
            {"file": "kb/plans_pricing.md", "lines": "9-10"},
            {"file": "kb/limits_edge_cases.md", "lines": "1-2"},
        ],
    ),
]


# ---------------------------------------------------------------------------
# Shared: run the CLI via main() with a mocked LLM
# ---------------------------------------------------------------------------

def _run_via_cli(fixture_path: str, responses: list[dict], out_dir: str) -> None:
    """Call agent.run.main() — the actual CLI entry point — with a mocked LLM."""
    response_iter = iter(_llm_msg(r) for r in responses)
    with patch("agent.graph.ChatOpenAI") as MockLLM:
        MockLLM.return_value.invoke.side_effect = lambda *a, **kw: next(response_iter)
        main(["--fixture", fixture_path, "--kb", KB_DIR, "--out", out_dir])


# ---------------------------------------------------------------------------
# Helper: load answer.json
# ---------------------------------------------------------------------------

def _load_answers(out_dir: str) -> list[dict]:
    with open(os.path.join(out_dir, "answer.json")) as fh:
        return json.load(fh)


def _load_memory(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "memory.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Schema validation helpers (reused across fixture tests)
# ---------------------------------------------------------------------------

REQUIRED_ANSWER_KEYS = {"final_answer", "citations", "assumptions", "memory_updates", "tool_calls"}


def assert_answer_schema(entry: dict) -> None:
    """Assert a single turn entry in answer.json satisfies the required schema."""
    assert "turn" in entry, "missing 'turn'"
    assert isinstance(entry["turn"], int), "'turn' must be int"

    missing = REQUIRED_ANSWER_KEYS - entry.keys()
    assert not missing, f"missing keys: {missing}"

    assert isinstance(entry["final_answer"], str) and entry["final_answer"], "final_answer must be non-empty string"
    assert isinstance(entry["citations"], list), "citations must be a list"
    assert isinstance(entry["assumptions"], list), "assumptions must be a list"
    assert isinstance(entry["memory_updates"], list), "memory_updates must be a list"
    assert isinstance(entry["tool_calls"], list), "tool_calls must be a list"

    for c in entry["citations"]:
        assert "file" in c and "lines" in c, f"citation missing keys: {c}"


def assert_citations_valid(citations: list[dict], kb_dir: str = KB_DIR) -> None:
    """Assert every citation points to a real KB file at a valid line range."""
    for c in citations:
        file_path = c["file"]
        assert os.path.isfile(file_path), f"cited file does not exist: {file_path}"

        with open(file_path) as fh:
            total_lines = sum(1 for _ in fh)

        parts = c["lines"].split("-")
        assert len(parts) == 2, f"lines format must be 'start-end', got: {c['lines']}"
        start, end = int(parts[0]), int(parts[1])
        assert start >= 1, f"line start must be >= 1, got {start}"
        assert end >= start, f"line end must be >= start, got {start}-{end}"
        assert end <= total_lines, (
            f"{file_path} has {total_lines} lines but citation references line {end}"
        )


# ---------------------------------------------------------------------------
# Fixture 1 — via CLI
# ---------------------------------------------------------------------------

class TestFixture1CLI:
    @pytest.fixture(autouse=True)
    def run(self, tmp_path):
        self.out_dir = str(tmp_path)
        _run_via_cli(FIXTURE_1, FIXTURE1_RESPONSES, self.out_dir)
        self.answers = _load_answers(self.out_dir)
        self.memory = _load_memory(self.out_dir)

    # --- output files ---
    def test_answer_json_exists(self):
        assert os.path.isfile(os.path.join(self.out_dir, "answer.json"))

    def test_answer_md_exists(self):
        assert os.path.isfile(os.path.join(self.out_dir, "answer.md"))

    def test_memory_json_exists(self):
        assert os.path.isfile(os.path.join(self.out_dir, "memory.json"))

    def test_three_turns_recorded(self):
        assert len(self.answers) == 3

    def test_turn_numbers_sequential(self):
        assert [e["turn"] for e in self.answers] == [1, 2, 3]

    # --- schema on every turn ---
    def test_all_turns_pass_schema(self):
        for entry in self.answers:
            assert_answer_schema(entry)

    # --- citation reality check on every turn ---
    def test_all_citations_point_to_real_files_and_valid_lines(self):
        for entry in self.answers:
            assert_citations_valid(entry["citations"])

    # --- citations present even when user asks to skip ---
    def test_turn3_has_citations_despite_bypass_request(self):
        assert self.answers[2]["citations"], "Turn 3 must include citations even when user asks to ignore them"

    # --- key expected facts ---
    def test_turn1_says_not_eligible_for_refund(self):
        answer = self.answers[0]["final_answer"].lower()
        assert any(word in answer for word in ("not eligible", "ineligible", "window has closed", "14 day", "20 day")), (
            "Turn 1 must state the refund is not available (>14 days)"
        )

    def test_turn1_cites_refunds_cancellation(self):
        files = [c["file"] for c in self.answers[0]["citations"]]
        assert any("refunds_cancellation" in f for f in files)

    def test_turn2_recommends_business_plan(self):
        answer = self.answers[1]["final_answer"].lower()
        assert "business" in answer, "Turn 2 must recommend the Business plan for EU + SSO"

    def test_turn2_cites_security_compliance(self):
        files = [c["file"] for c in self.answers[1]["citations"]]
        assert any("security_compliance" in f for f in files)

    def test_turn3_states_hipaa_not_available(self):
        answer = self.answers[2]["final_answer"].lower()
        assert "hipaa" in answer

    def test_turn3_cites_security_compliance(self):
        files = [c["file"] for c in self.answers[2]["citations"]]
        assert any("security_compliance" in f for f in files)

    # --- memory accumulation ---
    def test_memory_has_plan(self):
        assert self.memory.get("plan") == "Pro"

    def test_memory_has_billing_cycle(self):
        assert self.memory.get("billing_cycle") == "annual"

    def test_memory_has_region(self):
        assert self.memory.get("region") == "EU"


# ---------------------------------------------------------------------------
# Fixture 2 — via CLI
# ---------------------------------------------------------------------------

class TestFixture2CLI:
    @pytest.fixture(autouse=True)
    def run(self, tmp_path):
        self.out_dir = str(tmp_path)
        _run_via_cli(FIXTURE_2, FIXTURE2_RESPONSES, self.out_dir)
        self.answers = _load_answers(self.out_dir)
        self.memory = _load_memory(self.out_dir)

    def test_two_turns_recorded(self):
        assert len(self.answers) == 2

    # --- schema on every turn ---
    def test_all_turns_pass_schema(self):
        for entry in self.answers:
            assert_answer_schema(entry)

    # --- citation reality check ---
    def test_all_citations_point_to_real_files_and_valid_lines(self):
        for entry in self.answers:
            assert_citations_valid(entry["citations"])

    # --- key expected facts ---
    def test_turn1_recommends_business_plan(self):
        answer = self.answers[0]["final_answer"].lower()
        assert "business" in answer, "Turn 1 must recommend Business plan"

    def test_turn1_cites_plans_pricing(self):
        files = [c["file"] for c in self.answers[0]["citations"]]
        assert any("plans_pricing" in f for f in files)

    def test_turn2_mentions_annual_discount(self):
        answer = self.answers[1]["final_answer"].lower()
        assert "annual" in answer or "discount" in answer or "15" in answer

    def test_turn2_mentions_seat_overage_behavior(self):
        answer = self.answers[1]["final_answer"].lower()
        assert any(word in answer for word in ("soft", "overage", "next cycle", "not prorated"))

    def test_turn2_cites_plans_pricing(self):
        files = [c["file"] for c in self.answers[1]["citations"]]
        assert any("plans_pricing" in f for f in files)

    def test_turn2_cites_limits_edge_cases(self):
        files = [c["file"] for c in self.answers[1]["citations"]]
        assert any("limits_edge_cases" in f for f in files)

    # --- memory ---
    def test_memory_has_plan(self):
        assert self.memory.get("plan") == "Business"

    def test_memory_has_seat_count(self):
        assert self.memory.get("seat_count") == 12


# ---------------------------------------------------------------------------
# Memory: persists across turns and is injected into turn-2 prompt
# ---------------------------------------------------------------------------

class TestMemoryAcrossTurns:
    def test_turn1_memory_injected_into_turn2_prompt(self, tmp_path):
        """Memory set in turn 1 must appear in the user prompt sent for turn 2."""
        out_dir = str(tmp_path)
        captured_prompts: list[str] = []
        responses = iter(_llm_msg(r) for r in FIXTURE1_RESPONSES)

        def capture_invoke(messages):
            # messages is a list of dicts with 'role'/'content'
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            captured_prompts.append(user_msg)
            return next(responses)

        with patch("agent.graph.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = capture_invoke
            main(["--fixture", FIXTURE_1, "--kb", KB_DIR, "--out", out_dir])

        # Turn 1 sets plan=Pro and billing_cycle=annual in memory.
        # Turn 2 prompt must contain those values.
        assert len(captured_prompts) == 3, "expected one prompt per turn"
        turn2_prompt = captured_prompts[1]
        assert "Pro" in turn2_prompt, "turn-2 prompt must include plan=Pro from turn-1 memory"
        assert "annual" in turn2_prompt, "turn-2 prompt must include billing_cycle=annual from turn-1 memory"

    def test_memory_json_contains_accumulated_keys_after_all_turns(self, tmp_path):
        """memory.json after fixture 1 must have keys from all turns that produced memory_updates."""
        out_dir = str(tmp_path)
        responses = iter(_llm_msg(r) for r in FIXTURE1_RESPONSES)
        with patch("agent.graph.ChatOpenAI") as MockLLM:
            MockLLM.return_value.invoke.side_effect = lambda *a, **kw: next(responses)
            main(["--fixture", FIXTURE_1, "--kb", KB_DIR, "--out", out_dir])

        mem = _load_memory(out_dir)
        assert "plan" in mem
        assert "billing_cycle" in mem
        assert "region" in mem, "turn-2 set region=EU; it must survive to the final memory.json"
