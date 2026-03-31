"""Tests for agent.output.save_outputs."""

import json
import os

import pytest

from agent.output import save_outputs
from agent.schema import Answer, Citation, MemoryUpdate, ToolCall


def _make_answer(**kwargs) -> Answer:
    defaults = dict(
        final_answer="Test answer.",
        citations=[Citation(file="kb/plans_pricing.md", lines="3-5")],
        assumptions=[],
        memory_updates=[],
        tool_calls=[ToolCall(tool="search_kb", input="test query")],
    )
    defaults.update(kwargs)
    return Answer(**defaults)


@pytest.fixture
def out_dir(tmp_path):
    return str(tmp_path)


class TestAnswerJson:
    def test_creates_file(self, out_dir):
        save_outputs(out_dir, 1, "What is Pro?", _make_answer())
        assert os.path.isfile(os.path.join(out_dir, "answer.json"))

    def test_first_turn_produces_list(self, out_dir):
        save_outputs(out_dir, 1, "Q1", _make_answer(final_answer="A1"))
        with open(os.path.join(out_dir, "answer.json")) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_appends_across_turns(self, out_dir):
        save_outputs(out_dir, 1, "Q1", _make_answer(final_answer="A1"))
        save_outputs(out_dir, 2, "Q2", _make_answer(final_answer="A2"))
        with open(os.path.join(out_dir, "answer.json")) as fh:
            data = json.load(fh)
        assert len(data) == 2
        assert data[0]["turn"] == 1
        assert data[1]["turn"] == 2

    def test_turn_number_stored(self, out_dir):
        save_outputs(out_dir, 3, "Q3", _make_answer())
        with open(os.path.join(out_dir, "answer.json")) as fh:
            data = json.load(fh)
        assert data[0]["turn"] == 3

    def test_final_answer_stored(self, out_dir):
        save_outputs(out_dir, 1, "Q", _make_answer(final_answer="Hello world"))
        with open(os.path.join(out_dir, "answer.json")) as fh:
            data = json.load(fh)
        assert data[0]["final_answer"] == "Hello world"

    def test_citations_stored(self, out_dir):
        ans = _make_answer(citations=[Citation(file="kb/plans_pricing.md", lines="1-3")])
        save_outputs(out_dir, 1, "Q", ans)
        with open(os.path.join(out_dir, "answer.json")) as fh:
            data = json.load(fh)
        assert data[0]["citations"][0]["file"] == "kb/plans_pricing.md"


class TestAnswerMd:
    def test_creates_file(self, out_dir):
        save_outputs(out_dir, 1, "What is Pro?", _make_answer())
        assert os.path.isfile(os.path.join(out_dir, "answer.md"))

    def test_contains_question(self, out_dir):
        save_outputs(out_dir, 1, "What is the refund policy?", _make_answer())
        content = open(os.path.join(out_dir, "answer.md")).read()
        assert "What is the refund policy?" in content

    def test_contains_answer(self, out_dir):
        save_outputs(out_dir, 1, "Q", _make_answer(final_answer="No refund after 14 days."))
        content = open(os.path.join(out_dir, "answer.md")).read()
        assert "No refund after 14 days." in content

    def test_appends_separator_between_turns(self, out_dir):
        save_outputs(out_dir, 1, "Q1", _make_answer(final_answer="A1"))
        save_outputs(out_dir, 2, "Q2", _make_answer(final_answer="A2"))
        content = open(os.path.join(out_dir, "answer.md")).read()
        assert "A1" in content
        assert "A2" in content
        assert "---" in content

    def test_turn_heading_present(self, out_dir):
        save_outputs(out_dir, 2, "Q", _make_answer())
        content = open(os.path.join(out_dir, "answer.md")).read()
        assert "Turn 2" in content
