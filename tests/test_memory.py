"""Tests for agent.memory helpers."""

import json
import os

import pytest

from agent.memory import read_memory, update_memory, write_memory


@pytest.fixture
def out_dir(tmp_path):
    return str(tmp_path)


class TestReadMemory:
    def test_returns_empty_dict_when_file_absent(self, out_dir):
        assert read_memory(out_dir) == {}

    def test_returns_stored_values(self, out_dir):
        write_memory(out_dir, {"plan": "Pro", "billing_cycle": "annual"})
        mem = read_memory(out_dir)
        assert mem == {"plan": "Pro", "billing_cycle": "annual"}


class TestWriteMemory:
    def test_creates_file(self, out_dir):
        write_memory(out_dir, {"k": "v"})
        assert os.path.isfile(os.path.join(out_dir, "memory.json"))

    def test_file_is_valid_json(self, out_dir):
        write_memory(out_dir, {"plan": "Starter"})
        with open(os.path.join(out_dir, "memory.json")) as fh:
            data = json.load(fh)
        assert data == {"plan": "Starter"}

    def test_overwrites_existing(self, out_dir):
        write_memory(out_dir, {"plan": "Pro"})
        write_memory(out_dir, {"plan": "Business"})
        assert read_memory(out_dir) == {"plan": "Business"}

    def test_creates_out_dir_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "nested" / "out")
        write_memory(new_dir, {"k": "v"})
        assert os.path.isfile(os.path.join(new_dir, "memory.json"))


class TestUpdateMemory:
    def test_merges_new_keys(self, out_dir):
        write_memory(out_dir, {"plan": "Pro"})
        result = update_memory(out_dir, {"billing_cycle": "annual"})
        assert result == {"plan": "Pro", "billing_cycle": "annual"}

    def test_overwrites_existing_key(self, out_dir):
        write_memory(out_dir, {"plan": "Pro"})
        result = update_memory(out_dir, {"plan": "Business"})
        assert result["plan"] == "Business"

    def test_persists_to_disk(self, out_dir):
        update_memory(out_dir, {"seat_count": 12})
        assert read_memory(out_dir)["seat_count"] == 12

    def test_returns_full_merged_dict(self, out_dir):
        write_memory(out_dir, {"a": 1, "b": 2})
        result = update_memory(out_dir, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}
