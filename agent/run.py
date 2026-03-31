"""
Entry point for the AcmeCloud KB agent.

Usage:
    python -m agent.run --fixture fixtures/conversation_1.json --kb kb --out out
"""

import argparse
import json
import os
import sys

from agent.graph import run_conversation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixture conversation against the AcmeCloud KB agent."
    )
    parser.add_argument(
        "--fixture",
        required=True,
        help="Path to the fixture JSON file (e.g. fixtures/conversation_1.json)",
    )
    parser.add_argument(
        "--kb",
        required=True,
        help="Path to the knowledge-base directory (e.g. kb)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory where output files will be written (e.g. out)",
    )
    return parser.parse_args(argv)


def load_fixture(fixture_path: str) -> dict:
    if not os.path.isfile(fixture_path):
        sys.exit(f"Fixture file not found: {fixture_path}")
    with open(fixture_path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Fixture : {args.fixture}")
    print(f"KB dir  : {args.kb}")
    print(f"Out dir : {args.out}")

    fixture = load_fixture(args.fixture)
    conversation_id = fixture.get("conversation_id", os.path.basename(args.fixture))
    turns = fixture.get("turns", [])

    print(f"\nConversation: {conversation_id}  ({len(turns)} turn(s))")

    run_conversation(turns=turns, kb_dir=args.kb, out_dir=args.out)

    print(f"\nOutputs written to {args.out}/")


if __name__ == "__main__":
    main()
