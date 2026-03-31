"""
Entry point for the AcmeCloud KB agent.

Usage:
    python -m agent.run --fixture fixtures/conversation_1.json --kb kb --out out
"""

import argparse
import json
import os
import sys

from agent.retriever import KeywordSearchRetriever


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

    retriever = KeywordSearchRetriever(args.kb)

    for turn in turns:
        query = turn["message"]
        print(f"\n{'='*70}")
        print(f"Turn {turn['turn']} [{turn['role']}]: {query}")
        print(f"{'='*70}")

        results = retriever.search(query)
        if not results:
            print("  (no KB results found)")
        else:
            for r in results:
                print(
                    f"  [{r.match_type:7}] score={r.score:.2f}  "
                    f"{r.file}:{r.lines}  terms={r.matched_terms}"
                )
                print(f"    > {r.text[:120].replace(chr(10), ' ')}")

    os.makedirs(args.out, exist_ok=True)


if __name__ == "__main__":
    main()
