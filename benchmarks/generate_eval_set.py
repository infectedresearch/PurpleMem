#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tiny public eval set from sample data"
    )
    parser.add_argument("--input", default="sample_data/sample_memories.json")
    parser.add_argument("--output", default="benchmarks/results/sample_eval_set.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f:
        records = json.load(f)
    _VERB_TEMPLATES = {
        "likes": "like",
        "loves": "love",
        "prefers": "prefer",
        "plays": "play",
        "works_at": "work at",
        "lives_in": "live in",
        "hates": "hate",
        "dislikes": "dislike",
    }
    questions = []
    for idx, record in enumerate(records):
        predicate = record.get("predicate")
        if predicate:
            verb = _VERB_TEMPLATES.get(predicate, predicate.replace("_", " "))
            questions.append(
                {
                    "id": f"q_{idx:03d}",
                    "question": f"What does {record['username']} {verb}?",
                    "expected_answers": [record.get("object_text", record["text"])],
                    "expected_memory_ids": [record["id"]],
                }
            )
    with open(args.output, "w") as f:
        json.dump({"questions": questions}, f, indent=2)
    print(f"Wrote {len(questions)} questions to {args.output}")


if __name__ == "__main__":
    main()
