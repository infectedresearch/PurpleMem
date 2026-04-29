#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmarks.metrics import ndcg_score
from purplemem.retrieval import MemoryRecord, RetrievalEngine
from purplemem.scoring import SearchConfig


def load_records(sample_path: str) -> list[MemoryRecord]:
    with open(sample_path) as f:
        raw = json.load(f)
    records = []
    for item in raw:
        records.append(
            MemoryRecord(
                id=item["id"],
                text=item["text"],
                username=item.get("username"),
                memory_type=item.get("memory_type", "episodic"),
                topic=item.get("topic", "general"),
                confidence=float(item.get("confidence", 0.7)),
                evidence_count=int(item.get("evidence_count", 1)),
                last_seen_at=float(item.get("last_seen_at", 0.0)),
                valid_from=item.get("valid_from"),
                valid_to=item.get("valid_to"),
            )
        )
    return records


def evaluate_question(
    engine: RetrievalEngine, question: dict, config: SearchConfig
) -> dict:
    results = engine.search(
        question["question"],
        username=question.get("username_filter"),
        config=config,
        limit=10,
    )
    expected_answers = question.get("expected_answers", [])
    expected_ids = set(question.get("expected_memory_ids", []))
    hits = {}
    first_hit_rank = None
    for k in (1, 3, 5, 10):
        hit = False
        for idx, item in enumerate(results[:k]):
            if item.get("id") in expected_ids:
                hit = True
                if first_hit_rank is None:
                    first_hit_rank = idx + 1
                break
            text = item.get("document", "").lower()
            if any(ans.lower() in text for ans in expected_answers):
                hit = True
                if first_hit_rank is None:
                    first_hit_rank = idx + 1
                break
        hits[k] = hit
    ranked_ids = [item.get("id") for item in results]
    return {
        "question_id": question["id"],
        "category": question.get("category", "unknown"),
        "hits": hits,
        "first_hit_rank": first_hit_rank,
        "ndcg@10": ndcg_score(ranked_ids, expected_ids, 10) if expected_ids else None,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    out = {"count": total}
    for k in (1, 3, 5, 10):
        out[f"r@{k}"] = round(
            100.0 * sum(1 for r in results if r["hits"].get(k)) / max(total, 1), 1
        )
    out["mrr"] = round(
        sum((1.0 / r["first_hit_rank"]) for r in results if r["first_hit_rank"])
        / max(total, 1),
        3,
    )
    ndcg_values = [r["ndcg@10"] for r in results if r.get("ndcg@10") is not None]
    out["ndcg@10"] = (
        round(sum(ndcg_values) / max(len(ndcg_values), 1), 3) if ndcg_values else None
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PurpleMem custom evaluation")
    parser.add_argument("dataset", help="Path to generated eval set")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--records", default="sample_data/sample_memories.json")
    args = parser.parse_args()
    with open(args.dataset) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    records = load_records(args.records)
    engine = RetrievalEngine(
        sqlite_path="purplemem_eval.sqlite", collection_name="purplemem_eval"
    )
    engine.reset()
    engine.ingest(records)
    cfg = SearchConfig()
    results = [evaluate_question(engine, q, cfg) for q in questions]
    summary = summarize(results)
    out = {
        "name": args.name,
        "overall": summary,
        "total_questions": len(questions),
        "per_question": results,
    }
    results_dir = ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Loaded {len(questions)} questions for run '{args.name}'.")
    print(
        f"R@1={summary['r@1']}  R@3={summary['r@3']}  R@5={summary['r@5']}  "
        f"R@10={summary['r@10']}  MRR={summary['mrr']}  NDCG@10={summary['ndcg@10']}"
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
