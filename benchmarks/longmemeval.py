#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qdrant_client import QdrantClient, models

from benchmarks.metrics import dcg, ndcg_score
from purplemem.topics import detect_topic

_BENCH_COLLECTION = "_purplemem_longmem"
_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_client = QdrantClient(url=_QDRANT_URL, check_compatibility=False)

_STOP_WORDS = {
    "what",
    "when",
    "where",
    "who",
    "how",
    "which",
    "did",
    "do",
    "was",
    "were",
    "have",
    "has",
    "had",
    "is",
    "are",
    "the",
    "a",
    "an",
    "my",
    "me",
    "i",
    "you",
    "your",
    "their",
    "it",
    "its",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "ago",
    "last",
    "that",
    "this",
    "there",
    "about",
    "get",
    "got",
    "give",
    "gave",
    "buy",
    "bought",
    "made",
    "make",
}


def extract_keywords(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOP_WORDS]


def keyword_overlap(doc_text: str, query_keywords: list[str]) -> float:
    if not query_keywords:
        return 0.0
    doc_lower = doc_text.lower()
    hits = sum(1 for kw in query_keywords if kw in doc_lower)
    return hits / len(query_keywords)


def ensure_bench_collection() -> None:
    try:
        _client.delete_collection(_BENCH_COLLECTION)
    except Exception:
        pass
    sparse_config = None
    if hasattr(_client, "get_fastembed_sparse_vector_params"):
        sparse_config = _client.get_fastembed_sparse_vector_params()
    _client.create_collection(
        collection_name=_BENCH_COLLECTION,
        vectors_config=_client.get_fastembed_vector_params(),
        sparse_vectors_config=sparse_config,
    )


def cleanup_bench_collection() -> None:
    try:
        _client.delete_collection(_BENCH_COLLECTION)
    except Exception:
        pass


def evaluate_entry(entry: dict, hybrid_weight: float = 0.0, topic_boost: float = 0.0):
    question = entry["question"]
    answer_sids = set(entry["answer_session_ids"])
    question_type = entry["question_type"]
    corpus = []
    corpus_ids = []
    for session, sess_id in zip(
        entry["haystack_sessions"], entry["haystack_session_ids"]
    ):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if user_turns:
            corpus.append("\n".join(user_turns))
            corpus_ids.append(sess_id)
    if not corpus:
        return None
    ensure_bench_collection()
    try:
        vector_name = _client.get_vector_field_name()
        _client.upsert(
            collection_name=_BENCH_COLLECTION,
            points=[
                models.PointStruct(
                    id=i,
                    vector={
                        vector_name: models.Document(
                            text=doc,
                            model=_client.embedding_model_name,
                        )
                    },
                    payload={"corpus_id": cid, "document": doc},
                )
                for i, (doc, cid) in enumerate(zip(corpus, corpus_ids))
            ],
            wait=True,
        )
        results = _client.query_points(
            collection_name=_BENCH_COLLECTION,
            query=models.Document(
                text=question,
                model=_client.embedding_model_name,
            ),
            using=_client.get_vector_field_name(),
            limit=min(50, len(corpus)),
            with_payload=True,
        )
        ranked = []
        for point in results.points:
            idx = point.id
            ranked.append(
                {
                    "score": point.score,
                    "corpus_id": corpus_ids[idx],
                    "doc_text": corpus[idx],
                }
            )
        if hybrid_weight > 0:
            query_kws = extract_keywords(question)
            for r in ranked:
                overlap = keyword_overlap(r["doc_text"], query_kws)
                r["score"] = r["score"] * (1.0 + hybrid_weight * overlap)
        if topic_boost > 0:
            query_topic = detect_topic(question)
            if query_topic != "general":
                for r in ranked:
                    if detect_topic(r["doc_text"]) == query_topic:
                        r["score"] += topic_boost
        ranked.sort(key=lambda r: r["score"], reverse=True)
        ranked_cids = [r["corpus_id"] for r in ranked]
        seen = set(ranked_cids)
        for cid in corpus_ids:
            if cid not in seen:
                ranked_cids.append(cid)
    finally:
        cleanup_bench_collection()
    result = {"question_type": question_type, "question_id": entry["question_id"]}
    for k in (1, 3, 5, 10):
        result[f"r@{k}"] = (
            1.0 if any(sid in set(ranked_cids[:k]) for sid in answer_sids) else 0.0
        )
    result["ndcg@10"] = ndcg_score(ranked_cids, answer_sids, 10)
    return result


def run_benchmark(
    data_file: str,
    name: str,
    limit: int = 0,
    throttle: float = 0.1,
    hybrid_weight: float = 0.0,
    topic_boost: float = 0.0,
):
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"LongMemEval dataset not found at {data_file}.\n"
            "Download it first, for example:\n"
            "  mkdir -p benchmarks/data\n"
            "  curl -fsSL -o benchmarks/data/longmemeval_s_cleaned.json "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
        )
    with open(data_file) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]
    cleanup_bench_collection()
    metrics = defaultdict(list)
    type_metrics = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    print(f"\nPurpleMem LongMemEval: {name}")
    print(f"Questions: {len(data)}")
    print(f"Embedding: {_client.embedding_model_name}")
    print(f"Hybrid weight: {hybrid_weight}")
    print(f"Topic boost: {topic_boost if topic_boost else 'disabled'}")
    for i, entry in enumerate(data):
        result = evaluate_entry(
            entry, hybrid_weight=hybrid_weight, topic_boost=topic_boost
        )
        if result is None:
            continue
        qtype = result["question_type"]
        for k in (1, 3, 5, 10):
            metrics[f"r@{k}"].append(result[f"r@{k}"])
            type_metrics[qtype][f"r@{k}"].append(result[f"r@{k}"])
        metrics["ndcg@10"].append(result["ndcg@10"])
        type_metrics[qtype]["ndcg@10"].append(result["ndcg@10"])
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (len(data) - i - 1) / max(rate, 1e-9)
            r5 = sum(metrics["r@5"]) / len(metrics["r@5"]) * 100
            print(
                f"  [{i + 1:4d}/{len(data)}] R@5={r5:.1f}%  ({rate:.1f} q/s, ETA {eta:.0f}s)"
            )
        if throttle > 0:
            time.sleep(throttle)
    elapsed = time.time() - t0
    final = {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}
    print(f"\nResults: {name}")
    print(f"  R@1:      {final.get('r@1', 0) * 100:5.1f}%")
    print(f"  R@3:      {final.get('r@3', 0) * 100:5.1f}%")
    print(f"  R@5:      {final.get('r@5', 0) * 100:5.1f}%")
    print(f"  R@10:     {final.get('r@10', 0) * 100:5.1f}%")
    print(f"  NDCG@10:  {final.get('ndcg@10', 0):.3f}")
    os.makedirs("benchmarks/results", exist_ok=True)
    outpath = f"benchmarks/results/longmemeval_{name}.json"
    with open(outpath, "w") as f:
        json.dump(
            {
                "name": name,
                "benchmark": "LongMemEval",
                "embedding_model": _client.embedding_model_name,
                "overall": {k: round(v * 100, 1) for k, v in final.items()},
                "per_type": {
                    qtype: {
                        k: round(sum(vals) / len(vals) * 100, 1) if vals else 0.0
                        for k, vals in tm.items()
                    }
                    for qtype, tm in type_metrics.items()
                },
                "wall_time_s": elapsed,
                "total_questions": len(data),
            },
            f,
            indent=2,
        )
    print(f"Saved to {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval against PurpleMem")
    parser.add_argument("--data", default="benchmarks/data/longmemeval_s_cleaned.json")
    parser.add_argument("--name", default="purplemem")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--throttle", type=float, default=0.1)
    parser.add_argument("--hybrid-weight", type=float, default=0.0)
    parser.add_argument("--topic-boost", type=float, default=0.0)
    args = parser.parse_args()
    try:
        run_benchmark(
            args.data,
            args.name,
            args.limit,
            args.throttle,
            args.hybrid_weight,
            args.topic_boost,
        )
    except FileNotFoundError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
