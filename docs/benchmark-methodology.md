# Benchmark Methodology

PurpleMem is benchmarked in two ways.

## 1. LongMemEval

LongMemEval is a retrieval benchmark over long conversation histories. Each question comes with a haystack of prior sessions and a set of correct session IDs. The task is to rank the sessions so the answer-bearing sessions appear near the top.

In this repo, the LongMemEval run measures the retrieval layer on verbatim session text. It does not measure a full end-to-end agent pipeline.

Metrics:

- Recall@5
- Recall@10
- NDCG@10

The benchmark adapter uses a temporary Qdrant collection per question. This keeps the run isolated and reproducible and avoids mixing benchmark data with any production collection.

Published numbers in this repo are generated with:

- embedding model: `BAAI/bge-small-en` via Qdrant fastembed
- raw run: no lexical reranking, no topic signal
- hybrid run: keyword reranking weight `0.3`
- hybrid+topic run: keyword reranking `0.3`, topic boost `0.05`

## 2. Custom evaluation

The custom evaluation harness generates question / answer pairs from a structured memory corpus. It is useful for measuring ablations on:

- preference recall
- entity relationship queries
- topic-filtered recall
- temporal correctness
- conversation recall
- adversarial precision

This benchmark is not comparable to LongMemEval. It exists to evaluate the stack under a different workload shape.

## Fairness note

Raw LongMemEval runs in this repo should be compared to other retrieval-layer systems, not to full production agents.

If you use PurpleMem in a larger agent system, benchmark the full system separately.
