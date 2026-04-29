# PurpleMem

Benchmark-first memory retrieval for long-horizon AI systems.

Used daily in production on [PurpleTron](https://twitch.tv/purple_tron), a Twitch/Discord AI bot. This repo contains the retrieval core, the benchmark harness, and documented ablations.

## Benchmark

LongMemEval, session retrieval (500 questions, `BAAI/bge-small-en`):

| Variant | R@5 | R@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Raw retrieval | 92.8 | 96.4 | 0.846 |
| Hybrid lexical reranking | 96.8 | 98.0 | 0.920 |
| Hybrid + topic signal | 97.0 | 98.4 | 0.922 |

This evaluates the retrieval layer on verbatim session text. It does not claim full end-to-end agent parity.

The main result: hybrid lexical reranking closes the gap to published baselines.

Frozen result files are in `benchmarks/results/`. See `docs/benchmark-results.md` for the summary.

## What PurpleMem is

- a retrieval stack built around Qdrant and SQLite
- hybrid lexical + semantic ranking
- temporal validity for changing facts
- topic-aware retrieval and browse
- an optional lightweight entity graph
- a benchmark harness with reproducible results

## What PurpleMem is not

- not an agent framework
- not a chatbot product
- not a universal memory SDK

## Quick start

```bash
uv sync
uv run python examples/quickstart_sqlite_qdrant.py
uv run python benchmarks/generate_eval_set.py
uv run python benchmarks/run_eval.py benchmarks/results/sample_eval_set.json --name sample_eval
```

## Repo map

- `src/purplemem/` — retrieval core
- `benchmarks/` — LongMemEval adapter and evaluation tools
- `docs/` — methodology, architecture, ablations
- `examples/` — minimal integration path
- `sample_data/` — tiny sanitized dataset

## Reproduce the benchmark

```bash
mkdir -p benchmarks/data
curl -fsSL -o benchmarks/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

uv run python benchmarks/longmemeval.py --name raw
uv run python benchmarks/longmemeval.py --name hybrid --hybrid-weight 0.3
uv run python benchmarks/longmemeval.py --name hybrid_topic --hybrid-weight 0.3 --topic-boost 0.05
uv run python benchmarks/compare_runs.py benchmarks/results/longmemeval_*.json
```

## Design notes

PurpleMem treats memory as a retrieval-systems problem.

The core ideas:

- rank by semantic similarity
- sharpen with lexical overlap when the query has strong keyword anchors
- keep temporal history instead of overwriting facts
- use topic-aware fallbacks when semantic search is sparse
- optionally layer a small typed graph on top of retrieval

See `docs/` for the details.
