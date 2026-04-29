# Getting started

## Install

```bash
uv sync
```

## Run the sample example

```bash
uv run python examples/quickstart_sqlite_qdrant.py
```

## Run the sample eval flow

```bash
uv run python benchmarks/generate_eval_set.py
uv run python benchmarks/run_eval.py benchmarks/results/sample_eval_set.json --name sample_eval
uv run python benchmarks/compare_runs.py benchmarks/results/sample_eval.json
```

## Run a LongMemEval subset

```bash
mkdir -p benchmarks/data
curl -fsSL -o benchmarks/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

uv run python benchmarks/longmemeval.py --limit 20 --name smoke
```

## What to integrate first

If you want to use PurpleMem in another project, start with:

1. `SearchConfig`
2. topic detection
3. temporal validity
4. hybrid lexical fusion

The graph layer is useful, but optional.
