#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(paths: list[str]) -> list[dict]:
    out = []
    expanded = []
    for path in paths:
        expanded.extend(glob.glob(path))
    for path in sorted(set(expanded)):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = path
        out.append(data)
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: compare_runs.py benchmarks/results/*.json")
    runs = load(sys.argv[1:])
    if not runs:
        raise SystemExit("no result files found")
    print("Comparison")
    for run in runs:
        overall = run.get("overall", {})
        print(
            f"- {run.get('name')}: R@5={overall.get('r@5')}  R@10={overall.get('r@10')}  NDCG@10={overall.get('ndcg@10')}"
        )


if __name__ == "__main__":
    main()
