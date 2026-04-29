from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purplemem.retrieval import MemoryRecord, RetrievalEngine


def main():
    engine = RetrievalEngine()
    records = [
        MemoryRecord(
            id="m1",
            text="alex likes sushi and ramen",
            username="alex",
            topic="food",
            memory_type="preference",
            confidence=0.9,
            last_seen_at=1712000000,
        ),
        MemoryRecord(
            id="m2",
            text="alex plays valorant every weekend",
            username="alex",
            topic="gaming",
            memory_type="profile",
            confidence=0.8,
            last_seen_at=1712000100,
        ),
        MemoryRecord(
            id="m3",
            text="maya works at google in chicago",
            username="maya",
            topic="work",
            memory_type="profile",
            confidence=0.85,
            last_seen_at=1712000200,
        ),
    ]
    ranked = engine.rerank_records("what does alex like to eat?", records)
    for item in ranked:
        print(f"{item['final_score']:.3f}  {item['document']}")


if __name__ == "__main__":
    main()
