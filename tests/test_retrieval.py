from pathlib import Path

from purplemem.retrieval import MemoryRecord, RetrievalEngine
from purplemem.scoring import SearchConfig


def make_engine(tmp_path: Path) -> RetrievalEngine:
    return RetrievalEngine(
        sqlite_path=str(tmp_path / "purplemem_test.sqlite"),
        collection_name="purplemem_test_collection",
    )


def test_rerank_records_prefers_relevant_text(tmp_path):
    engine = make_engine(tmp_path)
    records = [
        MemoryRecord(
            id="m1",
            text="alex likes sushi and ramen",
            username="alex",
            topic="food",
            confidence=0.9,
            evidence_count=2,
            last_seen_at=1712000000,
        ),
        MemoryRecord(
            id="m2",
            text="alex plays valorant every weekend",
            username="alex",
            topic="gaming",
            confidence=0.8,
            evidence_count=1,
            last_seen_at=1712000100,
        ),
    ]
    ranked = engine.rerank_records("what does alex like to eat?", records)
    assert ranked[0]["id"] == "m1"


def test_topic_browse_empty_general(tmp_path):
    engine = make_engine(tmp_path)
    assert engine.browse_by_topic("general") == []


def test_ingest_and_search(tmp_path):
    engine = make_engine(tmp_path)
    engine.reset()
    engine.ingest(
        [
            MemoryRecord(
                id="m1",
                text="maya works at google",
                username="maya",
                topic="work",
                memory_type="profile",
                confidence=0.9,
                evidence_count=2,
                last_seen_at=1712000000,
            ),
            MemoryRecord(
                id="m2",
                text="alex likes sushi",
                username="alex",
                topic="food",
                memory_type="preference",
                confidence=0.9,
                evidence_count=2,
                last_seen_at=1712000001,
            ),
        ]
    )
    results = engine.search("where does maya work?", username="maya")
    assert results
    assert results[0]["id"] == "m1"


def test_temporal_penalty_historical_facts(tmp_path):
    engine = make_engine(tmp_path)
    records = [
        MemoryRecord(
            id="old",
            text="alex likes pizza",
            username="alex",
            topic="food",
            confidence=0.9,
            evidence_count=2,
            last_seen_at=1712000000,
            valid_to=1712001000,
        ),
        MemoryRecord(
            id="new",
            text="alex likes sushi",
            username="alex",
            topic="food",
            confidence=0.8,
            evidence_count=1,
            last_seen_at=1712000001,
            valid_to=None,
        ),
    ]
    ranked = engine.rerank_records(
        "what does alex like?", records, SearchConfig(temporal_penalty=0.25)
    )
    assert ranked[0]["id"] == "new"
