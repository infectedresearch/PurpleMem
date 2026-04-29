from __future__ import annotations

from dataclasses import dataclass
import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client import models

from .parsing import expand_query_variants, normalize_username
from .scoring import SearchConfig, lexical_match_bonus, score_memory_details
from .storage.schema import schema_sql
from .storage.sqlite import connect, execute, execute_many, query
from .temporal import parse_temporal_reference
from .topics import detect_topic


@dataclass
class MemoryRecord:
    id: str
    text: str
    username: str | None = None
    memory_type: str = "episodic"
    topic: str = "general"
    confidence: float = 0.7
    evidence_count: int = 1
    last_seen_at: float = 0.0
    valid_from: float | None = None
    valid_to: float | None = None


class RetrievalEngine:
    """Retrieval engine for PurpleMem.

    Intentionally narrow. Focuses on scoring and ranking,
    not on reproducing a full production storage stack.
    """

    def __init__(
        self,
        sqlite_path: str = "purplemem.sqlite",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "purplemem",
    ):
        self.sqlite_path = sqlite_path
        self.client = QdrantClient(url=qdrant_url, check_compatibility=False)
        self.collection_name = collection_name
        self.conn = connect(sqlite_path)
        self.ensure_storage()

    def ensure_storage(self) -> None:
        for stmt in schema_sql():
            execute(self.conn, stmt)
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            sparse_config = None
            if hasattr(self.client, "get_fastembed_sparse_vector_params"):
                sparse_config = self.client.get_fastembed_sparse_vector_params()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.client.get_fastembed_vector_params(),
                sparse_vectors_config=sparse_config,
            )

    def reset(self) -> None:
        execute(self.conn, "DELETE FROM memories")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.ensure_storage()

    def ingest(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        now = time.time()
        sqlite_rows = []
        docs = []
        ids = []
        payloads = []
        for record in records:
            rid = record.id or str(uuid.uuid4())
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"purplemem:{rid}"))
            topic = record.topic or detect_topic(record.text)
            sqlite_rows.append(
                (
                    rid,
                    record.text,
                    normalize_username(record.username),
                    record.memory_type,
                    topic,
                    record.confidence,
                    record.evidence_count,
                    record.last_seen_at or now,
                    record.valid_from or now,
                    record.valid_to,
                    None,
                    None,
                    None,
                )
            )
            docs.append(record.text)
            ids.append(qdrant_id)
            payloads.append(
                {
                    "source_id": rid,
                    "document": record.text,
                    "username": normalize_username(record.username),
                    "memory_type": record.memory_type,
                    "topic": topic,
                    "confidence": record.confidence,
                    "evidence_count": record.evidence_count,
                    "last_seen_at": record.last_seen_at or now,
                    "valid_from": record.valid_from or now,
                    "valid_to": record.valid_to,
                }
            )
        execute_many(
            self.conn,
            "INSERT OR REPLACE INTO memories (id, text, username, memory_type, topic, confidence, evidence_count, last_seen_at, valid_from, valid_to, subject, predicate, object_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sqlite_rows,
        )
        vector_name = self.client.get_vector_field_name()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=pid,
                    vector={
                        vector_name: models.Document(
                            text=doc,
                            model=self.client.embedding_model_name,
                        )
                    },
                    payload=payload,
                )
                for pid, doc, payload in zip(ids, docs, payloads)
            ],
            wait=True,
        )

    def _fetch_qdrant_hits(self, query_text: str, limit: int = 20) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query_text,
                model=self.client.embedding_model_name,
            ),
            using=self.client.get_vector_field_name(),
            limit=limit,
            with_payload=True,
        )
        hits = []
        for point in results.points:
            payload = (
                getattr(point, "payload", None)
                or getattr(point, "metadata", None)
                or {}
            )
            document = payload.get("document") if isinstance(payload, dict) else None
            if not document:
                document = getattr(point, "document", "") or ""
            hits.append(
                {
                    "id": payload.get("source_id", str(point.id)),
                    "score": point.score,
                    "document": document,
                    "username": payload.get("username"),
                    "topic": payload.get("topic", "general"),
                    "memory_type": payload.get("memory_type", "episodic"),
                    "confidence": payload.get("confidence", 0.7),
                    "evidence_count": payload.get("evidence_count", 1),
                    "last_seen_at": payload.get("last_seen_at", 0.0),
                    "valid_to": payload.get("valid_to"),
                }
            )
        return hits

    def browse_by_topic(
        self, topic: str, username: str | None = None, limit: int = 10
    ) -> list[dict]:
        if not topic or topic == "general":
            return []
        if username:
            rows = query(
                self.conn,
                "SELECT * FROM memories WHERE topic = ? AND username = ? AND valid_to IS NULL ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (topic, normalize_username(username), limit),
            )
        else:
            rows = query(
                self.conn,
                "SELECT * FROM memories WHERE topic = ? AND valid_to IS NULL ORDER BY confidence DESC, evidence_count DESC LIMIT ?",
                (topic, limit),
            )
        records = [
            MemoryRecord(
                id=row["id"],
                text=row["text"],
                username=row["username"],
                memory_type=row["memory_type"],
                topic=row["topic"],
                confidence=row["confidence"],
                evidence_count=row["evidence_count"],
                last_seen_at=row["last_seen_at"],
                valid_from=row["valid_from"],
                valid_to=row["valid_to"],
            )
            for row in rows
        ]
        return self.rerank_records(
            f"topic:{topic}",
            records,
            SearchConfig(topic_boost=0.0, lexical_mode="additive"),
        )

    def search(
        self,
        query: str,
        username: str | None = None,
        config: SearchConfig | None = None,
        limit: int = 10,
    ) -> list[dict]:
        cfg = config or SearchConfig()
        hits = self._fetch_qdrant_hits(query, limit=max(limit * 3, 20))
        if username:
            normalized = normalize_username(username)
            hits = [
                h for h in hits if normalize_username(h.get("username")) == normalized
            ]
        records = [
            MemoryRecord(
                id=h["id"],
                text=h["document"],
                username=h.get("username"),
                memory_type=h.get("memory_type", "episodic"),
                topic=h.get("topic", detect_topic(h["document"])),
                confidence=float(h.get("confidence", 0.7) or 0.7),
                evidence_count=int(h.get("evidence_count", 1) or 1),
                last_seen_at=float(h.get("last_seen_at", 0.0) or 0.0),
                valid_to=h.get("valid_to"),
            )
            for h in hits
        ]
        ranked = self.rerank_records(query, records, cfg)
        if cfg.topic_browse_supplement and len(ranked) < 3:
            topic = detect_topic(query)
            ranked.extend(self.browse_by_topic(topic, username=username, limit=5))
        deduped = []
        seen = set()
        for item in ranked:
            doc = item["document"]
            if doc in seen:
                continue
            seen.add(doc)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def rerank_records(
        self,
        query: str,
        records: list[MemoryRecord],
        config: SearchConfig | None = None,
    ) -> list[dict]:
        cfg = config or SearchConfig()
        query_variants = expand_query_variants(query)
        query_topic = detect_topic(query)
        temporal_target = None
        if cfg.temporal_query_boost:
            temporal_ref = parse_temporal_reference(query)
            if temporal_ref:
                days_offset, window_days = temporal_ref
                temporal_target = (
                    time.time() - days_offset * 86400,
                    window_days * 86400,
                )

        ranked = []
        for record in records:
            base_score = 0.5
            score_details = score_memory_details(
                base_score=base_score,
                confidence=record.confidence,
                evidence_count=record.evidence_count,
                last_seen_at=record.last_seen_at,
                status="active",
                memory_type=record.memory_type,
            )
            raw_lexical = lexical_match_bonus(record.text, query_variants)
            if cfg.lexical_mode == "multiplicative" and raw_lexical > 0:
                overlap_fraction = raw_lexical / 0.25
                lexical_adjustment = (
                    score_details["final_score"] * cfg.lexical_weight * overlap_fraction
                )
            else:
                lexical_adjustment = raw_lexical

            topic_bonus = (
                cfg.topic_boost
                if cfg.topic_boost > 0
                and query_topic != "general"
                and record.topic == query_topic
                else 0.0
            )
            temporal_penalty = (
                cfg.temporal_penalty if record.valid_to is not None else 0.0
            )
            temporal_query_boost = 0.0
            if temporal_target and record.last_seen_at:
                target_time, window_secs = temporal_target
                time_diff = abs(float(record.last_seen_at) - target_time)
                if time_diff < window_secs:
                    temporal_query_boost = 0.3 * (1.0 - time_diff / window_secs)

            final_score = (
                score_details["final_score"]
                + lexical_adjustment
                + topic_bonus
                + temporal_query_boost
                - temporal_penalty
            )
            ranked.append(
                {
                    "id": record.id,
                    "document": record.text,
                    "username": record.username,
                    "topic": record.topic,
                    "memory_type": record.memory_type,
                    "final_score": final_score,
                }
            )

        ranked.sort(key=lambda item: item["final_score"], reverse=True)
        return ranked
