from __future__ import annotations

from dataclasses import dataclass
import math
import time


HALF_LIFE_BY_TYPE = {
    "core": 100 * 365 * 24 * 60 * 60,
    "episodic": 14 * 24 * 60 * 60,
    "profile": 180 * 24 * 60 * 60,
    "preference": 120 * 24 * 60 * 60,
    "social": 60 * 24 * 60 * 60,
    "task": 7 * 24 * 60 * 60,
    "reflection": 30 * 24 * 60 * 60,
    "ambient": 21 * 24 * 60 * 60,
    "conversation": 30 * 24 * 60 * 60,
    "decision": 30 * 24 * 60 * 60,
    "milestone": 90 * 24 * 60 * 60,
    "problem": 14 * 24 * 60 * 60,
    "emotional": 60 * 24 * 60 * 60,
}


@dataclass
class SearchConfig:
    lexical_mode: str = "multiplicative"
    lexical_weight: float = 0.25
    topic_boost: float = 0.0
    temporal_penalty: float = 0.25
    temporal_query_boost: bool = False
    topic_browse_supplement: bool = True


def lexical_match_bonus(document: str, query_variants: list[str]) -> float:
    normalized_doc = " ".join((document or "").strip().lower().split())
    if not normalized_doc:
        return 0.0
    bonus = 0.0
    for variant in query_variants:
        normalized_variant = " ".join((variant or "").strip().lower().split())
        if not normalized_variant:
            continue
        if normalized_variant in normalized_doc:
            bonus = max(bonus, 0.25)
        else:
            tokens = [token for token in normalized_variant.split() if len(token) >= 3]
            if not tokens:
                continue
            overlap = sum(1 for token in tokens if token in normalized_doc)
            if overlap:
                bonus = max(bonus, min(0.2, 0.06 * overlap))
    return bonus


def score_memory(
    base_score, confidence, evidence_count, last_seen_at, status, memory_type
):
    confidence = float(confidence or 0.0)
    evidence_count = int(evidence_count or 1)
    last_seen_at = float(last_seen_at or 0.0)
    age_seconds = max(0.0, time.time() - last_seen_at)
    half_life = HALF_LIFE_BY_TYPE.get(
        memory_type or "episodic", HALF_LIFE_BY_TYPE["episodic"]
    )
    decay = math.exp(-age_seconds / half_life)
    confidence_boost = 0.25 * max(0.0, min(1.0, confidence))
    evidence_boost = min(0.12, 0.03 * math.log1p(max(1, evidence_count)))
    recency_boost = 0.2 * decay
    dispute_penalty = (
        0.35 if status == "disputed" else 0.2 if status == "stale" else 0.0
    )
    return (
        base_score + confidence_boost + evidence_boost + recency_boost - dispute_penalty
    )


def score_memory_details(
    base_score, confidence, evidence_count, last_seen_at, status, memory_type
):
    final_score = score_memory(
        base_score, confidence, evidence_count, last_seen_at, status, memory_type
    )
    return {
        "semantic_score": base_score,
        "final_score": final_score,
    }
