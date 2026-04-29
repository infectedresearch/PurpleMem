from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .storage.sqlite import query, execute
from .storage.schema import ENTITIES_SQL, RELATIONSHIPS_SQL, INDEXES


GAME_NAMES = {
    "valorant",
    "minecraft",
    "fortnite",
    "elden ring",
    "stardew valley",
    "resident evil",
    "rocket league",
}
COMPANY_NAMES = {
    "google",
    "amazon",
    "apple",
    "microsoft",
    "meta",
    "discord",
    "github",
    "openai",
    "anthropic",
}
STREAMER_NAMES = {"xqc", "shroud", "pokimane", "ludwig"}

_SKIP_CAPITALIZED_WORDS = {
    "The",
    "This",
    "That",
    "What",
    "When",
    "Where",
    "How",
    "Why",
    "Will",
    "Grace",
    "May",
    "Joy",
    "Hope",
    "Chance",
    "Star",
    "Sky",
    "River",
    "Rose",
    "Lily",
    "Ivy",
    "Ash",
    "Sage",
    "Monday",
    "Tuesday",
    "January",
    "February",
    "March",
    "April",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "But",
    "And",
    "Also",
    "There",
}

RELATIONSHIP_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:works?|working|employed|job)\s+(?:at|for)\s+(.+?)(?:\s+as|\s*[.,!]|$)",
            re.I,
        ),
        "works_at",
    ),
    (
        re.compile(
            r"\b(?:lives?|living|moved|relocated)\s+(?:in|to)\s+(.+?)(?:\s*[.,!]|$)",
            re.I,
        ),
        "lives_in",
    ),
    (
        re.compile(
            r"\b(?:married|engaged|dating|partner)\s+(?:to|with)\s+(.+?)(?:\s*[.,!]|$)",
            re.I,
        ),
        "partner_of",
    ),
    (
        re.compile(
            r"\b(?:friends?|besties|bros?)\s+(?:with|of)\s+(.+?)(?:\s*[.,!]|$)", re.I
        ),
        "friends_with",
    ),
    (
        re.compile(
            r"\b(?:plays?|playing|mains?)\s+(.+?)(?:\s+on|\s+a lot|\s*[.,!]|$)", re.I
        ),
        "plays",
    ),
    (
        re.compile(
            r"\b(?:born|from|originally)\s+(?:in|from)\s+(.+?)(?:\s*[.,!]|$)", re.I
        ),
        "from",
    ),
    (
        re.compile(
            r"\b(?:is|are)\s+(?:a|an)\s+(\w+(?:\s+\w+){0,2})(?:\s*[.,!]|$)", re.I
        ),
        "is_a",
    ),
]


def entity_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("'", "")


def classify_entity(name: str) -> str:
    lower = name.strip().lower()
    if lower in GAME_NAMES:
        return "game"
    if lower in COMPANY_NAMES:
        return "company"
    if lower in STREAMER_NAMES:
        return "streamer"
    return "concept"


def extract_entities(text: str, username: str | None = None) -> list[dict]:
    entities = []
    seen = set()
    if username:
        entities.append({"name": username, "type": "person"})
        seen.add(entity_id(username))
    lower_text = (text or "").lower()
    words = set(re.findall(r"\b\w+\b", lower_text))

    def name_in_text(name: str) -> bool:
        if len(name) <= 3:
            return name in words
        return name in lower_text

    for game in GAME_NAMES:
        if name_in_text(game) and entity_id(game) not in seen:
            entities.append({"name": game, "type": "game"})
            seen.add(entity_id(game))
    for company in COMPANY_NAMES:
        if name_in_text(company) and entity_id(company) not in seen:
            entities.append({"name": company, "type": "company"})
            seen.add(entity_id(company))
    for streamer in STREAMER_NAMES:
        if name_in_text(streamer) and entity_id(streamer) not in seen:
            entities.append({"name": streamer, "type": "streamer"})
            seen.add(entity_id(streamer))

    cap_pattern = re.compile(r"(?<=[.!?\s])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
    for match in cap_pattern.finditer(text or ""):
        name = match.group(1).strip()
        eid = entity_id(name)
        if eid in seen or len(name) <= 3:
            continue
        if name.split()[0] in _SKIP_CAPITALIZED_WORDS:
            continue
        entities.append({"name": name, "type": "person"})
        seen.add(eid)
    return entities


def extract_relationships(text: str, username: str | None = None) -> list[dict]:
    relationships = []
    if not text or not username:
        return relationships
    for pattern, rel_type in RELATIONSHIP_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        obj_name = match.group(1).strip().rstrip(".,!?;:'\"")
        if (
            obj_name
            and len(obj_name) > 1
            and len(obj_name) <= 50
            and len(obj_name.split()) <= 4
        ):
            relationships.append(
                {
                    "subject": username,
                    "predicate": rel_type,
                    "object": obj_name,
                }
            )
    return relationships


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str = "concept"


class EntityGraph:
    """Small optional graph layer for PurpleMem.

    The graph is intentionally lightweight. It stores entities and typed
    relationships with basic temporal validity so callers can answer
    questions that are awkward for pure vector retrieval.
    """

    def __init__(self, conn):
        self.conn = conn
        self.ensure_schema()

    def ensure_schema(self) -> None:
        execute(self.conn, ENTITIES_SQL)
        execute(self.conn, RELATIONSHIPS_SQL)
        for stmt in INDEXES:
            if "memories" in stmt:
                continue
            execute(self.conn, stmt)

    def upsert_entity(self, name: str, entity_type: str | None = None) -> str:
        now = time.time()
        eid = entity_id(name)
        entity_type = entity_type or classify_entity(name)
        existing = query(
            self.conn, "SELECT id, entity_type FROM entities WHERE id = ?", (eid,)
        )
        if existing:
            current_type = existing[0]["entity_type"]
            next_type = entity_type if entity_type != "concept" else current_type
            execute(
                self.conn,
                "UPDATE entities SET name = ?, entity_type = ?, last_seen = ? WHERE id = ?",
                (name.strip(), next_type, now, eid),
            )
        else:
            execute(
                self.conn,
                "INSERT INTO entities (id, name, entity_type, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
                (eid, name.strip(), entity_type, now, now),
            )
        return eid

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.7,
        source_memory_id: str | None = None,
    ) -> None:
        now = time.time()
        subject_id = self.upsert_entity(subject, "person")
        object_id = self.upsert_entity(obj)
        execute(
            self.conn,
            "INSERT OR IGNORE INTO relationships (subject_id, predicate, object_id, valid_from, valid_to, confidence, source_memory_id, created_at) VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
            (subject_id, predicate, object_id, now, confidence, source_memory_id, now),
        )

    def process_text(
        self,
        text: str,
        username: str | None = None,
        source_memory_id: str | None = None,
    ) -> None:
        for entity in extract_entities(text, username):
            self.upsert_entity(entity["name"], entity["type"])
        for rel in extract_relationships(text, username):
            self.add_relationship(
                rel["subject"],
                rel["predicate"],
                rel["object"],
                source_memory_id=source_memory_id,
            )

    def query_entity(self, name: str) -> dict | None:
        eid = entity_id(name)
        entities = query(
            self.conn, "SELECT id, name, entity_type FROM entities WHERE id = ?", (eid,)
        )
        if not entities:
            return None
        row = entities[0]
        outgoing = query(
            self.conn,
            "SELECT e.name as object_name, r.predicate, r.valid_from, r.valid_to, r.confidence "
            "FROM relationships r JOIN entities e ON r.object_id = e.id "
            "WHERE r.subject_id = ? AND r.valid_to IS NULL ORDER BY r.created_at DESC",
            (eid,),
        )
        incoming = query(
            self.conn,
            "SELECT e.name as subject_name, r.predicate, r.valid_from, r.valid_to, r.confidence "
            "FROM relationships r JOIN entities e ON r.subject_id = e.id "
            "WHERE r.object_id = ? AND r.valid_to IS NULL ORDER BY r.created_at DESC",
            (eid,),
        )
        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "outgoing": [dict(r) for r in outgoing],
            "incoming": [dict(r) for r in incoming],
        }
