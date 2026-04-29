from __future__ import annotations


MEMORIES_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    username TEXT,
    memory_type TEXT,
    topic TEXT,
    confidence REAL DEFAULT 0.7,
    evidence_count INTEGER DEFAULT 1,
    last_seen_at REAL,
    valid_from REAL,
    valid_to REAL,
    subject TEXT,
    predicate TEXT,
    object_text TEXT
)
"""

ENTITIES_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT DEFAULT 'concept',
    first_seen REAL,
    last_seen REAL
)
"""

RELATIONSHIPS_SQL = """
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    valid_from REAL,
    valid_to REAL,
    confidence REAL DEFAULT 0.7,
    source_memory_id TEXT,
    created_at REAL NOT NULL
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_username ON memories(username)",
    "CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)",
    "CREATE INDEX IF NOT EXISTS idx_memories_valid_to ON memories(valid_to)",
    "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
    "CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(object_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_active_unique ON relationships(subject_id, predicate, object_id) WHERE valid_to IS NULL",
]


def schema_sql() -> list[str]:
    return [MEMORIES_SQL, ENTITIES_SQL, RELATIONSHIPS_SQL, *INDEXES]
