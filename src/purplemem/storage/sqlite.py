from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def execute(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> None:
    conn.execute(sql, params)
    conn.commit()


def execute_many(conn: sqlite3.Connection, sql: str, params_seq: list[tuple]) -> None:
    conn.executemany(sql, params_seq)
    conn.commit()
