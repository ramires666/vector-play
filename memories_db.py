"""Shared LAN-wide gallery for Word2Vec equations, persisted in SQLite.

All records are common to visitors of this server; no browser-local storage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import word2vec_engine as engine

DB = Path(__file__).resolve().parent / "memories.sqlite3"
MAX_MEMORIES = int(os.getenv("VECTOR_CALC_MAX_MEMORIES", "50000"))
_write_lock = threading.Lock()


def connect():
    conn = sqlite3.connect(DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db():
    with closing(connect()) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            terms_json TEXT NOT NULL,
            answer TEXT NOT NULL,
            similarity REAL NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS memories_created_idx ON memories(id DESC)")


def _prune(conn):
    """Keep the newest MAX_MEMORIES rows so the shared gallery cannot fill the disk."""
    if MAX_MEMORIES <= 0:
        return
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total <= MAX_MEMORIES:
        return
    conn.execute("""DELETE FROM memories WHERE id < (
                        SELECT MIN(id) FROM (SELECT id FROM memories ORDER BY id DESC LIMIT ?))""",
                 (MAX_MEMORIES,))


def validate_terms(terms):
    if not isinstance(terms, list) or not 1 <= len(terms) <= 12:
        raise ValueError("Нужно от 1 до 12 слов")
    checked = []
    for item in terms:
        if not isinstance(item, dict):
            raise ValueError("Некорректная переменная")
        text = item.get("text")
        weight = item.get("weight")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 120:
            raise ValueError("Для коллекции каждое выражение — от 1 до 120 символов")
        if type(weight) not in (int, float) or weight not in (-1, 1):
            raise ValueError("В коллекции поддерживаются только сложение и вычитание")
        checked.append({"text": text.strip(), "weight": int(weight)})
    return checked


def signature(terms):
    canonical = [{"text": term["text"].casefold(), "weight": term["weight"]} for term in terms]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def as_item(row):
    return {"id": row["id"], "created_at": row["created_at"],
            "terms": json.loads(row["terms_json"]),
            "answer": row["answer"], "similarity": row["similarity"]}


def list_memories(after=0, limit=50):
    if type(after) is not int or after < 0:
        raise ValueError("Некорректный идентификатор обновления")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Лимит должен быть от 1 до 100")
    with closing(connect()) as conn:
        # Newest first, both for initial display and incremental refresh.
        rows = conn.execute("""SELECT id,created_at,terms_json,answer,similarity FROM memories
                               WHERE id>? ORDER BY id DESC LIMIT ?""", (after, limit)).fetchall()
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        last_id = conn.execute("SELECT COALESCE(MAX(id),0) FROM memories").fetchone()[0]
    return {"items": [as_item(row) for row in rows], "total": count, "latest_id": last_id}


def add_memory(raw_terms):
    terms = validate_terms(raw_terms)
    digest = signature(terms)
    with closing(connect()) as conn:
        existing = conn.execute("SELECT * FROM memories WHERE signature=?", (digest,)).fetchone()
        if existing:
            return as_item(existing), False

    # Calculate independently on the server; never trust a result sent by a browser.
    result = engine.calculate(terms, top_k=1, exclude_terms=True)
    if not result["results"]:
        raise ValueError("Нет результата для сохранения")
    best = result["results"][0]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _write_lock:
        with closing(connect()) as conn, conn:
            conn.execute("""INSERT OR IGNORE INTO memories
                         (signature,created_at,terms_json,answer,similarity)
                         VALUES(?,?,?,?,?)""", (
                digest, created_at, json.dumps(terms, ensure_ascii=False),
                best["text"].split(" · ")[0], best["similarity"]
            ))
            row = conn.execute("SELECT * FROM memories WHERE signature=?", (digest,)).fetchone()
            if row is None:
                raise RuntimeError("Не удалось записать пример")
            _prune(conn)
            return as_item(row), row["created_at"] == created_at and row["terms_json"] == json.dumps(terms, ensure_ascii=False)


init_db()
