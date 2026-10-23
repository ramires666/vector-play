"""Shared gallery for word equations, persisted in SQLite.

Records are common to visitors of this server and are kept per model: the same
equation gives different answers in different vector spaces, so the model id is
part of the record signature and of the listing filter.
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
LEGACY_MODEL = "ru-rusvectores"
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
            model TEXT NOT NULL DEFAULT 'ru-rusvectores',
            status TEXT NOT NULL DEFAULT 'approved',
            moderation TEXT,
            created_at TEXT NOT NULL,
            terms_json TEXT NOT NULL,
            answer TEXT NOT NULL,
            similarity REAL NOT NULL
        )""")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        if "model" not in columns:
            # Базы, созданные до появления английских моделей: все записи русские.
            conn.execute(f"ALTER TABLE memories ADD COLUMN model TEXT NOT NULL DEFAULT '{LEGACY_MODEL}'")
        if "status" not in columns:
            # Записи до появления модерации считаем одобренными (их можно перепроверить скриптом).
            conn.execute("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'approved'")
        if "moderation" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN moderation TEXT")
        if "group_name" not in columns:
            # Группа допуска: kids / teen / adult (NULL — ещё не оценивали).
            conn.execute("ALTER TABLE memories ADD COLUMN group_name TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS memories_created_idx ON memories(id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS memories_model_idx ON memories(model, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS memories_status_idx ON memories(status, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS memories_group_idx ON memories(group_name, id DESC)")
        _migrate_signatures(conn)


def _migrate_signatures(conn):
    """Пересчитать подписи записей, созданных до появления нескольких моделей.

    Раньше сигнатура считалась без модели, из-за чего после обновления повторное
    сохранение того же уравнения не находило старую запись и создавало дубль.
    """
    updated = merged = 0
    for row in conn.execute("SELECT id,signature,model,terms_json FROM memories").fetchall():
        try:
            terms = json.loads(row["terms_json"])
            if not isinstance(terms, list) or not terms:
                continue
            fresh = signature(terms, row["model"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if fresh == row["signature"]:
            continue
        try:
            conn.execute("UPDATE memories SET signature=? WHERE id=?", (fresh, row["id"]))
            updated += 1
        except sqlite3.IntegrityError:
            # Такая же запись (по новой подписи) уже есть — старую убираем.
            conn.execute("DELETE FROM memories WHERE id=?", (row["id"],))
            merged += 1
    if updated or merged:
        print(f"memories.sqlite3: подписей обновлено {updated}, дублей слито {merged}", flush=True)


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


def validate_model(model_id):
    if not isinstance(model_id, str) or not 1 <= len(model_id) <= 64:
        raise ValueError("Некорректная модель")
    return model_id


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


def signature(terms, model_id):
    canonical = [{"text": term["text"].casefold(), "weight": term["weight"]} for term in terms]
    payload = model_id + "|" + json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_item(row):
    return {"id": row["id"], "model": row["model"], "status": row["status"], "group": row["group_name"],
            "moderation": json.loads(row["moderation"]) if row["moderation"] else None,
            "created_at": row["created_at"], "terms": json.loads(row["terms_json"]),
            "answer": row["answer"], "similarity": row["similarity"]}


def set_assessment(memory_id, moderation=None, group_name=None, status=None):
    """Записать оценку модерации, группу допуска и/или статус (в том числе вручную)."""
    fields, params = [], []
    if moderation is not None:
        fields.append("moderation = ?")
        params.append(json.dumps(moderation, ensure_ascii=False))
    if group_name is not None:
        fields.append("group_name = ?")
        params.append(group_name)
    if status is not None:
        fields.append("status = ?")
        params.append(status)
    if not fields:
        return None
    params.append(memory_id)
    with _write_lock:
        with closing(connect()) as conn, conn:
            conn.execute(f"UPDATE memories SET {', '.join(fields)} WHERE id=?", params)
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    return as_item(row) if row else None


def list_memories(after=0, limit=50, model=None, models=None, status=None):
    if type(after) is not int or after < 0:
        raise ValueError("Некорректный идентификатор обновления")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Лимит должен быть от 1 до 100")
    if model is not None:
        model = validate_model(model)
        models = None
    elif models:
        models = [validate_model(item) for item in models]
    if status is not None and status not in ("approved", "pending", "rejected"):
        raise ValueError("Некорректный статус")
    filters, filter_params = [], []
    if model is not None:
        filters.append("model = ?")
        filter_params.append(model)
    elif models:
        filters.append("model IN (" + ",".join("?" for _ in models) + ")")
        filter_params.extend(models)
    if status is not None:
        filters.append("status = ?")
        filter_params.append(status)
    where_rows = " AND ".join(["id > ?"] + filters)
    where_any = " AND ".join(filters) if filters else "1=1"
    with closing(connect()) as conn:
        # Newest first, both for initial display and incremental refresh.
        rows = conn.execute(f"""SELECT id,model,status,group_name,moderation,created_at,terms_json,answer,similarity
                               FROM memories WHERE {where_rows} ORDER BY id DESC LIMIT ?""",
                            (after, *filter_params, limit)).fetchall()
        count = conn.execute(f"SELECT COUNT(*) FROM memories WHERE {where_any}", filter_params).fetchone()[0]
        last_id = conn.execute(f"SELECT COALESCE(MAX(id),0) FROM memories WHERE {where_any}", filter_params).fetchone()[0]
    return {"items": [as_item(row) for row in rows], "total": count, "latest_id": last_id, "model": model}


def find_existing(raw_terms, model_id=None):
    """Найти такую же запись (по подписи) — используется для дедупликации и модерации."""
    model_id = validate_model(model_id or engine.DEFAULT_MODEL)
    terms = validate_terms(raw_terms)
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM memories WHERE signature=?", (signature(terms, model_id),)).fetchone()
    return as_item(row) if row else None


def add_memory(raw_terms, model_id=None, status="approved", moderation=None, group_name=None):
    model_id = validate_model(model_id or engine.DEFAULT_MODEL)
    if status not in ("approved", "pending", "rejected"):
        raise ValueError("Некорректный статус")
    if group_name is not None and group_name not in ("kids", "teen", "adult"):
        raise ValueError("Некорректная группа допуска")
    terms = validate_terms(raw_terms)
    digest = signature(terms, model_id)
    moderation_json = json.dumps(moderation, ensure_ascii=False) if moderation else None
    with closing(connect()) as conn:
        existing = conn.execute("SELECT * FROM memories WHERE signature=?", (digest,)).fetchone()
        if existing:
            return as_item(existing), False

    # Calculate independently on the server; never trust a result sent by a browser.
    result = engine.calculate(terms, top_k=1, exclude_terms=True, model_id=model_id)
    if not result["results"]:
        raise ValueError("Нет результата для сохранения")
    best = result["results"][0]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _write_lock:
        with closing(connect()) as conn, conn:
            conn.execute("""INSERT OR IGNORE INTO memories
                         (signature,model,status,group_name,moderation,created_at,terms_json,answer,similarity)
                         VALUES(?,?,?,?,?,?,?,?,?)""", (
                digest, model_id, status, group_name, moderation_json, created_at,
                json.dumps(terms, ensure_ascii=False),
                best["text"].split(" · ")[0], best["similarity"]
            ))
            row = conn.execute("SELECT * FROM memories WHERE signature=?", (digest,)).fetchone()
            if row is None:
                raise RuntimeError("Не удалось записать пример")
            _prune(conn)
            return as_item(row), row["created_at"] == created_at and row["terms_json"] == json.dumps(terms, ensure_ascii=False)


def set_status(memory_id, status, moderation=None):
    """Решение человека по записи: approved / rejected / pending."""
    if status not in ("approved", "pending", "rejected"):
        raise ValueError("Некорректный статус")
    with _write_lock:
        with closing(connect()) as conn, conn:
            conn.execute("UPDATE memories SET status=?, moderation=COALESCE(?, moderation) WHERE id=?",
                         (status, json.dumps(moderation, ensure_ascii=False) if moderation else None, memory_id))
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    return as_item(row) if row else None


init_db()
