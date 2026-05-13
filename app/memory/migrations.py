"""SQLite schema 迁移：版本号记录在 schema_migrations 表。"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

# 迁移脚本按 version 升序执行；每条脚本须幂等（依赖 IF NOT EXISTS 等）。
MIGRATIONS: Mapping[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        kind TEXT NOT NULL,
        source TEXT NOT NULL,
        title TEXT,
        summary TEXT,
        body TEXT,
        payload_json TEXT,
        importance REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT,
        deleted_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_memories_session_created
        ON memories (session_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_memories_kind_deleted
        ON memories (kind, deleted_at);

    CREATE TABLE IF NOT EXISTS memory_milvus_refs (
        memory_id TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        milvus_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, collection_name, milvus_id),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS memory_tags (
        memory_id TEXT NOT NULL,
        tag TEXT NOT NULL COLLATE NOCASE,
        PRIMARY KEY (memory_id, tag),
        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
    );
    """,
}


def apply_pending_migrations(conn: sqlite3.Connection) -> None:
    """执行尚未记录的迁移版本。"""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY NOT NULL)"
    )
    applied = {
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version in sorted(MIGRATIONS):
        if version in applied:
            continue
        conn.executescript(MIGRATIONS[version])
        conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))
    conn.commit()
