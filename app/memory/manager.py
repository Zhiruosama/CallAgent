"""记忆管理器：添加与检索（一期仅 SQLite，不向量化）。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from app.config import config
from app.memory.migrations import apply_pending_migrations

ALLOWED_KINDS = frozenset({"episodic", "semantic_ref", "scratch", "meta"})
ALLOWED_SOURCES = frozenset({"user", "agent", "system"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017


@dataclass
class MemoryAddInput:
    """写入一条记忆的入参（一期最小字段集）。"""

    kind: str
    source: str
    session_id: str | None = None
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    payload: dict[str, Any] | None = None
    importance: float | None = None
    expires_at: str | None = None
    tags: list[str] = field(default_factory=list)
    milvus_refs: list[tuple[str, str]] = field(default_factory=list)
    """(collection_name, milvus_id) 列表，一期仅占位写入，不向量化。"""


class MemoryManager:
    """SQLite 记忆管理。方法为同步实现；在 async 路由中请使用 asyncio.to_thread 包装。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path or config.memory_db_path)

    @contextmanager
    def _session(self):
        """打开连接，迁移，提交/回滚后始终 close（避免 ResourceWarning）。"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        apply_pending_migrations(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_memory(self, data: MemoryAddInput) -> str:
        """插入一条记忆及可选标签、Milvus 引用。返回 memory id。"""
        if data.kind not in ALLOWED_KINDS:
            raise ValueError(f"invalid kind: {data.kind!r}, expected one of {sorted(ALLOWED_KINDS)}")
        if data.source not in ALLOWED_SOURCES:
            raise ValueError(
                f"invalid source: {data.source!r}, expected one of {sorted(ALLOWED_SOURCES)}"
            )
        memory_id = str(uuid.uuid4())
        now = _utc_now_iso()
        payload_json = json.dumps(data.payload, ensure_ascii=False) if data.payload is not None else None

        tags_norm = [_normalize_tag(t) for t in data.tags if t and t.strip()]
        if len(tags_norm) != len(set(tags_norm)):
            raise ValueError("duplicate tags after normalization")

        for collection_name, milvus_id in data.milvus_refs:
            if not collection_name or not milvus_id:
                raise ValueError(
                    "milvus_refs entries must have non-empty collection_name and milvus_id"
                )

        with self._session() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, session_id, kind, source, title, summary, body,
                    payload_json, importance, created_at, updated_at, expires_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    memory_id,
                    data.session_id,
                    data.kind,
                    data.source,
                    data.title,
                    data.summary,
                    data.body,
                    payload_json,
                    data.importance,
                    now,
                    now,
                    data.expires_at,
                ),
            )
            for tag in tags_norm:
                conn.execute(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                    (memory_id, tag),
                )
            for collection_name, milvus_id in data.milvus_refs:
                conn.execute(
                    """
                    INSERT INTO memory_milvus_refs (memory_id, collection_name, milvus_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (memory_id, collection_name, milvus_id, now),
                )

        logger.info(f"memory added id={memory_id} kind={data.kind} session_id={data.session_id}")
        return memory_id

    def retrieve_memories(
        self,
        *,
        query: str | None = None,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 20,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """按条件检索记忆（一期：仅 SQLite，标题/摘要/正文 LIKE）。"""
        if kind is not None and kind not in ALLOWED_KINDS:
            raise ValueError(f"invalid kind: {kind!r}")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")

        conditions: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            conditions.append("deleted_at IS NULL")

        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)

        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)

        if query and query.strip():
            like = f"%{query.strip()}%"
            conditions.append("(title LIKE ? OR summary LIKE ? OR body LIKE ?)")
            params.extend([like, like, like])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT id, session_id, kind, source, title, summary, body, payload_json,
                   importance, created_at, updated_at, expires_at, deleted_at
            FROM memories
            WHERE {where_clause}
            ORDER BY (importance IS NOT NULL) DESC, importance DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._session() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [_row_to_dict(row) for row in rows]

    def soft_delete_memory(self, memory_id: str) -> bool:
        """按 id 软删一条记忆。不存在或已删则返回 False。"""
        if not memory_id or not str(memory_id).strip():
            raise ValueError("memory_id 不能为空")
        now = _utc_now_iso()
        with self._session() as conn:
            cur = conn.execute(
                """
                UPDATE memories
                SET deleted_at = ?, updated_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, now, memory_id.strip()),
            )
            ok = cur.rowcount > 0
        if ok:
            logger.info(f"memory soft-deleted id={memory_id}")
        return bool(ok)

    def purge_session_memories(self, session_id: str) -> int:
        """软删指定 session_id 下所有未删除的记忆，返回影响行数。"""
        if not session_id or not str(session_id).strip():
            raise ValueError("session_id 不能为空")
        now = _utc_now_iso()
        with self._session() as conn:
            cur = conn.execute(
                """
                UPDATE memories
                SET deleted_at = ?, updated_at = ?
                WHERE session_id = ? AND deleted_at IS NULL
                """,
                (now, now, session_id.strip()),
            )
            n = cur.rowcount
        logger.info(f"purge_session_memories session_id={session_id} rows={n}")
        return int(n)

    def get_memory_stats(self) -> dict[str, Any]:
        """库内记忆条数统计（不含 FTS）。"""
        with self._session() as conn:
            active = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
                ).fetchone()[0]
            )
            soft_deleted = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL"
                ).fetchone()[0]
            )
            rows = conn.execute(
                """
                SELECT kind, COUNT(*) AS c
                FROM memories
                WHERE deleted_at IS NULL
                GROUP BY kind
                """
            ).fetchall()
            sessions = int(
                conn.execute(
                    """
                    SELECT COUNT(DISTINCT session_id) FROM memories
                    WHERE deleted_at IS NULL AND session_id IS NOT NULL AND TRIM(session_id) != ''
                    """
                ).fetchone()[0]
            )
        by_kind = {str(r["kind"]): int(r["c"]) for r in rows}
        return {
            "active_total": active,
            "soft_deleted_total": soft_deleted,
            "distinct_session_ids": sessions,
            "active_by_kind": by_kind,
        }


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    raw = d.pop("payload_json", None)
    if raw is not None:
        try:
            d["payload"] = json.loads(raw)
        except json.JSONDecodeError:
            d["payload"] = None
    else:
        d["payload"] = None
    return d


_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """进程内单例，便于与 FastAPI 生命周期对齐（后续可改为依赖注入）。"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
