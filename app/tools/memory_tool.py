"""会话记忆工具：依赖 ContextVar 绑定当前请求的 session_id（由 RagAgentService 设置）。"""

from __future__ import annotations

import contextvars
import json
from typing import Any

from langchain_core.tools import tool
from loguru import logger

from app.memory import MemoryAddInput, get_memory_manager

# 每个 asyncio Task 独立，避免并发请求串会话
_memory_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "memory_session_id", default=None
)


def memory_session_token_set(session_id: str) -> contextvars.Token[str | None]:
    """在单次 Agent 调用前绑定 session_id，返回用于 reset 的 token。"""
    return _memory_session_id.set(session_id)


def memory_session_token_reset(token: contextvars.Token[str | None]) -> None:
    """调用结束后恢复上下文。"""
    _memory_session_id.reset(token)


def _current_session_id() -> str | None:
    return _memory_session_id.get()


def get_memory_session_id_for_tools() -> str | None:
    """供统一检索等模块读取当前请求绑定的会话 ID（未绑定时为 None）。"""
    return _current_session_id()


_ALLOWED_SAVE_KINDS = frozenset({"episodic", "semantic_ref", "scratch", "meta", "note"})


@tool
def save_session_memory(
    summary: str,
    title: str | None = None,
    kind: str = "episodic",
) -> str:
    """将本轮值得记住的信息写入长期记忆（SQLite），供后续对话检索。

    当用户明确要求记住、或达成需要跨轮保留的约定/事实时使用。
    summary 应简短准确；kind 一般为 episodic（事件）、note（人工短记）或 meta（偏好/配置类）。

    Args:
        summary: 要保存的摘要（必填）
        title: 可选短标题
        kind: episodic | semantic_ref | scratch | meta | note，默认 episodic
    """
    sid = _current_session_id()
    if not sid:
        return "错误：当前未绑定会话，无法保存记忆。"

    if kind not in _ALLOWED_SAVE_KINDS:
        return f"错误：kind 必须是 {sorted(_ALLOWED_SAVE_KINDS)} 之一。"

    if not summary or not summary.strip():
        return "错误：summary 不能为空。"

    try:
        mid = get_memory_manager().add_memory(
            MemoryAddInput(
                kind=kind,
                source="agent",
                session_id=sid,
                title=title.strip() if title else None,
                summary=summary.strip(),
            )
        )
    except Exception as e:
        logger.exception(f"save_session_memory 失败: {e}")
        return f"保存失败: {e}"

    logger.info(f"save_session_memory session={sid} id={mid}")
    return f"已保存记忆，id={mid}"


@tool
def recall_session_memories(query: str, limit: int = 8) -> str:
    """从当前会话已存储的记忆中检索与 query 相关的条目（SQLite）。

    当需要回忆用户在本会话中曾保存的事实、偏好或约定时使用。
    与知识库向量检索不同：仅查本会话落库的记忆。

    Args:
        query: 检索关键词或短句
        limit: 返回条数上限，默认 8，最大 20
    """
    sid = _current_session_id()
    if not sid:
        return "错误：当前未绑定会话，无法检索记忆。"

    try:
        lim_raw = int(limit)
    except (TypeError, ValueError):
        lim_raw = 8
    lim = max(1, min(lim_raw, 20))
    q = query.strip() if query else ""

    try:
        rows = get_memory_manager().retrieve_memories(
            query=q if q else None,
            session_id=sid,
            limit=lim,
        )
    except Exception as e:
        logger.exception(f"recall_session_memories 失败: {e}")
        return f"检索失败: {e}"

    if not rows:
        return "未找到与当前查询匹配的本会话记忆。"

    lines: list[str] = []
    for i, r in enumerate(rows, 1):
        title = r.get("title") or ""
        summary = r.get("summary") or ""
        kind = r.get("kind") or ""
        created = r.get("created_at") or ""
        pay: Any = r.get("payload")
        extra = ""
        if pay is not None:
            extra = f" payload={json.dumps(pay, ensure_ascii=False)[:200]}"
        head = f"{i}. [{kind}] {created}"
        if title:
            head += f" {title}"
        lines.append(f"{head}\n   {summary}{extra}")
    return "\n".join(lines)
