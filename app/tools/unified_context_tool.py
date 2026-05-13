"""统一上下文检索：向量知识库（Milvus）+ 本会话 SQLite 记忆。"""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.memory import get_memory_manager
from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.memory_tool import get_memory_session_id_for_tools


def _knowledge_block(query: str) -> str:
    try:
        raw = retrieve_knowledge.invoke({"query": query})
    except Exception as e:
        logger.exception(f"retrieve_enriched_context 知识库部分失败: {e}")
        return f"（知识库检索失败：{e}）"
    if isinstance(raw, tuple):
        text = raw[0] if raw else ""
        return text if text else "没有找到相关信息。"
    return str(raw) if raw else "没有找到相关信息。"


def _memories_block(query: str, memory_limit: int) -> str:
    sid = get_memory_session_id_for_tools()
    if not sid:
        return "（当前未绑定会话，已跳过本会话记忆检索。）"
    try:
        rows = get_memory_manager().retrieve_memories(
            query=query.strip() if query.strip() else None,
            session_id=sid,
            limit=memory_limit,
        )
    except Exception as e:
        logger.exception(f"retrieve_enriched_context 记忆部分失败: {e}")
        return f"（本会话记忆检索失败：{e}）"
    if not rows:
        return "未检索到与查询相关的本会话已保存记忆。"
    lines: list[str] = []
    for i, r in enumerate(rows, 1):
        title = r.get("title") or ""
        summary = r.get("summary") or ""
        kind = r.get("kind") or ""
        created = r.get("created_at") or ""
        head = f"{i}. [{kind}] {created}"
        if title:
            head += f" {title}"
        lines.append(f"{head}\n   {summary}")
    return "\n".join(lines)


@tool
def retrieve_enriched_context(query: str, memory_limit: int | None = None) -> str:
    """同时检索向量知识库与本会话已保存记忆，合并为一段上下文。

    当问题既可能命中上传的文档/知识库、又可能命中用户在对话中要求记住的事实时，优先使用本工具；
    若仅需其一，也可单独使用 retrieve_knowledge 或 recall_session_memories。

    Args:
        query: 检索用自然语言查询
        memory_limit: 本会话记忆最多返回条数；默认使用服务端配置 memory_merge_limit（上限 20）
    """
    default_lim = config.memory_merge_limit
    if memory_limit is not None:
        try:
            lim_i = int(memory_limit)
        except (TypeError, ValueError):
            lim_i = default_lim
    else:
        lim_i = default_lim
    lim_i = max(1, min(lim_i, 20))

    q = query.strip() if query else ""
    if not q:
        return "错误：query 不能为空。"

    kb = _knowledge_block(q)
    mem = _memories_block(q, lim_i)

    parts = [
        "## 知识库检索结果",
        kb.strip(),
        "",
        "## 本会话记忆检索结果",
        mem.strip(),
    ]
    out = "\n".join(parts)
    logger.info(
        "retrieve_enriched_context 已执行 session=%s memory_limit=%s",
        get_memory_session_id_for_tools(),
        lim_i,
    )
    return out
