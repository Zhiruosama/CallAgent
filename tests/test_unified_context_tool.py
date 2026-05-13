"""retrieve_enriched_context 合并检索测试。"""

from unittest.mock import patch

from app.memory import MemoryAddInput, MemoryManager
from app.tools.memory_tool import memory_session_token_reset, memory_session_token_set
from app.tools.unified_context_tool import retrieve_enriched_context


def test_enriched_requires_non_empty_query() -> None:
    out = retrieve_enriched_context.invoke({"query": "   ", "memory_limit": None})
    assert "不能为空" in out


def test_enriched_kb_only_when_no_session(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "u.sqlite")
    with patch("app.tools.unified_context_tool.get_memory_manager", return_value=mgr):
        with patch(
            "app.tools.unified_context_tool._knowledge_block",
            return_value="文档片段A",
        ):
            out = retrieve_enriched_context.invoke({"query": "CPU 告警", "memory_limit": 5})
    assert "## 知识库检索结果" in out
    assert "文档片段A" in out
    assert "## 本会话记忆检索结果" in out
    assert "未绑定会话" in out or "跳过" in out


def test_enriched_merges_kb_and_memory(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "u.sqlite")
    mgr.add_memory(
        MemoryAddInput(
            kind="episodic",
            source="user",
            session_id="merge-s1",
            summary="合并测试记忆关键词 alpha-beta",
        )
    )
    with patch("app.tools.unified_context_tool.get_memory_manager", return_value=mgr):
        with patch(
            "app.tools.unified_context_tool._knowledge_block",
            return_value="知识库关于 alpha 的说明",
        ):
            tok = memory_session_token_set("merge-s1")
            try:
                out = retrieve_enriched_context.invoke(
                    {"query": "alpha-beta", "memory_limit": 5}
                )
            finally:
                memory_session_token_reset(tok)
    assert "知识库关于 alpha" in out
    assert "合并测试记忆" in out
