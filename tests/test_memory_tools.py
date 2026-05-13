"""会话记忆工具与 ContextVar 绑定测试。"""

from unittest.mock import patch

from app.memory import MemoryManager
from app.tools.memory_tool import (
    memory_session_token_reset,
    memory_session_token_set,
    recall_session_memories,
    save_session_memory,
)


def test_save_session_memory_requires_context(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "t.sqlite")
    with patch("app.tools.memory_tool.get_memory_manager", return_value=mgr):
        out = save_session_memory.invoke({"summary": "x", "title": None, "kind": "episodic"})
    assert "未绑定会话" in out


def test_save_and_recall_with_context(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "t.sqlite")
    with patch("app.tools.memory_tool.get_memory_manager", return_value=mgr):
        tok = memory_session_token_set("tool-sess-1")
        try:
            out_save = save_session_memory.invoke(
                {"summary": "用户喜欢短句回答", "title": "偏好", "kind": "meta"}
            )
            assert "已保存" in out_save
            out_recall = recall_session_memories.invoke({"query": "短句", "limit": 5})
        finally:
            memory_session_token_reset(tok)
    assert "短句" in out_recall
