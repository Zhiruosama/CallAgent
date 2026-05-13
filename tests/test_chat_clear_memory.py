"""清空对话时可选清理 SQLite 记忆。"""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat import router as chat_router
from app.memory import MemoryAddInput, MemoryManager


def test_chat_clear_with_wipe_purges_memories(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "clear_mem.sqlite")
    mgr.add_memory(
        MemoryAddInput(
            kind="episodic",
            source="user",
            session_id="s-clear-1",
            summary="to be purged",
        )
    )
    app = FastAPI()
    app.include_router(chat_router, prefix="/api")
    mock_rag = MagicMock()
    mock_rag.clear_session.return_value = True
    with patch("app.api.chat.rag_agent_service", mock_rag):
        with patch("app.api.chat.get_memory_manager", return_value=mgr):
            client = TestClient(app)
            res = client.post(
                "/api/chat/clear",
                json={"sessionId": "s-clear-1", "wipeSqliteMemories": True},
            )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["memories_purged"] >= 1
    assert body["data"]["wipe_sqlite_memories"] is True
    active = mgr.retrieve_memories(session_id="s-clear-1", limit=10)
    assert active == []


def test_chat_clear_without_wipe_keeps_memories(tmp_path) -> None:
    mgr = MemoryManager(tmp_path / "clear_mem2.sqlite")
    mgr.add_memory(
        MemoryAddInput(
            kind="meta",
            source="user",
            session_id="s-clear-2",
            summary="keep me",
        )
    )
    app = FastAPI()
    app.include_router(chat_router, prefix="/api")
    mock_rag = MagicMock()
    mock_rag.clear_session.return_value = True
    with patch("app.api.chat.rag_agent_service", mock_rag):
        with patch("app.api.chat.get_memory_manager", return_value=mgr):
            client = TestClient(app)
            res = client.post(
                "/api/chat/clear",
                json={"sessionId": "s-clear-2"},
            )
    assert res.status_code == 200
    assert res.json()["data"]["memories_purged"] == 0
    rows = mgr.retrieve_memories(session_id="s-clear-2", limit=10)
    assert len(rows) == 1
