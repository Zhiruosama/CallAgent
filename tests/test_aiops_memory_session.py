"""AIOps 执行期间应绑定与 RAG 相同的记忆会话 ContextVar。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.aiops_service import AIOpsService, NODE_PLANNER
from app.tools.memory_tool import get_memory_session_id_for_tools


@pytest.mark.asyncio
async def test_execute_binds_memory_session_for_graph_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = AIOpsService()
    captured: dict[str, str | None] = {}

    async def fake_astream(*args, **kwargs):
        captured["during_stream"] = get_memory_session_id_for_tools()
        yield {NODE_PLANNER: {"plan": ["仅测试步骤"]}}

    def fake_get_state(config):
        m = MagicMock()
        m.values = {"response": ""}
        return m

    monkeypatch.setattr(svc.graph, "astream", fake_astream)
    monkeypatch.setattr(svc.graph, "get_state", fake_get_state)

    out: list = []
    async for ev in svc.execute("test input", session_id="aiops-sess-xyz"):
        out.append(ev)

    assert captured.get("during_stream") == "aiops-sess-xyz"
    assert get_memory_session_id_for_tools() is None
    assert any(e.get("type") == "complete" for e in out)
