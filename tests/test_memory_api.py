"""记忆 API 轻量测试（独立 FastAPI 子应用 + patch get_memory_manager）。"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.memory import router
from app.memory import MemoryManager


@pytest.fixture
def memory_client(tmp_path) -> TestClient:
    mgr = MemoryManager(tmp_path / "api_memory.sqlite")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    with patch("app.api.memory.get_memory_manager", return_value=mgr):
        yield TestClient(app)


def test_post_memory_success(memory_client: TestClient) -> None:
    res = memory_client.post(
        "/api/memory",
        json={
            "kind": "episodic",
            "source": "user",
            "session_id": "s1",
            "summary": "hello api",
            "tags": ["t1"],
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["code"] == 200
    assert body["data"]["success"] is True
    assert "id" in body["data"]


def test_post_memory_bad_kind(memory_client: TestClient) -> None:
    res = memory_client.post(
        "/api/memory",
        json={"kind": "nope", "source": "user", "summary": "x"},
    )
    assert res.status_code == 200
    assert res.json()["code"] == 400


def test_get_memory_list(memory_client: TestClient) -> None:
    memory_client.post(
        "/api/memory",
        json={"kind": "meta", "source": "agent", "summary": "unique-xyz-123"},
    )
    res = memory_client.get("/api/memory", params={"query": "unique-xyz-123"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["count"] >= 1


def test_memory_stats(memory_client: TestClient) -> None:
    memory_client.post(
        "/api/memory",
        json={"kind": "episodic", "source": "user", "session_id": "sx", "summary": "s"},
    )
    res = memory_client.get("/api/memory/stats")
    assert res.status_code == 200
    d = res.json()["data"]
    assert d["success"] is True
    assert d["active_total"] >= 1


def test_delete_and_purge_flow(memory_client: TestClient) -> None:
    r = memory_client.post(
        "/api/memory",
        json={
            "kind": "scratch",
            "source": "user",
            "session_id": "spx",
            "summary": "one",
        },
    )
    mid = r.json()["data"]["id"]
    dres = memory_client.delete(f"/api/memory/{mid}")
    assert dres.json()["code"] == 200
    pres = memory_client.post("/api/memory/purge", params={"session_id": "spx"})
    assert pres.json()["code"] == 200
    assert pres.json()["data"]["purged_count"] == 0


def test_purge_with_delete_method_returns_hint(memory_client: TestClient) -> None:
    res = memory_client.delete("/api/memory/purge")
    assert res.status_code == 405
    body = res.json()
    assert body["code"] == 405
    assert "POST" in body["data"]["error"]
