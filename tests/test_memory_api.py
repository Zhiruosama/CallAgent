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
