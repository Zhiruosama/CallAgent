"""MemoryManager 基础行为测试（使用临时 SQLite 文件）。"""

import pytest

from app.memory.manager import MemoryAddInput, MemoryManager


@pytest.fixture
def mgr(tmp_path) -> MemoryManager:
    return MemoryManager(tmp_path / "test_memory.sqlite")


def test_add_memory_invalid_kind(mgr: MemoryManager) -> None:
    with pytest.raises(ValueError, match="invalid kind"):
        mgr.add_memory(
            MemoryAddInput(kind="unknown", source="user", summary="x"),
        )


def test_add_memory_invalid_source(mgr: MemoryManager) -> None:
    with pytest.raises(ValueError, match="invalid source"):
        mgr.add_memory(
            MemoryAddInput(kind="episodic", source="hacker", summary="x"),
        )


def test_add_and_retrieve_by_session(mgr: MemoryManager) -> None:
    mid = mgr.add_memory(
        MemoryAddInput(
            kind="episodic",
            source="agent",
            session_id="sess-a",
            title="告警处理",
            summary="CPU 高负载已排查",
            tags=["ops", "cpu"],
        )
    )
    assert len(mid) == 36

    rows = mgr.retrieve_memories(session_id="sess-a", limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == mid
    assert rows[0]["kind"] == "episodic"
    assert rows[0]["session_id"] == "sess-a"
    assert rows[0]["title"] == "告警处理"
    assert rows[0]["payload"] is None


def test_retrieve_text_query(mgr: MemoryManager) -> None:
    mgr.add_memory(
        MemoryAddInput(
            kind="semantic_ref",
            source="user",
            session_id="s1",
            summary="偏好使用深色主题",
        )
    )
    mgr.add_memory(
        MemoryAddInput(
            kind="episodic",
            source="system",
            session_id="s2",
            summary="无关内容",
        )
    )
    rows = mgr.retrieve_memories(query="深色", limit=10)
    assert len(rows) == 1
    assert rows[0]["summary"] == "偏好使用深色主题"


def test_soft_delete_not_returned_by_default(mgr: MemoryManager) -> None:
    # 一期尚未实现 delete API，直接用 SQL 模拟软删列
    mid = mgr.add_memory(
        MemoryAddInput(kind="scratch", source="user", summary="tmp"),
    )
    with mgr._session() as conn:
        conn.execute(
            "UPDATE memories SET deleted_at = ? WHERE id = ?",
            ("2099-01-01T00:00:00+00:00", mid),
        )
        conn.commit()

    assert mgr.retrieve_memories(query="tmp", limit=10) == []
    rows = mgr.retrieve_memories(query="tmp", limit=10, include_deleted=True)
    assert len(rows) == 1


def test_milvus_ref_roundtrip(mgr: MemoryManager) -> None:
    mid = mgr.add_memory(
        MemoryAddInput(
            kind="episodic",
            source="agent",
            summary="with vector ref",
            milvus_refs=[("biz", "uuid-chunk-1")],
        )
    )
    with mgr._session() as conn:
        refs = conn.execute(
            "SELECT collection_name, milvus_id FROM memory_milvus_refs WHERE memory_id = ?",
            (mid,),
        ).fetchall()
    assert len(refs) == 1
    assert refs[0][0] == "biz"
    assert refs[0][1] == "uuid-chunk-1"


def test_duplicate_tags_rejected(mgr: MemoryManager) -> None:
    with pytest.raises(ValueError, match="duplicate tags"):
        mgr.add_memory(
            MemoryAddInput(
                kind="meta",
                source="user",
                summary="x",
                tags=["a", "A"],
            ),
        )


def test_payload_roundtrip(mgr: MemoryManager) -> None:
    mgr.add_memory(
        MemoryAddInput(
            kind="meta",
            source="user",
            summary="cfg",
            payload={"k": 1, "nested": {"x": True}},
        )
    )
    rows = mgr.retrieve_memories(query="cfg", limit=5)
    assert rows[0]["payload"] == {"k": 1, "nested": {"x": True}}
