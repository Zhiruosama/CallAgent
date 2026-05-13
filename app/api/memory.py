"""记忆 HTTP 接口（最小 CRUD：写入与列表检索）"""

import asyncio
from typing import Any

from fastapi import APIRouter, Query
from loguru import logger

from app.memory import MemoryAddInput, get_memory_manager
from app.models.memory import MemoryCreateRequest

router = APIRouter()


def _to_add_input(body: MemoryCreateRequest) -> MemoryAddInput:
    refs = [(r.collection_name, r.milvus_id) for r in body.milvus_refs]
    return MemoryAddInput(
        kind=body.kind,
        source=body.source,
        session_id=body.session_id,
        title=body.title,
        summary=body.summary,
        body=body.body,
        payload=body.payload,
        importance=body.importance,
        expires_at=body.expires_at,
        tags=body.tags,
        milvus_refs=refs,
    )


@router.post("/memory")
async def create_memory(body: MemoryCreateRequest) -> dict[str, Any]:
    """写入一条记忆（SQLite）。同步存储在 asyncio 线程池中执行。"""
    mgr = get_memory_manager()
    try:
        memory_id = await asyncio.to_thread(mgr.add_memory, _to_add_input(body))
    except ValueError as e:
        logger.warning(f"创建记忆参数错误: {e}")
        return {
            "code": 400,
            "message": "invalid_request",
            "data": {"success": False, "error": str(e)},
        }
    except Exception as e:
        logger.exception(f"创建记忆失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {"success": False, "error": str(e)},
        }

    logger.info(f"API 创建记忆 id={memory_id} kind={body.kind}")
    return {
        "code": 200,
        "message": "success",
        "data": {"success": True, "id": memory_id, "db_path": str(mgr._db_path)},
    }


@router.get("/memory")
async def list_memories(
    query: str | None = Query(None, description="标题/摘要/正文 子串匹配"),
    session_id: str | None = Query(None),
    kind: str | None = Query(None, description="episodic | semantic_ref | scratch | meta"),
    limit: int = Query(20, ge=1, le=500),
    include_deleted: bool = Query(False),
) -> dict[str, Any]:
    """检索记忆列表（仅 SQLite，一期不做与 Milvus 聚合）。"""
    mgr = get_memory_manager()
    try:
        items = await asyncio.to_thread(
            mgr.retrieve_memories,
            query=query,
            session_id=session_id,
            kind=kind,
            limit=limit,
            include_deleted=include_deleted,
        )
    except ValueError as e:
        return {
            "code": 400,
            "message": "invalid_request",
            "data": {"success": False, "error": str(e)},
        }
    except Exception as e:
        logger.exception(f"检索记忆失败: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {"success": False, "error": str(e)},
        }

    return {
        "code": 200,
        "message": "success",
        "data": {
            "success": True,
            "count": len(items),
            "items": items,
            "db_path": str(mgr._db_path),
        },
    }
