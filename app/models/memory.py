"""记忆 API 请求/响应模型"""

from typing import Any

from pydantic import BaseModel, Field


class MemoryMilvusRefItem(BaseModel):
    """单条 Milvus 向量引用"""

    collection_name: str = Field(..., description="集合名，如 biz")
    milvus_id: str = Field(..., description="与 LangChain Milvus 写入 id 一致")


class MemoryCreateRequest(BaseModel):
    """创建记忆请求体"""

    kind: str = Field(..., description="episodic | semantic_ref | scratch | meta")
    source: str = Field(..., description="user | agent | system")
    session_id: str | None = Field(None, description="会话/线程 ID")
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    payload: dict[str, Any] | None = None
    importance: float | None = Field(None, ge=0.0, le=1.0)
    expires_at: str | None = Field(None, description="ISO8601，可选")
    tags: list[str] = Field(default_factory=list)
    milvus_refs: list[MemoryMilvusRefItem] = Field(default_factory=list)
