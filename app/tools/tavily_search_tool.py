"""Tavily 联网搜索：调用官方 REST API（仅需 httpx，与项目依赖一致）。"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool
from loguru import logger

from app.config import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _tavily_api_key() -> str:
    return (config.tavily_api_key or "").strip()


def _format_tavily_response(data: dict[str, Any]) -> str:
    parts: list[str] = []
    ans = data.get("answer")
    if ans:
        parts.append(f"## 摘要\n{ans}\n")
    results = data.get("results") or []
    if not results:
        parts.append("未返回具体结果条目。")
        return "\n".join(parts)
    parts.append("## 搜索结果\n")
    lim = max(1, min(config.tavily_max_results, 20))
    for i, r in enumerate(results[:lim], 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        content = (r.get("content") or "").strip()
        parts.append(f"### {i}. {title}\n- 链接: {url}\n- 摘要: {content}\n")
    return "\n".join(parts)


@tool
def tavily_web_search(query: str) -> str:
    """使用 Tavily 搜索公开网页，获取新闻、百科、文档等互联网上的最新或通用信息。

    当问题明显超出本地知识库、需要核实近期事实或公开资料时使用。
    不要用于查询本系统内部私有数据（应优先用 retrieve_enriched_context / retrieve_knowledge）。

    Args:
        query: 搜索关键词或一句自然语言问题
    """
    key = _tavily_api_key()
    if not key:
        return (
            "错误：未配置 Tavily API Key。请在 .env 中设置 TAVILY_API_KEY，"
            "或在应用配置中填写 tavily_api_key。"
        )

    q = (query or "").strip()
    if not q:
        return "错误：搜索 query 不能为空。"

    max_res = max(1, min(config.tavily_max_results, 20))
    depth = (config.tavily_search_depth or "basic").strip().lower()
    if depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        depth = "basic"

    payload: dict[str, Any] = {
        "api_key": key,
        "query": q,
        "max_results": max_res,
        "search_depth": depth,
        "include_answer": True,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:500]
        logger.warning("Tavily HTTP 错误: {} {}", e.response.status_code, body)
        return f"Tavily 搜索失败（HTTP {e.response.status_code}）: {body[:300]}"
    except Exception as e:
        logger.exception("Tavily 请求异常: {}", e)
        return f"Tavily 搜索失败: {e!s}"

    return _format_tavily_response(data)
