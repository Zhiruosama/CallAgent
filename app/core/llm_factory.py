"""LLM 工厂类

使用 LangChain ChatOpenAI 对接 **OpenAI 兼容协议**（/v1/chat/completions）。

典型用法：
- DeepSeek 官方：`base_url=https://api.deepseek.com/v1` + DeepSeek API Key
- 阿里云 DashScope 兼容：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- OpenAI、Azure OpenAI 等同理，只需改 base_url 与 api_key

Agent 对话统一走 `create_agent_chat_model()`；嵌入向量仍使用 DashScope Embeddings（见 vector_embedding_service）。
"""

from __future__ import annotations

from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config


def _agent_base_host_is_deepseek(base_url: str) -> bool:
    """识别 DeepSeek 官方 OpenAI 兼容域名，用于默认关闭 thinking（避免 reasoning_content 多轮问题）。"""
    raw = (base_url or "").strip().lower()
    if not raw:
        return False
    if "://" not in raw:
        raw = "https://" + raw
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return False
    return host == "api.deepseek.com" or host.endswith(".deepseek.com")


class LLMFactory:
    """LLM 工厂类 - 使用 OpenAI 兼容模式"""

    # 阿里云 DashScope OpenAI 兼容模式 URL
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key

        # 参考：https://help.aliyun.com/zh/model-studio/getting-started/models
        extra_body = {}
        extra_body["stream"] = streaming

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body if extra_body else None,
        )

        return llm

    @staticmethod
    def create_agent_chat_model(
        temperature: float = 0.7,
        streaming: bool = True,
    ) -> ChatOpenAI:
        """RAG / AIOps 等 Agent 使用的对话模型（OpenAI 兼容，可接 DeepSeek 官方或 DashScope 兼容模式）。"""
        base_url = (config.agent_openai_base_url or "").strip() or LLMFactory.DASHSCOPE_BASE_URL
        api_key = (config.agent_openai_api_key or "").strip() or config.dashscope_api_key
        model = (config.agent_openai_model or "").strip() or config.rag_model
        bu = base_url.rstrip("/")
        deepseek_host = _agent_base_host_is_deepseek(bu)
        auto_off = deepseek_host and config.agent_deepseek_auto_disable_thinking
        send_thinking_disabled = config.agent_thinking_disabled or auto_off
        extra_body: dict | None = None
        if send_thinking_disabled:
            # https://api-docs.deepseek.com/guides/thinking_mode
            extra_body = {"thinking": {"type": "disabled"}}
            logger.info(
                "Agent ChatOpenAI: extra_body thinking=disabled "
                "(agent_thinking_disabled={} deepseek_host={} auto_disable={}) base_url={}",
                config.agent_thinking_disabled,
                deepseek_host,
                config.agent_deepseek_auto_disable_thinking,
                bu,
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=bu,
            api_key=api_key,
            extra_body=extra_body,
        )

# 全局 LLM 工厂实例
llm_factory = LLMFactory()
