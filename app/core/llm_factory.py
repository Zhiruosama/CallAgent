"""LLM 工厂类

使用 LangChain ChatOpenAI 对接 **OpenAI 兼容协议**（/v1/chat/completions）。

典型用法：
- DeepSeek 官方：`base_url=https://api.deepseek.com/v1` + DeepSeek API Key
- 阿里云 DashScope 兼容：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- OpenAI、Azure OpenAI 等同理，只需改 base_url 与 api_key

Agent 对话统一走 `create_agent_chat_model()`；嵌入向量仍使用 DashScope Embeddings（见 vector_embedding_service）。
"""

from langchain_openai import ChatOpenAI

from app.config import config
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
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
        )

# 全局 LLM 工厂实例
llm_factory = LLMFactory()
