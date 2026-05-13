"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 未单独配置 agent_openai_model 时，Agent 对话默认使用的模型名

    # Agent 对话 LLM（OpenAI 兼容协议 /chat/completions）
    # 接 DeepSeek 官方：设置 AGENT_OPENAI_BASE_URL=https://api.deepseek.com/v1 与 AGENT_OPENAI_API_KEY、AGENT_OPENAI_MODEL
    # 留空 api_key 或 model 时分别回退为 dashscope_api_key、rag_model（便于沿用原百炼配置）
    agent_openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    agent_openai_api_key: str = ""
    agent_openai_model: str = ""

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # PDF 入库：文本抽取上限（防恶意超大文件）
    pdf_max_pages: int = 200
    pdf_max_extract_chars: int = 500_000

    # 记忆系统（SQLite，路径扩展名一般为 .sqlite，已在 .gitignore 中忽略）
    memory_db_path: str = "data/memory.sqlite"
    # 统一检索工具中，本会话记忆最多返回条数（1–20）
    memory_merge_limit: int = 8

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
