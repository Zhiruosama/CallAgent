# 记忆系统后续实施计划（规划稿）

> **决策**：一期 **不引入 Neo4j**。语义检索继续 **Milvus**；结构化长期记忆 **SQLite**；Working 仍以 LangGraph 会话状态为主，可选进程内 TTL。  
> 本文含：**完整系统架构图**、**记忆子系统流程图**、**架构自查**、**SQLite 表设计讨论与推荐草案**、**实施记录（第 8 节）**。  
> **状态（2026-05）**：对话侧 **记忆系统一期主体已落地**（SQLite + API + RAG 工具 + 合并检索 + 管理与清空联动）；**AIOps 图内挂记忆** 仍列为后续项。

---

## 0. 一期范围摘要

| 层级 | 一期做法 |
|------|----------|
| 语义 / 文档 RAG | 不变：`biz`、上传索引、`retrieve_knowledge` |
| 长期 / 事件记忆 | 新增：**SQLite** 存条目与元数据；检索由 **MemoryManager** 聚合 |
| 记忆向量（可选） | 若 episodic 需向量召回：Milvus **同库扩展 metadata** 或 **独立 collection**（实施前再定） |
| 图数据库 | **不采用 Neo4j**；复杂关联优先 **SQLite 表 + 索引**，必要时递归 CTE |
| 租户 / 用户 | **不做区分**（个人使用）；不按用户隔离，仅用 `session_id` 等区分会话 |

---

## 1. 完整系统架构图（原系统 + 一期记忆）

下图在先前「FastAPI / 双 Agent / Milvus / MCP / DashScope」整体结构上，叠加 **MemoryManager** 与 **SQLite**，**不含 Neo4j**。

```mermaid
flowchart TB
  subgraph Client["客户端"]
    Web["静态页 static/"]
    API_Caller["HTTP / SSE 调用方"]
  end

  subgraph FastAPI["FastAPI app.main"]
    H["/health"]
    CH["/api/chat · chat_stream · clear"]
    FI["/api/upload · index_directory"]
    AI["/api/aiops SSE"]
    MEM_API["/api/memory 已落地\nstats·purge·CRUD"]
  end

  subgraph MemoryLayer["一期新增：统一记忆层"]
    MM["MemoryManager\nadd / retrieve / manage"]
    SQLITE[("SQLite\n长期记忆元数据")]
    TTL["可选：进程内 TTL\nWorking 辅助"]
  end

  subgraph Services["app/services"]
    RAG["RagAgentService"]
    AIO["AIOpsService"]
    VIDX["VectorIndexService"]
    VEMB["VectorEmbeddingService"]
    VSM["VectorStoreManager"]
    Other["DocumentSplitter / ResumeParser …"]
  end

  subgraph AgentLayer["app/agent"]
    MCP["mcp_client"]
    subgraph AIOpsNodes["app/agent/aiops"]
      PL["planner"]
      EX["executor"]
      RP["replanner"]
    end
  end

  subgraph Tools["app/tools"]
    RK["retrieve_knowledge"]
    TM["get_current_time"]
  end

  subgraph Core["app/core"]
    MIL["milvus_client"]
  end

  subgraph External["外部依赖"]
    DS["DashScope"]
    MV[("Milvus biz")]
    MCP_CLS["MCP: cls"]
    MCP_MON["MCP: monitor"]
  end

  Web --> FastAPI
  API_Caller --> FastAPI

  CH --> RAG
  FI --> VIDX
  AI --> AIO
  CH --> MM
  MEM_API --> MM
  AI -.规划集成.-> MM

  MM --> SQLITE
  MM --> TTL
  MM --> VSM

  RAG --> DS
  RAG --> MCP
  RAG --> RK
  RAG --> TM

  AIO --> PL --> EX --> RP
  PL --> DS
  PL --> MCP
  PL --> RK
  PL --> TM
  EX --> DS
  EX --> MCP
  EX --> RK
  EX --> TM
  RP --> DS
  RP --> MCP
  RP --> RK
  RP --> TM

  RK --> VSM
  VIDX --> VSM
  VSM --> VEMB
  VSM --> MIL
  MIL --> MV
  VEMB --> DS

  MCP --> MCP_CLS
  MCP --> MCP_MON
```

**读图说明**

- **实线**：现有主数据流；**对话**经 `/api/chat` 与 RAG Agent 已可写/读 SQLite 记忆（见第 8 节）。  
- **MemoryManager → VSM**：仅在「某类记忆需要向量化」或「retrieve 要与 Milvus 联合 Top-K」时发生；纯结构化记忆可只读写 SQLite。  
- **AIOps**：与 `MemoryManager` 的虚线挂钩仍为 **后续**（当前未在 planner/executor 绑 `session_id` 工具）。

---

## 2. 记忆子系统内部流程（对齐参考图，一期落地）

```mermaid
flowchart TB
  UI["用户输入 / Agent 钩子"]
  JUDGE{"操作类型判断"}

  subgraph ADD["添加记忆"]
    MA["Manager.add_memory"]
    CLS{"自动分类 / 路由"}
    WM["Working 辅助\n内存 TTL 可选"]
    EP["Episodic\nSQLite 主存"]
    SM["Semantic 指针\nSQLite + 现有 Milvus 文档"]
    MM_SKIP["Perceptual\n一期不做"]
    EMB["统一嵌入：复用\nVectorEmbeddingService\nDashScope"]
  end

  subgraph SEARCH["搜索记忆"]
    MR["Manager.retrieve_memories"]
    PAR["跨源并行：\nSQLite 条件/全文 +\nMilvus 向量可选"]
    AGG["聚合与排序"]
    TOPK["Top-K 记忆"]
  end

  subgraph MGMT["管理操作"]
    MG["Manager.manage"]
    STR{"管理策略"}
    CON["记忆整合\n摘要写入 Episodic 等"]
    FORG["智能遗忘\nTTL / 配额 / 软删"]
    STAT["统计分析"]
  end

  UI --> JUDGE
  JUDGE -->|添加| MA --> CLS
  CLS -->|临时| WM
  CLS -->|事件| EP
  CLS -->|抽象知识指针| SM
  CLS -->|多模态| MM_SKIP
  EP --> EMB
  SM --> EMB

  JUDGE -->|检索| MR --> PAR --> AGG --> TOPK
  PAR --> SQLITE_DB[("SQLite")]
  PAR --> MV_DB[("Milvus")]

  JUDGE -->|管理| MG --> STR
  STR --> CON
  STR --> FORG
  STR --> STAT
  CON --> SQLITE_DB
  FORG --> SQLITE_DB
  STAT --> SQLITE_DB
```

**与参考图的差异（刻意简化）**

- 无 **Neo4j**；「语义」仍以 **Milvus 文档块** 为准，SQLite 存引用与业务标签。  
- **Episodic** 一期以 SQLite 行为主；是否每条都打向量见下文表设计讨论。

---

## 3. 整体架构自查（是否有问题）

| 关注点 | 说明 | 建议 |
|--------|------|------|
| 双写一致性 | 同一条「记忆」若同时落 SQLite 与 Milvus，需约定 **主键来源**（如 `memory_id` + `milvus_point_id`）与失败回滚策略 | 先 **SQLite 提交成功** 再异步/同步写 Milvus，或仅对「需向量检索」子集写 Milvus |
| 检索语义重叠 | `retrieve_knowledge` 与 `MemoryManager.retrieve` 可能返回相似片段 | Manager 层做 **去重、时间/来源加权**；对外可暴露统一 `context_pack` |
| 多进程 / 多副本 | `MemorySaver` 与进程内 TTL **不跨 worker** | 一期单实例可接受；上多 worker 时 TTL 迁 **Redis** 或只用 SQLite 时间字段驱动过期任务 |
| API 面 | 记忆若仅内嵌 Agent，调试困难 | 规划 **MEM_API** 或管理端最小 CRUD，便于排查与合规删除 |
| 安全与合规 | 记忆可能含 PII | 表级字段 **脱敏级别**、**按用户/租户隔离**、导出与硬删流程写进 P3 |

当前结论：**架构闭环成立**；主要风险在 **一致性、检索聚合、多实例**，可通过一期约束（单实例 + 明确主从存储）控制范围。

---

## 4. SQLite 库表设计：讨论与推荐草案

### 4.1 设计原则（建议先对齐）

1. **一条业务记忆 = 一行主表**（`memories`），扩展字段用 JSON 要克制，避免全表不可查。  
2. **类型用枚举**：`kind` = `episodic` | `semantic_ref` | `meta`（或再拆 `preference`），与规划中的「自动分类」一致；**不必**把 working 长期落库，若落库则 `kind=scratch` + 短 `expires_at`。  
3. **软删优先**：`deleted_at`，便于「遗忘」与审计。  
4. **与 Milvus 弱耦合**：单独关联表，避免在 `memories` 上堆多个可空列。  
5. **全文**：若需关键词检索，再上 **FTS5 虚拟表** 或依赖应用层；一期可仅用 `LIKE` + 索引，数据量上来再加 FTS。

### 4.2 可选路线对比

| 路线 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A. 宽表 + JSON | `memories` 含 `payload_json` | 迭代快 | 查询、索引难；易变「垃圾抽屉」 |
| B. 主表 + 关联表 | 核心列固定，扩展放 `memory_attributes(key,value)` | 灵活又可查部分维度 | 多表 join |
| C. 图式边表 | `entities` + `relations` + `memory_entity` | 不引入 Neo4j 也能表达关系 | 一期建模与 UI 成本高 |

**推荐一期**：**主表 `memories` + 关联表 `memory_milvus_refs` + 可选 `memory_tags`**；若 AIOps 强依赖「服务—告警」边，再在 **P2** 加极简 `relations`（两端 entity 用字符串 id 即可），仍不引入 Neo4j。

### 4.3 推荐表结构（草案，供评审）

**（1）`memories` — 主表**

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | TEXT UUID PK | 全局主键 |
| `session_id` | TEXT NULL | 对话 / 诊断线程（个人项目用会话区分即可，不做用户/租户字段） |
| `kind` | TEXT NOT NULL | `episodic` / `semantic_ref` / `scratch` / … |
| `source` | TEXT NOT NULL | `user` / `agent` / `system` |
| `title` | TEXT NULL | 短标题，便于列表展示 |
| `summary` | TEXT NULL | 检索与展示的摘要（建议控制长度） |
| `body` | TEXT NULL | 可选正文 |
| `payload_json` | TEXT NULL | 结构化扩展（JSON），宜有 schema 约定 |
| `importance` | REAL NULL | 0–1，供排序 |
| `created_at` | TEXT ISO8601 | 创建时间 |
| `updated_at` | TEXT ISO8601 | 更新时间 |
| `expires_at` | TEXT NULL | 过期时间；`scratch` / TTL 策略用 |
| `deleted_at` | TEXT NULL | 软删 |

**索引建议**：`(session_id, created_at DESC)`、`(kind, deleted_at)`。  
（已拍板：**不做** `tenant_id` / 用户区分；若日后改为多人共用，再迁移增加 `user_id` 或 `tenant_id` 及对应索引。）

**（2）`memory_milvus_refs` — 与向量库关联（可选行）**

| 列名 | 类型 | 说明 |
|------|------|------|
| `memory_id` | TEXT FK | 关联 `memories.id` |
| `collection_name` | TEXT | 如 `biz` 或未来 `memory_episodic` |
| `milvus_id` | TEXT | 与 LangChain Milvus 写入的 id 对齐 |
| `created_at` | TEXT | 写入时间 |

**约束**：同一 `memory_id` 可多条（一条记忆对应多个 chunk），聚合时在应用层合并。

**（3）`memory_tags` — 标签（可选）**

| 列名 | 类型 | 说明 |
|------|------|------|
| `memory_id` | TEXT FK | |
| `tag` | TEXT | 小写归一化 |
| PK | `(memory_id, tag)` | |

**（4）远期可选：`entities` / `relations`**

仅当产品确认要在 SQLite 内做「轻图」时再开表；一期 **可不建**，避免过度设计。

### 4.4 待讨论问题（需要你拍板）

1. **租户 / 用户区分**（已拍板）：**不做**。本项目按 **个人使用** 设计，不引入 `tenant_id`、不按登录用户隔离数据；仅用 **`session_id`** 等区分不同对话/诊断线程即可。若以后部署为多人服务，再在 schema 迁移中增加隔离字段。  
2. **Episodic 是否向量化**：每条 episodic 都打 Milvus，还是仅「用户点收藏 / 标重要」才打？（影响成本与 `memory_milvus_refs` 填充率）  
3. **`semantic_ref` 与上传文档**：是否允许 `memories` 只存 `source_file` + 段落锚点，而不重复存 `body`？（减少与 Milvus 内容重复）  
4. **FTS5**：是否坚持一期就上，还是 **P2** 再加？

---

## 5. 目标与路线图（与实施状态对照）

| 操作类型 | 能力概要 | 实施状态 |
|----------|----------|----------|
| 添加记忆 | 路由后写 SQLite；可选 `memory_milvus_refs` | **已落地**：`MemoryManager.add_memory`、REST `POST /api/memory`、工具 `save_session_memory` |
| 搜索记忆 | SQLite 条件 + LIKE；与知识库合并入口 | **已落地**：`retrieve_memories`、`GET /api/memory`、工具 `recall_session_memories`、**`retrieve_enriched_context`**（Milvus + 本会话记忆） |
| 管理操作 | 遗忘、统计、按会话清理 | **已落地**：`GET /api/memory/stats`、`DELETE /api/memory/{id}` 软删、`POST /api/memory/purge?session_id=` 批量软删；误用 `DELETE …/purge` 返回 **405** 提示改用 POST |
| 与对话清空一致 | 清 LangGraph 时可选用时清 SQLite 同会话 | **已落地**：`POST /api/chat/clear` 增加 **`wipeSqliteMemories`**（默认 `false`） |

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | 表结构 + `schema_migrations` 迁移 | **完成**（`app/memory/migrations.py` v1） |
| P1 | `MemoryManager` + 单元测试 | **完成**（`app/memory/manager.py`，`tests/test_memory_manager.py`） |
| P2 | 与 `retrieve_knowledge` 聚合 | **完成**（`app/tools/unified_context_tool.py`，`memory_merge_limit` 配置） |
| P3 | 过期、整合 consolidate、多策略清理 | **部分完成**（软删 + purge + stats；**TTL 自动任务 / consolidate 摘要** 未做） |
| P4 | 专用图库评估 | **未做**（仍不引入 Neo4j） |

**配置项（环境变量可读）**：`memory_db_path`、`memory_merge_limit`（见 `app/config.py`）。

---

## 6. 风险与约束（保留）

- 双源检索一致性、PII、多 worker 与 TTL，见第 3 节。

---

## 7. 待确认事项（更新）

1. **租户 / 用户**：**已定** — 个人项目，不做用户或租户区分（无 `tenant_id`）。  
2. **写入策略**：仅用户显式「记住」 / 允许 Agent 自动摘要（及配额）— **仍为产品策略，代码未强制**。  
3. **Episodic 向量**：独立 collection / `biz` + metadata 区分 / 一期不向量化 — **未强制**；当前 `milvus_refs` 仅占位写入。

---

## 8. 实施记录（与仓库代码一致，便于追溯）

以下按模块罗列 **已实现** 内容（不含 AIOps 侧记忆挂钩）。

### 8.1 数据与核心逻辑

| 项 | 说明 |
|----|------|
| SQLite 文件 | 默认 `data/memory.sqlite`（`memory_db_path`），`data/.gitkeep` 保留目录 |
| 表 | `memories`、`memory_milvus_refs`、`memory_tags`、`schema_migrations` |
| `MemoryManager` | `add_memory`、`retrieve_memories`、`soft_delete_memory`、`purge_session_memories`、`get_memory_stats`；连接 `_session` WAL + `PRAGMA foreign_keys` |

### 8.2 HTTP（`app/api/memory.py`，前缀 `/api`）

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/memory` | 创建记忆 |
| GET | `/memory` | 列表检索（Query：`query`、`session_id`、`kind`、`limit`、`include_deleted`） |
| GET | `/memory/stats` | 统计活跃/软删/按 kind/去重会话数 |
| DELETE | `/memory/{memory_id}` | 单条软删 |
| POST | `/memory/purge?session_id=` | **按会话批量软删**（必须用 POST） |
| DELETE | `/memory/purge` | 防呆：返回 **405**，提示勿用 DELETE 调 purge |

### 8.3 对话与清空（`app/api/chat.py`、`app/models/request.py`）

| 项 | 说明 |
|----|------|
| `ClearRequest` | 可选 **`wipe_sqlite_memories`**，请求 JSON 可用 **`wipeSqliteMemories`**（Pydantic alias），默认 `false` |
| `POST /api/chat/clear` | checkpointer 清空成功后，若 `wipeSqliteMemories=true` 则 `purge_session_memories(session_id)`；`data` 返回 `memories_purged` 等 |

### 8.4 RAG Agent 工具（`app/services/rag_agent_service.py` + `app/tools/`）

| 项 | 说明 |
|----|------|
| `ContextVar` | `memory_tool.memory_session_token_set/reset`，在 `query` / `query_stream` 包一层，绑定 **`session_id` = 请求的 thread** |
| 工具 | `save_session_memory`、`recall_session_memories`、`retrieve_enriched_context`、`retrieve_knowledge`、MCP 等 |
| 合并检索 | `retrieve_enriched_context`：先知识库再本会话记忆，两段 Markdown 标题输出 |
| 辅助 | `get_memory_session_id_for_tools()` 供统一检索读会话 |

### 8.5 测试

| 文件 | 覆盖 |
|------|------|
| `tests/test_memory_manager.py` | CRUD、软删、purge、stats、标签、payload 等 |
| `tests/test_memory_api.py` | REST 子应用 + purge 防呆 DELETE→405 |
| `tests/test_memory_tools.py` | 工具与 ContextVar |
| `tests/test_unified_context_tool.py` | 合并检索 |
| `tests/test_chat_clear_memory.py` | clear + wipe |

### 8.6 明确未做 / 后续

- **AIOps**：planner/executor/replanner **未**绑 `session_id` 与记忆工具（需求优先级延后）。  
- **自动 TTL / consolidate**、**FTS5**、**记忆条目强制写 Milvus**：未做。  
- **前端**：未改静态页；仅靠 API / Apifox / Navicat 调试。

---

*文档版本：规划稿 + 实施记录（2026-05）。*
