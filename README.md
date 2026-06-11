# 🍳 Graph RAG 烹饪助手系统

> 基于知识图谱（Neo4j）的 RAG 智能烹饪问答系统，支持智能查询路由、图多跳推理、三路混合检索。
> 原链接：[https://github.com/datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)

**技术栈**：RAG · Neo4j · Milvus · BGE-small-zh · LangChain ·  BM25 · RRF融合 · 图谱 RAG 检索


**核心特性**：
- 🕸️ 图RAG检索 — 基于 Neo4j 的多跳遍历与子图推理
- 🔀 智能路由 — LLM 驱动的查询复杂度分析，自动选择检索策略
- 🔍 三路混合检索 — Milvus向量 + BM25关键词 + 图KV索引，RRF融合
- 📉 降级保护 — 图RAG失败自动降级到传统检索，保证可用性


## Graph RAG 烹饪助手系统 — 完整架构分析

### 一、系统总览

这是一个**基于图数据库（Neo4j）的 RAG 智能烹饪问答系统**

| 维度 | 技术选型 |
|------|---------|
| 图数据库 | Neo4j（存储菜谱/食材/步骤/分类的实体与关系） |
| 向量数据库 | Milvus（HNSW索引，COSINE相似度） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（512维） |
| LLM | GLM-4-Flash（智谱API） |
| 关键词检索 | BM25Okapi + jieba分词 + 中文停用词过滤 |
| 融合策略 | RRF (Reciprocal Rank Fusion, k=60) |

运行流程可以参考 drawio 流程图：[Graph_RAG_system_flow.drawio](Graph_RAG_system_flow.drawio)

---

### 二、系统启动流程

流程图的顶部描述了**系统初始化与知识库构建**，对应 [main.py](code/main.py) 的 `main()` → `AdvancedGraphRAGSystem`：

```
main() 
  → AdvancedGraphRAGSystem()           # 加载 GraphRAGConfig
  → initialize_system()                 # 初始化6个核心模块
  → build_knowledge_base()              # 构建/加载知识库
  → run_interactive()                   # 进入交互问答循环
```

#### 阶段 1：初始化 6 大模块（[main.py#initialize_system](code/main.py#L73-L122)）

```
1. GraphDataPreparationModule    ← Neo4j连接
2. MilvusIndexConstructionModule ← Milvus连接
3. GenerationIntegrationModule   ← LLM客户端
4. HybridRetrievalModule         ← 传统检索（依赖1,2,3）
5. GraphRAGRetrieval             ← 图RAG检索（依赖config, LLM）
6. IntelligentQueryRouter        ← 智能路由（依赖4,5,LLM）
```

#### 阶段 2：知识库构建（[main.py#build_knowledge_base](code/main.py#L124-L189)）

```
检查Milvus集合是否存在？
  ├── 已存在 → load_collection() + load_graph_data() + build_recipe_documents() + chunk_documents() + _initialize_retrievers()
  └── 不存在 → 完整构建流程：
        1. load_graph_data()          ← 从Neo4j读取 Recipe/Ingredient/CookingStep 节点
        2. build_recipe_documents()   ← 查询每个菜谱的食材(REQUIRES)和步骤(CONTAINS_STEP)，组装结构化文档
        3. chunk_documents()          ← 按章节/长度分块（chunk_size=500, overlap=50）
        4. build_vector_index()       ← BGE嵌入 → Milvus HNSW索引
        5. _initialize_retrievers()   ← 初始化传统检索器 + 图RAG检索器
```

---

### 三、核心模块详解

#### 模块 1：图数据准备（[graph_data_preparation.py](code/rag_modules/graph_data_preparation.py)）  

**Neo4j 用途：**

1. 数据源头 — 系统启动时从这里加载所有菜谱数据，组装成结构化文档
2. 图RAG检索 — 执行 Cypher 多跳遍历查询（ MATCH path=(source)-[*1..3]-(target) ），发现隐含的实体关联关系
3. 图键值索引 — 构建 (K,V) 键值对，K 是实体名称，V 是详细描述

**Neo4j 图数据 → 组装成 Markdown 文档 → 分块 → 向量化 → 存入 Milvus**

**职责**：从 Neo4j 读取图数据，转换为 LangChain Document

- `load_graph_data()` — 通过 Cypher 查询加载三种节点：
  - `Recipe`（菜谱）：包含 name, category, cuisineType, difficulty 等
  - `Ingredient`（食材）：包含 name, category, description 等
  - `CookingStep`（步骤）：包含 name, description, methods, tools 等

- `build_recipe_documents()` — 对每个 Recipe，查询其关联关系：
  - `(Recipe)-[:REQUIRES]->(Ingredient)` → 食材列表（含用量）
  - `(Recipe)-[:CONTAINS_STEP]->(CookingStep)` → 步骤列表（按 stepOrder 排序）
  - 组装成 Markdown 格式的结构化文档

- `chunk_documents()` — 双策略分块：
  - 短文档（≤500字）→ 整体作为一个 chunk
  - 长文档 → 按 `## ` 标题章节分割，或按长度强制分割

---

#### 模块 2：Milvus 向量索引（[milvus_index_construction.py](code/rag_modules/milvus_index_construction.py)）

**Milvus — 向量数据库（存储语义嵌入）**

存储的是 文档分块的向量表示 ：
- 每个菜谱文档被切成 chunk → 用 BGE-small-zh-v1.5 编码成 512维浮点向量 → 存入 Milvus
- 索引类型： HNSW （近似最近邻），距离度量： COSINE  `简历项目\graph_cook_rag\HNSW.md`
- 同时存储元数据字段（recipe_name, category, cuisine_type, difficulty 等）
用途 ：

1. 语义相似度搜索 — 用户查询 "番茄炒蛋怎么做"，先编码成向量，在 Milvus 中找语义最接近的菜谱文档
2. 作为传统混合检索的一路 — 与 BM25 关键词检索、图KV索引检索 三路融合（RRF）


**职责**：向量嵌入 + Milvus 集合管理 + 相似度搜索

- Schema 设计：12个字段（id, vector, text, node_id, recipe_name, node_type, category, cuisine_type, difficulty, doc_type, chunk_id, parent_id）
- 索引类型：**HNSW**（M=16, efConstruction=200），COSINE 距离
- `build_vector_index()` — 批量嵌入（batch_size=100）→ 插入 → 建索引 → 加载到内存
- `similarity_search()` — 支持 filter 表达式，返回 top-k 结果

---

#### 模块 3：图键值索引（[graph_indexing.py](code/rag_modules/graph_indexing.py)）

**职责**：构建**实体/关系的键值对索引**（LightRAG 风格的 (K,V) 结构），用于传统三路检索中的**图键值索引检索（实体级 + 主题级双层）**

- **实体 KV**：`K = 实体名称`（唯一索引键），`V = 详细描述段落`
  - 三种实体类型：Recipe, Ingredient, CookingStep
  
- **关系 KV**：`K = 多个索引键`（关系类型 + 主题关键词），`V = 关系描述`
  - 例如 `REQUIRES` 关系的索引键：`["REQUIRES", "食材搭配", "烹饪原料", "宫保鸡丁_食材", "花生"]`
  - 支持 LLM 增强生成主题关键词（可选）

- `key_to_entities` / `key_to_relations` — 双向映射，支持 O(1) 索引查找
- `deduplicate_entities_and_relations()` — 基于名称/签名去重

---

#### 模块 4：混合检索（[hybrid_retrieval.py](code/rag_modules/hybrid_retrieval.py)）

**职责**：传统三路检索 + RRF 融合

核心方法 `hybrid_search()`（在截断部分，但从 `dual_level_retrieval` 和整体逻辑可推断）执行三路召回：

```
                    ┌─── Milvus 向量检索（语义相似度）
用户查询 → BM25检索 ┤─── BM25 关键词检索（jieba分词 + 停用词过滤）
                    └─── 图键值索引检索（实体级 + 主题级双层）
                              │
                              ▼
                    RRF 融合 (k=60) → top_k 结果
```

**双层检索范式**：
```
用户查询："有哪些用番茄做的家常菜？"
    │
    ▼ LLM 提取
entity_keywords: ["番茄"]           ← 具体的食材/菜品名
topic_keywords:  ["家常菜"]          ← 抽象的概念/分类/风格
```
**关键词提取**：调用 LLM 将查询拆分为 `entity_keywords`（实体级）和 `topic_keywords`（主题级）

- `entity_level_retrieval()` — 具体实体匹配（如"番茄"→匹配 Ingredient 节点 + 一跳邻居）
- `topic_level_retrieval()` — 抽象主题匹配（如"家常菜"→匹配 Category 关系）



---

#### 模块 5：图 RAG 检索（[graph_rag_retrieval.py](code/rag_modules/graph_rag_retrieval.py)）

**职责**：基于图结构的深度推理检索 — 这是系统的**核心差异化能力**

**关键区别**：图RAG检索**不依赖预构建的向量索引**，而是直接在 Neo4j 上执行 Cypher 遍历查询，利用图拓扑结构发现隐含关系。只在内存中缓存实体/关系的元信息用于加速查找。
```
graph_rag_search(query)
  │
  ├── 1. understand_graph_query()     ← LLM分析查询意图 → GraphQuery
  │     识别5种查询类型：
  │     - ENTITY_RELATION（实体关系）
  │     - MULTI_HOP（多跳遍历）
  │     - SUBGRAPH（子图提取）
  │     - PATH_FINDING（路径查找）
  │     - CLUSTERING（聚类查询）
  │
  ├── 2. 根据类型执行不同策略：
  │     ├── MULTI_HOP / PATH_FINDING → multi_hop_traversal()
  │     │     Cypher: MATCH path=(source)-[*1..depth]-(target)
  │     │     路径评分 = 1/路径长度 + 节点度数均值 + 关系类型匹配加分
  │     │
  │     ├── SUBGRAPH / CLUSTERING → extract_knowledge_subgraph()
  │     │     提取指定深度的邻居子图 + 计算图密度指标
  │     │     → graph_structure_reasoning() 推理链生成
  │     │
  │     └── ENTITY_RELATION → _find_entity_relations()
  │
  └── 3. _rank_by_graph_relevance() ← 图结构相关性排序
```



---

#### 模块 6：智能查询路由（[intelligent_query_router.py](code_modules/intelligent_query_router.py)）

**职责**：根据查询特征自动选择最优检索策略

**Combined 策略**：Round-robin 交替合并两路结果（`traditional_k = top_k // 2`, `graph_k = top_k - traditional_k`）

```
route_query(query, top_k)
  │
  ├── analyze_query() ← LLM 4维度分析
  │     ├── query_complexity（0-1）
  │     ├── relationship_intensity（0-1）
  │     ├── reasoning_required（bool）
  │     └── entity_count（int）
  │
  ├── 路由决策规则：
  │     ├── 复杂度<0.4 且 关系密集度<0.4 → hybrid_traditional
  │     ├── 复杂度>0.4 或 关系密集度>0.4 → graph_rag
  │     └── 其他 → combined
  │
  └── 执行检索 + 降级保护（graph_rag失败 → 降级到传统检索）
```


---

#### 模块 7：生成集成（[generation_integration.py](code/rag_modules/generation_integration.py)）

**职责**：基于检索结果生成最终回答

- `generate_adaptive_answer()` — 标准生成
- `generate_adaptive_answer_stream()` — 流式生成（带3次重试机制）
- 提示词策略：LightRAG 风格统一提示词，自动适应不同查询类型

---

### 四、完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        系统启动阶段                              │
│                                                                 │
│  Neo4j ─load_graph_data()─→ Recipe/Ingredient/CookingStep 节点  │
│       ─build_recipe_documents()─→ 结构化 Markdown 文档          │
│       ─chunk_documents()─→ 分块文档 (500字/块)                  │
│                                                                 │
│  分块文档 ─BGE嵌入─→ 512维向量 ─Milvus HNSW─→ 向量索引         │
│  Neo4j关系 ─GraphIndexing─→ 实体KV + 关系KV 键值索引           │
│  分块文档 ─jieba分词─→ BM25倒排索引                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        查询处理阶段                              │
│                                                                 │
│  用户查询                                                       │
│    │                                                            │
│    ▼                                                            │
│  IntelligentQueryRouter.analyze_query()                         │
│    │  LLM 4维度分析：复杂度 / 关系密集度 / 推理需求 / 实体数    │
│    ▼                                                            │
│  ┌──────────────┬───────────────┬──────────────┐                │
│  │ hybrid_      │ graph_rag     │ combined     │                │
│  │ traditional  │               │              │                │
│  └──────┬───────┴───────┬───────┴──────┬───────┘                │
│         │               │              │                         │
│         ▼               ▼              ▼                         │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Milvus向量 │  │ Neo4j Cypher│  │ 两者合并    │              │
│  │ BM25关键词 │  │ 多跳遍历    │  │ Round-robin │              │
│  │ 图KV索引  │  │ 子图提取    │  │ 去重排序    │              │
│  │ RRF融合   │  │ 图结构推理  │  │             │              │
│  └─────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│        │               │               │                        │
│        └───────────────┼───────────────┘                        │
│                        ▼                                        │
│              GenerationIntegrationModule                         │
│                LLM 统一提示词生成                                │
│                        │                                        │
│                        ▼                                        │
│                   流式/标准输出                                  │
│              更新路由统计 → 自适应学习                           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 五、降级策略

```
图RAG检索失败 → 降级到传统混合检索（hybrid_traditional）
传统混合检索失败 → 系统异常（无更低级降级）
LLM分析失败 → _rule_based_analysis()（关键词匹配降级）
流式生成失败 → 3次重试 → 降级到标准非流式生成
```

---

### 六、代码修复记录（graph_rag_retrieval.py）

原项目 `all-in-rag-main` 的图 RAG 检索存在一个 bug：对于 "番茄红酱怎么做" 这类查询，图 RAG 检索始终返回 0 条结果，导致无法生成回答。

#### 问题原因

[graph_rag_retrieval.py](code/rag_modules/graph_rag_retrieval.py) 的 `multi_hop_traversal()` 中，LLM 将用户查询翻译为图查询时，会输出抽象概念作为 target_entities（如 "做法"、"菜谱"、"步骤"），而 Cypher 查询用这些词做 `target.name CONTAINS` 过滤。但图中的节点名称都是具体实体（如 "芹菜"、"热锅凉油"），没有任何节点的 name 包含这些抽象词，导致所有路径被过滤掉。

```
原流程：
  LLM 输出: source=["番茄红酱"], target=["做法"]
  Cypher: MATCH path=(source)-[*1..2]-(target)
          WHERE target.name CONTAINS '做法'   ← 没有节点匹配，0 条路径
```

#### 修复内容

**修复 1：`understand_graph_query()` 提示词约束**（[graph_rag_retrieval.py#L147](code/rag_modules/graph_rag_retrieval.py#L147)）

在提示词中增加约束，明确要求 target_entities 必须是图中实际存在的具体实体名称，禁止填抽象概念：

```
修复前：target_entities: 目标实体列表（不确定则为[]）
修复后：target_entities 必须是图中实际存在的具体实体名称（如["鸡蛋","川菜"]），
       绝对不能填抽象概念（如"做法"、"菜谱"、"步骤"、"食材"等）。
       如果查询没有明确指向另一个具体实体，target_entities 必须设为空列表 []。
```

同时将示例默认值从 `multi_hop` 改为 `subgraph`，引导 LLM 在不确定时优先使用子图查询（不需要 target）。

**修复 2：`multi_hop_traversal()` 增加退化机制**（[graph_rag_retrieval.py#L232](code/rag_modules/graph_rag_retrieval.py#L232)）

当有 target 关键词时，先尝试带 target 过滤的查询；如果返回 0 条结果，自动退化为不带 target 的纯子图遍历：

```
修复前：
  有 target → 拼 AND 过滤 → 查询 → 返回结果（可能为 0，直接结束）

修复后：
  有 target → 带 AND 过滤查询
    ├── 有结果 → 返回
    └── 0 条结果 → 自动退化，去掉 target 重新查询
  无 target → 直接执行纯子图遍历
```

#### 修复效果

```
修复前：番茄红酱怎么做 → multi_hop(target="做法") → AND 过滤 → 0 条路径 → 无回答
修复后：番茄红酱怎么做 → subgraph(target=[])     → 子图提取 → 成功 → 正常回答
```

提示词修复让 LLM 在大多数情况下不再输出抽象 target；即使偶尔输出，Cypher 容错机制也能保证不返回空结果。

---

### 七、配置一览（[config.py](code/config.py)）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `neo4j_uri` | bolt://localhost:7687 | Neo4j连接 |
| `milvus_host/port` | localhost:19530 | Milvus连接 |
| `embedding_model` | BAAI/bge-small-zh-v1.5 | 512维嵌入模型 |
| `llm_model` | glm-4-flash | 智谱GLM模型 |
| `top_k` | 5 | 检索返回数量 |
| `chunk_size` | 500 | 文档分块大小 |
| `chunk_overlap` | 50 | 分块重叠 |
| `max_graph_depth` | 2 | 图遍历最大深度 |

---


### 九、系统启动指南

#### 前置条件

| 服务 | 版本要求 | 说明 |
|------|---------|------|
| Python | >= 3.10 | 推荐 3.12 |
| Docker & Docker Compose | — | 用于运行 Neo4j 和 Milvus |
| 智谱 API Key | — | 用于 LLM 调用（glm-4-flash） |

#### 第 1 步：启动 Milvus 向量数据库

```bash
# 进入 data 目录（已包含 docker-compose.yml）
cd graph_cook_rag/data

# 启动 Milvus（包含 etcd + minio + standalone 三个容器）
docker-compose up -d

# 检查服务状态
docker-compose ps
```

> 如果前面已经启动过了可以跳过此步，通过 `docker-compose ps` 确认 Milvus 服务正在运行即可。

#### 第 2 步：启动 Neo4j 图数据库并导入数据

```bash
# 启动 Neo4j 容器
docker run -d --name neo4j-db \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/all-in-rag \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v "graph_cook_rag/data/cypher:/import" \
  neo4j:5

# 等待 Neo4j 启动完成（约 10-20 秒）
# 然后导入菜谱知识图谱数据
docker exec -it neo4j-db cypher-shell -u neo4j -p all-in-rag -f /import/neo4j_import.cypher
```

启动成功后，可以通过以下方式访问：
- **Web界面**：http://localhost:7474
- **用户名**：neo4j
- **密码**：all-in-rag

> 当前网址为本地访问，如果你是部署在远程服务器上，需要将 `localhost` 修改为你的服务器IP地址。

**导入的数据包括**：
- **菜谱节点**：包含菜名、难度、烹饪时间、菜系等信息
- **食材节点**：包含食材名称、分类、营养信息等
- **烹饪步骤节点**：包含步骤描述、烹饪方法、所需工具等
- **关系网络**：菜谱与食材（REQUIRES）、步骤（CONTAINS_STEP）、分类（BELONGS_TO_CATEGORY）等关系

#### 第 3 步：创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
conda create -n graph-rag python=3.12.7
conda activate graph-rag
# 或使用 venv
# python -m venv venv && venv\Scripts\activate

# 进入代码目录并安装依赖
cd graph_cook_rag/code
pip install -r requirements.txt
```

> 首次运行时，`BAAI/bge-small-zh-v1.5` 嵌入模型会自动下载到本地缓存。后续运行设置了 `HF_HUB_OFFLINE=1`，使用离线模式。

#### 第 4 步：配置环境变量

编辑 `code/.env` 文件（参考 `.env.example`）：

```env
# 智谱 API Key（必填，用于 LLM 调用）
MOONSHOT_API_KEY=your_zhipu_api_key_here

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=all-in-rag
NEO4J_DATABASE=neo4j

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 模型配置
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LLM_MODEL=glm-4-flash
```

#### 第 5 步：启动系统

```bash
cd graph_cook_rag/code
python main.py
```

启动后系统会依次执行：

```
启动高级图RAG系统...
初始化数据准备模块...          ← 连接 Neo4j
初始化Milvus向量索引...        ← 连接 Milvus
初始化生成模块...              ← 初始化 LLM 客户端
初始化传统混合检索...
初始化图RAG检索引擎...
初始化智能查询路由器...
✅ 高级图RAG系统初始化完成！

检查知识库状态...
  ├── 首次运行：构建新知识库（从 Neo4j 加载 → 分块 → 向量化 → 入库 Milvus）
  └── 再次运行：加载已有知识库（直接从 Milvus 加载向量索引）

✅ 知识库构建完成！
欢迎使用尝尝咸淡RAG烹饪助手！
```

#### 第 6 步：交互问答

进入交互模式后直接输入问题：

```
您的问题: 川菜有哪些特色菜？
您的问题: 如何制作宫保鸡丁？
您的问题: 减肥期间适合吃什么菜？
```

**内置命令**：
- `stats` — 查看系统统计（路由分布、知识库规模）
- `rebuild` — 重建知识库（清除 Milvus 集合并重新构建）
- `quit` — 退出系统

#### 启动流程总结

```
docker-compose up -d（Milvus: 19530）
docker run neo4j（Neo4j: 7687）+ 导入 cypher 数据
        │
        ▼
配置 code/.env（API Key / 数据库连接）
        │
        ▼
pip install -r requirements.txt
        │
        ▼
python main.py
  ├── 首次：Neo4j图数据 → 文档分块 → BGE向量化 → Milvus索引
  └── 再次：直接加载 Milvus 已有索引
        │
        ▼
交互问答（智能路由 → 三路检索/图RAG → LLM生成 → 流式输出）
```

#### 项目完整目录结构

```
graph_cook_rag/
├── README.md                        # 本文档
├── Graph_RAG_system_flow.drawio     # 系统架构流程图
├── HNSW.md                          # HNSW 算法说明
├── data/                            # 数据与 Docker 配置
│   ├── docker-compose.yml           # Milvus docker-compose（etcd + minio + standalone）
│   └── cypher/                      # Neo4j 数据导入脚本
│       ├── neo4j_import.cypher      # 建表 + 导入 Cypher 脚本
│       ├── nodes.csv                # 节点数据（Recipe/Ingredient/CookingStep）
│       └── relationships.csv        # 关系数据（REQUIRES/CONTAINS_STEP 等）
└── code/                            # 项目代码（无需改动）
    ├── main.py                      # 入口：AdvancedGraphRAGSystem
    ├── config.py                    # GraphRAGConfig 配置类
    ├── .env                         # 环境变量（API Key 等）
    ├── .env.example                 # 环境变量模板
    ├── requirements.txt             # Python 依赖
    ├── rag_modules/
    │   ├── graph_data_preparation.py    # Neo4j 数据读取 → 结构化文档 → 分块
    │   ├── milvus_index_construction.py # 向量嵌入 → Milvus 索引 → 相似度搜索
    │   ├── graph_indexing.py            # 实体/关系 KV 键值索引（LightRAG 风格）
    │   ├── hybrid_retrieval.py          # 传统三路检索 + RRF 融合
    │   ├── graph_rag_retrieval.py       # 图RAG 检索（多跳遍历/子图/推理）
    │   ├── intelligent_query_router.py  # 智能路由（LLM 分析 → 策略选择）
    │   └── generation_integration.py    # LLM 答案生成（流式/标准）
    └── agent/                           # AI Agent 扩展（独立模块）
```
        
