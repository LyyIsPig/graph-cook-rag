# 🍳 尝尝咸淡 · Graph RAG 烹饪问答系统

> 基于知识图谱（Neo4j）+ 向量库（Milvus）的生产级 RAG 烹饪问答服务：智能路由、多路混合检索、图谱关系推理、四级 Redis 缓存、异步高并发、Prometheus 监控、可视化前端。
>
> 在 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 教程基础上做的**工程化重构与评测驱动优化**——把"能跑的二次开发"升级成"有量化指标、能扛面试追问的生产级项目"。

![tech](https://img.shields.io/badge/RAG-Graph%20%2B%20Hybrid-orange) ![stack](https://img.shields.io/badge/stack-Neo4j%20%7C%20Milvus%20%7C%20Redis%20%7C%20FastAPI-blue) ![llm](https://img.shields.io/badge/LLM-GLM--4--Flash-green) ![eval](https://img.shields.io/badge/eval-Recall%2FMRR%2FNDCG%2FRAGAS-purple)

---

## 📊 一、核心成果

用一套 112 题的能力维度 golden 测试集（Cypher 真值、可审计）驱动优化，关键指标：

### 1. graph_rag 关系查询翻盘（[B] 目标 Cypher 改造）

通用子图检索在"关系类"问题（如 *和可乐鸡翅一样用了鸡翅中的菜还有哪些？*）上召回恒为 **0**（无法施加分类/工具过滤）。改成把自然语言关系查询**编译成精确 Cypher** 后：

| | Recall@5 | MRR | NDCG@5 | 延迟 |
|---|---|---|---|---|
| hybrid（对比） | 0.449 | 0.487 | 0.420 | 2096 ms |
| **graph_rag [B]** | **0.909** | **1.000** | **0.976** | **12 ms** |

> 关系召回 **0 → 0.91**，反超 hybrid 2 倍，且纯 Cypher 12ms（快 170×）。详见 `docs/问题与解决方案记录.md` → P2-[B]。

### 2. 纵向三系统对比（C8 普通 RAG → C9 Graph 基线 → 当前）

同语料同题（32 题×4 能力）对比进化收益，出现一个**反直觉发现**：

| 系统 | lookup | list | relation | reasoning | **overall Recall@5** |
|---|---|---|---|---|---|
| C8 普通 RAG（向量） | 1.00 | 0.35 | 0.26 | 0.75 | **0.59** |
| C9 Graph 基线 | 0.00 | 0.08 | 0.00 | 0.00 | **0.02** ⚠️ |
| **当前系统** | 1.00 | 0.41 | **0.95** | **1.00** | **0.84** |

> **盲目加一个坏掉的图谱 + 错误路由，反而比什么都不加的普通 RAG 差 30 倍（0.59→0.02）**；诊断根因、修好图谱 + 数据驱动路由后，0.02 → 0.84。完整的"发现问题→定位→修复→量化"闭环。

### 3. 防幻觉拒答

| | 负样本拒答率 | 正样本误拒率 |
|---|---|---|
| 修前 | 0%（全瞎编） | 0% |
| **向量闸门 + 提示词拒答** | **70%** | **0%** |

### 4. Redis 四级缓存 + 异步并发

| 指标 | 冷启动 | 缓存命中 |
|---|---|---|
| 精确缓存（L1/L2）p50 | 9834 ms | **0.0 ms** |
| 语义缓存（L3，BGE 余弦≥0.92）p50 | 9834 ms | **385 ms** |

- **Prometheus 每步耗时定位瓶颈**：把关 0.5s / 检索 4.0s / **生成 19.6s**（生成是瓶颈，不是检索）。
- **全异步管道**：FastAPI async 端点 + `asyncio.gather` 三路检索并发 + AsyncOpenAI。
- **Redis Lua 令牌桶限流**保护下游 LLM API。

---

## 🏗️ 二、系统架构

```
                         用户提问
                            │
        ┌──────────── FastAPI (async) ────────────┐
        │  精确缓存 L1/L2  →  拒答闸门  →  语义缓存 L3 │   ← P1 Redis 四级缓存
        │              │ (miss)                      │
        │        智能路由器 ──┬─▶ 混合检索(向量+BM25+图KV, RRF k=60)
        │   (数据驱动, P2-2c) └─▶ 图谱检索 [B](关系→目标Cypher)   ← [B] 关系翻盘
        │              │                           │
        │        GLM 生成 (AsyncOpenAI, P2-3 拒答)  │
        └──────────────────┬───────────────────────┘
                           │
        ┌────── 数据/基础设施 ──────┐
        │  Neo4j (知识图谱)          │
        │  Milvus (HNSW 向量, BGE)   │
        │  Redis (缓存 / 限流)       │
        └───────────────────────────┘
        Prometheus 监控 · /metrics · 工坊前端
```

完整数据流图见 `Graph_RAG_system_flow.png`。

---

## ✨ 三、核心特性

| 模块 | 能力 |
|---|---|
| **P0 工程化** | 配置统一收口（.env 注入）、FastAPI 服务层、结构化 JSON 日志 + request_id 全链路追踪 |
| **P2 评测** | 112 题能力维度测试集（lookup/list/relation/negative/reasoning，Cypher 真值）、Recall/MRR/NDCG、LLM-as-judge、纵向对比 |
| **[B] 图谱检索** | 关系查询编译成目标 Cypher，关系召回 0→0.91；含拒答防幻觉 |
| **P1 缓存** | L1 进程 LRU / L2 Redis 精确 / L3 语义(BGE 余弦) / L4 路由决策 + embedding 缓存，优雅降级 |
| **P4 异步+限流** | async 端点 + 三路 gather 并发 + AsyncOpenAI + Redis Lua 令牌桶限流 |
| **P5 监控** | Prometheus 指标（HTTP/缓存/路由/每步耗时）+ `/metrics`，定位瓶颈 |
| **P3 前端** | "检索工坊"可视化（对话 / 检索溯源 X 光 / 知识图谱 / 监控），impeccable 设计 |

---

## 📁 四、项目结构

```
graph_cook_rag/
├── README.md · LICENSE · requirements.txt · PRODUCT.md
├── Graph_RAG_system_flow.png        # 架构图
├── code/                            # RAG 问答系统
│   ├── main.py                      # 系统编排 + async 问答
│   ├── config.py                    # 统一配置（.env 注入）
│   ├── api/server.py                # FastAPI + SSE + 监控 + 限流 + 前端托管
│   ├── rag_modules/                 # 数据准备/检索/路由/生成
│   ├── core/                        # logging / metrics / rate_limiter / stats
│   ├── cache/                       # redis_client / cache_manager（四级缓存）
│   ├── eval/                        # 评测体系（指标/测试集/对比脚本）
│   └── static/                      # 工坊前端（index.html / styles.css / app.js）
├── data_pipeline/                   # 知识图谱构建（LLM 解析菜谱→Neo4j）
├── data/                            # 种子数据 + docker-compose
│   ├── cypher/                      # Neo4j 导入文件（nodes/relationships/csv）
│   └── docker-compose.yml           # Milvus + Redis
└── docs/                            # 开发记录（路线图/问题排查/改进待办/HNSW）
```

---

## 🚀 五、快速开始

### 1. 起基础设施（Neo4j + Milvus + Redis）

```bash
cd data
docker compose up -d                 # 起 Milvus(etcd/minio/standalone) + Redis
# Neo4j 单独起（或加入 compose），导入 data/cypher/neo4j_import.cypher
```

### 2. 装依赖 + 配密钥

```bash
pip install -r requirements.txt
cp code/.env.example code/.env        # 填入 LLM_API_KEY（智谱 GLM）、DEEPSEEK_API_KEY（评测判官）
```

### 3. 跑起来

```bash
cd code
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000  →  检索工坊前端
```

> 首次启动会加载 BGE 嵌入模型并连接知识库（已存在则直接加载）。

---

## 📡 六、API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 检索工坊前端 |
| POST | `/api/ask` | 异步问答，返回答案 + 路由 + 溯源 + 缓存命中层 + 每步耗时 |
| POST | `/api/ask/stream` | SSE 流式问答 |
| GET | `/api/graph` | 给菜谱列表，返回 1 跳图谱子图（前端用） |
| GET | `/api/stats` | 监控 JSON（QPS / p99 / 缓存命中 / 路由分布） |
| POST | `/api/cache/invalidate` | 清空缓存命名空间 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/health` | 健康检查 |

---

## 📈 七、评测

```bash
cd code
python -m eval.build_testset_v2               # 生成/更新 golden 测试集
python -m eval.run_retrieval_eval_v2          # 6 路策略 × 能力切片 检索评测
python -m eval.run_generation_eval            # LLM-as-judge 生成评测
python -m eval.run_refusal_eval               # 拒答/防幻觉评测
python -m eval.compare_longitudinal           # C8/C9/当前 纵向对比
python -m eval.bench_cache                    # 缓存冷/热/语义 基准
python -m eval.load_test --users 10 --total 60 --scenario repeat   # 压测
```

测试集与各指标定义见 `code/eval/`，详细排查过程与结论见 `docs/问题与解决方案记录.md`。

---

## 🛠️ 八、技术栈

| 层 | 选型 |
|---|---|
| 图数据库 | Neo4j 5 |
| 向量库 | Milvus 2.5（HNSW + COSINE） |
| 嵌入 | BAAI/bge-small-zh-v1.5（512 维） |
| LLM | GLM-4-Flash（智谱） · 判官 DeepSeek |
| 关键词 | BM25Okapi + jieba + 中文停用词 |
| 融合 | RRF（Reciprocal Rank Fusion, k=60） |
| 服务 | FastAPI + Uvicorn（async）+ SSE |
| 缓存/限流 | Redis（四级缓存 + Lua 令牌桶） |
| 监控 | prometheus_client |
| 前端 | 原生 HTML / CSS / JS（impeccable 设计） |

---

## 🙏 致谢

- 基础教程：[datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)
- 菜谱语料：datawhalechina 菜谱库

## 📄 License

MIT（代码），菜谱语料遵循其原始协议。本项目仅用于学习与求职作品展示。
