# 知识图谱构建流水线（data_pipeline）

> 本目录是**数据准备**部分：用 LLM（Kimi / GLM）把原始菜谱 Markdown 解析成结构化数据，导入 Neo4j 成为知识图谱。
> 与 `code/` 下的 RAG 问答系统相互独立——RAG 系统跑在已经建好的图谱上，本流水线只在"造数据"时用一次。

## 它做了什么

读入 `dishes/` 下的菜谱 Markdown（按 `素菜/荤菜/水产/主食/汤类/甜品/...` 分类），调用 LLM 抽取：
- 菜谱实体（名称、难度、分类）
- 食材实体与 `REQUIRES` 关系
- 烹饪步骤与 `CONTAINS_STEP` 关系（含工具/做法）
- `BELONGS_TO_CATEGORY` 关系

输出 Neo4j 导入文件 / CSV，最终落到仓库 `data/cypher/`（`nodes.csv`、`relationships.csv`、`neo4j_import.cypher`）—— **这些导入文件已经提交在仓库里，所以普通使用者不需要重跑本流水线**。

## 主要文件

| 文件 | 作用 |
|---|---|
| `recipe_ai_agent.py` | 核心：LLM 菜谱解析 + 知识图谱构建器 |
| `batch_manager.py` | 批量处理、并发控制、断点续跑 |
| `amount_normalizer.py` | 食材用量归一化（"少许/适量"→标准） |
| `run_ai_agent.py` | 命令行入口 |
| `config.json` | 模型/批处理/输出/分类配置（**密钥请用环境变量，勿填真实 key**） |
| `recipe_ontology_design.md` | 本体设计（实体/关系 schema） |

## 运行（仅当需要重建知识库时）

```bash
pip install -r data_pipeline/requirements.txt
# 在 config.json 配好模型端点（或改成读环境变量）
python data_pipeline/run_ai_agent.py
```

> 详见 `AI_AGENT_README.md`。
