# 本地运行：两种模式

项目在本地有**两种跑法**，共享同一套基础设施（Neo4j + Milvus + Redis），互不冲突。

## 核心思想：一套设施，应用两种跑法

不维护两套数据库（会端口/容器名冲突）。而是：
- **一套基础设施**：`docker compose` 起的 neo4j + milvus + redis（数据在 compose 卷里，共享）。
- **应用两种模式**：① 容器里跑 ② 本地 conda 跑。
- 同一时刻只跑**一种应用模式**（都用 8000 端口），切换时停一个、起另一个；**基础设施不动**。

| | 模式 A：容器（演示/部署） | 模式 B：本地（开发/评测） |
|---|---|---|
| 应用 | `graph-rag-app` 容器 | 本地 conda 跑 `uvicorn` |
| Neo4j | compose 的 `graph-rag-neo4j` (localhost:7687) | 同左（共享） |
| Milvus | compose 的 `milvus-standalone` (localhost:19530) | 同左（共享） |
| Redis | compose 的 `graph-rag-redis`（容器内部） | 宿主 **Memurai** (localhost:6379) |
| 用途 | 给人演示、截图、验证一键部署 | 改代码、跑 `eval/` 评测、快速迭代 |
| 改代码要重 build？ | **要**（改 code/ 后 `docker compose build app`） | **不要**（直接重启 uvicorn） |

---

## 模式 A：容器（默认，演示用）

```bash
docker compose up -d            # 一键起全栈（含 app）
# 浏览器 http://localhost:8000
docker compose down             # 全停（数据卷保留）
```

改了 `code/` 后：`docker compose up -d --build app`。

## 模式 B：本地开发（评测/调试用）

```bash
# 1. 起基础设施（不含 app 容器）+ 确保 Memurai 在跑(6379)
docker compose up -d neo4j etcd minio standalone
# （graph-rag-redis 不用起；本地 uvicorn 用宿主 Memurai）

# 2. 本地 conda 跑应用
conda activate graph-rag
cd code
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
# 浏览器 http://localhost:8000
```

跑评测（也在模式 B 下，连同一套设施）：
```bash
cd code
python -m eval.run_retrieval_eval_v2     # 检索评测
python -m eval.compare_longitudinal      # 纵向对比
python -m eval.load_test --users 10 --total 60 --scenario repeat   # 压测
```

---

## 切换（A ↔ B）

两种模式的 app 都占 8000，切换时停掉另一个 app 即可，**设施不用动**：

```bash
# A → B：停容器 app，起本地 uvicorn
docker compose stop app
cd code && python -m uvicorn api.server:app --port 8000

# B → A：停本地 uvicorn (Ctrl+C)，起容器 app
docker compose start app
```

## 注意

- 两模式共享同一份 Neo4j/Milvus 数据（compose 卷），所以 dev 下改的数据容器也能看到，反之亦然。
- 模式 B 的 Redis 是 Memurai（6379），模式 A 的 Redis 是 compose 容器（内部）——两者数据不共享，但缓存丢了无影响（重新算即可）。
- 若要彻底独立的两套数据（不共享），需要给 dev 单独一套端口+卷，不推荐——维护成本高且易冲突。
