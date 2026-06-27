"""
P1 缓存基准（cold / warm / semantic 三档），给简历/面试的量化数字。
- cold：invalidate 后首跑，全量检索+生成+路由LLM；
- warm：原样再跑，走 L1/L2 精确命中（应 ~0ms）；
- semantic：跑改写版 query，走 L3 语义命中（验证阈值有效）。
报：各档命中率、p50/p99 延迟、LLM 调用次数对比。

前置：Neo4j + Milvus + Redis 都在跑（Redis 见 data/docker-compose.yml）。
用法（在 code/ 下）：
    python -m eval.bench_cache
"""

import os
import sys
import time
from collections import Counter

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem

# 基准查询：跨能力 + 几组改写对（验证语义缓存）
QUERIES = [
    "宫保鸡丁怎么做",                # lookup
    "番茄炒蛋怎么做",                # lookup
    "用鸡蛋做的菜有哪些",            # list
    "和可乐鸡翅一样用了鸡翅中的菜还有哪些",  # relation (graph_rag [B])
    "用了青辣椒的素菜有哪些",        # relation
    "需要砂锅的菜有哪些",            # relation
    "宫保鸡丁和番茄炒蛋哪个难度大",  # reasoning
]
# 语义改写：与上面某条同义不同字面，验证 L3（需 BGE 余弦 ≥ 阈值）
PARAPHRASES = [
    "宫保鸡丁如何做",                # ≈ "宫保鸡丁怎么做"
    "宫保鸡丁的做法",                # ≈ 同上
    "哪些菜用鸡蛋做的",              # ≈ "用鸡蛋做的菜有哪些"
]


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def llm_call_count():
    """粗略统计本进程对 LLM endpoint 的调用次数（httpx 计数，仅作趋势参考）。"""
    # 通过 logger 无法精确计数，这里用一个简易探针：检查 cache metrics 的 route_miss
    return None


def run_batch(system, queries, label):
    lat, hits = [], Counter()
    print(f"\n--- {label}（{len(queries)} 条）---")
    for q in queries:
        t0 = time.time()
        # 1) 精确缓存（生产顺序：置于拒答闸门【之前】，命中免跑闸门）
        cached = system.cache.get_answer(q) if system.cache else None
        if cached:
            layer = cached[1]; hits[layer.split("(")[0]] += 1
            ms = (time.time() - t0) * 1000; lat.append(ms)
            print(f"  [{layer}] {ms:>7.1f}ms  {q}")
            continue
        # 2) 拒答闸门
        answerable, reason, conf = system.check_answerable(q)
        if not answerable:
            lat.append((time.time() - t0) * 1000); hits["refused"] += 1
            continue
        # 3) 语义缓存（闸门通过后）
        layer = None
        if system.cache:
            sem = system.cache.get_semantic_answer(q)
            if sem:
                layer = f"L3({sem[1]:.2f})"; hits["L3"] += 1
                ms = (time.time() - t0) * 1000; lat.append(ms)
                print(f"  [{layer}] {ms:>7.1f}ms  {q}")
                continue
        # 4) 全链路（检索 + 生成）
        docs, analysis = system.query_router.route_query(q, system.config.top_k)
        ans = system.generation_module.generate_adaptive_answer(q, docs)
        ms = (time.time() - t0) * 1000; lat.append(ms); hits["MISS"] += 1
        if system.cache:
            payload = {"answer": ans, "strategy": getattr(analysis.recommended_strategy, "value", None),
                       "latency_ms": round(ms, 1)}
            system.cache.set_answer(q, payload)
            system.cache.register_semantic(q, payload)
        print(f"  [MISS] {ms:>7.1f}ms  {q}")
    return lat, hits


def main():
    print("初始化系统（需 Neo4j+Milvus+Redis）...")
    system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    system.initialize_system(); system.build_knowledge_base()
    cache = system.cache
    if cache is None or not cache.redis_on:
        print("⚠️ Redis 未连接，语义/精确缓存无法生效；仅能测降级路径。建议先 docker-compose up redis。")

    # COLD：清缓存后首跑
    system.invalidate_cache()
    cold_lat, cold_hits = run_batch(system, QUERIES, "COLD（冷启动，无缓存）")

    # WARM：原样再跑 → 精确命中
    warm_lat, warm_hits = run_batch(system, QUERIES, "WARM（精确缓存命中）")

    # SEMANTIC：改写版 → 语义命中
    sem_lat, sem_hits = run_batch(system, PARAPHRASES, "SEMANTIC（语义缓存命中）")

    # 汇总
    print("\n" + "=" * 72)
    print(f"{'档':<10}{'命中率':>10}{'p50(ms)':>10}{'p99(ms)':>10}{'命中分布'}")
    for label, lat, h in [("COLD", cold_lat, cold_hits), ("WARM", warm_lat, warm_hits), ("SEMANTIC", sem_lat, sem_hits)]:
        total = sum(h.values()) or 1
        hit = total - h.get("MISS", 0) - h.get("refused", 0)
        print(f"{label:<10}{hit/total:>10.0%}{pct(lat,50):>10.1f}{pct(lat,99):>10.1f}    {dict(h)}")
    print("=" * 72)
    stats = system.cache_stats()
    print(f"缓存统计: enabled={stats['enabled']} redis={stats['redis_available']} "
          f"answer_hit_rate={stats['answer_hit_rate']} L1={stats['l1_size']} sem_idx={stats['semantic_index_size']}")
    print(f"埋点: {stats['metrics']}")
    if warm_lat and cold_lat:
        speedup = pct(cold_lat, 50) / max(pct(warm_lat, 50), 0.001)
        print(f"\n⚡ 精确缓存命中使 p50 延迟从 {pct(cold_lat,50):.0f}ms → {pct(warm_lat,50):.1f}ms（≈{speedup:.0f}x）")
    system._cleanup()


if __name__ == "__main__":
    main()
