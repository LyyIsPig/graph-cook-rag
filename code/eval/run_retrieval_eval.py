"""
检索消融评测（P2）。
对每种检索策略在 golden 测试集上独立打分：Recall@1/3/5、MRR、NDCG@5、平均延迟。
策略间彼此隔离，可量化每一路（向量/BM25/图KV/RRF融合/图RAG/智能路由）的贡献。

用法（在 code/ 下，需 Neo4j + Milvus 已启动）：
    python -m eval.run_retrieval_eval --limit 10            # 快速验证（前10题）
    python -m eval.run_retrieval_eval                       # 全量
    python -m eval.run_retrieval_eval --strategies vector bm25 hybrid
输出：eval/results/retrieval_report.md + retrieval_metrics.csv，原始结果缓存到 cache.json
"""

import os
import sys
import time
import json
import argparse

# ---- bootstrap ----
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem
from eval.testset import load_testset
from eval.metrics import ranked_names_from_docs, recall_at_k, mrr, ndcg_at_k, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 3, 5]


def build_strategies(system):
    """返回 {策略名: callable(query, top_k) -> List[Document]}。"""
    tr = system.traditional_retrieval
    gr = system.graph_rag_retrieval
    return {
        "vector": lambda q, k: tr.vector_search_enhanced(q, k),
        "bm25": lambda q, k: tr.bm25_search(q, k),
        "dual_level": lambda q, k: tr.dual_level_retrieval(q, k),   # 图KV双层（含LLM关键词提取）
        "hybrid": lambda q, k: tr.hybrid_search(q, k),               # 向量+BM25+图KV，RRF融合
        "graph_rag": lambda q, k: gr.graph_rag_search(q, k),
        "routed": lambda q, k: system.query_router.route_query(q, k)[0],  # 线上智能路由
    }


def run_strategy(fn, query, k, retries=1):
    """带一次重试的执行；返回 (docs, latency_ms, error)。"""
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            docs = fn(query, k)
            return docs, (time.time() - t0) * 1000.0, None
        except Exception as e:
            if attempt == retries:
                return [], (time.time() - t0) * 1000.0, str(e)
    return [], 0.0, "unreachable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 题（0=全量）")
    ap.add_argument("--strategies", nargs="*", default=None, help="指定策略子集")
    ap.add_argument("--k", type=int, default=5, help="检索 top_k（用于调用各策略）")
    ap.add_argument("--testset", default=os.path.join(HERE, "testset.jsonl"))
    args = ap.parse_args()

    print("初始化系统（连接 Neo4j/Milvus + 加载知识库）...")
    sys = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    sys.initialize_system()
    sys.build_knowledge_base()

    items = load_testset(args.testset)
    if args.limit:
        items = items[: args.limit]
    print(f"测试集: {len(items)} 条")

    strategies = build_strategies(sys)
    if args.strategies:
        strategies = {k: v for k, v in strategies.items() if k in set(args.strategies)}
    print(f"策略: {list(strategies.keys())}")

    # 菜谱名 universe：知识库菜谱 + 测试集 relevant（兜底）
    universe = {r.name for r in sys.data_module.recipes if r.name}
    for it in items:
        universe.update(it.relevant_recipe_names)

    # per-strategy 累加器
    acc = {name: {"recall": {k: [] for k in KS}, "mrr": [], "ndcg5": [], "ms": [], "fail": 0} for name in strategies}
    detail_rows = []  # 缓存原始结果

    for qi, it in enumerate(items, 1):
        rel = set(it.relevant_recipe_names)
        print(f"\n[{qi}/{len(items)}] {it.id} ({it.query_type}) {it.query}  relevant={len(rel)}")
        for name, fn in strategies.items():
            docs, ms, err = run_strategy(fn, it.query, args.k)
            if err:
                acc[name]["fail"] += 1
                print(f"   {name:12} FAIL ({ms:.0f}ms) {err[:80]}")
                detail_rows.append({"id": it.id, "strategy": name, "ranked": [], "ms": ms, "error": err})
                continue
            ranked = ranked_names_from_docs(docs, universe)
            for k in KS:
                acc[name]["recall"][k].append(recall_at_k(ranked, rel, k))
            acc[name]["mrr"].append(mrr(ranked, rel))
            acc[name]["ndcg5"].append(ndcg_at_k(ranked, rel, 5))
            acc[name]["ms"].append(ms)
            detail_rows.append({"id": it.id, "strategy": name,
                                "ranked": ranked[:10], "ms": round(ms, 1), "error": None})

    # 汇总
    print("\n" + "=" * 90)
    header = f"{'策略':<12}{'n':>4}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>9}{'NDCG@5':>9}{'avg_ms':>9}{'fail':>5}"
    print(header)
    print("-" * 90)
    summary = []
    for name in strategies:
        a = acc[name]
        n = len(a["mrr"])
        row = {
            "strategy": name, "n": n,
            "recall@1": aggregate(a["recall"][1])["mean"],
            "recall@3": aggregate(a["recall"][3])["mean"],
            "recall@5": aggregate(a["recall"][5])["mean"],
            "mrr": aggregate(a["mrr"])["mean"],
            "ndcg@5": aggregate(a["ndcg5"])["mean"],
            "avg_ms": aggregate(a["ms"])["mean"] if a["ms"] else 0.0,
            "fail": a["fail"],
        }
        summary.append(row)
        print(f"{name:<12}{n:>4}{row['recall@1']:>10.3f}{row['recall@3']:>10.3f}{row['recall@5']:>10.3f}"
              f"{row['mrr']:>9.3f}{row['ndcg@5']:>9.3f}{row['avg_ms']:>9.0f}{row['fail']:>5}")

    # 写报告
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrieval_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("strategy,n,recall@1,recall@3,recall@5,mrr,ndcg@5,avg_ms,fail\n")
        for r in summary:
            f.write(f"{r['strategy']},{r['n']},{r['recall@1']:.4f},{r['recall@3']:.4f},"
                    f"{r['recall@5']:.4f},{r['mrr']:.4f},{r['ndcg@5']:.4f},{r['avg_ms']:.1f},{r['fail']}\n")
    with open(os.path.join(out_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(detail_rows, f, ensure_ascii=False, indent=1)
    print(f"\n✅ 报告写入 {out_dir}/retrieval_metrics.csv 与 cache.json")

    sys._cleanup()


if __name__ == "__main__":
    main()
