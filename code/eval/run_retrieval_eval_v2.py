"""
检索评测 v2（P2-2）：在能力维度测试集上、按 capability 切片打分。
- 非 negative 查询：算 Recall@1/3/5、MRR、NDCG@5、延迟，按 lookup/list/relation/reasoning 切片；
- negative 查询：relevant 为空，召回无意义 → 改报「误检索率」（检索出 ≥1 条的比例，越低越好）。

用法（在 code/ 下，需 Neo4j+Milvus）：
    python -m eval.run_retrieval_eval_v2 --per-cap 6           # 快速平衡子集（每能力取6条）
    python -m eval.run_retrieval_eval_v2                       # 全量 112
    python -m eval.run_retrieval_eval_v2 --strategies vector bm25 hybrid graph_rag routed
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict

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
from eval.testset_v2 import load_testset_v2
from eval.metrics import ranked_names_from_docs, recall_at_k, mrr, ndcg_at_k, aggregate

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 3, 5]
CAPS = ["lookup", "list", "relation", "reasoning"]   # 负样本单独处理


def build_strategies(system):
    tr = system.traditional_retrieval
    gr = system.graph_rag_retrieval
    return {
        "vector": lambda q, k: tr.vector_search_enhanced(q, k),
        "bm25": lambda q, k: tr.bm25_search(q, k),
        "dual_level": lambda q, k: tr.dual_level_retrieval(q, k),
        "hybrid": lambda q, k: tr.hybrid_search(q, k),
        "graph_rag": lambda q, k: gr.graph_rag_search(q, k),
        "routed": lambda q, k: system.query_router.route_query(q, k)[0],
    }


def run_once(fn, query, k):
    t0 = time.time()
    try:
        docs = fn(query, k)
        return docs, (time.time() - t0) * 1000.0, None
    except Exception as e:
        return [], (time.time() - t0) * 1000.0, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cap", type=int, default=0, help="每个能力取前 N 条（0=全量）")
    ap.add_argument("--neg-cap", type=int, default=0, help="负样本取前 N 条（0=全量）")
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--testset", default=os.path.join(HERE, "testset.v2.jsonl"))
    args = ap.parse_args()

    print("初始化系统 ...")
    system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    system.initialize_system()
    system.build_knowledge_base()
    # 评测要测冷启动的真实检索/路由成本，关掉 P1 L4 路由缓存（warm 才命中，会失真）
    system.query_router.cache = None

    items = load_testset_v2(args.testset)
    # 按能力分组 + 截断
    by_cap = defaultdict(list)
    for it in items:
        by_cap[it.capability].append(it)
    if args.per_cap:
        for c in CAPS:
            by_cap[c] = by_cap[c][: args.per_cap]
    neg = by_cap["negative"][: args.neg_cap] if args.neg_cap else by_cap["negative"]
    scored = [it for c in CAPS for it in by_cap[c]]
    print(f"评测：非负样本 {len(scored)} 条（lookup {len(by_cap['lookup'])}/list {len(by_cap['list'])}/"
          f"relation {len(by_cap['relation'])}/reasoning {len(by_cap['reasoning'])}），负样本 {len(neg)} 条")

    strategies = build_strategies(system)
    if args.strategies:
        strategies = {k: v for k, v in strategies.items() if k in set(args.strategies)}
    print(f"策略: {list(strategies.keys())}")

    universe = {r.name for r in system.data_module.recipes if r.name}
    for it in items:
        universe.update(it.relevant_recipe_names)

    # 累加器：strategy -> capability -> metric list
    acc = {s: {c: {"recall": {k: [] for k in KS}, "mrr": [], "ndcg5": [], "ms": []} for c in CAPS} for s in strategies}
    # 负样本：strategy -> 每条检索到的文档数
    neg_docs = {s: [] for s in strategies}
    detail = []

    # 1) 非负样本：召回/排序指标
    for qi, it in enumerate(scored, 1):
        rel = set(it.relevant_recipe_names)
        print(f"[{qi}/{len(scored)}] {it.id} ({it.capability}/{it.query_type}) {it.query}  rel={len(rel)}")
        for name, fn in strategies.items():
            docs, ms, err = run_once(fn, it.query, args.k)
            if err:
                detail.append({"id": it.id, "strategy": name, "error": err})
                continue
            ranked = ranked_names_from_docs(docs, universe)
            for k in KS:
                acc[name][it.capability]["recall"][k].append(recall_at_k(ranked, rel, k))
            acc[name][it.capability]["mrr"].append(mrr(ranked, rel))
            acc[name][it.capability]["ndcg5"].append(ndcg_at_k(ranked, rel, 5))
            acc[name][it.capability]["ms"].append(ms)
            detail.append({"id": it.id, "capability": it.capability, "strategy": name,
                           "ranked": ranked[:8], "ms": round(ms, 1)})

    # 2) 负样本：误检索率（检索出 ≥1 条 → 误检索）
    if neg:
        print(f"\n[负样本 {len(neg)} 条] 统计误检索率 ...")
        for ni, it in enumerate(neg, 1):
            for name, fn in strategies.items():
                docs, ms, err = run_once(fn, it.query, args.k)
                neg_docs[name].append(len(docs))

    # ===== 汇总：每策略 × 每能力 =====
    def mean(xs):
        xs = [float(x) for x in xs]
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 100)
    print(f"{'策略':<12}{'能力':<10}{'n':>4}{'Rec@1':>8}{'Rec@3':>8}{'Rec@5':>8}{'MRR':>8}{'NDCG@5':>9}{'avg_ms':>9}")
    print("-" * 100)
    rows = []
    for name in strategies:
        for c in CAPS:
            a = acc[name][c]
            n = len(a["mrr"])
            if n == 0:
                continue
            r = {"strategy": name, "capability": c, "n": n,
                 "rec@1": mean(a["recall"][1]), "rec@3": mean(a["recall"][3]),
                 "rec@5": mean(a["recall"][5]), "mrr": mean(a["mrr"]),
                 "ndcg@5": mean(a["ndcg5"]), "avg_ms": mean(a["ms"])}
            rows.append(r)
            print(f"{name:<12}{c:<10}{n:>4}{r['rec@1']:>8.3f}{r['rec@3']:>8.3f}{r['rec@5']:>8.3f}"
                  f"{r['mrr']:>8.3f}{r['ndcg@5']:>9.3f}{r['avg_ms']:>9.0f}")

    # 负样本误检索率
    if neg:
        print("-" * 100)
        print(f"{'策略':<12}{'负样本误检索率(越低越好)':<28}{'平均检索条数':>12}")
        for name in strategies:
            ds = neg_docs[name]
            if not ds:
                continue
            fp = sum(1 for d in ds if d > 0) / len(ds)
            print(f"{name:<12}{fp:<28.1%}{sum(ds)/len(ds):>12.1f}")
            rows.append({"strategy": name, "capability": "negative", "n": len(ds),
                         "false_positive_rate": round(fp, 3), "avg_docs": round(sum(ds)/len(ds), 1)})

    # 写报告
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "retrieval_v2_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("strategy,capability,n,rec@1,rec@3,rec@5,mrr,ndcg@5,avg_ms\n")
        for r in rows:
            if r.get("capability") == "negative":
                continue
            f.write(f"{r['strategy']},{r['capability']},{r['n']},{r['rec@1']:.4f},{r['rec@3']:.4f},"
                    f"{r['rec@5']:.4f},{r['mrr']:.4f},{r['ndcg@5']:.4f},{r['avg_ms']:.1f}\n")
    with open(os.path.join(out_dir, "cache_v2.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=1)
    print(f"\n✅ 结果写入 eval/results/retrieval_v2_metrics.csv + cache_v2.json")
    system._cleanup()


if __name__ == "__main__":
    main()
