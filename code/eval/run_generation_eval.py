"""
生成评测（P2-3）：公正判决 graph_rag 作为【生成上下文】的价值。
对每道题，分别用不同检索策略的上下文喂给 GLM 生成答案，再用 DeepSeek 判官打分：
  - 非 negative：faithfulness（忠于上下文/防幻觉）+ relevancy（切题）
  - negative：refusal（正确拒答而非编造）
这是 graph_rag 的公平赛场——它在检索召回层已认输（relation 0），但作为生成上下文是否有价值，由这里判决。

用法（在 code/ 下，需 Neo4j+Milvus + DEEPSEEK_API_KEY）：
    python -m eval.run_generation_eval --per-cap 3 --neg 4      # 小样本快速验证
    python -m eval.run_generation_eval                          # 默认规模
    python -m eval.run_generation_eval --strategies hybrid graph_rag
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
from eval.llm_judge import LLMJudge
from eval.metrics import parse_judge_score

HERE = os.path.dirname(os.path.abspath(__file__))
CAPS = ["lookup", "list", "relation", "reasoning"]


def build_context_strategies(system):
    tr = system.traditional_retrieval
    gr = system.graph_rag_retrieval
    return {
        "vector": lambda q, k: tr.vector_search_enhanced(q, k),
        "hybrid": lambda q, k: tr.hybrid_search(q, k),
        "graph_rag": lambda q, k: gr.graph_rag_search(q, k),
    }


def doc_context(docs, per_doc=300, total=1500):
    """拼上下文文本（每篇截断，总量受控），喂给判官 faithfulness 用。"""
    parts = []
    for d in docs:
        parts.append(d.page_content[:per_doc])
        if sum(len(p) for p in parts) > total:
            break
    return "\n".join(parts)[:total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cap", type=int, default=4, help="每个能力取前 N 条")
    ap.add_argument("--neg", type=int, default=6, help="负样本取前 N 条")
    ap.add_argument("--strategies", nargs="*", default=["hybrid", "graph_rag", "vector"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--testset", default=os.path.join(HERE, "testset.v2.jsonl"))
    args = ap.parse_args()

    print("初始化系统 ...")
    system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    system.initialize_system(); system.build_knowledge_base()

    print("初始化判官 ...")
    judge = LLMJudge()
    print(f"  判官 provider = {judge.provider}, model = {judge.model}")

    items = load_testset_v2(args.testset)
    by_cap = defaultdict(list)
    for it in items:
        by_cap[it.capability].append(it)
    scored = [it for c in CAPS for it in by_cap[c][: args.per_cap]]
    negs = by_cap["negative"][: args.neg]
    print(f"评测：非负样本 {len(scored)} 条，负样本 {len(negs)} 条；策略 {args.strategies}")

    strategies = {k: v for k, v in build_context_strategies(system).items() if k in set(args.strategies)}

    # acc: strategy -> {'faith':[], 'rel':[], 'refusal':[], 'ms':[]}
    acc = {s: {"faith": [], "rel": [], "refusal": [], "ms": []} for s in strategies}
    detail = []

    def run_one(query, strat_name, fn, negative):
        t0 = time.time()
        try:
            docs = fn(query, args.k)
            answer = system.generation_module.generate_adaptive_answer(query, docs)
        except Exception as e:
            return None, (time.time() - t0) * 1000, str(e)
        ms = (time.time() - t0) * 1000
        ctx = doc_context(docs)
        if negative:
            raw = judge.score_refusal(query, answer)
            score = {"refusal": parse_judge_score(raw, 0.0)}
        else:
            f = parse_judge_score(judge.score_faithfulness(query, ctx, answer), 0.0)
            r = parse_judge_score(judge.score_relevancy(query, answer), 0.0)
            score = {"faith": f, "rel": r}
        return {"answer": answer[:200], "score": score, "raw_n_docs": len(docs)}, ms, None

    # 1) 非负样本
    for qi, it in enumerate(scored, 1):
        print(f"\n[{qi}/{len(scored)}] {it.id} ({it.capability}) {it.query}")
        for name, fn in strategies.items():
            res, ms, err = run_one(it.query, name, fn, negative=False)
            if err:
                print(f"   {name:<10} FAIL {err[:60]}")
                continue
            acc[name]["faith"].append(res["score"]["faith"])
            acc[name]["rel"].append(res["score"]["rel"])
            acc[name]["ms"].append(ms)
            detail.append({"id": it.id, "strategy": name, "capability": it.capability, **res["score"], "ms": round(ms, 1)})
            print(f"   {name:<10} faith={res['score']['faith']} rel={res['score']['rel']} ({ms:.0f}ms)")

    # 2) 负样本
    if negs:
        print(f"\n[负样本 {len(negs)} 条] 测拒答 ...")
        for ni, it in enumerate(negs, 1):
            for name, fn in strategies.items():
                res, ms, err = run_one(it.query, name, fn, negative=True)
                if err:
                    continue
                acc[name]["refusal"].append(res["score"]["refusal"])
                detail.append({"id": it.id, "strategy": name, "capability": "negative",
                               "refusal": res["score"]["refusal"], "ms": round(ms, 1)})

    # 汇总
    def mean(xs):
        xs = [float(x) for x in xs]
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 80)
    print(f"{'策略':<12}{'n':>4}{'Faith(0-2)':>12}{'Relev(0-2)':>12}{'Refusal(0-2)':>14}{'avg_ms':>9}")
    print("-" * 80)
    rows = []
    for name in strategies:
        a = acc[name]
        rows.append({
            "strategy": name,
            "n_faith": len(a["faith"]), "faith": mean(a["faith"]),
            "rel": mean(a["rel"]),
            "refusal": mean(a["refusal"]),
            "avg_ms": mean(a["ms"]),
        })
        print(f"{name:<12}{len(a['faith']):>4}{mean(a['faith']):>12.2f}{mean(a['rel']):>12.2f}"
              f"{mean(a['refusal']):>14.2f}{mean(a['ms']):>9.0f}")
    print("\n注：Faith/Relev 越高越好（满分2）；Refusal 越高越好（正确拒答，负样本场景）。")

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generation_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("strategy,n,faithfulness,relevancy,refusal,avg_ms\n")
        for r in rows:
            f.write(f"{r['strategy']},{r['n_faith']},{r['faith']:.3f},{r['rel']:.3f},{r['refusal']:.3f},{r['avg_ms']:.1f}\n")
    with open(os.path.join(out_dir, "cache_generation.json"), "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=1)
    print(f"\n✅ 结果写入 eval/results/generation_metrics.csv + cache_generation.json")
    system._cleanup()


if __name__ == "__main__":
    main()
