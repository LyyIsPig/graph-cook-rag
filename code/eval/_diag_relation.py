"""
[B] 关系查询目标 Cypher 诊断（纯 Cypher，不依赖 LLM）。
对 testset.v2 的 relation 子集逐条：检测子型 → 编译 Cypher → 执行 → 对比真值，
输出每种子型的 检出率/召回率/命中规模，确认 graph_rag 在 relation 上翻盘。

用法（在 code/ 下，只需 Neo4j）：
    python -m eval._diag_relation
"""

import os
import sys
from collections import defaultdict

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
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval, detect_relation_pattern
from eval.testset_v2 import load_testset_v2

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    gr = GraphRAGRetrieval(DEFAULT_CONFIG, llm_client=None)
    gr.initialize()

    items = [it for it in load_testset_v2(os.path.join(HERE, "testset.v2.jsonl"))
             if it.capability == "relation"]
    print(f"relation 子集 {len(items)} 条\n")

    # 按子型聚合
    by_subtype = defaultdict(lambda: {"n": 0, "detected": 0, "recall_sum": 0.0,
                                      "prec_sum": 0.0, "hit0": 0})
    detail = []
    for it in items:
        st = it.query_type  # shared_ingredient / ingredient_category / by_tool / by_method
        agg = by_subtype[st]
        agg["n"] += 1
        detected = detect_relation_pattern(it.query)
        gold = set(it.relevant_recipe_names)

        if not detected:
            detail.append((it.query, st, "未检出", [], gold, 0.0))
            continue
        agg["detected"] += 1
        docs = gr.relational_search(it.query, 5)
        got = set(d.metadata.get("recipe_name") for d in docs if d.metadata.get("recipe_name"))
        if not got:
            agg["hit0"] += 1
        tp = len(gold & got)
        recall = tp / len(gold) if gold else 0.0
        prec = tp / len(got) if got else 0.0
        agg["recall_sum"] += recall
        agg["prec_sum"] += prec
        detail.append((it.query, detected[0], f"命中{len(got)}条", sorted(got)[:5], gold, recall))

    # 明细
    print("=" * 78)
    for q, st, tag, got, gold, rec in detail:
        miss = sorted(gold - set(got))[:3]
        miss_s = (" 漏:" + "/".join(miss)) if miss else ""
        print(f"[{st:<20}] {tag:<8} R={rec:.2f}  {q[:30]}{miss_s}")
    print("=" * 78)

    # 汇总
    print(f"\n{'子型':<22}{'总数':>5}{'检出':>6}{'0命中':>7}{'Recall':>9}{'Precision':>11}")
    tot_n = tot_rec = 0.0
    for st, a in by_subtype.items():
        rec = a["recall_sum"] / a["n"] if a["n"] else 0.0
        prec = a["prec_sum"] / a["n"] if a["n"] else 0.0
        tot_n += a["n"]; tot_rec += a["recall_sum"]
        print(f"{st:<22}{a['n']:>5}{a['detected']:>6}{a['hit0']:>7}{rec:>9.3f}{prec:>11.3f}")
    print("-" * 60)
    print(f"{'relation 总体':<22}{tot_n:>5}{'':>6}{'':>7}{(tot_rec/tot_n if tot_n else 0):>9.3f}")
    gr.close()
    print("\n解读：Recall 应显著 >0（修前 graph_rag relation=0），Precision 应接近 1.0（精确 Cypher）。")


if __name__ == "__main__":
    main()
