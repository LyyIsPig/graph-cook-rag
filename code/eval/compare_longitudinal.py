"""
P2-3 纵向对比：C8(普通RAG) / C9(Graph基线) / 当前系统，同题同库比召回/排序。
数据来源：
  - C8：eval/results/long_c8.jsonl  (ranked)
  - C9：eval/results/long_c9.jsonl  (ranked_graph, ranked_routed)
  - 当前：eval/results/cache_v2.json (strategy=vector/graph_rag/routed 的 ranked)
  - gold：eval/testset.v2.jsonl
对比集合 = C9 子样本的 id（C9 最慢、最小，三者取交集才公平）。
用法：python -m eval.compare_longitudinal
"""

import json
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

from eval.metrics import recall_at_k, mrr, ndcg_at_k  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CAPS = ["lookup", "list", "relation", "reasoning"]


def load_jsonl(path):
    out = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            out.append(json.loads(l))
    return out


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    testset = load_jsonl(os.path.join(HERE, "testset.v2.jsonl"))
    gold = {it["id"]: set(it.get("relevant_recipe_names", [])) for it in testset}
    cap_of = {it["id"]: it.get("capability") for it in testset}

    # 各系统的 ranked：sys_name -> id -> ranked_names
    systems = {}

    # C8
    c8 = {r["id"]: r.get("ranked", []) for r in load_jsonl(os.path.join(RES, "long_c8.jsonl"))}
    systems["C8 普通 RAG(向量)"] = c8

    # C9
    c9rows = load_jsonl(os.path.join(RES, "long_c9.jsonl"))
    systems["C9 Graph基线(graph_rag)"] = {r["id"]: r.get("ranked_graph", []) for r in c9rows}
    systems["C9 Graph基线(routed)"] = {r["id"]: r.get("ranked_routed", []) for r in c9rows}

    # 当前（cache_v2）
    cache = json.load(open(os.path.join(RES, "cache_v2.json"), encoding="utf-8"))
    for strat, label in [("vector", "当前(vector)"), ("graph_rag", "当前(graph_rag)"),
                         ("routed", "当前(routed)")]:
        systems[label] = {r["id"]: r.get("ranked", []) for r in cache
                          if r.get("strategy") == strat and "ranked" in r}

    # 对比集合 = C9 的 id（三者交集）
    common = sorted(systems["C9 Graph基线(graph_rag)"].keys())
    print(f"对比集合：{len(common)} 条（C9 子样本；三系统同题）\n")

    # 按 capability 切片
    by_cap_ids = defaultdict(list)
    for qid in common:
        by_cap_ids[cap_of.get(qid, "?")].append(qid)

    KS = [1, 3, 5]
    # 表头
    def print_table(metric_fn, name, k=None):
        print("=" * 92)
        print(f"{name}" + (f"@{k}" if k else ""))
        hdr = f"{'系统':<26}" + "".join(f"{c:>12}" for c in CAPS) + f"{'overall':>12}"
        print(hdr)
        print("-" * 92)
        for sname, ranked in systems.items():
            cells = []
            all_vals = []
            for c in CAPS:
                vals = []
                for qid in by_cap_ids.get(c, []):
                    rel = gold.get(qid, set())
                    if not rel:
                        continue
                    val = metric_fn(ranked.get(qid, []), rel, k) if k else metric_fn(ranked.get(qid, []), rel)
                    vals.append(val); all_vals.append(val)
                cells.append(mean(vals))
            overall = mean(all_vals)
            print(f"{sname:<26}" + "".join(f"{x:>12.3f}" for x in cells) + f"{overall:>12.3f}")
        print()

    print_table(lambda r, rel, k: recall_at_k(r, rel, k), "Recall", 5)
    print_table(lambda r, rel, k: recall_at_k(r, rel, k), "Recall", 1)
    print_table(lambda r, rel: mrr(r, rel), "MRR")
    print_table(lambda r, rel, k: ndcg_at_k(r, rel, k), "NDCG", 5)

    # 写 CSV（Recall@5 为主）
    with open(os.path.join(RES, "longitudinal_compare.csv"), "w", encoding="utf-8") as f:
        f.write("system,capability,n,rec@1,rec@5,mrr,ndcg@5\n")
        for sname, ranked in systems.items():
            for c in CAPS + ["overall"]:
                vals = []
                ids = by_cap_ids.get(c, []) if c != "overall" else common
                for qid in ids:
                    rel = gold.get(qid, set())
                    if not rel:
                        continue
                    vals.append((recall_at_k(ranked.get(qid, []), rel, 1),
                                 recall_at_k(ranked.get(qid, []), rel, 5),
                                 mrr(ranked.get(qid, []), rel),
                                 ndcg_at_k(ranked.get(qid, []), rel, 5)))
                if not vals:
                    continue
                r1 = mean([v[0] for v in vals]); r5 = mean([v[1] for v in vals])
                mm = mean([v[2] for v in vals]); nd = mean([v[3] for v in vals])
                f.write(f"{sname},{c},{len(vals)},{r1:.4f},{r5:.4f},{mm:.4f},{nd:.4f}\n")
    print(f"✅ CSV → eval/results/longitudinal_compare.csv")


if __name__ == "__main__":
    main()
