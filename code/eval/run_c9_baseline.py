"""
P2-3 纵向对比 · C9 基线（Graph RAG 基线，当前项目前身）。
【独立子进程运行】——C9 与当前项目同栈同名模块，同进程会串台。
本脚本实例化 C9 的 AdvancedGraphRAGSystem（复用同一 Neo4j+Milvus 数据），对测试集子样本
每条查询跑 graph_rag_search（纯图）+ route_query（端到端路由），提取 recipe_name，写出
{id, capability, ranked_graph, ranked_routed}。

用法：
    python eval/run_c9_baseline.py --per-cap 8
"""

import json
import os
import sys
import argparse
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

C9_ROOT = r"D:/learn/实习计划/C9"
CODE_ROOT = r"D:/learn/实习计划/graph_cook_rag/code"
TESTSET = os.path.join(CODE_ROOT, "eval", "testset.v2.jsonl")
OUT = os.path.join(CODE_ROOT, "eval", "results", "long_c9.jsonl")

sys.path.insert(0, C9_ROOT)
os.chdir(C9_ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(C9_ROOT, ".env"))

from main import AdvancedGraphRAGSystem  # noqa: E402

CAPS = ["lookup", "list", "relation", "reasoning"]


def names_from_docs(docs):
    out = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        nm = md.get("recipe_name") or md.get("name")
        if nm and nm not in out:
            out.append(nm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cap", type=int, default=8, help="每个能力取前 N 条（C9 慢，默认 8）")
    args = ap.parse_args()

    print("初始化 C9 系统（复用同库 Neo4j+Milvus）...")
    sys = AdvancedGraphRAGSystem()
    sys.initialize_system()
    sys.build_knowledge_base()
    print("✅ C9 就绪")

    items = [json.loads(l) for l in open(TESTSET, encoding="utf-8")]
    by_cap = defaultdict(list)
    for it in items:
        if it.get("capability") in CAPS:
            by_cap[it["capability"]].append(it)
    subset = []
    for c in CAPS:
        subset.extend(by_cap[c][: args.per_cap])
    print(f"子样本 {len(subset)} 条（每能力 {args.per_cap}）")

    with open(OUT, "w", encoding="utf-8") as fout:
        for i, it in enumerate(subset, 1):
            q = it["query"]
            # (a) 纯图检索 graph_rag_search
            try:
                gdocs = sys.graph_rag_retrieval.graph_rag_search(q, 5)
            except Exception as e:
                print(f"  graph_rag 异常 {it['id']}: {e}"); gdocs = []
            # (b) 端到端路由 route_query（返回元组）
            try:
                rdocs, _ = sys.query_router.route_query(q, 5)
            except Exception as e:
                print(f"  route 异常 {it['id']}: {e}"); rdocs = []
            rec = {
                "id": it["id"], "capability": it["capability"],
                "query_type": it.get("query_type"),
                "ranked_graph": names_from_docs(gdocs),
                "ranked_routed": names_from_docs(rdocs),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  [{i}/{len(subset)}] {it['id']} ({it['capability']}) graph={len(rec['ranked_graph'])} routed={len(rec['ranked_routed'])}")
    print(f"✅ C9 基线写出 {len(subset)} 条 → {OUT}")
    sys._cleanup()


if __name__ == "__main__":
    main()
