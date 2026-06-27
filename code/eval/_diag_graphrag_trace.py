"""
定位 graph_rag 空子图的根因。
对每个查询逐层追踪：
  T1. understand_graph_query 抽取的 source_entities / query_type / max_depth 是什么
  T2. 这些 entity 能否 CONTAINS 匹配到真实节点（含哪种 Label）
  T3. 子图 Cypher 在【带】和【不带】`size(neighbors)<=max_nodes` 过滤下的命中情况
  T4. 把 max_depth 降到 1 是否就有结果
据此判断根因是"实体抽不到"、"子图太大被整块丢弃"、还是其它。
"""
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path: sys.path.insert(0, _CODE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))

from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem

QUERIES = [
    "用了胡椒粉的主食有哪些？",
    "和炸鲜奶一样用了玉米淀粉的菜还有哪些？",
    "需要高压锅的菜有哪些？",
]

print("初始化系统 ...")
system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
system.initialize_system(); system.build_knowledge_base()
gr = system.graph_rag_retrieval
drv = gr.driver
RF = "n.nodeId >= '200000000'"


def cy(cypher, **p):
    with drv.session() as s:
        return [dict(r) for r in s.run(cypher, p)]


for q in QUERIES:
    print("\n" + "=" * 90)
    print(f"查询: {q}")

    # T1: LLM 抽取
    gq = gr.understand_graph_query(q)
    print(f"[T1] query_type={gq.query_type.value}  source_entities={gq.source_entities}  "
          f"target={gq.target_entities}  max_depth={gq.max_depth}  max_nodes={gq.max_nodes}")

    # T2: 每个 entity 能否匹配到节点
    for e in gq.source_entities:
        rows = cy(f"MATCH (n) WHERE n.name CONTAINS $e RETURN labels(n) AS lbl, n.name AS nm LIMIT 5", e=e)
        print(f"[T2] entity={e!r}  命中节点 {len(rows)} 个: {[(r['lbl'], r['nm']) for r in rows[:3]]}")

    # T3: 子图 Cypher —— 不带 size 过滤，看每个 source 的邻居规模
    if gq.source_entities:
        for depth in (gq.max_depth, 1):
            rows = cy(
                f"UNWIND $es AS en MATCH (source) WHERE source.name CONTAINS en "
                f"MATCH (source)-[*1..{depth}]-(neighbor) "
                f"RETURN source.name AS src, count(DISTINCT neighbor) AS ncnt "
                f"ORDER BY ncnt DESC LIMIT 5",
                es=gq.source_entities)
            print(f"[T3] depth={depth}  各 source 的邻居数: {rows}")
        # 带 size<=max_nodes 过滤（复刻 extract_knowledge_subgraph 的行为）
            kept = cy(
                f"UNWIND $es AS en MATCH (source) WHERE source.name CONTAINS en "
                f"MATCH (source)-[*1..{depth}]-(neighbor) "
                f"WITH source, collect(DISTINCT neighbor) AS nb WHERE size(nb) <= $m "
                f"RETURN source.name AS src, size(nb) AS ncnt",
                es=gq.source_entities, m=gq.max_nodes)
            print(f"       加 size<={gq.max_nodes} 过滤后保留的 source: {kept}")
    else:
        print("[T2/T3] source_entities 为空 → Cypher 无输入，必然空子图（根因=LLM没抽出实体）")

system._cleanup()
