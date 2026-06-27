"""探针：定位 shared_ingredient 命中 0 的根因。"""
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path: sys.path.insert(0, _CODE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))
from config import DEFAULT_CONFIG
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval, detect_relation_pattern
from neo4j import GraphDatabase

drv = GraphDatabase.driver(DEFAULT_CONFIG.neo4j_uri, auth=(DEFAULT_CONFIG.neo4j_user, DEFAULT_CONFIG.neo4j_password))

q = "和可乐鸡翅一样用了鸡翅中的菜还有哪些？"
det = detect_relation_pattern(q)
print("检出:", det)
anchor = det[1]["anchor"]; ing = det[1]["ingredient"]

gr = GraphRAGRetrieval(DEFAULT_CONFIG, llm_client=None); gr.initialize()
print("resolve anchor(Recipe):", repr(gr._resolve_name(anchor, "Recipe")))
print("resolve ing(Ingredient):", repr(gr._resolve_name(ing, "Ingredient")))

with drv.session() as s:
    # anchor 是否存在 + nodeId
    rows = list(s.run("MATCH (r:Recipe{name:$n}) RETURN r.name AS name, r.nodeId AS nid", n=anchor))
    print("anchor 节点:", [(r["name"], r["nid"]) for r in rows])
    # ingredient 是否存在
    rows = list(s.run("MATCH (i:Ingredient{name:$n}) RETURN i.name AS name", n=ing))
    print("ing 节点:", [r["name"] for r in rows])
    # 直接跑真值 cypher（字符串插值，与 testset 完全一致）
    truth = list(s.run(
        "MATCH (:Recipe{name:$a})-[:REQUIRES]->(:Ingredient{name:$i})<-[:REQUIRES]-(r:Recipe) "
        "WHERE r.name<>$a RETURN DISTINCT r.name AS name", a=anchor, i=ing))
    print("真值 cypher 命中:", [r["name"] for r in truth])
    # 检查 anchor 是否真的 REQUIRES 这个 ingredient
    rel = list(s.run(
        "MATCH (r:Recipe{name:$a})-[rel:REQUIRES]->(i:Ingredient) RETURN i.name AS name", a=anchor))
    print(f"{anchor} 的 REQUIRES 食材:", [r["name"] for r in rel])
gr.close(); drv.close()
