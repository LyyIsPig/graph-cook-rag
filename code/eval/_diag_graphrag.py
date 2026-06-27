"""
诊断：graph_rag 在关系查询上到底返回了什么？
区分两种可能：
  (a) 找到了相关菜谱但没写进 recipe_name/正文（度量/序列化问题）→ graph_rag 其实有用
  (b) 子图里压根没这些菜谱（真的没找到）→ graph_rag 在这类查询上没用
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

CASES = [
    ("和炸鲜奶一样用了玉米淀粉的菜还有哪些？", ["玛格丽特饼干", "芋泥雪媚娘", "脆皮豆腐"]),
    ("用了胡椒粉的主食有哪些？", ["皮蛋瘦肉粥", "汤面", "炒河粉", "热干面", "蛋炒饭"]),
]

print("初始化系统 ...")
system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
system.initialize_system(); system.build_knowledge_base()

for q, relevant in CASES:
    print("\n" + "=" * 80)
    print(f"查询: {q}")
    print(f"真值 relevant: {relevant}")
    docs = system.graph_rag_retrieval.graph_rag_search(q, 5)
    print(f"graph_rag 返回 {len(docs)} 个文档:")
    for i, d in enumerate(docs):
        print(f"\n  --- doc[{i}] metadata ---")
        for k, v in d.metadata.items():
            print(f"     {k}: {str(v)[:150]}")
        print(f"  --- doc[{i}] page_content (前 400 字) ---")
        print(f"     {d.page_content[:400]}")
    # 检查 relevant 是否出现在任何文档的元数据或正文
    found_in = []
    for nm in relevant:
        for d in docs:
            if nm in d.page_content or nm == d.metadata.get("recipe_name"):
                found_in.append(nm); break
    print(f"\n  >>> relevant 在返回结果里出现的: {found_in}  (共 {len(relevant)} 个真值)")

system._cleanup()
