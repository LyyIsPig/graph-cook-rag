"""探向量相似度分数分布，为拒答阈值定标。
对比：存在菜谱(高分组) / 不存在菜谱(应低分) / 泛化查询 的 top1 cosine。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
from dotenv import load_dotenv; load_dotenv()
from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem

CASES = [
    ("宫保鸡丁怎么做", True),        # 存在
    ("番茄炒蛋怎么做", True),        # 存在
    ("北京烤鸭怎么做", False),       # 不存在(KB外)
    ("佛跳墙怎么做", False),         # 不存在
    ("怎么用Python写爬虫", False),   # 跨域
    ("有哪些川菜", True),            # 存在(分类)
]

print("初始化 ...")
s = AdvancedGraphRAGSystem(DEFAULT_CONFIG); s.initialize_system(); s.build_knowledge_base()
print(f"\n{'query':<22}{'应可答':<8}{'top1_score':<12}{'top1_name'}")
for q, ans in CASES:
    res = s.index_module.similarity_search(q, k=3)
    if res:
        top = res[0]
        score = top.get("score", top.get("distance", "?"))
        name = (top.get("metadata") or {}).get("recipe_name", "?")
        print(f"{q:<22}{str(ans):<8}{str(score)[:10]:<12}{name}")
    else:
        print(f"{q:<22}{str(ans):<8}(空)")
s._cleanup()
