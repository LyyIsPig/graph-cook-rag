"""[B] 端到端冒烟：关系查询走完整 ask 链路，确认不被拒答闸门误杀、生成正确列表。"""
import os, sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path: sys.path.insert(0, _CODE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))
from config import DEFAULT_CONFIG
from main import AdvancedGraphRAGSystem

queries = [
    "和可乐鸡翅一样用了鸡翅中的菜还有哪些？",   # shared_ingredient
    "用了青辣椒的素菜有哪些？",                  # ingredient_category
    "需要砂锅的菜有哪些？",                       # by_tool
]
sys = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
sys.initialize_system(); sys.build_knowledge_base()
for q in queries:
    ok, reason, score = sys.check_answerable(q)
    tag = "放行" if ok else f"拒答({reason})"
    print(f"\n{'='*70}\n问: {q}\n闸门: {tag} (score={score:.2f})")
    if not ok:
        print("  → 被拒，跳过生成"); continue
    ans = sys.ask_question_with_routing(q)
    print(f"答: {str(ans)[:300]}")
sys._cleanup()
