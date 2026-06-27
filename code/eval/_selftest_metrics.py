"""metrics.py 自测（纯函数）。预期全绿。"""
from types import SimpleNamespace
from eval.metrics import recall_at_k, mrr, ndcg_at_k, names_from_doc, ranked_names_from_docs, aggregate

def doc(name=None, content=""):
    return SimpleNamespace(metadata=({"recipe_name": name} if name else {}), page_content=content)

# recall@k
assert recall_at_k(["A", "B", "C"], {"A", "C"}, k=3) == 1.0
assert recall_at_k(["A", "B", "C"], {"A", "C"}, k=1) == 0.5
assert recall_at_k(["A", "B", "C"], set(), k=3) == 0.0

# mrr
assert mrr(["A", "B", "C"], {"B"}) == 0.5
assert mrr(["A", "B", "C"], {"A"}) == 1.0
assert mrr(["A", "B", "C"], {"D"}) == 0.0

# ndcg@k
def approx(a, b, tol=1e-4): return abs(a - b) < tol
assert approx(ndcg_at_k(["B", "A"], {"B"}, k=2), 1.0)              # 完美排序
assert approx(ndcg_at_k(["A", "B"], {"B"}, k=2), 1 / __import__("math").log2(3))  # 相关项排第2 → 0.6309
assert approx(ndcg_at_k(["A", "B"], {"A", "B"}, k=2), 1.0)         # 全中且完美
assert ndcg_at_k(["A"], {"D"}, k=2) == 0.0

# names_from_doc：metadata 命中 + 正文扫描命中（图RAG 子图场景）
U = {"番茄炒蛋", "宫保鸡丁", "鸡蛋"}
assert names_from_doc(doc("番茄炒蛋"), U) == ["番茄炒蛋"]
# 子图文档没有 recipe_name，但正文里提到菜谱名 → 按出现位置扫出来
assert names_from_doc(doc(None, "关于 鸡蛋 的知识网络，宫保鸡丁 与之相关"), U) == ["鸡蛋", "宫保鸡丁"]

# ranked_names_from_docs 跨文档去重、保序
docs = [doc("番茄炒蛋"), doc("番茄炒蛋", "宫保鸡丁"), doc("宫保鸡丁")]
assert ranked_names_from_docs(docs, U) == ["番茄炒蛋", "宫保鸡丁"]

# aggregate
agg = aggregate([0.0, 0.5, 1.0])
assert agg["n"] == 3 and approx(agg["mean"], 0.5)

print("=== metrics.py 自测全部通过 ===")
