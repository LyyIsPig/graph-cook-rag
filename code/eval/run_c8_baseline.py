"""
P2-3 纵向对比 · C8 基线（普通 RAG / FAISS 纯向量检索）。
【独立子进程运行】——C8 与当前项目都有 rag_modules/config.py，同进程会模块串台。
本脚本把 C8 根目录插到 sys.path 最前 + chdir，加载 C8 的 FAISS 索引，对测试集每条查询做
similarity_search，提取 dish_name（去掉"的做法"后缀），写出 {id, capability, ranked}。

用法（从任意目录，用 graph-rag 环境的 python）：
    python -m eval.run_c8_baseline
或直接： python eval/run_c8_baseline.py
"""

import json
import os
import re
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

C8_ROOT = r"D:/learn/实习计划/C8"
CODE_ROOT = r"D:/learn/实习计划/graph_cook_rag/code"
TESTSET = os.path.join(CODE_ROOT, "eval", "testset.v2.jsonl")
OUT = os.path.join(CODE_ROOT, "eval", "results", "long_c8.jsonl")

# 必须在 import C8 模块【之前】把 C8 放到 path 最前 + 切 cwd（FAISS 用相对路径）
sys.path.insert(0, C8_ROOT)
os.chdir(C8_ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rag_modules.index_construction import IndexConstructionModule  # noqa: E402


def clean_name(s):
    if not s:
        return None
    return re.sub(r"的(做法|做法步骤|制作方法)$", "", s).strip()


def main():
    print("加载 C8 FAISS 索引 ...")
    # FAISS(C++) 在 Windows 读不了含中文的路径 → 拷到 ASCII 临时目录再加载
    src = os.path.join(C8_ROOT, "vector_index")
    tmp = tempfile.mkdtemp(prefix="c8faiss_")  # ASCII 路径
    for fn in os.listdir(src):
        shutil.copy(os.path.join(src, fn), os.path.join(tmp, fn))
    print(f"  索引拷到临时目录: {tmp}")
    im = IndexConstructionModule(
        model_name="BAAI/bge-small-zh-v1.5",
        index_save_path=tmp,
    )
    vs = im.load_index()
    if vs is None:
        print("❌ C8 FAISS 加载失败"); sys.exit(1)
    print("✅ C8 索引就绪")

    items = [json.loads(l) for l in open(TESTSET, encoding="utf-8")]
    n_out = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for it in items:
            if it.get("capability") == "negative":
                continue
            try:
                docs = vs.similarity_search(it["query"], k=5)
            except Exception as e:
                print(f"  检索异常 {it['id']}: {e}")
                docs = []
            names = []
            for d in docs:
                md = getattr(d, "metadata", {}) or {}
                nm = clean_name(md.get("dish_name")) or clean_name(md.get("主标题"))
                if nm and nm not in names:
                    names.append(nm)
            fout.write(json.dumps(
                {"id": it["id"], "capability": it["capability"],
                 "query_type": it.get("query_type"), "ranked": names},
                ensure_ascii=False) + "\n")
            fout.flush()
            n_out += 1
            if n_out % 20 == 0:
                print(f"  已处理 {n_out} 条")
    print(f"✅ C8 基线写出 {n_out} 条 → {OUT}")


if __name__ == "__main__":
    main()
