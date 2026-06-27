"""
构建 golden 测试集（P2）。
真值直接来自 Neo4j 知识图谱（非 LLM 臆造），三类查询：
  - single_recipe : "{菜名}怎么做"            → relevant = [该菜]
  - ingredient    : "有哪些用{食材}做的菜？"   → relevant = 所有 REQUIRES 该食材的菜
  - category      : "有哪些{分类}？"           → relevant = 该分类下的菜

用法（在 code/ 下）：
    python -m eval.build_testset                      # 默认规模
    python -m eval.build_testset --single 40 --ingredient 15 --category 8
生成：eval/testset.jsonl  （建议人工抽检/微调后再用于评测）
"""

import os
import sys
import random
import argparse
from collections import defaultdict

# ---- bootstrap：让本脚本无论怎么跑都能 import code/ 下的包 ----
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))

from config import DEFAULT_CONFIG
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from eval.testset import EvalQuery, save_testset

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch_ingredient_to_recipes(driver, min_cnt=3, max_cnt=15, limit=40):
    """食材 → 菜谱名列表（按频率降序，过滤数量过少/过多的食材）。"""
    cypher = """
    MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient)
    WHERE r.nodeId >= '200000000' AND i.name IS NOT NULL
    RETURN i.name AS ing, collect(DISTINCT r.name) AS recipes
    """
    result = {}
    with driver.session() as session:
        for rec in session.run(cypher):
            ing = rec["ing"]
            recipes = [r for r in rec["recipes"] if r]
            if min_cnt <= len(recipes) <= max_cnt:
                result[ing] = recipes
    # 按菜谱数降序取前 limit
    return dict(sorted(result.items(), key=lambda kv: len(kv[1]), reverse=True)[:limit])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, default=40)
    ap.add_argument("--ingredient", type=int, default=15)
    ap.add_argument("--category", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "testset.jsonl"))
    args = ap.parse_args()
    random.seed(args.seed)

    print("连接 Neo4j 加载菜谱...")
    dm = GraphDataPreparationModule(
        uri=DEFAULT_CONFIG.neo4j_uri, user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password, database=DEFAULT_CONFIG.neo4j_database,
    )
    dm.load_graph_data()
    recipes = [r for r in dm.recipes if r.name and len(r.name) >= 2]
    print(f"菜谱总数: {len(recipes)}")

    items = []

    # 1) single_recipe
    sampled = random.sample(recipes, min(args.single, len(recipes)))
    for i, r in enumerate(sorted(sampled, key=lambda x: x.name), 1):
        items.append(EvalQuery(
            id=f"q_single_{i:03d}",
            query=f"{r.name}怎么做",
            query_type="single_recipe",
            relevant_recipe_names=[r.name],
            notes=f"菜系={r.properties.get('cuisineType','?')} 难度={r.properties.get('difficulty','?')}",
        ))

    # 2) ingredient
    ing_map = fetch_ingredient_to_recipes(dm.driver)
    ings = list(ing_map.items())[:args.ingredient]
    for i, (ing, recipes_for_ing) in enumerate(ings, 1):
        items.append(EvalQuery(
            id=f"q_ingr_{i:03d}",
            query=f"有哪些用{ing}做的菜？",
            query_type="ingredient",
            relevant_recipe_names=recipes_for_ing,
            notes=f"该食材共 {len(recipes_for_ing)} 道菜",
        ))

    # 3) category
    cat_map = defaultdict(list)
    for r in recipes:
        cat = r.properties.get("category") or "未知"
        cat_map[cat].append(r.name)
    cats = sorted(cat_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:args.category]
    for i, (cat, names) in enumerate(cats, 1):
        if cat == "未知":
            continue
        items.append(EvalQuery(
            id=f"q_cat_{i:03d}",
            query=f"有哪些{cat}？",
            query_type="category",
            relevant_recipe_names=names,
            notes=f"该分类共 {len(names)} 道菜",
        ))

    dm.close()
    save_testset(items, args.out)
    print(f"\n✅ 已生成 {len(items)} 条 → {args.out}")
    by_type = defaultdict(int)
    for it in items:
        by_type[it.query_type] += 1
    for t, c in by_type.items():
        print(f"   {t}: {c}")
    print("\n下一步：人工抽检 eval/testset.jsonl，删除/修改不自然条目，再运行 run_retrieval_eval.py")


if __name__ == "__main__":
    main()
