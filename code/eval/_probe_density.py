"""
关系/负样本数据密度探测（P2-2 前置）。
只读 Neo4j，统计 4 个 relation 子型与负样本的可生成种子数量，
决定 build_testset_v2 保留哪些子型（目标：每条查询结果集落在 2~10）。
"""

import os
import sys
from collections import Counter

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from dotenv import load_dotenv
load_dotenv(os.path.join(_CODE, ".env"))

from config import DEFAULT_CONFIG
from rag_modules.graph_data_preparation import GraphDataPreparationModule

# 负样本候选菜名（期望多数不在 KB）
NEG_CANDIDATES = [
    "北京烤鸭", "佛跳墙", "叫花鸡", "东坡肉", "夫妻肺片", "蚂蚁上树", "狮子头",
    "松鼠鳜鱼", "九转大肠", "剁椒鱼头", "回锅肉", "水煮肉片", "腊味合蒸",
    "糖醋排骨", "酸菜鱼", "蒜泥白肉", "梅菜扣肉", "辣子鸡", "香辣虾", "清蒸武昌鱼",
]


def run(driver, cypher, **params):
    with driver.session() as s:
        return [dict(r) for r in s.run(cypher, params)]


def main():
    dm = GraphDataPreparationModule(
        uri=DEFAULT_CONFIG.neo4j_uri, user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password, database=DEFAULT_CONFIG.neo4j_database)
    drv = dm.driver

    # 0) 基础规模
    n_recipe = run(drv, "MATCH (r:Recipe) WHERE r.nodeId>='200000000' RETURN count(r) AS c")[0]["c"]
    n_ing = run(drv, "MATCH (i:Ingredient) WHERE i.nodeId>='200000000' RETURN count(i) AS c")[0]["c"]
    n_step = run(drv, "MATCH (s:CookingStep) WHERE s.nodeId>='200000000' RETURN count(s) AS c")[0]["c"]
    print(f"\n== 基础规模 ==  Recipe={n_recipe}  Ingredient={n_ing}  CookingStep={n_step}")

    # 分类分布
    cats = run(drv, "MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(c:Category) WHERE r.nodeId>='200000000' "
                    "RETURN c.name AS cat, count(r) AS c ORDER BY c DESC")
    print("\n== 分类分布 ==")
    for r in cats:
        print(f"  {r['cat']:<8} {r['c']}")

    # 1) 共用食材：每食材被多少菜用（决定 shared_ingredient 查询结果集大小）
    ing_freq = run(drv, "MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient) WHERE r.nodeId>='200000000' "
                        "RETURN i.name AS ing, count(DISTINCT r) AS c ORDER BY c DESC")
    freq_dist = Counter(r["c"] for r in ing_freq)
    viable_ing = [r for r in ing_freq if 3 <= r["c"] <= 10]
    print(f"\n== 1) 共用食材 ==  食材总数={len(ing_freq)}  结果集3~10的食材={len(viable_ing)}（即可生成 {len(viable_ing)} 条查询）")
    print(f"   频次分布(前若干): {dict(sorted(freq_dist.items())[:12])}")
    print(f"   示例: {[(r['ing'], r['c']) for r in viable_ing[:5]]}")

    # 2) 食材×分类组合
    pairs = run(drv, "MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient), (r)-[:BELONGS_TO_CATEGORY]->(c:Category) "
                     "WHERE r.nodeId>='200000000' RETURN i.name AS ing, c.name AS cat, count(DISTINCT r) AS c ORDER BY c DESC")
    viable_pairs = [p for p in pairs if 2 <= p["c"] <= 10]
    print(f"\n== 2) 食材×分类 ==  总(食材,分类)对={len(pairs)}  结果集2~10的对={len(viable_pairs)}（可生成 {len(viable_pairs)} 条）")
    print(f"   示例: {[(p['ing'], p['cat'], p['c']) for p in viable_pairs[:5]]}")

    # 3) 同分类避用某食材：每分类大小 + 选避用后落在2~10
    print(f"\n== 3) 同分类避用某食材 ==")
    viable_excl = 0
    excl_samples = []
    for cat_row in cats[:8]:  # 取较大分类
        cat = cat_row["cat"]
        cat_total = cat_row["c"]
        # 该分类下各食材使用数
        usage = run(drv, "MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(:Category{name:$cat}), (r)-[:REQUIRES]->(i:Ingredient) "
                         "WHERE r.nodeId>='200000000' RETURN i.name AS ing, count(DISTINCT r) AS c ORDER BY c DESC",
                    cat=cat)
        for u in usage:
            excl = cat_total - u["c"]
            if 2 <= excl <= 10:
                viable_excl += 1
                if len(excl_samples) < 5:
                    excl_samples.append((cat, u["ing"], excl))
    print(f"   可生成(分类,避用食材)结果集2~10 的种子数={viable_excl}")
    print(f"   示例: {excl_samples}")

    # 4) 按工具/做法：CookingStep.tools / methods 密度
    dens = run(drv, "MATCH (s:CookingStep) WHERE s.nodeId>='200000000' "
                    "RETURN count(s) AS total, count(s.tools) AS has_tools, count(s.methods) AS has_methods")[0]
    print(f"\n== 4) 按工具/做法 ==  CookingStep总数={dens['total']}  有tools={dens['has_tools']}  有methods={dens['has_methods']}")
    tools_top = run(drv, "MATCH (s:CookingStep) WHERE s.nodeId>='200000000' AND s.tools IS NOT NULL "
                         "RETURN s.tools AS t, count(s) AS c ORDER BY c DESC LIMIT 8")
    print(f"   tools 取值TOP: {[(r['t'][:20], r['c']) for r in tools_top]}")

    # 5) 负样本：候选中不在 KB 的
    found = {r["name"] for r in run(drv, "UNWIND $names AS n MATCH (r:Recipe{name:n}) RETURN r.name AS name", names=NEG_CANDIDATES)}
    negatives = [n for n in NEG_CANDIDATES if n not in found]
    print(f"\n== 5) 负样本 ==  候选{len(NEG_CANDIDATES)}个，确认不在KB的={len(negatives)}: {negatives}")

    dm.close()
    print("\n=== 探测完成，据此决定保留子型 ===")


if __name__ == "__main__":
    main()
