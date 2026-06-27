"""
重构 golden 测试集 v2（P2-2）。
按"系统能力维度"组织，真值优先 Cypher 标注（可审计可复现），每条带 cypher 字段。
目标 120+ 条：lookup / list / relation / negative / reasoning。

用法（在 code/ 下，只需 Neo4j）：
    python -m eval.build_testset_v2
输出：eval/testset.v2.jsonl  （建议人工抽检 relation/reasoning 后再用于评测）
"""

import os
import sys
import random
import argparse

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
from eval.testset_v2 import EvalQueryV2, save_testset_v2

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE_FILTER = "r.nodeId >= '200000000'"

# 稀有工具/做法候选（按 recipe 级计数再筛 2~10）
TOOL_CANDIDATES = ["烤箱", "空气炸锅", "蒸笼", "砂锅", "高压锅", "平底锅", "电饭煲", "微波炉", "漏勺", "擀面杖"]
METHOD_CANDIDATES = ["炸", "烤", "蒸", "焖", "炖", "煮", "煎", "拌", "腌", "炒"]
NEG_DISHES = ["北京烤鸭", "佛跳墙", "叫花鸡", "东坡肉", "夫妻肺片", "狮子头", "松鼠鳜鱼",
              "九转大肠", "剁椒鱼头", "腊味合蒸", "酸菜鱼", "蒜泥白肉", "辣子鸡", "香辣虾", "清蒸武昌鱼"]
NEG_CROSS = [
    ("怎么用Python写一个网络爬虫？", "跨域：编程"),
    ("不需要任何食材的菜谱有哪些？", "条件矛盾"),
    ("永动机的制作方法？", "跨域：物理"),
    ("红烧肉不放酱油还能叫红烧肉吗？", "语义矛盾/开放"),
    ("明天深圳天气怎么样？", "跨域：天气"),
]


def run(driver, cypher, **p):
    with driver.session() as s:
        return [dict(r) for r in s.run(cypher, p)]


def gen_lookup(driver, n, seed):
    rows = run(driver, f"MATCH (r:Recipe) WHERE {RECIPE_FILTER} AND r.name IS NOT NULL "
                       f"RETURN r.name AS name ORDER BY size(r.name) DESC")
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for i, r in enumerate(chosen, 1):
        nm = r["name"]
        out.append(EvalQueryV2(
            id=f"q_lookup_{i:03d}", query=f"{nm}怎么做", capability="lookup",
            query_type="single_recipe", relevant_recipe_names=[nm], negative=False,
            label_source="cypher", cypher=f"MATCH (r:Recipe{{name:'{nm}'}}) RETURN r.name",
            notes="单菜how-to"))
    return out


def gen_list(driver, n, seed):
    rng = random.Random(seed)
    out = []
    # (a) 纯食材列表
    ings = run(driver, f"MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient) WHERE {RECIPE_FILTER} "
                      f"RETURN i.name AS ing, collect(DISTINCT r.name) AS rs")
    ings = [r for r in ings if 3 <= len(r["rs"]) <= 10]
    rng.shuffle(ings)
    for r in ings[: n // 3]:
        out.append(EvalQueryV2(
            id=f"q_list_ing_{len(out)+1:03d}", query=f"用{r['ing']}做的菜有哪些？", capability="list",
            query_type="ingredient", relevant_recipe_names=r["rs"], negative=False, label_source="cypher",
            cypher=f"MATCH (r:Recipe)-[:REQUIRES]->(:Ingredient{{name:'{r['ing']}'}}) RETURN r.name",
            notes=f"食材关联 {len(r['rs'])} 道"))
    # (b) 纯分类列表
    cats = run(driver, f"MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(c:Category) WHERE {RECIPE_FILTER} "
                      f"RETURN c.name AS cat, collect(DISTINCT r.name) AS rs")
    cats = [r for r in cats if len(r["rs"]) >= 3]
    rng.shuffle(cats)
    for r in cats[: n // 3]:
        rs = r["rs"][:10]
        out.append(EvalQueryV2(
            id=f"q_list_cat_{len(out)+1:03d}", query=f"有哪些{r['cat']}？", capability="list",
            query_type="category", relevant_recipe_names=rs, negative=False, label_source="cypher",
            cypher=f"MATCH (r:Recipe)-[:BELONGS_TO_CATEGORY]->(:Category{{name:'{r['cat']}'}}) RETURN r.name",
            notes=f"分类共 {len(r['rs'])} 道（取前10）"))
    # (c) 食材+难度过滤
    for ing in (ings[:6]):
        for diff_label, op, val in [("低难度", "<=", 2), ("高难度", ">=", 4)]:
            rs = [x["r.name"] for x in run(driver, f"MATCH (r:Recipe)-[:REQUIRES]->(:Ingredient{{name:$ing}}) WHERE {RECIPE_FILTER} "
                            f"AND r.difficulty {op} $val RETURN DISTINCT r.name", ing=ing["ing"], val=val)]
            if 2 <= len(rs) <= 10:
                out.append(EvalQueryV2(
                    id=f"q_list_filt_{len(out)+1:03d}", query=f"用{ing['ing']}做的{diff_label}菜有哪些？",
                    capability="list", query_type="ingredient_difficulty", relevant_recipe_names=rs,
                    negative=False, label_source="cypher",
                    cypher=f"MATCH (r:Recipe)-[:REQUIRES]->(:Ingredient{{name:'{ing['ing']}'}}) WHERE r.difficulty {op} {val} RETURN r.name",
                    notes=f"食材×难度({diff_label}) {len(rs)} 道"))
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return out[:n]


def gen_relation(driver, n_shared, n_ingcat, n_tool, seed):
    rng = random.Random(seed)
    out = []

    # 共用食材：选被3~10道菜用的食材 + 一个锚点菜
    ings = run(driver, f"MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient) WHERE {RECIPE_FILTER} "
                      f"RETURN i.name AS ing, collect(DISTINCT r.name) AS rs")
    ings = [r for r in ings if 3 <= len(r["rs"]) <= 10]
    rng.shuffle(ings)
    cnt = 0
    for r in ings:
        if cnt >= n_shared:
            break
        rs = r["rs"]
        anchor = rs[0]                       # 锚点菜
        others = [x for x in rs if x != anchor]
        if 2 <= len(others) <= 9:
            out.append(EvalQueryV2(
                id=f"q_rel_shared_{cnt+1:03d}", query=f"和{anchor}一样用了{r['ing']}的菜还有哪些？",
                capability="relation", query_type="shared_ingredient", relevant_recipe_names=others,
                negative=False, label_source="cypher",
                cypher=f"MATCH (:Recipe{{name:'{anchor}'}})-[:REQUIRES]->(:Ingredient{{name:'{r['ing']}'}})<-[:REQUIRES]-(r:Recipe) WHERE r.name<>'{anchor}' RETURN DISTINCT r.name",
                notes=f"共用{r['ing']}，除锚点外 {len(others)} 道"))
            cnt += 1

    # 食材×分类
    pairs = run(driver, f"MATCH (r:Recipe)-[:REQUIRES]->(i:Ingredient), (r)-[:BELONGS_TO_CATEGORY]->(c:Category) "
                       f"WHERE {RECIPE_FILTER} RETURN i.name AS ing, c.name AS cat, collect(DISTINCT r.name) AS rs")
    pairs = [p for p in pairs if 2 <= len(p["rs"]) <= 10]
    rng.shuffle(pairs)
    for p in pairs[:n_ingcat]:
        out.append(EvalQueryV2(
            id=f"q_rel_ingcat_{cnt+1:03d}", query=f"用了{p['ing']}的{p['cat']}有哪些？",
            capability="relation", query_type="ingredient_category", relevant_recipe_names=p["rs"],
            negative=False, label_source="cypher",
            cypher=f"MATCH (r:Recipe)-[:REQUIRES]->(:Ingredient{{name:'{p['ing']}'}}), (r)-[:BELONGS_TO_CATEGORY]->(:Category{{name:'{p['cat']}'}}) RETURN DISTINCT r.name",
            notes=f"食材×分类 {len(p['rs'])} 道"))
        cnt += 1

    # 按工具/做法（recipe 级计数筛 2~10）
    seeds = []
    for t in TOOL_CANDIDATES:
        rs = [x["r.name"] for x in run(driver, f"MATCH (r:Recipe)-[:CONTAINS_STEP]->(s:CookingStep) WHERE {RECIPE_FILTER} "
                        f"AND s.tools CONTAINS $t RETURN DISTINCT r.name", t=t)]
        if 2 <= len(rs) <= 10:
            seeds.append((f"需要{t}的菜有哪些？", "by_tool", t, f"...s.tools CONTAINS '{t}'", rs))
    for m in METHOD_CANDIDATES:
        rs = [x["r.name"] for x in run(driver, f"MATCH (r:Recipe)-[:CONTAINS_STEP]->(s:CookingStep) WHERE {RECIPE_FILTER} "
                        f"AND s.methods CONTAINS $m RETURN DISTINCT r.name", m=m)]
        if 2 <= len(rs) <= 10:
            seeds.append((f"用到{m}这种做法的菜有哪些？", "by_method", m, f"...s.methods CONTAINS '{m}'", rs))
    rng.shuffle(seeds)
    for q, qt, tok, cy, rs in seeds[:n_tool]:
        out.append(EvalQueryV2(
            id=f"q_rel_{qt}_{cnt+1:03d}", query=q, capability="relation", query_type=qt,
            relevant_recipe_names=rs, negative=False, label_source="cypher",
            cypher=f"MATCH (r:Recipe)-[:CONTAINS_STEP]->(s:CookingStep) WHERE {cy} RETURN DISTINCT r.name",
            notes=f"{qt}={tok}，{len(rs)} 道"))
        cnt += 1
    return out


def gen_negative(driver, seed):
    rng = random.Random(seed)
    out = []
    # 菜名负样本：再校验一次不在 KB
    found = {r["name"] for r in run(driver, "UNWIND $names AS n MATCH (r:Recipe{name:n}) RETURN r.name", names=NEG_DISHES)}
    for i, dish in enumerate([d for d in NEG_DISHES if d not in found], 1):
        out.append(EvalQueryV2(
            id=f"q_neg_dish_{i:03d}", query=f"{dish}怎么做", capability="negative",
            query_type="unanswerable_dish", relevant_recipe_names=[], negative=True, label_source="cypher",
            cypher=f"MATCH (r:Recipe{{name:'{dish}'}}) RETURN r.name  // 应为空",
            notes="KB不存在的菜，测拒答/防幻觉"))
    for i, (q, note) in enumerate(NEG_CROSS, 1):
        out.append(EvalQueryV2(
            id=f"q_neg_cross_{i:03d}", query=q, capability="negative", query_type="unanswerable_cross",
            relevant_recipe_names=[], negative=True, label_source="human", cypher="", notes=note))
    return out


def gen_reasoning(driver, n, seed):
    rng = random.Random(seed)
    out = []
    rows = run(driver, f"MATCH (r:Recipe) WHERE {RECIPE_FILTER} AND r.name IS NOT NULL AND r.difficulty IS NOT NULL "
                      f"OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient) "
                      f"RETURN r.name AS name, r.difficulty AS diff, count(DISTINCT i) AS icount")
    rows = [r for r in rows if r["name"] and r["diff"] is not None]
    rng.shuffle(rows)
    # 难度比较
    i = 0
    while i < len(rows) - 1 and len(out) < n // 2:
        a, b = rows[i], rows[i + 1]
        i += 2
        if a["diff"] == b["diff"]:
            continue
        winner = a["name"] if a["diff"] > b["diff"] else b["name"]
        out.append(EvalQueryV2(
            id=f"q_reas_diff_{len(out)+1:03d}", query=f"{a['name']}和{b['name']}哪个难度更大？",
            capability="reasoning", query_type="difficulty_compare", relevant_recipe_names=[winner],
            negative=False, label_source="cypher",
            cypher=f"RETURN CASE WHEN r1.difficulty>r2.difficulty THEN r1.name ELSE r2.name END  // {a['name']}({a['diff']}) vs {b['name']}({b['diff']})",
            notes=f"难度比较：{a['name']}={a['diff']} {b['name']}={b['diff']}"))
    # 食材数比较
    cand = [r for r in rows if r["icount"] and r["icount"] > 0]
    rng.shuffle(cand)
    j = 0
    while j < len(cand) - 1 and len(out) < n:
        a, b = cand[j], cand[j + 1]
        j += 2
        if a["icount"] == b["icount"]:
            continue
        winner = a["name"] if a["icount"] > b["icount"] else b["name"]
        out.append(EvalQueryV2(
            id=f"q_reas_ing_{len(out)+1:03d}", query=f"{a['name']}和{b['name']}哪个用的食材更多？",
            capability="reasoning", query_type="ingredient_count_compare", relevant_recipe_names=[winner],
            negative=False, label_source="cypher",
            cypher=f"// 食材数比较：{a['name']}={a['icount']} {b['name']}={b['icount']}",
            notes=f"食材数比较：{a['name']}={a['icount']} {b['name']}={b['icount']}"))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", type=int, default=20)
    ap.add_argument("--list", type=int, default=25)
    ap.add_argument("--rel-shared", type=int, default=16)
    ap.add_argument("--rel-ingcat", type=int, default=16)
    ap.add_argument("--rel-tool", type=int, default=8)
    ap.add_argument("--reasoning", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "testset.v2.jsonl"))
    args = ap.parse_args()

    print("连接 Neo4j ...")
    dm = GraphDataPreparationModule(
        uri=DEFAULT_CONFIG.neo4j_uri, user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password, database=DEFAULT_CONFIG.neo4j_database)
    drv = dm.driver

    items = []
    items += gen_lookup(drv, args.lookup, args.seed)
    items += gen_list(drv, args.list, args.seed + 1)
    items += gen_relation(drv, args.rel_shared, args.rel_ingcat, args.rel_tool, args.seed + 2)
    items += gen_negative(drv, args.seed + 3)
    items += gen_reasoning(drv, args.reasoning, args.seed + 4)

    dm.close()
    save_testset_v2(items, args.out)

    from collections import Counter
    by_cap = Counter(it.capability for it in items)
    by_type = Counter(it.query_type for it in items)
    print(f"\n✅ 生成 {len(items)} 条 → {args.out}")
    print("按能力:", dict(by_cap))
    print("按子型:", dict(by_type))
    multi = sum(1 for it in items if len(it.relevant_recipe_names) >= 2)
    neg = sum(1 for it in items if it.negative)
    print(f"多relevant(≥2): {multi}  负样本: {neg}  平均relevant数: {sum(len(it.relevant_recipe_names) for it in items)/len(items):.1f}")
    print("\n下一步：人工抽检 relation/reasoning 几条，确认后用于评测")


if __name__ == "__main__":
    main()
