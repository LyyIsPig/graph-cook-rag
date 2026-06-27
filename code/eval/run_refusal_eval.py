"""
拒答机制验证（P2-3）：测量防幻觉效果。
- 负样本（不存在菜/跨域）：拒答率应高（之前 0% → 全幻觉；加机制后应大幅提升）
- 正样本（真实菜谱）：误拒率应低（不该把能答的也拒了）
拒答来源两处：① 向量低分闸门 check_answerable 直拒；② 生成层拒答提示词让 LLM 主动拒。

用法（在 code/ 下，需 Neo4j+Milvus + DEEPSEEK_API_KEY）：
    python -m eval.run_refusal_eval --pos-per-cap 4
"""

import os
import sys
import argparse
from collections import defaultdict

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
from main import AdvancedGraphRAGSystem, REFUSAL_MSG
from eval.testset_v2 import load_testset_v2
from eval.llm_judge import LLMJudge

HERE = os.path.dirname(os.path.abspath(__file__))
CAPS = ["lookup", "list", "relation", "reasoning"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-per-cap", type=int, default=4, help="每个能力取前 N 条正样本")
    ap.add_argument("--neg", type=int, default=0, help="负样本取前 N 条（0=全量）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--testset", default=os.path.join(HERE, "testset.v2.jsonl"))
    args = ap.parse_args()

    print("初始化系统 ...")
    system = AdvancedGraphRAGSystem(DEFAULT_CONFIG)
    system.initialize_system(); system.build_knowledge_base()
    judge = LLMJudge()
    print(f"判官 = {judge.provider}/{judge.model}")

    items = load_testset_v2(args.testset)
    by_cap = defaultdict(list)
    for it in items:
        by_cap[it.capability].append(it)
    positives = [it for c in CAPS for it in by_cap[c][: args.pos_per_cap]]
    negatives = by_cap["negative"][: args.neg] if args.neg else by_cap["negative"]
    print(f"正样本 {len(positives)} 条，负样本 {len(negatives)} 条\n")

    def evaluate(it_list, expect_answerable: bool):
        refused = 0
        gate_refused = 0
        for i, it in enumerate(it_list, 1):
            ok, reason, conf = system.check_answerable(it.query)
            if not ok:
                refused += 1; gate_refused += 1
                tag = f"闸门拒({reason},score={conf:.2f})"
                print(f"  [{i}/{len(it_list)}] {it.query[:24]:<24} {tag}")
                continue
            # 放行 → 生成 → 判官判是否拒答
            docs = system.query_router.route_query(it.query, args.k)[0]
            ans = system.generation_module.generate_adaptive_answer(it.query, docs)
            is_ref = judge.is_refusal(ans)
            if is_ref:
                refused += 1
            tag = "提示词拒" if is_ref else "作答"
            print(f"  [{i}/{len(it_list)}] {it.query[:24]:<24} {tag}  (score={conf:.2f})")
        return refused, gate_refused, len(it_list)

    print("=== 负样本（期望：尽量拒答）===")
    neg_refused, neg_gate, neg_n = evaluate(negatives, expect_answerable=False)
    print("\n=== 正样本（期望：尽量作答，别误拒）===")
    pos_refused, pos_gate, pos_n = evaluate(positives, expect_answerable=True)

    print("\n" + "=" * 70)
    print(f"负样本拒答率 : {neg_refused}/{neg_n} = {neg_refused/max(1,neg_n):.1%}  "
          f"(其中闸门直拒 {neg_gate})   【越高越好，修前为 0%】")
    print(f"正样本误拒率 : {pos_refused}/{pos_n} = {pos_refused/max(1,pos_n):.1%}  "
          f"(其中闸门误拒 {pos_gate})   【越低越好】")
    print("=" * 70)
    print("解读：负样本拒答率越高=防幻觉越好；正样本误拒率越低=没误伤正常提问。")
    system._cleanup()


if __name__ == "__main__":
    main()
