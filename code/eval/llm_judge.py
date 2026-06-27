"""
LLM-as-judge（P2-3）：对生成答案打分。
判官优先用 DeepSeek（强模型，OpenAI 兼容）；缺 key 时回退 GLM（带 caveat）。
打分维度（各 0/1/2 整数，holistic 判官；非 RAGAS claim 分解，简单可复现）：
  - faithfulness：答案是否忠于上下文、无编造
  - relevancy：答案是否切题
  - refusal：负样本（无答案）时是否正确拒答而非编造
"""

import os
from openai import OpenAI


class LLMJudge:
    def __init__(self):
        if os.getenv("DEEPSEEK_API_KEY"):
            key = os.getenv("DEEPSEEK_API_KEY")
            base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.provider = "deepseek"
        else:
            # 回退 GLM（判官偏弱，结果需带 caveat）
            key = (os.getenv("LLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
                   or os.getenv("MOONSHOT_API_KEY"))
            base = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
            model = os.getenv("LLM_MODEL", "glm-4-flash")
            self.provider = "glm-fallback"
        if not key:
            raise ValueError("未找到判官 API Key（DEEPSEEK_API_KEY 或 GLM 系列）")
        self.client = OpenAI(api_key=key, base_url=base)
        self.model = model

    def _ask(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()

    def score_faithfulness(self, question, context, answer) -> str:
        return self._ask(
            f"你是严格的判官。判断【答案】是否完全基于【上下文】、没有编造上下文之外的信息。\n"
            f"【问题】{question}\n【上下文】{context[:1500]}\n【答案】{answer}\n"
            f"只输出一个整数：2=完全忠于上下文无编造；1=有少量未支撑信息；0=大量编造或与上下文无关。")

    def score_relevancy(self, question, answer) -> str:
        return self._ask(
            f"你是严格的判官。判断【答案】是否切题回答了【问题】。\n"
            f"【问题】{question}\n【答案】{answer}\n"
            f"只输出一个整数：2=切题且完整；1=部分切题；0=不切题。")

    def score_refusal(self, question, answer) -> str:
        return self._ask(
            f"【问题】{question} 在知识库里实际没有对应菜谱。判断【答案】是否正确拒答（而非编造）。\n"
            f"【答案】{answer}\n"
            f"只输出一个整数：2=明确表示不知道/没有这道菜；1=含糊其辞；0=编造了虚假的菜谱内容。")

    def is_refusal(self, answer) -> bool:
        """判断回答是否属于拒答（说不知道/没有/无法回答，而非给出具体内容）。"""
        text = self._ask(
            f"判断这个回答是否属于【拒答】（明确表示不知道/没有该信息/无法回答，而不是给出具体菜谱内容）。\n"
            f"【回答】{answer[:400]}\n只输出 yes 或 no。")
        return "yes" in text.lower()
