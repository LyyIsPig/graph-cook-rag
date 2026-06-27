"""
检索与生成评测指标（P2）。
全部为纯函数，不依赖运行中的系统，便于单元测试与离线复算。

支持策略间公平对比的关键设计：
- 不同检索策略返回的 Document 形态不一（向量/BM25/图KV 返回菜谱文档；
  图RAG 返回子图/路径描述）。统一用 names_from_doc 把每个文档映射成
  「命中的菜谱名列表」，再按检索顺序展平成 ranked_names，最后算指标。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def normalize_name(name: str) -> str:
    """菜谱名归一化：去首尾空白； None 安全。"""
    if not name:
        return ""
    return str(name).strip()


def names_from_doc(doc, universe: Set[str]) -> List[str]:
    """
    从一个检索结果 Document 中提取命中的菜谱名（按出现顺序、去重）。
    - 优先用 metadata['recipe_name'] / metadata['name']（向量/BM25/图KV 路径）；
    - 其次在 page_content 里扫描 universe 中的菜谱名（图RAG 子图描述常把菜谱名写进正文）。
    """
    md = getattr(doc, "metadata", {}) or {}
    matched: List[str] = []

    primary = normalize_name(md.get("recipe_name") or md.get("name") or "")
    if primary and primary in universe:
        matched.append(primary)

    content = getattr(doc, "page_content", "") or ""
    if content:
        # 按在正文中出现的位置排序，保证确定性（否则 set 遍历顺序不稳）
        found = []
        for name in universe:
            n = normalize_name(name)
            if n and n != primary and n in content:
                found.append((content.find(n), n))
        found.sort(key=lambda x: x[0])
        for _, n in found:
            if n not in matched:
                matched.append(n)
    return matched


def ranked_names_from_docs(docs: Sequence, universe: Set[str]) -> List[str]:
    """把检索结果序列展平成「按相关度排序的菜谱名列表」（跨文档去重，保持首次出现顺序）。"""
    ranked: List[str] = []
    seen: Set[str] = set()
    for doc in docs:
        for name in names_from_doc(doc, universe):
            if name not in seen:
                seen.add(name)
                ranked.append(name)
    return ranked


# ===== 检索指标（二元相关性） =====

def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@k = |relevant ∩ retrieved[:k]| / |relevant|。relevant 为空时返回 0。"""
    rel = {normalize_name(r) for r in relevant if normalize_name(r)}
    if not rel:
        return 0.0
    topk = {normalize_name(n) for n in ranked[:k]}
    return len(rel & topk) / len(rel)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """MRR = 1 / 第一个相关项的排名；无相关项则为 0。"""
    rel = {normalize_name(r) for r in relevant if normalize_name(r)}
    for i, name in enumerate(ranked, start=1):
        if normalize_name(name) in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """NDCG@k（二元相关性）：DCG@k / IDCG@k。"""
    rel = {normalize_name(r) for r in relevant if normalize_name(r)}
    if not rel:
        return 0.0

    def dcg(gains: Sequence[int]) -> float:
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    gains = [1 if normalize_name(n) in rel else 0 for n in ranked[:k]]
    dcg_val = dcg(gains)
    ideal_hits = min(len(rel), k)
    idcg_val = dcg([1] * ideal_hits)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def aggregate(values: Iterable[float]) -> dict:
    """对一组样本指标做均值统计。"""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": 0.0}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"n": n, "mean": mean, "std": math.sqrt(var)}


# ===== 生成指标（LLM-as-judge，判定 0/1/2 分） =====

def parse_judge_score(text: str, default: float = 0.0) -> float:
    """从 LLM 判官回复里抽 0/1/2 分（兼容 JSON 或纯文本）。"""
    import re
    if not text:
        return default
    m = re.search(r"\b([012])\b", text)
    return float(m.group(1)) if m else default
