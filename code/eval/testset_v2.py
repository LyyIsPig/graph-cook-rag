"""
golden 测试集 v2 数据结构 + 读写（P2-2）。
相比 v1 增加：capability（能力维度，用于切片报告）、negative、label_source、cypher（可审计）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List


@dataclass
class EvalQueryV2:
    id: str
    query: str
    capability: str                       # lookup | list | relation | negative | reasoning
    query_type: str                       # 细分子型
    relevant_recipe_names: List[str] = field(default_factory=list)
    negative: bool = False                # True=不可答（relevant 空），测拒答/防幻觉
    label_source: str = "cypher"          # cypher | human | llm_assisted
    cypher: str = ""                      # 标注用的 Cypher，可复跑审计
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQueryV2":
        return cls(
            id=d["id"], query=d["query"], capability=d.get("capability", ""),
            query_type=d.get("query_type", ""), relevant_recipe_names=d.get("relevant_recipe_names", []),
            negative=d.get("negative", False), label_source=d.get("label_source", "cypher"),
            cypher=d.get("cypher", ""), notes=d.get("notes", ""),
        )


def load_testset_v2(path) -> List[EvalQueryV2]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(EvalQueryV2.from_dict(json.loads(line)))
    return out


def save_testset_v2(items: List[EvalQueryV2], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
