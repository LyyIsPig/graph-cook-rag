"""
golden 测试集读写（P2）。
每条记录：{id, query, query_type, relevant_recipe_names, notes}
relevant_recipe_names 直接来自 Neo4j 知识图谱，是 KG-grounded 真值，非 LLM 臆造。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List


@dataclass
class EvalQuery:
    id: str
    query: str
    query_type: str                         # single_recipe | ingredient | category | cuisine
    relevant_recipe_names: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalQuery":
        return cls(
            id=d["id"],
            query=d["query"],
            query_type=d.get("query_type", ""),
            relevant_recipe_names=d.get("relevant_recipe_names", []),
            notes=d.get("notes", ""),
        )


def load_testset(path) -> List[EvalQuery]:
    p = Path(path)
    if not p.exists():
        return []
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(EvalQuery.from_dict(json.loads(line)))
    return items


def save_testset(items: List[EvalQuery], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
