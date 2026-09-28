from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

QueryType = Literal["factual", "multi_hop", "temporal", "reasoning"]

SAMPLE_GOLD_SET: list[dict] = [
    # Factual
    {
        "question": "Luật Đầu tư công",
        "expected_doc_ids": [
            "134516_68",
            "134516_74",
            "134516_55",
        ],
        "answer_ground_truth": "",
        "query_type": "factual",
    },
    {
        "question": "Kho bạc nhà nước",
        "expected_doc_ids": [
            "134516_34",
            "134516_9",
            "134516_33",
        ],
        "answer_ground_truth": "",
        "query_type": "factual",
    },

    # Multi-hop
    {
        "question": "dự án đầu tư",
        "expected_doc_ids": [
            "134516_55",
            "134516_49",
            "134516_62",
            "134516_68",
            "134516_16",
        ],
        "answer_ground_truth": "",
        "query_type": "multi_hop",
    },
]
DEFAULT_GOLD_SET_PATH = "data/gold_set.json"

def load_gold_set(path: str | None = None) -> list[dict]:
    """Load gold set from file, falling back to the built-in sample."""
    
    p = Path(path or DEFAULT_GOLD_SET_PATH)

    if p.exists():
        return json.loads(
            p.read_text(encoding="utf-8")
        )

    return SAMPLE_GOLD_SET


def save_gold_set(
    entries: list[dict],
    path: str | None = None,
) -> None:
    
    p = Path(path or DEFAULT_GOLD_SET_PATH)

    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    p.write_text(
        json.dumps(
            entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

def filter_by_type(entries: list[dict], query_type: QueryType) -> list[dict]:
    return [e for e in entries if e.get("query_type") == query_type]


def sample_gold_set(entries: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(entries, min(n, len(entries)))