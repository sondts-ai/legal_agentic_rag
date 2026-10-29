from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from configs import get_config

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
DEFAULT_GOLD_SET_PATH: Path | None = None

def load_gold_set(path: str | Path | None = None) -> list[dict]:
    """Load gold set from file, falling back to the built-in sample."""

    p = Path(path) if path else get_config().evaluation.gold_set_path

    if p.exists():
        data = json.loads(
            p.read_text(encoding="utf-8")
        )
        if not isinstance(data, list):
            raise ValueError(f"Gold set must be a JSON list: {p}")
        return data

    return SAMPLE_GOLD_SET


def validate_gold_set(entries: list[dict]) -> None:
    """Reject malformed entries before they can produce misleading metrics."""
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Gold entry {index} must be an object")
        question = entry.get("question")
        expected = entry.get("expected_doc_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Gold entry {index} has no non-empty question")
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"Gold entry {index} must contain a non-empty expected_doc_ids list"
            )
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in expected):
            raise ValueError(f"Gold entry {index} contains an invalid document id")


def save_gold_set(
    entries: list[dict],
    path: str | Path | None = None,
) -> None:
    p = Path(path) if path else get_config().evaluation.gold_set_path

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