import json
from pathlib import Path

from configs import get_config
from src.evaluation.gold_set import SAMPLE_GOLD_SET, save_gold_set


def main() -> None:
    out_path = get_config().evaluation.gold_set_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_gold_set(SAMPLE_GOLD_SET, out_path)

    print(f"Saved {len(SAMPLE_GOLD_SET)} sample queries to {out_path}")

    data = json.loads(Path(out_path).read_text(encoding="utf-8"))

    print(f"Entries: {len(data)}")

    from collections import Counter

    types = Counter(e["query_type"] for e in data)

    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()