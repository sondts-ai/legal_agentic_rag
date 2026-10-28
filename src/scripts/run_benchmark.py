"""CLI: run strategy benchmarks against the configured gold QA set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from configs import get_config


_CONFIG = get_config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG strategy benchmarks")
    parser.add_argument(
        "--strategy",
        choices=["naive", "hybrid", "reranker", "graph", "agentic", "all"],
        default="all",
        help="Which strategy to benchmark",
    )
    parser.add_argument("--sample", type=int, default=0, help="Number of gold queries (0 = all)")
    parser.add_argument(
        "--recall-k",
        type=int,
        default=_CONFIG.evaluation.recall_k,
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=_CONFIG.evaluation.ndcg_k,
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Generate LLM answers and run RAGAS metrics (faithfulness, relevancy, context precision)",
    )
    args = parser.parse_args()

    from src.evaluation.benchmark import run_benchmark, STRATEGIES

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]
    n = args.sample if args.sample > 0 else None

    run_benchmark(
        strategies=strategies,
        gold_path=str(_CONFIG.evaluation.gold_set_path),
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        sample_n=n,
        use_ragas=args.ragas,
    )


if __name__ == "__main__":
    main()