
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from configs import get_config
from src.evaluation.gold_set import load_gold_set, validate_gold_set
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_ragas_metrics,
    ndcg_at_k,
    recall_at_k,
)


def aggregate_metrics_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Run RAGAS and prefix keys to distinguish from retrieval metrics."""
    raw = compute_ragas_metrics(
        questions,
        answers,
        contexts,
        ground_truths,
    )

    if "ragas_error" in raw:
        print(f"  RAGAS error: {raw['ragas_error']}")
        return {}

    return {
        f"ragas_{k}": v
        for k, v in raw.items()
        if isinstance(v, float)
    }


def _retrieve_naive(
    query: str,
    k: int = 5,
) -> list[Document]:
    from openai import APIConnectionError

    from src.retrieval.dense import dense_search
    from src.tools.retrieval_tools import _get_bm25, _get_store

    try:
        return dense_search(_get_store(), query, k=k)
    except APIConnectionError:
        return _get_bm25().search(query, k=k)


def _retrieve_hybrid(
    query: str,
    k: int = 5,
) -> list[Document]:
    from src.retrieval.hybrid import hybrid_search
    from src.tools.retrieval_tools import _get_bm25, _get_store

    return hybrid_search(
        _get_store(),
        _get_bm25(),
        query,
        k=k,
    )


def _retrieve_reranker(
    query: str,
    k: int = 5,
) -> list[Document]:
    from src.retrieval.hybrid import hybrid_search
    from configs import get_config
    from src.tools.retrieval_tools import _get_bm25, _get_store

    candidate_k = max(get_config().retrieval.reranker_candidate_k, k)
    candidates = hybrid_search(
        _get_store(),
        _get_bm25(),
        query,
        k=candidate_k,
    )

    try:
        from src.retrieval.reranker import rerank
    except ModuleNotFoundError:
        return candidates[:k]
    return rerank(query, candidates, k=k)


def _retrieve_graph(
    query: str,
    k: int = 5,
) -> list[Document]:
    from openai import APIConnectionError

    from src.tools.retrieval_tools import _get_bm25, _get_graph, _get_store
    try:
        from src.retrieval.graph import graph_search
        return graph_search(_get_store(), _get_graph(), query, k=k)
    except (ModuleNotFoundError, APIConnectionError):
        return _get_bm25().search(query, k=k)


def _retrieve_agentic(
    query: str,
    k: int = 5,
) -> list[Document]:
    """Classify the query and execute the same route as the API agent."""
    from openai import APIConnectionError

    from src.retrieval.hybrid import hybrid_search
    from configs import get_config
    from src.tools.retrieval_tools import (
        _get_bm25,
        _get_graph,
        _get_store,
    )

    store = _get_store()
    bm25 = _get_bm25()
    q_lower = query.lower()

    if any(
        keyword in q_lower
        for keyword in ("sửa đổi", "thay thế", "bãi bỏ", "tham chiếu", "được quy định tại")
    ):
        try:
            from src.retrieval.graph import graph_search
            return graph_search(
                store,
                _get_graph(),
                query,
                k=k,
                initial_k=min(3, k),
                max_hops=2,
            )
        except (ModuleNotFoundError, APIConnectionError):
            return bm25.search(query, k=k)

    if any(
        keyword in q_lower
        for keyword in ("còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm", "từ ngày")
    ):
        return hybrid_search(store, bm25, query, k=k)

    candidates = hybrid_search(
        store,
        bm25,
        query,
        k=max(get_config().retrieval.reranker_candidate_k, k),
    )
    try:
        from src.retrieval.reranker import rerank
    except ModuleNotFoundError:
        return candidates[:k]

    docs = rerank(query, candidates, k=k)
    if len(docs) >= 2:
        return docs

    try:
        from src.retrieval.dense import dense_search
        return dense_search(store, query, k=k)
    except APIConnectionError:
        return docs


STRATEGIES: dict[str, Callable] = {
    "naive": _retrieve_naive,
    "hybrid": _retrieve_hybrid,
    "reranker": _retrieve_reranker,
    "graph": _retrieve_graph,
    "agentic": _retrieve_agentic,
}


def _generate_answer(
    question: str,
    docs: list[Document],
) -> str:
    """Generate an answer from retrieved docs using the configured LLM."""
    from src.tools.retrieval_tools import generate_answer_tool

    return generate_answer_tool.invoke(
        {
            "query": question,
            "docs": [
                {
                    "page_content": d.page_content,
                    "metadata": d.metadata,
                }
                for d in docs
            ],
        }
    )


def run_benchmark(
    strategies: list[str] | None = None,
    gold_path: str | None = None,
    recall_k: int | None = None,
    ndcg_k: int | None = None,
    output_dir: str | None = None,
    sample_n: int | None = None,
    use_ragas: bool = False,
) -> dict[str, dict]:
    """
    Run each strategy against the gold set and return per-strategy metric summaries.

    Also writes a timestamped CSV to output_dir.

    When use_ragas=True, generates LLM answers for each query and runs RAGAS metrics.
    """

    # Gold set path is fixed for this evaluation
    evaluation = get_config().evaluation
    _gold_path = gold_path or evaluation.gold_set_path
    _recall_k = recall_k if recall_k is not None else evaluation.recall_k
    _ndcg_k = ndcg_k if ndcg_k is not None else evaluation.ndcg_k
    _output_dir = output_dir or evaluation.output_dir

    entries = load_gold_set(_gold_path)
    validate_gold_set(entries)

    if sample_n is not None:
        if sample_n < 1:
            raise ValueError("sample_n must be greater than zero")
        entries = entries[:sample_n]

    selected = strategies or list(STRATEGIES.keys())

    all_results: dict[str, dict] = {}
    rows: list[dict] = []

    for strategy_name in selected:
        if strategy_name not in STRATEGIES:
            print(
                f"Unknown strategy: {strategy_name}, skipping."
            )
            continue

        retrieve_fn = STRATEGIES[strategy_name]

        per_query: list[dict[str, float]] = []
        latencies: list[float] = []

        ragas_questions: list[str] = []
        ragas_answers: list[str] = []
        ragas_contexts: list[list[str]] = []
        ragas_ground_truths: list[str] = []
        generation_latencies: list[float] = []

        # Resource/model construction is not query latency. Warm the strategy
        # once so the first measured query does not include initialization.
        if entries:
            retrieve_fn(entries[0]["question"], k=max(_recall_k, _ndcg_k))

        for entry in entries:
            question = entry["question"]

            relevant_ids: list[str] = entry.get(
                "expected_doc_ids",
                [],
            )

            retrieval_start = time.perf_counter()

            docs = retrieve_fn(
                question,
                k=max(_recall_k, _ndcg_k),
            )

            retrieval_latency_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

            if use_ragas:
                generation_start = time.perf_counter()
                answer = _generate_answer(
                    question,
                    docs,
                )
                generation_latency_ms = (
                    time.perf_counter() - generation_start
                ) * 1000

            # Gold entries identify chunks as doc_id_chunk_index. Keeping the
            # chunk index preserves the distinction between multiple chunks
            # from the same legal document.
            retrieved_ids = list(
                dict.fromkeys(
                    (
                        f"{d.metadata['doc_id']}_{d.metadata.get('chunk_index', 0)}"
                    )
                    for d in docs
                    if d.metadata.get("doc_id")
                )
            )

            per_query.append(
                {
                    f"recall@{_recall_k}": recall_at_k(
                        retrieved_ids,
                        relevant_ids,
                        _recall_k,
                    ),
                    f"ndcg@{_ndcg_k}": ndcg_at_k(
                        retrieved_ids,
                        relevant_ids,
                        _ndcg_k,
                    ),
                }
            )

            latencies.append(retrieval_latency_ms)
            if use_ragas:
                generation_latencies.append(generation_latency_ms)

            if use_ragas:
                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(
                    [d.page_content for d in docs]
                )
                ragas_ground_truths.append(
                    entry.get(
                        "answer_ground_truth",
                        "",
                    )
                )

        summary = aggregate_metrics(per_query)

        summary["avg_latency_ms"] = (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        )
        if use_ragas:
            summary["avg_generation_latency_ms"] = (
                sum(generation_latencies) / len(generation_latencies)
                if generation_latencies
                else 0.0
            )

        if use_ragas:
            print(
                f"  Running RAGAS for "
                f"{strategy_name} "
                f"({len(ragas_questions)} queries)..."
            )

            ragas_scores = aggregate_metrics_ragas(
                ragas_questions,
                ragas_answers,
                ragas_contexts,
                ragas_ground_truths,
            )

            summary.update(ragas_scores)

        all_results[strategy_name] = summary

        rows.append(
            {
                "strategy": strategy_name,
                **summary,
            }
        )

    # Write CSV
    Path(_output_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        Path(_output_dir)
        / f"benchmark_{ts}.csv"
    )

    if rows:
        fieldnames = [
            "strategy"
        ] + [
            k
            for k in rows[0]
            if k != "strategy"
        ]

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    print(f"Results saved to {csv_path}")

    # Print table
    _print_table(all_results)

    return all_results


def _print_table(
    results: dict[str, dict],
) -> None:
    if not results:
        return

    strategies = list(results.keys())

    metrics = list(
        next(iter(results.values())).keys()
    )

    col_w = 18

    header = (
        f"{'Strategy':<15}"
        + "".join(
            f"{m:>{col_w}}"
            for m in metrics
        )
    )

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for strategy, scores in results.items():
        row = (
            f"{strategy:<15}"
            + "".join(
                f"{scores[m]:>{col_w}.4f}"
                for m in metrics
            )
        )

        print(row)

    print("=" * len(header) + "\n")
