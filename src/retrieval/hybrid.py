from __future__ import annotations

from configs import RetrievalConfig
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import APIConnectionError
from src.indexing.bm25_index import BM25Index
from src.retrieval.bm25 import bm25_search
from src.retrieval.dense import dense_search


def _rrf_score(rank: int, rrf_k: int) -> float:
    if rrf_k <= 0:
        raise ValueError("RRF k must be greater than zero")
    return 1.0 / (rrf_k + rank + 1)


def hybrid_search(
    store: Chroma,
    bm25_index: BM25Index,
    query: str,
    k: int | None = None,
    bm25_k: int | None = None,
    dense_k: int | None = None,
    rrf_k: int | None = None,
    metadata_filter: dict | None = None,
    trace: list | None = None,
    config: RetrievalConfig | None = None,
) -> list[Document]:
    if config is None:
        from configs import get_config

        config = get_config().retrieval

    k = config.k if k is None else k
    bm25_k = config.bm25_k if bm25_k is None else bm25_k
    dense_k = config.dense_k if dense_k is None else dense_k
    rrf_k = config.rrf_k if rrf_k is None else rrf_k

    if k <= 0 or bm25_k <= 0 or dense_k <= 0:
        raise ValueError("Hybrid retrieval values k must be greater than zero")

    bm25_results = bm25_search(bm25_index, query, k=bm25_k)
    dense_error: str | None = None
    try:
        dense_results = dense_search(
            store,
            query,
            k=dense_k,
            metadata_filter=metadata_filter,
        )
    except APIConnectionError as exc:
        dense_results = []
        dense_error = (
            "Dense search unavailable; using BM25-only results. "
            f"Embedding service error: {exc}"
        )
    if trace is not None:
        trace.append({"icon": "📖", "label": "BM25 search", "detail": f"{len(bm25_results)} candidates (keyword)"})
        trace.append({
            "icon": "🔍",
            "label": "Dense search",
            "detail": dense_error or f"{len(dense_results)} candidates (semantic)",
        })

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_results):
        key = f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, rrf_k)
        doc_map[key] = doc

    for rank, doc in enumerate(dense_results):
        key = f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index', 0)}"
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank, rrf_k)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = [doc_map[key] for key, _ in ranked[:k]]

    if trace is not None:
        trace.append({"icon": "🔀", "label": "RRF fusion", "detail": f"{len(scores)} unique candidates → top {len(result)} docs"})

    return result