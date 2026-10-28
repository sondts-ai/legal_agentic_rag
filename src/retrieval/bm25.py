from __future__ import annotations

from configs import RetrievalConfig
from langchain_core.documents import Document
from src.indexing.bm25_index import BM25Index


def bm25_search(
    index: BM25Index,
    query: str,
    k: int | None = None,
    config: RetrievalConfig | None = None,
) -> list[Document]:
    if k is None:
        from configs import get_config

        k = get_config().retrieval.bm25_k

    if k <= 0:
        raise ValueError("BM25 retrieval k must be greater than zero")

    return index.search(query, k=k)