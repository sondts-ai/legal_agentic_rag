from __future__ import annotations

from configs import RetrievalConfig
from langchain_chroma import Chroma
from langchain_core.documents import Document


def dense_search(
    store: Chroma,
    query: str,
    k: int | None = None,
    metadata_filter: dict | None = None,
    config: RetrievalConfig | None = None,
) -> list[Document]:
    if k is None:
        from configs import get_config

        k = get_config().retrieval.k

    if k <= 0:
        raise ValueError("Dense retrieval k must be greater than zero")

    kwargs: dict = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter

    return store.similarity_search(query, **kwargs)