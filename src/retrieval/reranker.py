from __future__ import annotations

from configs import RetrievalConfig
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_cached: dict[tuple[str, str], CrossEncoder] = {}

def _get_model(model_name: str, device: str) -> CrossEncoder:
    cache_key = (model_name, device)
    if cache_key not in _cached:
        _cached[cache_key] = CrossEncoder(model_name, device=device)
    return _cached[cache_key]

def rerank(
    query: str,
    docs: list[Document],
    k: int | None = None,
    model_name: str | None = None,
    config: RetrievalConfig | None = None,
) -> list[Document]:
    if not docs:
        return []

    if config is None:
        from configs import get_config

        config = get_config().retrieval

    k = config.k if k is None else k
    model_name = model_name or config.reranker_model

    if k <= 0:
        raise ValueError("Reranker k must be greater than zero")

    model = _get_model(model_name, config.reranker_device)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]