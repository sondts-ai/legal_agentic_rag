from __future__ import annotations

import pickle
from pathlib import Path

from configs import Config, get_config
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def _tokenizer(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, chunks: list[Document]):
        self.chunks=chunks
        corpus = [_tokenizer(chunk.page_content) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list[Document]:
        query_tokens = _tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        return [self.chunks[i] for i in top_indices]

    def save(self, path: str | Path | None = None, config: Config | None = None):
        app_config = config or get_config()
        output_path = Path(path or app_config.paths.bm25_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        config: Config | None = None,
    ) -> "BM25Index":
        app_config = config or get_config()
        input_path = Path(path or app_config.paths.bm25_index)

        with input_path.open("rb") as f:
            return pickle.load(f)


def build_bm25_index(
    chunks: list[Document],
    save_path: str | Path | None = None,
    config: Config | None = None,
) -> BM25Index:
    index = BM25Index(chunks)
    index.save(save_path, config=config)
    return index
