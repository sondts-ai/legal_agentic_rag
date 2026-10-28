from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from configs import Config, get_config


class VLLMEmbedding:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        workers: int | None = None,
        config: Config | None = None,
    ):
        app_config = config or get_config()
        embedding_config = app_config.embedding

        base_url = base_url or embedding_config.base_url
        model = model or embedding_config.model
        batch_size = batch_size or embedding_config.batch_size
        workers = workers or embedding_config.workers

        if batch_size <= 0:
            raise ValueError("Embedding batch_size must be greater than zero")
        if workers <= 0:
            raise ValueError("Embedding workers must be greater than zero")

        self.client = OpenAI(
            api_key=api_key or app_config.hf_token,
            base_url=base_url,
        )
        self.model = model
        self.batch_size = batch_size
        self.workers = workers

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

        return [data.embedding for data in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        results = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._embed_batch, batch): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()

        return [
            embedding
            for batch in results
            for embedding in batch
        ]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[query],
            model=self.model,
        )

        return response.data[0].embedding
