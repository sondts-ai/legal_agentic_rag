from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


class VLLMEmbedding:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 32,
        workers: int = 4,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.batch_size = batch_size
        self.workers = workers

    def _embed_batch(self, texts: list[str]):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

        return [data.embedding for data in response.data]

    def embed_documents(self, texts: list[str]):
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

    def embed_query(self, query: str):
        response = self.client.embeddings.create(
            input=[query],
            model=self.model,
        )

        return response.data[0].embedding

