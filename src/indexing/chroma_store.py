import json
from logging import config
import chromadb
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.indexing.embeddings import VLLMEmbedding

def get_store() -> Chroma:
    """Return a Chroma instance backed by the HTTP server."""

    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
    )

    embedding_function = VLLMEmbedding(
        api_key=os.getenv("HF_TOKEN"),
        base_url="http://localhost:8080/v1",
        model="intfloat/multilingual-e5-small",
    )

    return Chroma(
        client=client,
        collection_name="legal_documents",
        embedding_function=embedding_function,
    )

def upsert_documents(
    store: Chroma,
    chunks: list[Document],
    embeddings: VLLMEmbedding,
    segment_size: int = 10000,
    batch_size: int = 100,
    upsert_workers: int = 4,
    start_offset: int = 0,
    progress_path: str | None = None,
):
    total = len(chunks)

    for seg_start in range(start_offset, total, segment_size):
        seg_end = min(seg_start + segment_size, total)
        segment = chunks[seg_start:seg_end]

        texts = [chunk.page_content for chunk in segment]

        ids = [
            f"{chunk.metadata['doc_id']}_chunk_{chunk.metadata['chunk_index']}"
            for chunk in segment
        ]

        metadatas = [chunk.metadata for chunk in segment]

        print(f"Embedding [{seg_start}:{seg_end}]...")
        vectors = embeddings.embed_documents(texts)

        def _upsert(i: int):
            end = min(i + batch_size, len(segment))

            store._collection.upsert(
                ids=ids[i:end],
                embeddings=vectors[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

            return end - i

        offsets = range(0, len(segment), batch_size)

        with tqdm(
            total=len(segment),
            desc=f"Upserting [{seg_start}:{seg_end}]",
            unit="chunk",
        ) as bar:

            with ThreadPoolExecutor(
                max_workers=upsert_workers
            ) as executor:

                futures = [
                    executor.submit(_upsert, i)
                    for i in offsets
                ]

                for future in as_completed(futures):
                    bar.update(future.result())

        if progress_path:
            Path(progress_path).write_text(
                json.dumps({
                    "last_offset": seg_end,
                    "total": total,
                })
            )


def build_store_from_chunks(
    chunks: list[Document],
    progress_path: str = "progress.json",
) -> Chroma:

    start_offset = 0

    if Path(progress_path).exists():
        progress = json.loads(
            Path(progress_path).read_text()
        )

        start_offset = progress["last_offset"]
        print(f"Resuming from chunk {start_offset}")

    embeddings = VLLMEmbedding(
        api_key=os.getenv("HF_TOKEN"),
        base_url="http://localhost:8080/v1",
        model="intfloat/multilingual-e5-small",
    )

    # Kết nối Chroma Server
    store = get_store()

    upsert_documents(
        store=store,
        chunks=chunks,
        embeddings=embeddings,
        start_offset=start_offset,
        progress_path=progress_path,
    )

    return store