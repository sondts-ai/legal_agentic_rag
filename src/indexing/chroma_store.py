import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
from configs import Config, get_config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from src.indexing.embeddings import VLLMEmbedding


def get_store(config: Config | None = None) -> Chroma:
    """Return a Chroma instance backed by the configured persistence mode."""
    app_config = config or get_config()
    chroma_config = app_config.chroma
    embedding_config = app_config.embedding

    if chroma_config.mode == "persistent":
        chroma_config.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_config.path))
    elif chroma_config.mode == "http":
        client = chromadb.HttpClient(
            host=chroma_config.host,
            port=chroma_config.port,
        )
    else:
        raise ValueError(
            f"Unsupported Chroma mode={chroma_config.mode!r}; "
            "expected 'persistent' or 'http'"
        )

    embedding_function = VLLMEmbedding(
        api_key=app_config.hf_token,
        base_url=embedding_config.base_url,
        model=embedding_config.model,
        batch_size=embedding_config.batch_size,
        workers=embedding_config.workers,
    )

    return Chroma(
        client=client,
        collection_name=chroma_config.collection_name,
        collection_metadata={"hnsw:space": chroma_config.distance_metric},
        embedding_function=embedding_function,
    )


def upsert_documents(
    store: Chroma,
    chunks: list[Document],
    embeddings: VLLMEmbedding,
    segment_size: int | None = None,
    batch_size: int | None = None,
    upsert_workers: int | None = None,
    start_offset: int = 0,
    progress_path: str | Path | None = None,
    config: Config | None = None,
) -> None:
    app_config = config or get_config()
    ingestion_config = app_config.ingestion

    segment_size = segment_size or ingestion_config.segment_size
    batch_size = batch_size or ingestion_config.batch_size
    upsert_workers = upsert_workers or ingestion_config.upsert_workers

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

        def _upsert(i: int) -> int:
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
            progress = Path(progress_path)
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(
                json.dumps({
                    "last_offset": seg_end,
                    "total": total,
                }),
                encoding="utf-8",
            )


def build_store_from_chunks(
    chunks: list[Document],
    progress_path: str | Path | None = None,
    config: Config | None = None,
) -> Chroma:
    app_config = config or get_config()
    progress = Path(progress_path or app_config.paths.chroma_progress)

    start_offset = 0

    if progress.exists():
        progress_state = json.loads(progress.read_text(encoding="utf-8"))
        start_offset = progress_state["last_offset"]
        print(f"Resuming from chunk {start_offset}")

    embedding_config = app_config.embedding
    embeddings = VLLMEmbedding(
        api_key=app_config.hf_token,
        base_url=embedding_config.base_url,
        model=embedding_config.model,
        batch_size=embedding_config.batch_size,
        workers=embedding_config.workers,
    )

    store = get_store(app_config)

    upsert_documents(
        store=store,
        chunks=chunks,
        embeddings=embeddings,
        start_offset=start_offset,
        progress_path=progress,
        config=app_config,
    )

    return store
