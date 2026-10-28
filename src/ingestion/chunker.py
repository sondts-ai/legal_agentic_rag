from __future__ import annotations

from configs import ChunkingConfig
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        keep_separator=True,
    )


def chunk_documents(
    docs: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    splitter = build_splitter(config)

    chunks: list[Document] = []

    for doc in docs:
        splits = splitter.split_documents([doc])

        for index, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = index
            chunks.append(chunk)

    return chunks