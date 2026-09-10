from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 100
    separators: list[str] = None


class Config:
    def __init__(self):
        self.chunking = ChunkingConfig(
            separators=["\n\n", "\n", " ", ""]
        )


def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nĐiều ", "\nKhoản ", "\nĐiểm ", "\n\n", "\n", " "],
        keep_separator=True,
    )


def chunk_documents(docs, config):
    splitter = build_splitter()

    chunks = []

    for doc in docs:
        splits = splitter.split_documents([doc])

        for i, chunk in enumerate(splits):
            chunk.metadata["chunk_index"] = i
            chunks.append(chunk)

    return chunks