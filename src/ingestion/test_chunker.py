from dataclasses import dataclass

from langchain_core.documents import Document

from chunker import chunk_documents


@dataclass
class ChunkingConfig:
    chunk_size: int = 50
    chunk_overlap: int = 10
    separators: list[str] = None


@dataclass
class Config:
    chunking: ChunkingConfig = None


config = Config(
    chunking=ChunkingConfig(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
    )
)


docs = [
    Document(
        page_content=(
            "Đây là một văn bản pháp luật rất dài để test chunking. "
            "Văn bản này được dùng để kiểm tra chương trình "
            "có thể chia văn bản thành nhiều chunk hay không."
        ),
        metadata={"source": "test.txt"},
    )
]


chunks = chunk_documents(docs, config)


print(f"Tổng số chunks: {len(chunks)}")

for chunk in chunks:
    print(f"Chunk index: {chunk.metadata['chunk_index']}")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("---")