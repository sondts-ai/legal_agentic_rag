"""LangChain @tool wrappers for the ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool


def create_ingestion_tools(
    dataset_name: str,
    data_raw: str,
    data_processed: str = "data/processed",
    token: str | None = None,
    clean_workers: int = 1,
    chunk_config=None,
    bm25_path: str = "data/indexes/bm25.pkl",
    graph_path: str = "data/indexes/graph.pkl",
    chroma_progress_path: str = "progress.json",
):
    """Create ingestion tools without global config."""

    data_processed = Path(data_processed)

    def load_docs(path: Path) -> list[Document]:
        data = json.loads(path.read_text(encoding="utf-8"))

        return [
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]

    def save_docs(docs: list[Document], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @tool
    def load_dataset_tool(sample_size: int = 0) -> str:
        """Load dataset and save raw documents."""

        from src.ingestion.loader import load_documents

        docs = load_documents(
            dataset_name=dataset_name,
            data_raw=data_raw,
            token=token,
            sample_size=sample_size or None,
        )

        path = data_processed / "raw_docs.json"
        save_docs(docs, path)

        return f"Loaded {len(docs)} documents → {path}"

    @tool
    def load_relationships_tool() -> str:
        """Load document relationships and save them."""

        from src.ingestion.loader import load_relationships

        raw_path = data_processed / "raw_docs.json"
        doc_ids = None

        if raw_path.exists():
            data = json.loads(raw_path.read_text(encoding="utf-8"))

            doc_ids = {
                item.get("metadata", {}).get("doc_id")
                for item in data
                if item.get("metadata", {}).get("doc_id")
            } or None

        relationships = load_relationships(
            dataset_name=dataset_name,
            data_raw=data_raw,
            token=token,
            doc_ids=doc_ids,
        )

        path = data_processed / "relationships.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(relationships, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return f"Loaded {len(relationships)} relationships → {path}"

    @tool
    def clean_docs_tool() -> str:
        """Clean raw documents."""

        from src.ingestion.cleaner import clean_documents

        raw_path = data_processed / "raw_docs.json"
        docs = load_docs(raw_path)

        cleaned = clean_documents(
            docs,
            workers=clean_workers,
        )

        path = data_processed / "cleaned_docs.json"
        save_docs(cleaned, path)

        return f"Cleaned {len(cleaned)} documents → {path}"

    @tool
    def chunk_docs_tool() -> str:
        """Chunk cleaned documents."""

        from src.ingestion.chunker import chunk_documents

        cleaned_path = data_processed / "cleaned_docs.json"
        docs = load_docs(cleaned_path)

        chunks = chunk_documents(
            docs,
            config=chunk_config,
        )

        path = data_processed / "chunks.json"
        save_docs(chunks, path)

        return f"Created {len(chunks)} chunks → {path}"

    @tool
    def build_chroma_tool() -> str:
        """Build Chroma vector store from chunks."""

        from src.indexing.chroma_store import build_store_from_chunks

        chunks_path = data_processed / "chunks.json"
        chunks = load_docs(chunks_path)

        build_store_from_chunks(
            chunks,
            progress_path=chroma_progress_path,
        )

        return f"Indexed {len(chunks)} chunks into Chroma"

    @tool
    def build_bm25_tool() -> str:
        """Build and save BM25 index."""

        from src.indexing.bm25_index import BM25Index

        chunks_path = data_processed / "chunks.json"
        chunks = load_docs(chunks_path)

        index = BM25Index(chunks)
        index.save(bm25_path)

        return f"Built BM25 index over {len(chunks)} chunks → {bm25_path}"

    @tool
    def build_graph_tool() -> str:
        """Build and save relationship graph."""

        from src.retrieval.graph import build_graph

        path = data_processed / "relationships.json"

        relationships = json.loads(
            path.read_text(encoding="utf-8")
        )

        graph = build_graph(
            relationships,
            save_path=graph_path,
        )

        return (
            f"Built graph: "
            f"{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges → {graph_path}"
        )

    return [
        load_dataset_tool,
        load_relationships_tool,
        clean_docs_tool,
        chunk_docs_tool,
        build_chroma_tool,
        build_bm25_tool,
        build_graph_tool,
    ]

