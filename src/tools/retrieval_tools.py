from langchain_chroma import Chroma
from langchain_core.tools import tool

from src.indexing.chroma_store import get_store
from src.indexing.bm25_index import BM25Index
from langchain_core.documents import Document
from src.retrieval.dense import dense_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.graph import load_graph, graph_search
import json
from src.llm import get_llm
from functools import lru_cache

BM25_INDEX_PATH=  "./data/indexes/bm25.pkl"

@lru_cache(maxsize=1)
def _get_store() -> Chroma:
    return get_store()

load_indexbm25=BM25Index.load(
        BM25_INDEX_PATH
    )

@lru_cache(maxsize=1)
def _get_bm25() -> BM25Index:
    return load_indexbm25

@lru_cache(maxsize=1)
def _get_graph():
    return load_graph(
        "data/graph.pkl"
    )

@tool
def dense_search_tool(query:str,k:int):
    """Search documents using dense retrieval."""
    docs=dense_search(store=_get_store(),
        query=query,
        k=k,)

    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

@tool
def bm25_search_tool(query:str,k:int):
    """Search documents using BM25 retrieval."""
    index = BM25Index.load(
        BM25_INDEX_PATH
    )

    # Truyền index vào hàm bm25_search
    docs = bm25_search(
        index=index,
        query=query,
        k=k,
    )

    return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

@tool
def hybrid_search_tool(query:str,k:int, bm25_k:int=10,
    dense_k: int = 10,
    metadata_filter_json: str = ""):
    """Search documents using hybrid dense and BM25 retrieval."""
    flt = json.loads(metadata_filter_json) if metadata_filter_json else None
    docs=hybrid_search(store=_get_store(),index=_get_bm25(),query=query,k=k,bm25_k=bm25_k,dense_k=dense_k,metadata_filter_json=flt)
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

@tool
def rerank_tool(
    query: str,
    docs: list[dict],
    k: int = 5,
) -> list[dict]:
    """Rerank documents using CrossEncoder."""
    documents = [
        Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"],
        )
        for doc in docs
    ]

    reranked_docs = rerank(
        query=query,
        docs=documents,
        k=k,
    )

    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in reranked_docs
    ]

@tool
def graph_traverse_tool(
    query: str,
    k: int = 5,
    initial_k: int = 3,
    max_hops: int = 2,
) -> list[dict]:
    """Traverse the graph to find relevant documents."""

    # Lấy Chroma store
    store = get_store()

    # Load graph
    graph = load_graph(
        "data/graph.pkl"
    )

    # Graph search
    docs = graph_search(
        store=store,
        graph=graph,
        query=query,
        k=k,
        initial_k=initial_k,
        max_hops=max_hops,
    )

    # Document -> dict
    return [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

@tool
def generate_answer_tool(
    query: str,
    docs: list[dict],
) -> str:
    """Generate an answer based on the provided documents and query."""
    context_parts=[]
    for i,doc in enumerate(docs,1):
        doc_id=doc["metadata"].get(
            "doc_id",
            "unknown",
        )
        context_parts.append(
            f"[{i}] (doc_id: {doc_id})\n"
            f"{doc['page_content']}"
        )
    context = "\n\n".join(context_parts)
    llm = get_llm()

    # Gửi câu hỏi + context cho LLM
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý pháp lý. "
                "Hãy trả lời dựa trên tài liệu được cung cấp. "
                "Nếu không có thông tin trong tài liệu, hãy nói rõ."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Tài liệu:\n\n{context}\n\n"
                f"Câu hỏi: {query}"
            ),
        },
    ]

    response = llm.invoke(messages)

    return response.content
