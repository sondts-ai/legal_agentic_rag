"""Graph-based multi-hop retrieval using the relationships NetworkX graph."""
from __future__ import annotations

import pickle
from pathlib import Path

from configs import RetrievalConfig
import networkx as nx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.retrieval.dense import dense_search


_EDGE_TYPES = {
    "Căn cứ",
    "Dẫn chiếu",
    "Sửa đổi, bổ sung",
    "Thay thế",
    "Bãi bỏ",
    "Văn bản hết hiệu lực",
    "Văn bản quy định hết hiệu lực 1 phần",
    "Văn bản căn cứ",
    "Văn bản dẫn chiếu",
    "Văn bản sửa đổi",
    "Văn bản được sửa đổi",
    "Văn bản bổ sung",
    "Văn bản được bổ sung",
    "Văn bản hết hiệu lực",
    "Văn bản quy định hết hiệu lực",
    "Văn bản HD, QĐ chi tiết",
    "Văn bản được HD, QĐ chi tiết",
    "Văn bản liên quan khác",
}


def build_graph(relationships: list[dict], save_path: str | None = None) -> nx.DiGraph:
    """Build a directed graph from the HF relationships config rows."""
    G = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", ""))
        dst = str(row.get("other_doc_id", ""))
        rel_type = row.get("relationship", "")
        if src and dst:
            G.add_edge(src, dst, rel_type=rel_type)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(G, f)
    return G


def load_graph(path: str) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_search(
    store: Chroma,
    graph: nx.DiGraph,
    query: str,
    k: int | None = None,
    initial_k: int | None = None,
    max_hops: int | None = None,
    edge_types: set[str] | None = None,
    trace: list | None = None,
    config: RetrievalConfig | None = None,
) -> list[Document]:
    """
    1. Dense-retrieve `initial_k` seed documents.
    2. Traverse the relationship graph up to `max_hops` hops.
    3. Dense-retrieve chunks for each reachable doc_id.
    4. Return the top-k merged results.
    """
    if config is None:
        from configs import get_config

        config = get_config().retrieval

    k = config.k if k is None else k
    initial_k = min(k, config.dense_k) if initial_k is None else initial_k
    max_hops = config.graph_max_hops if max_hops is None else max_hops
    allowed = edge_types or (set(config.graph_edge_types) | _EDGE_TYPES)

    if k <= 0 or initial_k <= 0:
        raise ValueError("Graph retrieval k values must be greater than zero")
    if max_hops < 0:
        raise ValueError("Graph retrieval max_hops cannot be negative")

    seed_docs = dense_search(store, query, k=initial_k)
    seed_ids = {d.metadata.get("doc_id", "") for d in seed_docs}

    if trace is not None:
        sample = ", ".join(list(seed_ids)[:3])
        suffix = "…" if len(seed_ids) > 3 else ""
        trace.append({"icon": "🔍", "label": "Dense seed search", "detail": f"{len(seed_docs)} seed nodes: {sample}{suffix}"})

    reachable: set[str] = set(seed_ids)
    frontier = set(seed_ids)
    for hop in range(max_hops):
        next_frontier: set[str] = set()
        edge_types_seen: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            for _, neighbor, data in graph.out_edges(node, data=True):
                rel = data.get("rel_type", "")
                if rel in allowed and neighbor not in reachable:
                    next_frontier.add(neighbor)
                    edge_types_seen.add(rel)
            for neighbor, _, data in graph.in_edges(node, data=True):
                rel = data.get("rel_type", "")
                if rel in allowed and neighbor not in reachable:
                    next_frontier.add(neighbor)
                    edge_types_seen.add(rel)
        reachable |= next_frontier
        frontier = next_frontier
        if trace is not None:
            if next_frontier:
                rels = ", ".join(list(edge_types_seen)[:2])
                suffix = "…" if len(edge_types_seen) > 2 else ""
                trace.append({"icon": "🕸️", "label": f"Graph hop {hop + 1}", "detail": f"+{len(next_frontier)} nodes via [{rels}{suffix}]"})
            else:
                trace.append({"icon": "🕸️", "label": f"Graph hop {hop + 1}", "detail": "no new nodes — frontier exhausted"})
        if not frontier:
            break

    all_docs: list[Document] = list(seed_docs)
    extra_ids = reachable - seed_ids
    if extra_ids:
        id_filter = {"doc_id": {"$in": list(extra_ids)}}
        extra_docs = dense_search(store, query, k=k * 2, metadata_filter=id_filter)
        all_docs.extend(extra_docs)
        if trace is not None:
            trace.append({"icon": "📥", "label": "Fetch expanded nodes", "detail": f"{len(extra_docs)} docs from {len(extra_ids)} expanded nodes"})

    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in all_docs:
        key = f"{doc.metadata.get('doc_id', '')}_{doc.metadata.get('chunk_index', 0)}"
        if key not in seen:
            seen.add(key)
            deduped.append(doc)

    result = deduped[:k]
    if trace is not None:
        trace.append({"icon": "📄", "label": "Graph result", "detail": f"{len(result)} docs from {len(reachable)} total reachable nodes"})

    return result