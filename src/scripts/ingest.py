"""CLI: download → clean → chunk → index all artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow both `python -m src.scripts.ingest` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs import get_config

_CONFIG = get_config()

# ============================================================
# Configured paths and settings
# ============================================================

STEPS = ["load", "clean", "chunk", "chroma", "bm25", "graph"]

_PROCESSED = _CONFIG.paths.data_processed


def _skip(label: str, path: Path) -> None:
    print(f"[skip] {label} — {path} already exists")


# ============================================================
# LOAD
# ============================================================

def step_load(sample_size: int | None, resume: bool) -> None:
    from src.ingestion import progress as p
    from src.ingestion.loader import load_documents, load_relationships

    raw_path = _PROCESSED / "raw_docs.json"
    rels_path = _PROCESSED / "relationships.json"

    if resume and raw_path.exists() and rels_path.exists():
        _skip("load", raw_path)
        p.step_skip("load", f"{raw_path} already exists")
        return

    _PROCESSED.mkdir(parents=True, exist_ok=True)

    p.step_start("load", total=sample_size or 0)

    docs = load_documents(
        config=_CONFIG,
        sample_size=sample_size,
    )

    raw_path.write_text(
        json.dumps(
            [
                {
                    "page_content": d.page_content,
                    "metadata": d.metadata,
                }
                for d in docs
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[load] {len(docs)} docs → {raw_path}")

    doc_ids = (
        {
            d.metadata.get("doc_id", "")
            for d in docs
            if d.metadata.get("doc_id")
        }
        if sample_size
        else None
    )

    rels = load_relationships(
        config=_CONFIG,
        doc_ids=doc_ids,
    )

    rels_path.write_text(
        json.dumps(
            rels,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[load] {len(rels)} relationships → {rels_path}")

    p.step_done(
        "load",
        count=len(docs),
        message=f"{len(docs)} docs, {len(rels)} relationships",
    )


# ============================================================
# CLEAN
# ============================================================

def step_clean(resume: bool, sample_size: int | None) -> None:
    from langchain_core.documents import Document
    from src.ingestion import progress as p
    from src.ingestion.cleaner import clean_documents

    raw_path = _PROCESSED / "raw_docs.json"
    out_path = _PROCESSED / "cleaned_docs.json"

    if resume and out_path.exists():
        _skip("clean", out_path)
        p.step_skip("clean", f"{out_path} already exists")
        return

    data = json.loads(
        raw_path.read_text(encoding="utf-8")
    )

    docs = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"],
        )
        for d in data
    ]

    if sample_size:
        docs = docs[:sample_size]

    p.step_start(
        "clean",
        total=len(docs),
    )

    cleaned = clean_documents(
        docs,
        workers=_CONFIG.chunking.clean_workers,
    )

    out_path.write_text(
        json.dumps(
            [
                {
                    "page_content": d.page_content,
                    "metadata": d.metadata,
                }
                for d in cleaned
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[clean] {len(cleaned)} docs → {out_path}")

    p.step_done(
        "clean",
        count=len(cleaned),
        message=f"{len(cleaned)} docs",
    )


# ============================================================
# CHUNK
# ============================================================

def step_chunk(resume: bool, sample_size: int | None) -> None:
    from langchain_core.documents import Document
    from src.ingestion import progress as p
    from src.ingestion.chunker import chunk_documents

    cleaned_path = _PROCESSED / "cleaned_docs.json"
    out_path = _PROCESSED / "chunks.json"

    if resume and out_path.exists():
        _skip("chunk", out_path)
        p.step_skip("chunk", f"{out_path} already exists")
        return

    data = json.loads(
        cleaned_path.read_text(encoding="utf-8")
    )

    docs = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"],
        )
        for d in data
    ]

    p.step_start(
        "chunk",
        total=len(docs),
    )

    chunks = chunk_documents(
        docs,
        config=_CONFIG.chunking,
    )

    out_path.write_text(
        json.dumps(
            [
                {
                    "page_content": c.page_content,
                    "metadata": c.metadata,
                }
                for c in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[chunk] {len(chunks)} chunks → {out_path}")

    p.step_done(
        "chunk",
        count=len(chunks),
        message=f"{len(chunks)} chunks",
    )


# ============================================================
# CHROMA
# ============================================================

def step_chroma(resume: bool, segment_size: int | None) -> None:
    from langchain_core.documents import Document

    from src.ingestion import progress as p
    from src.indexing.chroma_store import (
        get_store,
        upsert_documents,
    )
    from src.indexing.embeddings import VLLMEmbedding

    chunks_path = _PROCESSED / "chunks.json"

    progress_path = _CONFIG.paths.chroma_progress
    segment_size = segment_size or _CONFIG.ingestion.segment_size

    data = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    chunks = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"],
        )
        for d in data
    ]

    total = len(chunks)

    start_offset = 0

    if resume and progress_path.exists():
        prog = json.loads(
            progress_path.read_text(encoding="utf-8")
        )

        previous_total = prog.get("total")
        if previous_total is not None and previous_total != total:
            raise ValueError(
                "Chroma progress does not match chunks.json "
                f"(progress total={previous_total}, current total={total}); "
                "rerun without --resume or restart from the chroma step"
            )

        start_offset = prog.get(
            "last_offset",
            0,
        )

        if start_offset >= total:
            print(
                f"[skip] chroma — all {total} chunks already indexed"
            )

            p.step_skip(
                "chroma",
                f"all {total} chunks already indexed",
            )

            return

        print(
            f"[resume] chroma — "
            f"continuing from chunk {start_offset}/{total}"
        )

    else:
        # A fresh pipeline may use a smaller sample than a previous run.
        # Remove stale vectors so the Chroma collection matches chunks.json.
        store = get_store(_CONFIG)
        store._client.delete_collection(_CONFIG.chroma.collection_name)
        store = get_store(_CONFIG)
        progress_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        progress_path.write_text(
            json.dumps(
                {
                    "last_offset": 0,
                    "total": total,
                }
            ),
            encoding="utf-8",
        )

    p.step_start(
        "chroma",
        total=total,
    )

    store = get_store(_CONFIG)
    emb_fn = VLLMEmbedding(config=_CONFIG)

    upsert_documents(
        store=store,
        chunks=chunks,
        embeddings=emb_fn,
        start_offset=start_offset,
        progress_path=progress_path,
        segment_size=segment_size,
        config=_CONFIG,
    )

    n = total - start_offset

    print(
        f"[chroma] indexed {n} chunks "
        f"(total {total}) → "
        f"{_CONFIG.chroma.host}:{_CONFIG.chroma.port}"
    )

    p.step_done(
        "chroma",
        count=n,
        message=f"{n} chunks indexed",
    )


# ============================================================
# BM25
# ============================================================

def step_bm25(resume: bool) -> None:
    from langchain_core.documents import Document
    from src.ingestion import progress as p
    from src.indexing.bm25_index import build_bm25_index

    chunks_path = _PROCESSED / "chunks.json"
    out_path = _CONFIG.paths.bm25_index

    if resume and out_path.exists():
        _skip("bm25", out_path)
        p.step_skip(
            "bm25",
            f"{out_path} already exists",
        )
        return

    data = json.loads(
        chunks_path.read_text(encoding="utf-8")
    )

    chunks = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"],
        )
        for d in data
    ]

    p.step_start(
        "bm25",
        total=len(chunks),
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_bm25_index(
        chunks,
        save_path=out_path,
        config=_CONFIG,
    )

    print(
        f"[bm25] {len(chunks)} chunks → {out_path}"
    )

    p.step_done(
        "bm25",
        count=len(chunks),
        message=f"{len(chunks)} chunks",
    )


# ============================================================
# GRAPH
# ============================================================

def step_graph(resume: bool) -> None:
    from src.ingestion import progress as p
    from src.retrieval.graph import build_graph

    relationships_path = (
        _PROCESSED / "relationships.json"
    )

    out_path = _CONFIG.paths.graph_index

    if resume and out_path.exists():
        _skip("graph", out_path)

        p.step_skip(
            "graph",
            f"{out_path} already exists",
        )

        return

    rels = json.loads(
        relationships_path.read_text(
            encoding="utf-8"
        )
    )

    p.step_start(
        "graph",
        total=len(rels),
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = build_graph(
        rels,
        save_path=str(out_path),
    )

    print(
        f"[graph] "
        f"{graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges → "
        f"{out_path}"
    )

    p.step_done(
        "graph",
        count=graph.number_of_nodes(),
        message=(
            f"{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        ),
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    sample_size: int | None = None,
    resume: bool = False,
    from_step: str | None = None,
    segment_size: int | None = None,
) -> None:

    from src.ingestion import progress as p

    # Apply the configured development sample when the caller does not
    # explicitly choose a size. This prevents a default run from loading the
    # complete dataset.
    sample_size = (
        sample_size
        if sample_size is not None
        else _CONFIG.dataset.sample_size
    )

    if from_step is not None:
        if from_step not in STEPS:
            raise ValueError(
                f"Unknown step: {from_step}"
            )

        force_from = STEPS.index(from_step)

    else:
        force_from = len(STEPS)

    def should_resume(step_name: str) -> bool:
        if from_step is None:
            return resume

        idx = STEPS.index(step_name)

        if idx < force_from:
            return True

        if idx == force_from:
            return False

        return resume

 

    try:
        step_load(
            sample_size,
            resume=should_resume("load"),
        )

        step_clean(
            resume=should_resume("clean"),
            sample_size=sample_size,
        )

        step_chunk(
            resume=should_resume("chunk"),
            sample_size=sample_size,
        )

        step_chroma(
            resume=should_resume("chroma"),
            segment_size=segment_size,
        )

        step_bm25(
            resume=should_resume("bm25"),
        )

        step_graph(
            resume=should_resume("graph"),
        )

        p.pipeline_done()

        print("\nIngestion complete.")

    except Exception as exc:
        p.pipeline_error(str(exc))
        raise


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Ingest Vietnamese legal documents "
            "using the project configuration"
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=_CONFIG.dataset.sample_size or 0,
        help=(
            "Number of docs to load "
            f"(default: {_CONFIG.dataset.sample_size or 'full'}; 0 = full dataset)"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose output files already exist",
    )

    parser.add_argument(
        "--from-step",
        choices=STEPS,
        default=None,
        dest="from_step",
        help=(
            "Force-restart pipeline from this step "
            "(implies --resume for earlier steps)"
        ),
    )

    parser.add_argument(
        "--segment-size",
        type=int,
        default=None,
        dest="segment_size",
        help="Chunks per Chroma upsert segment",
    )

    args = parser.parse_args()

    if args.sample < 0:
        parser.error("--sample must be >= 0")
    if args.segment_size is not None and args.segment_size < 100:
        parser.error("--segment-size must be >= 100")

    run_pipeline(
        sample_size=(
            args.sample
            if args.sample > 0
            else None
        ),
        resume=args.resume,
        from_step=args.from_step,
        segment_size=args.segment_size,
    )


if __name__ == "__main__":
    main()