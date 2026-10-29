"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs import get_config
from src.api.routers import benchmark, ingest, query
from src.api.schemas import HealthResponse


_retrieval_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources when the API starts."""
    global _retrieval_error

    # Warm up retrieval resources
    try:
        from src.tools.retrieval_tools import (
            _get_bm25,
            _get_graph,
            _get_store,
        )

        _get_store()
        _get_bm25()
        _get_graph()
        _retrieval_error = None

    except Exception as e:
        _retrieval_error = str(e)
        print(f"Warning: failed to warm up retrieval resources: {_retrieval_error}")

    yield


def _indexes_ready() -> bool:
    config = get_config()
    if not (
        config.paths.bm25_index.exists()
        and config.paths.graph_index.exists()
        and config.paths.chroma_progress.exists()
    ):
        return False

    try:
        from src.tools.retrieval_tools import _get_store

        store = _get_store()
        return store._collection.count() > 0
    except Exception:
        return False


def create_app() -> FastAPI:
    app = FastAPI(
        title="AIO Agentic RAG — Vietnamese Legal Benchmark",
        description="Benchmarking 5 RAG strategies on Vietnamese legal documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(query.router)
    app.include_router(benchmark.router)
    app.include_router(ingest.router)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        indexes_ready = _indexes_ready()
        detail = _retrieval_error
        if not indexes_ready and detail is None:
            detail = "Indexes are not ready — run the ingestion pipeline."
        return HealthResponse(
            status="ok" if indexes_ready and detail is None else "degraded",
            indexes_ready=indexes_ready and detail is None,
            detail=detail,
        )

    return app


app = create_app()