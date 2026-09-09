"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import benchmark, ingest, query
from src.api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources when the API starts."""

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

    except Exception as e:
        print(f"Warning: failed to warm up retrieval resources: {e}")

    yield


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
        return HealthResponse(
            status="ok",
            indexes_ready=True,
        )

    return app


app = create_app()