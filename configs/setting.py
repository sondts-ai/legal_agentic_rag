from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


_CONFIG_PATH = Path(__file__).with_name("config.yaml")
_PROJECT_ROOT = _CONFIG_PATH.parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    metadata_config: str
    content_config: str
    relationships_config: str
    sample_size: int | None


@dataclass(frozen=True)
class PathsConfig:
    data_raw: Path
    data_processed: Path
    bm25_index: Path
    graph_index: Path
    chroma_progress: Path


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    clean_workers: int
    separators: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    batch_size: int
    workers: int


@dataclass(frozen=True)
class RetrievalConfig:
    k: int
    bm25_k: int
    dense_k: int
    rrf_k: int
    reranker_model: str
    reranker_candidate_k: int
    reranker_device: str
    graph_max_hops: int
    graph_edge_types: tuple[str, ...]


@dataclass(frozen=True)
class ChromaConfig:
    mode: Literal["persistent", "http"]
    path: Path
    distance_metric: str
    host: str
    port: int
    collection_name: str


@dataclass(frozen=True)
class IngestionConfig:
    segment_size: int
    batch_size: int
    upsert_workers: int


@dataclass(frozen=True)
class GenerationConfig:
    provider: Literal["openrouter"]
    model: str | None
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    openrouter_api_key: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    gold_set_path: Path
    output_dir: Path
    recall_k: int
    ndcg_k: int
    query_types: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    chroma: ChromaConfig
    ingestion: IngestionConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    hf_token: str | None = None


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _load_yaml() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {_CONFIG_PATH}")

    return raw


def _build_config(raw: dict) -> Config:
    dataset_raw = raw["dataset"]
    dataset_configs = dataset_raw["configs"]
    dataset = DatasetConfig(
        name=dataset_raw["name"],
        metadata_config=dataset_configs["metadata"],
        content_config=dataset_configs["content"],
        relationships_config=dataset_configs["relationships"],
        sample_size=dataset_raw.get("sample_size"),
    )

    paths_raw = raw["paths"]
    paths = PathsConfig(
        data_raw=_resolve_path(paths_raw["data_raw"]),
        data_processed=_resolve_path(paths_raw["data_processed"]),
        bm25_index=_resolve_path(paths_raw["bm25_index"]),
        graph_index=_resolve_path(paths_raw["graph_index"]),
        chroma_progress=_resolve_path(paths_raw["chroma_progress"]),
    )

    chunking_raw = raw["chunking"]
    chunking = ChunkingConfig(
        chunk_size=int(chunking_raw["chunk_size"]),
        chunk_overlap=int(chunking_raw["chunk_overlap"]),
        clean_workers=int(
            os.getenv("CLEAN_WORKERS", chunking_raw["clean_workers"])
        ),
        separators=tuple(chunking_raw["separators"]),
    )

    embedding_raw = raw["embedding"]
    embedding = EmbeddingConfig(
        provider=embedding_raw["provider"],
        model=os.getenv("EMBEDDING_MODEL", embedding_raw["model"]),
        base_url=os.getenv("EMBEDDING_BASE_URL", embedding_raw["base_url"]),
        batch_size=int(embedding_raw["batch_size"]),
        workers=int(embedding_raw["workers"]),
    )

    retrieval_raw = raw["retrieval"]
    retrieval = RetrievalConfig(
        k=int(retrieval_raw["k"]),
        bm25_k=int(retrieval_raw["bm25_k"]),
        dense_k=int(retrieval_raw["dense_k"]),
        rrf_k=int(retrieval_raw["rrf_k"]),
        reranker_model=retrieval_raw["reranker_model"],
        reranker_candidate_k=int(
            os.getenv(
                "RERANKER_CANDIDATE_K",
                retrieval_raw.get("reranker_candidate_k", 20),
            )
        ),
        reranker_device=os.getenv(
            "RERANKER_DEVICE",
            retrieval_raw.get("reranker_device", "cpu"),
        ),
        graph_max_hops=int(retrieval_raw["graph_max_hops"]),
        graph_edge_types=tuple(retrieval_raw["graph_edge_types"]),
    )

    chroma_raw = raw["chroma"]
    chroma = ChromaConfig(
        mode=os.getenv("CHROMA_MODE", chroma_raw.get("mode", "persistent")),
        path=_resolve_path(
            os.getenv("CHROMA_PATH", chroma_raw.get("path", "chroma"))
        ),
        distance_metric=chroma_raw["distance_metric"],
        host=os.getenv("CHROMA_HOST", chroma_raw["host"]),
        port=int(os.getenv("CHROMA_PORT", chroma_raw["port"])),
        collection_name=chroma_raw["collection_name"],
    )

    ingestion_raw = raw["ingestion"]
    ingestion = IngestionConfig(
        segment_size=int(ingestion_raw["segment_size"]),
        batch_size=int(ingestion_raw["batch_size"]),
        upsert_workers=int(ingestion_raw["upsert_workers"]),
    )

    generation_raw = raw["generation"]
    provider = generation_raw["provider"]
    if provider != "openrouter":
        raise ValueError(
            f"Unsupported generation provider={provider!r}; "
            "this project is configured for openrouter only"
        )

    generation = GenerationConfig(
        provider="openrouter",
        model=os.getenv("LLM_MODEL", generation_raw.get("model")),
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            generation_raw["base_url"],
        ),
        temperature=float(
            os.getenv("LLM_TEMPERATURE", generation_raw["temperature"])
        ),
        max_tokens=(
            int(os.environ["LLM_MAX_TOKENS"])
            if os.getenv("LLM_MAX_TOKENS")
            else generation_raw.get("max_tokens")
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    evaluation_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        gold_set_path=_resolve_path(evaluation_raw["gold_set_path"]),
        output_dir=_resolve_path(evaluation_raw["output_dir"]),
        recall_k=int(evaluation_raw["recall_k"]),
        ndcg_k=int(evaluation_raw["ndcg_k"]),
        query_types=tuple(evaluation_raw["query_types"]),
    )

    return Config(
        dataset=dataset,
        paths=paths,
        chunking=chunking,
        embedding=embedding,
        retrieval=retrieval,
        chroma=chroma,
        ingestion=ingestion,
        generation=generation,
        evaluation=evaluation,
        hf_token=os.getenv("HF_TOKEN"),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    load_dotenv()
    return _build_config(_load_yaml())


config = get_config()
