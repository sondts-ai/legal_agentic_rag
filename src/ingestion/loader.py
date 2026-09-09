"""Load Vietnamese legal documents from a Hugging Face dataset."""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import snapshot_download
from langchain_core.documents import Document
from tqdm import tqdm


_CONTENT_FEATURES = Features({
    "id": Value("large_string"),
    "content_html": Value("large_string"),
})


def _ensure_local_dataset(
    dataset_name: str, data_raw: str, token: str | None = None
) -> str:
    """Download dataset locally if parquet files are not available."""
    local = Path(data_raw) / dataset_name.replace("/", "--")

    if not any(local.rglob("*.parquet")):
        print(f"Downloading {dataset_name} → {local}")
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(local),
            token=token,
        )

    return str(local)


def _iter_content(source: str, sample_size: int | None = None) -> list[dict]:
    """Load content rows from the local dataset."""
    ds = load_dataset(
        source, "content", split="data", features=_CONTENT_FEATURES
    )

    total = sample_size if sample_size else len(ds)
    rows = []

    with tqdm(total=total, desc="Loading content", unit="doc") as bar:
        for i, row in enumerate(ds):
            if sample_size is not None and i >= sample_size:
                break
            rows.append(dict(row))
            bar.update(1)

    return rows


def _load_metadata_lookup(
    source: str, doc_ids: set[str] | None = None
) -> dict[str, dict]:
    """Load metadata and create an id → metadata lookup."""
    ds: Dataset = load_dataset(source, "metadata", split="data")
    remaining = set(doc_ids) if doc_ids is not None else None
    result: dict[str, dict] = {}

    for row in tqdm(ds, desc="Loading metadata", unit="doc"):
        row_id = str(row["id"])

        if remaining is None:
            result[row_id] = dict(row)
        elif row_id in remaining:
            result[row_id] = dict(row)
            remaining.discard(row_id)
            if not remaining:
                break

    return result


def _load_relationships(
    source: str, doc_ids: set[str] | None = None
) -> list[dict]:
    """Load relationships between documents."""
    ds: Dataset = load_dataset(source, "relationships", split="data")

    return [
        dict(row)
        for row in tqdm(ds, desc="Loading relationships", unit="rel")
        if (
            doc_ids is None
            or str(row.get("doc_id", "")) in doc_ids
            or str(row.get("other_doc_id", "")) in doc_ids
        )
    ]


def load_documents(
    dataset_name: str,
    data_raw: str,
    token: str | None = None,
    sample_size: int | None = None,
) -> list[Document]:
    """Load content + metadata and return LangChain Documents."""
    source = _ensure_local_dataset(dataset_name, data_raw, token)

    content_rows = _iter_content(source, sample_size)
    doc_ids = {str(row.get("id", "")) for row in content_rows}
    metadata_lookup = _load_metadata_lookup(source, doc_ids)

    documents = []

    for row in content_rows:
        doc_id = str(row.get("id", ""))
        meta = metadata_lookup.get(doc_id, {})

        documents.append(
            Document(
                page_content=row.get("content_html", ""),
                metadata={
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "doc_type": meta.get("loai_van_ban", ""),
                    "authority": meta.get("co_quan_ban_hanh", ""),
                    "issue_date": meta.get("ngay_ban_hanh", ""),
                    "effective_date": meta.get("ngay_co_hieu_luc", ""),
                    "expiry_date": meta.get("ngay_het_hieu_luc", ""),
                    "sector": meta.get("linh_vuc", ""),
                    "status": meta.get("tinh_trang_hieu_luc", ""),
                },
            )
        )

    return documents


def load_relationships(
    dataset_name: str,
    data_raw: str,
    token: str | None = None,
    doc_ids: set[str] | None = None,
) -> list[dict]:
    """Load relationships between documents."""
    source = _ensure_local_dataset(dataset_name, data_raw, token)
    return _load_relationships(source, doc_ids)