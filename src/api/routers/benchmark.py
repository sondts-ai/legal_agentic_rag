from __future__ import annotations

from fastapi import APIRouter, HTTPException

from configs import get_config
from src.api.schemas import BenchmarkRequest, BenchmarkResponse

router = APIRouter()

@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    config = get_config()
    try:
        from src.evaluation.benchmark import run_benchmark as _run
        n = request.sample_n if request.sample_n > 0 else None
        results= _run(
            strategies=list(request.strategies),
            recall_k=request.recall_k,
            ndcg_k=request.ndcg_k,
            sample_n=n,
            use_ragas=request.use_ragas,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    csv_files = sorted(
        config.evaluation.output_dir.glob("benchmark_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    csv_path = str(csv_files[0]) if csv_files else ""

    return BenchmarkResponse(results=results, csv_path=csv_path)