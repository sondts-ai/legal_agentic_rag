# Legal Agentic RAG - Demo

Demo hỏi đáp và retrieval trên văn bản pháp luật Việt Nam. Hệ thống gồm:

- **Dense retrieval** với `intfloat/multilingual-e5-small` qua vLLM.
- **BM25** cho tìm kiếm từ khóa.
- **Hybrid search**, knowledge graph và CrossEncoder reranker.
- **Agentic routing** chọn graph/hybrid/reranker theo loại câu hỏi.
- **FastAPI** backend, **Gradio** UI và benchmark retrieval.

> Đây là bản demo. Dataset mặc định trong `configs/config.yaml` chỉ dùng một
> mẫu nhỏ để khởi động nhanh. Kết quả benchmark chỉ có ý nghĩa trong phạm vi
> gold set và index đã được tạo cùng một phiên bản dữ liệu.

## 1. Kiến trúc và các cổng

```text
Browser ──> Gradio UI :7860 ──> FastAPI :8001
                                  ├──> Chroma :8000
                                  ├──> vLLM embedding :8080
                                  └──> BM25 / graph / CrossEncoder
```

| Thành phần | URL mặc định | Vai trò |
|---|---:|---|
| Gradio | http://localhost:7860 | Giao diện demo |
| FastAPI | http://localhost:8001 | API và Swagger |
| Chroma | http://localhost:8000 | Vector database |
| vLLM | http://localhost:8080/v1 | Embedding API |

## 2. Yêu cầu

- Docker và Docker Compose plugin.
- NVIDIA driver, NVIDIA Container Toolkit và GPU nếu chạy vLLM/reranker bằng
  GPU.
- API key cho LLM sinh câu trả lời, ví dụ `OPENROUTER_API_KEY`.
- Dữ liệu/index demo đã có trong `data/`, `chroma/` và `data/indexes/`.

Cài dependency local nếu không dùng Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,ui,ingest,dev]"
```

## 3. Chạy demo bằng Docker Compose

Tạo file `.env` ở thư mục gốc:

```dotenv
OPENROUTER_API_KEY=your-key
LLM_MODEL=your-openrouter-model
HF_TOKEN=
```

Khởi động toàn bộ stack:

```bash
docker compose up --build
```

Sau khi các container khởi động:

- Mở UI: http://localhost:7860
- Swagger: http://localhost:8001/docs
- Kiểm tra API: http://localhost:8001/health

Xem log riêng từng service:

```bash
docker compose logs -f vllm
docker compose logs -f api
docker compose logs -f ui
```

Dừng stack:

```bash
docker compose down
```

`docker-compose.yml` dùng đúng các tham số demo hiện tại:

```bash
chroma run --host 0.0.0.0 --port 8000

uvicorn src.api.app:app --host 0.0.0.0 --port 8001

vllm serve intfloat/multilingual-e5-small \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 512 \
  --max-num-seqs 1

python UI/app.py
```

Trong container, API dùng `CHROMA_HOST=chroma` và
`EMBEDDING_BASE_URL=http://vllm:8080/v1`; UI dùng
`API_BASE_URL=http://api:8001`. Vì vậy không cần sửa code khi chạy Compose.

## 4. Chạy từng service thủ công

Nếu không dùng Docker, mở bốn terminal.

### Terminal 1 - Chroma

```bash
chroma run --host 0.0.0.0 --port 8000
```

Khi dùng Chroma server, đặt:

```bash
export CHROMA_MODE=http
export CHROMA_HOST=127.0.0.1
export CHROMA_PORT=8000
```

### Terminal 2 - vLLM

```bash
vllm serve intfloat/multilingual-e5-small \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 512 \
  --max-num-seqs 1
```

### Terminal 3 - FastAPI

```bash
export EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1
export RERANKER_DEVICE=cuda
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

### Terminal 4 - Gradio

```bash
API_BASE_URL=http://127.0.0.1:8001 python UI/app.py
```

## 5. Index và ingestion

Pipeline mặc định dùng `sample_size: 200` trong config. Với bản demo, nên giữ
nguyên index có sẵn. Nếu cần tạo lại index từ đầu:

```bash
python -m src.scripts.ingest --sample 200
```

Không dùng `--resume` khi đổi số lượng mẫu hoặc đổi dataset. Lệnh này tạo lại
processed documents, Chroma vectors, BM25 index và graph theo cùng một lần
ingestion.

Kiểm tra index:

```bash
curl http://localhost:8001/health
```

Kết quả sẵn sàng có dạng:

```json
{"status":"ok","indexes_ready":true,"detail":null}
```

## 6. Thử query

Trong UI, chọn `agentic`, `k = 5`, rồi thử:

> Những văn bản nào sửa đổi, bổ sung Luật Đầu tư công và hiện còn hiệu lực?

Để kiểm tra trực tiếp reranker:

> Luật Đầu tư công quy định những nội dung nào về quản lý dự án đầu tư?

## 7. Benchmark

Chạy benchmark retrieval:

```bash
python -m src.scripts.run_benchmark --strategy all
```

Chạy một strategy:

```bash
python -m src.scripts.run_benchmark --strategy agentic --sample 3
```

Kết quả được lưu tại `results/benchmark_*.csv`.

Benchmark đo `recall@K`, `ndcg@K` và latency retrieval. Lần khởi tạo model/index
được warm-up trước khi đo; nếu bật RAGAS, latency sinh câu trả lời được ghi
riêng ở `avg_generation_latency_ms`.

### Diễn giải kết quả demo

`naive` có thể cao điểm hơn `agentic` trên gold set nhỏ. Đây chưa phải bằng
chứng naive tốt hơn trong production: agentic có thêm bước routing/reranking,
còn gold set demo yêu cầu các chunk cụ thể và benchmark đang đánh giá top-K.
Khi thay đổi dataset, chunking, candidate count hoặc device, cần rebuild index
và ghi lại cấu hình để so sánh công bằng.

## 8. Cấu hình quan trọng

Các thiết lập chính nằm trong `configs/config.yaml` và có thể override bằng
environment variables:

| Biến | Mặc định demo | Ý nghĩa |
|---|---|---|
| `CHROMA_MODE` | `persistent` | `persistent` local hoặc `http` server |
| `CHROMA_HOST` | `localhost` | Host Chroma |
| `CHROMA_PORT` | `8000` | Port Chroma |
| `EMBEDDING_BASE_URL` | `http://localhost:8080/v1` | vLLM OpenAI API |
| `RERANKER_DEVICE` | `cuda` | `cuda` hoặc `cpu` |
| `RERANKER_CANDIDATE_K` | `20` | Số candidate trước reranking |
| `API_BASE_URL` | `http://localhost:8001` | UI gọi FastAPI |
| `OPENROUTER_API_KEY` | - | API key sinh câu trả lời |
| `LLM_MODEL` | - | Model generation |

## 9. Xử lý lỗi nhanh

- **UI báo API unreachable:** kiểm tra `docker compose ps` và
  `docker compose logs api`.
- **API báo indexes not ready:** kiểm tra `data/indexes/`, `chroma/` và chạy
  ingestion nếu index chưa tồn tại.
- **vLLM không khởi động:** kiểm tra NVIDIA Container Toolkit, VRAM và log
  `docker compose logs vllm`.
- **Query timeout:** chờ vLLM tải model lần đầu, sau đó restart API nếu cần;
  UI đã cấu hình timeout query 600 giây.
- **Reranker lỗi CUDA:** tạm thời đặt `RERANKER_DEVICE=cpu` rồi restart API.
