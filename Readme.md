# ⚖️ Legal Agentic RAG

Hệ thống **Agentic Retrieval-Augmented Generation (RAG)** chuyên sâu cho lĩnh vực pháp lý, kết hợp cơ chế Multi-Agent điều phối thông minh, tìm kiếm lai (**Hybrid Search**: Dense + Sparse BM25 + Knowledge Graph), xếp hạng lại (**Reranking**), bộ công cụ đánh giá chuẩn hóa (**Benchmark**) và giao diện người dùng trực quan với **Gradio**.

## 📌 Mục lục

* [Tính năng nổi bật](#-tính-năng-nổi-bật)
* [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
* [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
* [Yêu cầu hệ thống & Cài đặt](#-yêu-cầu-hệ-thống--cài-đặt)
* [Cấu hình hệ thống](#-cấu-hình-hệ-thống)
* [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
  * [1. Nạp và xử lý dữ liệu văn bản pháp luật](#1-nạp-và-xử-lý-dữ-liệu-văn-bản-pháp-luật)
  * [2. Khởi chạy Backend API (FastAPI)](#2-khởi-chạy-backend-api-fastapi)
  * [3. Khởi chạy Giao diện Gradio](#3-khởi-chạy-giao-diện-gradio)
  * [4. Đánh giá chất lượng (Evaluation & Benchmark)](#4-đánh-giá-chất-lượng-evaluation--benchmark)

## 🚀 Tính năng nổi bật

* **Kiến trúc Multi-Agent thông minh:**
  * `orchestrator.py`: Định tuyến truy vấn, phân tích câu hỏi người dùng và điều phối các agent thành phần.
  * `rag_agent.py`: Trích xuất tài liệu ngữ cảnh, kiểm tra độ liên quan và tổng hợp câu trả lời có trích dẫn điều luật chính xác.
  * `ingestion_agent.py`: Quản lý pipeline làm sạch, phân đoạn (chunking) theo cấu trúc điều khoản luật và xây dựng chỉ mục.
* **Chiến lược Tìm kiếm Đa tầng (Advanced Retrieval):**
  * **Dense Retrieval**: Tìm kiếm ngữ nghĩa dựa trên vector embedding (ChromaDB).
  * **Sparse Retrieval**: Tìm kiếm từ khóa chính xác dựa trên BM25.
  * **Graph Retrieval**: Khai thác liên kết tham chiếu giữa các văn bản, thông tư, nghị định và điều luật.
  * **Hybrid Search & Reranking**: Kết hợp đa nguồn tìm kiếm và sử dụng Cross-Encoder để chấm điểm, tái sắp xếp tài liệu có độ liên quan cao nhất.
* **Đánh giá & Benchmark toàn diện:**
  * Xây dựng tập dữ liệu vàng (Gold Set) và đo lường các chỉ số retrieval / generation: MRR, Hit@K, Precision, Recall, Faithfulness,...
* **Giao diện tương tác Gradio trực quan:**
  * Giao diện thân thiện giúp hỏi đáp pháp lý theo thời gian thực, xem luồng suy luận của Agent và các trích dẫn pháp luật nguồn.

## 🏗️ Kiến trúc hệ thống

```text
       ┌────────────────────────┐
       │   Gradio UI / Client   │
       └───────────┬────────────┘
                   │ (HTTP / REST)
       ┌───────────▼────────────┐
       │    FastAPI Gateway     │
       └───────────┬────────────┘
                   │
       ┌───────────▼────────────┐
       │  Orchestrator Agent    │
       └─────┬────────────┬─────┘
             │            │
 ┌───────────▼──┐      ┌──▼────────────┐
 │  RAG Agent   │      │Ingestion Agent│
 └───────┬──────┘      └──┬────────────┘
         │                │
┌────────▼────────────────▼────────────┐
│ Retrieval & Indexing Layer           │
│  - Dense: ChromaDB (Vector Store)    │
│  - Sparse: BM25 Index                │
│  - Graph: Structural References      │
│  - Reranker: Cross-Encoder           │
└──────────────────────────────────────┘
```

## 📂 Cấu trúc thư mục

```text
legal_agentic_rag/
├── UI/
│   ├── api_client.py        # Client kết nối FastAPI cho giao diện
│   └── app.py               # Giao diện người dùng xây dựng bằng Gradio
├── configs/
│   ├── __init__.py
│   ├── config.yaml          # File cấu hình mô hình, chunk size, tham số RAG
│   └── setting.py           # Quản lý cấu hình bằng Pydantic / dataclass
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Điều phối toàn bộ luồng tác vụ
│   │   ├── rag_agent.py         # Agent truy vấn và sinh phản hồi
│   │   └── ingestion_agent.py   # Agent nạp và lập chỉ mục văn bản
│   ├── api/
│   │   ├── app.py               # Entrypoint FastAPI backend
│   │   ├── schemas.py           # Định nghĩa Pydantic Request/Response
│   │   └── routers/
│   │       ├── query.py         # Endpoint xử lý câu hỏi pháp lý
│   │       ├── ingest.py        # Endpoint nạp tài liệu
│   │       └── benchmark.py     # Endpoint chạy đánh giá
│   ├── evaluation/
│   │   ├── benchmark.py         # Pipeline benchmark
│   │   ├── gold_set.py          # Quản lý ground-truth / golden set
│   │   └── metrics.py           # Tính toán các chỉ số đánh giá
│   ├── indexing/
│   │   ├── bm25_index.py        # Xây dựng và quản lý BM25 Index
│   │   ├── chroma_store.py      # Quản lý Chroma Vector Database
│   │   └── embeddings.py        # Embedding models wrapper
│   ├── ingestion/
│   │   ├── loader.py            # Đọc tài liệu (PDF, Word, TXT,...)
│   │   ├── cleaner.py           # Làm sạch và tiền xử lý văn bản luật
│   │   ├── chunker.py           # Chia đoạn theo cấu trúc chương/mục/điều
│   │   └── progress.py          # Theo dõi tiến độ ingest
│   ├── retrieval/
│   │   ├── dense.py             # Dense retriever
│   │   ├── bm25.py              # BM25 retriever
│   │   ├── graph.py             # Knowledge Graph retriever
│   │   ├── hybrid.py            # Kết hợp kết quả (Hybrid Search)
│   │   └── reranker.py          # Cross-Encoder Reranker
│   ├── tools/                   # Bộ tool tích hợp cho Agent
│   │   ├── eval_tools.py
│   │   ├── ingestion_tools.py
│   │   └── retrieval_tools.py
│   ├── scripts/                 # CLI Scripts tiện ích
│   │   ├── ingest.py            # Script chạy nạp dữ liệu thủ công
│   │   ├── build_eval_set.py    # Tạo tập test đánh giá
│   │   └── run_benchmark.py     # Script chạy kiểm thử đánh giá
│   └── llm.py                   # Client quản lý LLM (OpenAI, Gemini, v.v.)
├── progress.json                # Lưu trạng thái tiến trình
├── pyproject.toml               # Quản lý dependencies & packaging
└── README.md
```

## ⚙️ Yêu cầu hệ thống & Cài đặt

### Yêu cầu:
* Python >= 3.10
* GPU (Khuyến nghị nếu chạy embedding / reranker model local)

### Các bước cài đặt:

1. **Clone repository:**
   ```bash
   git clone https://github.com/sondts-ai/legal_agentic_rag.git
   cd legal_agentic_rag
   ```

2. **Khởi tạo môi trường ảo:**
   ```bash
   python -m venv venv
   # Trên Linux/macOS:
   source venv/bin/activate
   # Trên Windows:
   venv\Scripts\activate
   ```

3. **Cài đặt thư viện dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

## 🔧 Cấu hình hệ thống

Tùy chỉnh thông số trong tệp `configs/config.yaml` và thiết lập các biến môi trường cần thiết:

```bash
# Thiết lập API Key cho LLM
export OPENAI_API_KEY="sk-..."
```

Nội dung cơ bản trong `configs/config.yaml`:
* **LLM Settings:** Model name, temperature, max tokens.
* **Embedding Settings:** Model name, dimension, batch size.
* **ChromaDB Path:** Đường dẫn lưu trữ vector DB cục bộ.
* **Retrieval Weights:** Tỷ lệ phân bổ giữa Dense, BM25 và Graph Search.

## 📖 Hướng dẫn sử dụng

### 1. Nạp và xử lý dữ liệu văn bản pháp luật

Để nạp tài liệu vào cơ sở dữ liệu vector và xây dựng chỉ mục BM25, chạy script ingest:

```bash
python -m src.scripts.ingest --data_dir /path/to/legal/documents
```

Quá trình xử lý:
1. `loader.py`: Phân tích và đọc văn bản.
2. `cleaner.py`: Loại bỏ ký tự thừa, chuẩn hóa định dạng.
3. `chunker.py`: Phân đoạn văn bản theo cấu trúc Điều/Khoản luật.
4. `chroma_store.py` & `bm25_index.py`: Lưu trữ vector embeddings và lập chỉ mục BM25.

### 2. Khởi chạy Backend API (FastAPI)

Khởi động dịch vụ API để phục vụ các yêu cầu truy vấn:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

* **Swagger UI:** Truy cập `http://localhost:8000/docs` để kiểm thử trực tiếp danh sách API endpoints.

### 3. Khởi chạy Giao diện Gradio

Mở một terminal mới (đảm bảo Backend API đang chạy) và khởi chạy giao diện Gradio:

```bash
python UI/app.py
```

* Ứng dụng Gradio sẽ mặc định mở tại: `http://127.0.0.1:7860`
* Giao diện cung cấp:
  * Khung hội thoại tra cứu tình huống pháp lý.
  * Tùy chọn nguồn tìm kiếm và hiển thị trích dẫn trực tiếp từ các Điều luật liên quan.

### 4. Đánh giá chất lượng (Evaluation & Benchmark)

Để đo lường hiệu năng của hệ thống RAG trên tập dữ liệu kiểm thử chuẩn:

```bash
# Tạo tập kiểm thử
python -m src.scripts.build_eval_set

# Chạy benchmark
python -m src.scripts.run_benchmark --config configs/config.yaml
```

Kết quả benchmark (MRR, Hit@K, Faithfulness, Answer Relevance,...) sẽ được hiển thị chi tiết trên console hoặc lưu vào file kết quả đánh giá.