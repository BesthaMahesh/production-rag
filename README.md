# Enterprise Production-Grade RAG Evaluation & Observability Platform

A modular, pre-production hybrid RAG platform upgraded in place with **dense vector search (ChromaDB + SentenceTransformers)**, **sparse lexical search (BM25Okapi)**, **Reciprocal Rank Fusion (RRF)**, **CrossEncoder reranking**, **grounded LLM generation (Groq / Ollama)**, **comprehensive offline evaluation benchmarks**, and **genuine OpenTelemetry distributed tracing**.

---

## Architecture Highlights

1. **Multi-Source Ingestion & Processing**:
   - Ingests PDF, DOCX, TXT, Markdown, CSV, and LOG documents.
   - Page-level extraction with metadata preservation (`document_name`, `page`, `section`, `file_type`).
   - Scanned PDF detection (zero-text warning + optional OCR hook).
   - Paragraph-aware chunking with stable SHA256 chunk IDs and content hashes to prevent duplicate chunks.
   - Hierarchical parent-child context window mapping.

2. **Multi-Representation Vector & Sparse Stores**:
   - **Persistent ChromaDB** vector database with cosine metric (`hnsw:space: cosine`).
   - In-memory / Numpy vector fallback.
   - **BM25Okapi** sparse lexical index.
   - In-memory SQLite table registration for structured CSV queries.

3. **Query Understanding & Routing**:
   - Classification into `policy`, `factual_lookup`, `comparison`, `multi_document`, and `unanswerable`.
   - Normalization and deterministic multi-query expansion.

4. **Hybrid Retrieval & Reranking**:
   - Dual dense and sparse candidate generation with configurable top-k.
   - **Reciprocal Rank Fusion (RRF)** with standard constant `rrf_k = 60`.
   - **Cross-Encoder Reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with relevance score thresholding.
   - Token-budgeted evidence selection preserving source metadata and citations.

5. **Grounded Generation with Real Citations**:
   - Prompt strictly grounds generation on retrieved evidence.
   - Explicit abstention for unanswerable or out-of-domain queries to eliminate hallucinations.
   - Real citations `[SOURCE n]` attributing specific document names and page numbers.

6. **Pre-Production Evaluation Pipeline**:
   - Evaluates against the 61-case ground-truth dataset (`production_rag_evaluation_set.csv`).
   - Metrics computed: **Source Hit Rate**, **Page Hit Rate**, **MRR (Mean Reciprocal Rank)**, **Precision@K**, **Recall@K**, **nDCG@K**, **Lexical Recall Proxy**, and **Abstention Accuracy**.
   - Downloadable evaluation reports in JSON and CSV formats.

7. **OpenTelemetry & Observability**:
   - Authentic OpenTelemetry tracer and spans wrapping every stage: ingestion, extraction, chunking, embeddings, dense search, BM25 search, hybrid fusion, reranking, context assembly, LLM generation, and evaluation.
   - Local in-memory span waterfall and latency distribution visualization in the Streamlit UI.
   - Optional export to remote OTLP gRPC collectors (e.g. Jaeger, Grafana Tempo, Datadog).
   - Append-only local audit logs in `data/runs.jsonl` and `data/feedback.jsonl`.

8. **Enterprise Streamlit Interface**:
   - Organized into 8 dedicated navigation sections:
     1. Executive Overview
     2. Knowledge Base & Ingestion
     3. Retrieval Playground
     4. RAG Assistant
     5. Evaluation Dashboard
     6. Feedback & Improvement
     7. Observability & Telemetry
     8. Configuration & System Health

---

## 1. Setup & Environment

### Create and Activate Virtual Environment
```bash
# Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### Install Requirements
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Secrets Configuration

Create `.streamlit/secrets.toml` in the project root:
```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Optional remote OpenTelemetry collector endpoint:
# OTEL_EXPORTER_OTLP_ENDPOINT = "localhost:4317"
```

*Note: `.streamlit/secrets.toml` is excluded by `.gitignore` to prevent credential leaks.*

---

## 3. Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 4. Indexing the Evaluation PDFs

Inside the running app:
1. Navigate to **2. Knowledge Base & Ingestion**.
2. Click **Index 3 Benchmark PDFs Now**.
   - The system ingests `RAG_Document_01_HR_Policy_Handbook.pdf`, `RAG_Document_02_IT_Security_Standard.pdf`, and `RAG_Document_03_Product_Support_Operations.pdf`.
   - Embeddings are persisted in `data/chromadb/` and chunks are registered in BM25.

---

## 5. Running the Benchmark Evaluation

1. Navigate to **5. Evaluation Dashboard**.
2. Select **Workspace Benchmark CSV** (`production_rag_evaluation_set.csv`).
3. Choose either **Retrieval Quality Only** (fast offline evaluation) or **Full Retrieval + LLM Generation**.
4. Click **Run Benchmark Evaluation**.
5. Inspect the scorecard (Source Hit Rate, Page Hit Rate, MRR, Precision, Recall, nDCG, Abstention Accuracy) and per-case results.
6. Export the complete report as JSON or CSV.

---

## 6. Running Tests

Run the full pytest suite:
```bash
python -m pytest tests -v
```

All 23 unit and integration tests validate:
- PDF and text extraction with page preservation.
- Semantic chunking, stable hashing, and deduplication.
- ChromaDB and BM25 indexing.
- Dense search, sparse search, RRF fusion, and CrossEncoder reranking.
- Citation extraction and abstention detection.
- OpenTelemetry span creation and in-memory buffering.
- Evaluation metrics (nDCG, lexical recall, evidence match).
- Feedback logging and reading.
