# RAG Benchmark Suite

## Overview
This project benchmarks RAG (Retrieval-Augmented Generation) across three frameworks using Weaviate as the vector DB and OpenAI embeddings:
- LangChain
- LlamaIndex
- Haystack

It ingests Paul Graham essays, builds a vector index, and benchmarks query performance with consistent settings across frameworks.

## Requirements
- Python 3.11
- A running Weaviate instance on `http://localhost:8080`
- Environment variable `OPENAI_API_KEY` set

## Setup
1. Clone the repository and create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv-py311
   . .venv-py311/bin/activate
   ```
2. Install core dependencies (LangChain + LlamaIndex + Weaviate client):
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
4. (Optional) Disk usage reporting: If you want on-disk size printed from Weaviate’s data dir, set:
   ```bash
   export WEAVIATE_DATA_DIR="/path/to/weaviate/data"
   ```
5. Haystack runs in its own virtual environment automatically from `run_benchmark.py` (created at `.venv-haystack/`). No manual steps needed.

## Running the Benchmark
Run the end-to-end suite for all three frameworks:
```bash
python run_benchmark.py
```
What it does:
- LangChain: ingest -> query benchmark
- LlamaIndex: ingest -> query benchmark
- Haystack: creates `.venv-haystack` (first run), installs deps, ingest -> query benchmark
- Prints a “Quick Summary” at the end and saves all metrics to `results/`

## Metrics Recorded
### Ingestion metrics (saved to `results/ingestion_metrics_<framework>.json`)
- ingestion_time_seconds
- document_count
- chunk_count (LangChain, LlamaIndex)
- peak_memory_mb
- memory_increase_mb
- index_size_mb (approx aggregate count-based)
- vector_bytes_estimate_mb (count × 1536 × 4 bytes)
- disk_usage_mb (optional; if `WEAVIATE_DATA_DIR` is set)

### Query metrics (saved to `results/query_metrics_<framework>.json`)
- total_queries
- queries_per_second (QPS)
- latency_stats: avg, median, p95, p99, min, max (ms)
- peak_memory_mb
- memory_increase_mb
- avg_precision_at_k, avg_recall_at_k (keyword-heuristic vs expected topics)

Notes:
- Precision/Recall are computed via a simple keyword-contains heuristic over expected topic terms for each query. For more rigorous evaluation, provide labeled relevance data or an LLM-based judge.

## Troubleshooting
- Ensure Weaviate is reachable at `http://localhost:8080`.
- For Haystack, the pipeline uses its own venv `.venv-haystack` and installs pinned versions compatible with Weaviate 3.x.
- If LlamaIndex complains about NLTK resources, this repo avoids downloads by using manual chunking. No extra setup required.
- If you need to rerun only a specific framework’s query benchmark:
  ```bash
  python scripts/query_benchmark.py --framework langchain --top_k 3
  python scripts/query_benchmark.py --framework llamaindex --top_k 3
  python scripts/query_benchmark.py --framework haystack --top_k 3
  ```