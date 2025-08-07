#!/usr/bin/env python3
"""
Ingestion pipeline using Haystack with WeaviateDocumentStore.
"""
import os
import time
import json
import psutil
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs
# Define a simple Document class for compatibility
class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

load_dotenv()

class IngestionMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0

    def start_tracking(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024

    def update_peak_memory(self):
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)

    def finish_tracking(self):
        self.end_time = time.time()

    def get_duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self, document_count: int, index_size_mb: float) -> Dict[str, Any]:
        return {
            "ingestion_time_seconds": self.get_duration(),
            "document_count": document_count,
            "index_size_mb": index_size_mb,
            "vector_bytes_estimate_mb": getattr(self, 'vector_bytes_estimate_mb', 0),
            "disk_usage_mb": getattr(self, 'disk_usage_mb', 0),
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - self.start_memory,
            "framework": "haystack"
        }

class QueryMetrics:
    def __init__(self):
        self.query_times: List[float] = []
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.total_queries = 0

    def start_tracking(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024

    def add_query_time(self, qtime: float):
        self.query_times.append(qtime)
        self.total_queries += 1
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)

    def to_dict(self, total_time: float) -> Dict[str, Any]:
        qps = self.total_queries / total_time if total_time > 0 else 0.0
        latency = {
            "avg_latency_ms": statistics.mean(self.query_times) * 1000 if self.query_times else 0,
            "median_latency_ms": statistics.median(self.query_times) * 1000 if self.query_times else 0,
        }
        return {
            "total_queries": self.total_queries,
            "queries_per_second": qps,
            "latency_stats": latency,
            "peak_memory_mb": self.peak_memory,
            "framework": "haystack"
        }

def get_benchmark_queries() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "query": "How to start a startup?", "expected_topics": ["startup", "entrepreneur", "business", "founder"]},
        {"id": 2, "query": "What makes a good programmer?", "expected_topics": ["programming", "coding", "developer", "software"]},
        {"id": 3, "query": "Venture capital and funding", "expected_topics": ["vc", "investor", "funding", "money", "capital"]}
    ]

def main():
    root = Path(__file__).parent.parent
    essays_dir = root / "data" / "essays"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    # Ingestion
    metrics_ing = IngestionMetrics()
    metrics_ing.start_tracking()

    # Load and chunk Markdown essays manually
    docs = []
    chunk_size = 1000
    chunk_overlap = 200
    for md_file in essays_dir.glob("*.md"):
        text = md_file.read_text()
        clean_text = clean_wiki_text(text)
        text_len = len(clean_text)
        start = 0
        idx = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = clean_text[start:end]
            docs.append({"content": chunk, "meta": {"source": md_file.name, "chunk": idx}})
            idx += 1
            start += chunk_size - chunk_overlap
    # Delete existing class in Weaviate to avoid schema conflict
    import weaviate as _weaviate_client_lib
    _wc = _weaviate_client_lib.Client(url="http://localhost:8080")
    try:
        _wc.schema.delete_class("PaulGrahamEssay")
        print("Deleted existing class: PaulGrahamEssay")
    except Exception:
        pass
    store = WeaviateDocumentStore(
        host="http://localhost",
        port=8080,
        index="PaulGrahamEssayHaystack",
        embedding_dim=1536,
        recreate_index=True
    )
    store.delete_documents()
    store.write_documents(docs)
    retriever = EmbeddingRetriever(
        document_store=store,
        embedding_model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_gpu=False
    )
    store.update_embeddings(retriever)

    metrics_ing.update_peak_memory()
    metrics_ing.finish_tracking()
    doc_count = len(docs)
    # Estimate index size + vector bytes and optional disk usage
    try:
        agg = _wc.query.aggregate("PaulGrahamEssayHaystack").with_meta_count().do()
        count = agg["data"]["Aggregate"]["PaulGrahamEssayHaystack"][0]["meta"]["count"]
        metrics_ing.vector_bytes_estimate_mb = (count * 1536 * 4) / (1024 * 1024)
        index_size_mb = metrics_ing.vector_bytes_estimate_mb
    except Exception:
        index_size_mb = 0
        metrics_ing.vector_bytes_estimate_mb = 0
    data_dir = os.getenv("WEAVIATE_DATA_DIR")
    metrics_ing.disk_usage_mb = 0.0
    if data_dir and Path(data_dir).exists():
        total = 0
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    pass
        metrics_ing.disk_usage_mb = total / (1024 * 1024)
    ingestion_metrics = metrics_ing.to_dict(doc_count, index_size_mb)
    ingestion_file = results_dir / "ingestion_metrics_haystack.json"
    with open(ingestion_file, 'w') as f:
        json.dump(ingestion_metrics, f, indent=2)
    print("\n" + "="*50)
    print("INGESTION (HAYSTACK) COMPLETED")
    print("="*50)
    print(f"Time: {metrics_ing.get_duration():.2f} seconds")
    print(f"Documents: {doc_count}")
    print(f"Peak memory: {metrics_ing.peak_memory:.1f} MB")
    print(f"Index size (approx): {index_size_mb:.1f} MB | Vector bytes est.: {metrics_ing.vector_bytes_estimate_mb:.1f} MB")
    if metrics_ing.disk_usage_mb > 0:
        print(f"Disk usage (WEAVIATE_DATA_DIR): {metrics_ing.disk_usage_mb:.1f} MB")
    print(f"Results saved to: {ingestion_file}")

    # Query Benchmarking
    metrics_q = QueryMetrics()
    metrics_q.start_tracking()
    queries = get_benchmark_queries()
    for q in queries:
        start = time.time()
        res = retriever.retrieve(q["query"], top_k=3)
        qtime = time.time() - start
        metrics_q.add_query_time(qtime)
    total_time = sum(metrics_q.query_times)
    query_metrics = metrics_q.to_dict(total_time)
    query_file = results_dir / "query_metrics_haystack.json"
    with open(query_file, 'w') as f:
        json.dump(query_metrics, f, indent=2)

    print("\n" + "="*50)
    print("QUERY (HAYSTACK) COMPLETED")
    print("="*50)
    print(f"Total queries: {metrics_q.total_queries}")
    print(f"Average latency: {query_metrics['latency_stats']['avg_latency_ms']:.1f} ms")
    print(f"Queries per second: {query_metrics['queries_per_second']:.1f}\n")
    print(f"Results saved to: {query_file}")

if __name__ == "__main__":
    main()
 
class HaystackVectorStore:
    def __init__(self, retriever: EmbeddingRetriever):
        self.retriever = retriever

    def similarity_search_with_score(self, query: str, k: int = 3):
        docs = self.retriever.retrieve(query, top_k=k)
        results = []
        for doc in docs:
            results.append((Document(page_content=doc.content, metadata=doc.meta), getattr(doc, 'score', 0)))
        return results

def setup_haystack_vectorstore():
    store = WeaviateDocumentStore(
        host="http://localhost",
        port=8080,
        index="PaulGrahamEssayHaystack",
        embedding_dim=1536,
        recreate_index=False
    )
    retriever = EmbeddingRetriever(
        document_store=store,
        embedding_model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        use_gpu=False
    )
    return HaystackVectorStore(retriever)