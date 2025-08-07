#!/usr/bin/env python3
"""
LlamaIndex-based ingestion and query setup for RAG benchmarking.
Indexes Paul Graham essays into Weaviate and provides a vectorstore-like
wrapper compatible with benchmark_utils.run_queries.
"""

import os
import time
import json
import psutil
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import weaviate

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding


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
        return (self.end_time or time.time()) - (self.start_time or time.time())

    def to_dict(self, document_count: int, chunk_count: int, index_size_mb: float) -> Dict[str, Any]:
        return {
            "ingestion_time_seconds": self.get_duration(),
            "document_count": document_count,
            "chunk_count": chunk_count,
            "index_size_mb": index_size_mb,
            "vector_bytes_estimate_mb": getattr(self, 'vector_bytes_estimate_mb', 0),
            "disk_usage_mb": getattr(self, 'disk_usage_mb', 0),
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - (self.start_memory or 0),
            "framework": "llamaindex",
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
        import statistics

        latency = {
            "avg_latency_ms": statistics.mean(self.query_times) * 1000 if self.query_times else 0,
            "median_latency_ms": statistics.median(self.query_times) * 1000 if self.query_times else 0,
        }
        return {
            "total_queries": self.total_queries,
            "queries_per_second": qps,
            "latency_stats": latency,
            "peak_memory_mb": self.peak_memory,
            "framework": "llamaindex",
        }


def _setup_weaviate_client() -> weaviate.Client:
    client = weaviate.Client(url="http://localhost:8080", timeout_config=(5, 15))
    if not client.is_ready():
        raise ConnectionError("Weaviate is not ready at localhost:8080")
    return client


def _create_llamaindex_schema(client: weaviate.Client, class_name: str):
    try:
        client.schema.delete_class(class_name)
        print(f"Deleted existing class: {class_name}")
    except Exception:
        pass

    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"], "description": "Chunk text"},
        ],
    }
    client.schema.create_class(class_obj)
    print(f"Created schema for class: {class_name}")


def _estimate_index_size_mb(client: weaviate.Client, class_name: str) -> float:
    try:
        result = client.query.aggregate(class_name).with_meta_count().do()
        count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]
        return count * 0.001
    except Exception:
        return 0.0


def main():
    project_root = Path(__file__).parent.parent
    essays_dir = project_root / "data" / "essays"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    if not essays_dir.exists():
        print(f"Essays directory not found: {essays_dir}")
        return

    metrics = IngestionMetrics()
    metrics.start_tracking()

    client = _setup_weaviate_client()
    class_name = "PaulGrahamEssayLlamaIndex"
    _create_llamaindex_schema(client, class_name)

    # Reader and manual chunking to avoid NLTK downloads
    documents = SimpleDirectoryReader(str(essays_dir), recursive=False, filename_as_id=False).load_data()
    nodes = []
    chunk_size = 1000
    chunk_overlap = 200
    for doc in documents:
        text = doc.get_content() if hasattr(doc, "get_content") else getattr(doc, "text", "")
        text_len = len(text)
        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            nodes.append(type("TmpNode", (), {"get_content": lambda self, s=text[start:end]: s, "text": text[start:end]})())
            start += max(1, chunk_size - chunk_overlap)

    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = embed_model

    print("Starting LlamaIndex ingestion into Weaviate (manual write)...")
    # Embed nodes and write manually using weaviate v3 client
    vectors = []
    for n in nodes:
        text = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
        vec = embed_model.get_text_embedding(text)
        vectors.append((text, vec))
    with client.batch as batch:
        batch.batch_size = 100
        for text, vec in vectors:
            props = {"content": text}
            client.batch.add_data_object(props, class_name, vector=vec)

    metrics.update_peak_memory()
    metrics.finish_tracking()

    index_size_mb = _estimate_index_size_mb(client, class_name)
    # Estimate vector bytes
    try:
        agg = client.query.aggregate(class_name).with_meta_count().do()
        count = agg["data"]["Aggregate"][class_name][0]["meta"]["count"]
        metrics.vector_bytes_estimate_mb = (count * 1536 * 4) / (1024 * 1024)
    except Exception:
        metrics.vector_bytes_estimate_mb = 0.0
    # Optional disk usage via WEAVIATE_DATA_DIR
    data_dir = os.getenv("WEAVIATE_DATA_DIR")
    metrics.disk_usage_mb = 0.0
    if data_dir and Path(data_dir).exists():
        total = 0
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    pass
        metrics.disk_usage_mb = total / (1024 * 1024)

    results = metrics.to_dict(document_count=len(documents), chunk_count=len(nodes), index_size_mb=index_size_mb)

    out_file = results_dir / "ingestion_metrics_llamaindex.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("INGESTION (LLAMAINDEX) COMPLETED")
    print("=" * 50)
    print(f"Time: {metrics.get_duration():.2f} seconds")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(nodes)}")
    print(f"Index size (approx): {index_size_mb:.1f} MB | Vector bytes est.: {metrics.vector_bytes_estimate_mb:.1f} MB")
    if metrics.disk_usage_mb > 0:
        print(f"Disk usage (WEAVIATE_DATA_DIR): {metrics.disk_usage_mb:.1f} MB")
    print(f"Peak memory: {metrics.peak_memory:.1f} MB")
    print(f"Results saved to: {out_file}")


class LlamaIndexVectorStore:
    def __init__(self, client: weaviate.Client, class_name: str, embed_model: OpenAIEmbedding):
        self.client = client
        self.class_name = class_name
        self.embed_model = embed_model

    def similarity_search_with_score(self, query: str, k: int = 3):
        from types import SimpleNamespace
        vector = self.embed_model.get_query_embedding(query)
        result = (
            self.client.query.get(self.class_name, ["content"])
            .with_near_vector({"vector": vector})
            .with_additional(["certainty"])
            .with_limit(k)
            .do()
        )
        objs = result.get("data", {}).get("Get", {}).get(self.class_name, []) or []
        out = []
        for obj in objs:
            content = obj.get("content", "")
            score = obj.get("_additional", {}).get("certainty", 0)
            out.append((SimpleNamespace(page_content=content, metadata={}), score))
        return out


def setup_llamaindex_vectorstore():
    client = _setup_weaviate_client()
    class_name = "PaulGrahamEssayLlamaIndex"
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = embed_model
    return LlamaIndexVectorStore(client, class_name, embed_model)


if __name__ == "__main__":
    main()


