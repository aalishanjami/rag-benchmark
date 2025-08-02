#!/usr/bin/env python3
"""
Query benchmarking script for RAG systems.
Tests query latency, throughput, and quality metrics.
"""

import os
import time
import json
import psutil
import weaviate
import statistics
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

load_dotenv()

class WeaviateVectorStore:
    def __init__(self, client, index_name: str, text_key: str, embedding):
        self.client = client
        self.index_name = index_name
        self.text_key = text_key
        self.embedding = embedding

    def similarity_search_with_score(self, query: str, k: int = 10):
        vector = self.embedding.embed_query(query)
        
        builder = (
            self.client.query.get(self.index_name, [self.text_key])
            .with_near_vector({"vector": vector})
            .with_limit(k)
            .with_additional("distance")
        )
        
        result = builder.do()
        
        items = result.get("data", {}).get("Get", {}).get(self.index_name, [])
        
        results = []
        for item in items:
            content = item.get(self.text_key, "")
            metadata = {}
            distance = item.get("_additional", {}).get("distance", 0)
            results.append((Document(page_content=content, metadata=metadata), distance))
        
        return results

class QueryMetrics:
    def __init__(self):
        self.query_times = []
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.total_queries = 0
        self.process = psutil.Process()
        self.framework = "langchain"
    
    def start_tracking(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
    
    def add_query_time(self, query_time: float):
        self.query_times.append(query_time)
        self.total_queries += 1
        self.update_peak_memory()
    
    def update_peak_memory(self):
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_latency_stats(self) -> Dict[str, float]:
        if not self.query_times:
            return {}
        
        sorted_times = sorted(self.query_times)
        return {
            "avg_latency_ms": statistics.mean(self.query_times) * 1000,
            "median_latency_ms": statistics.median(self.query_times) * 1000,
            "p95_latency_ms": self._percentile(sorted_times, 95) * 1000,
            "p99_latency_ms": self._percentile(sorted_times, 99) * 1000,
            "min_latency_ms": min(self.query_times) * 1000,
            "max_latency_ms": max(self.query_times) * 1000
        }
    
    def get_throughput(self, total_time: float) -> float:
        return self.total_queries / total_time if total_time > 0 else 0
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        if not data:
            return 0
        index = (percentile / 100) * (len(data) - 1)
        if index.is_integer():
            return data[int(index)]
        else:
            lower = data[int(index)]
            upper = data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def to_dict(self, total_time: float) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "queries_per_second": self.get_throughput(total_time),
            "latency_stats": self.get_latency_stats(),
            "memory_usage_mb": {
                "start": self.start_memory,
                "peak": self.peak_memory,
                "increase": self.peak_memory - self.start_memory
            }
        }

def get_benchmark_queries() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "query": "How to start a startup?", "description": "Fundamental startup advice", "expected_topics": ["startup", "entrepreneur", "business", "founder"]},
        {"id": 2, "query": "What makes a good programmer?", "description": "Programming skills and qualities", "expected_topics": ["programming", "coding", "developer", "software"]},
        {"id": 3, "query": "Venture capital and funding", "description": "Investment and funding advice", "expected_topics": ["vc", "investor", "funding", "money", "capital"]}
    ]

def setup_weaviate_vectorstore():
    client = weaviate.Client(
        url="http://localhost:8080",
        timeout_config=(5, 15)
    )
    
    if not client.is_ready():
        raise ConnectionError("Weaviate is not ready at localhost:8080")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = WeaviateVectorStore(
        client,
        "PaulGrahamEssay",
        "content",
        embeddings,
    )
    
    return vectorstore

def calculate_precision_at_k(results: List[Dict], expected_topics: List[str], k: int = 5) -> float:
    if not results or k == 0:
        return 0.0
    
    relevant_count = 0
    for i, result in enumerate(results[:k]):
        content = result[0].page_content.lower()
        if any(topic.lower() in content for topic in expected_topics):
            relevant_count += 1
    
    return relevant_count / k

def calculate_recall_at_k(results: List[Dict], expected_topics: List[str], k: int = 5) -> float:
    if not results or k == 0:
        return 0.0
    
    relevant_in_k = 0
    total_relevant = len([r for r in results if any(topic.lower() in r[0].page_content.lower() for topic in expected_topics)])
    
    if total_relevant == 0:
        return 0.0
    
    for result in results[:k]:
        content = result[0].page_content.lower()
        if any(topic.lower() in content for topic in expected_topics):
            relevant_in_k += 1
    
    return relevant_in_k / total_relevant

def run_single_query(vectorstore, query_data: Dict, k: int = 3) -> Dict[str, Any]:
    query_text = query_data["query"]
    start_time = time.time()
    result_docs = vectorstore.similarity_search_with_score(query_text, k=k)
    query_time = time.time() - start_time
    
    precision_at_k = calculate_precision_at_k(result_docs, query_data["expected_topics"], k=k)
    recall_at_k = calculate_recall_at_k(result_docs, query_data["expected_topics"], k=k)
    
    return {
        "query_id": query_data["id"],
        "query": query_text,
        "latency_seconds": query_time,
        "result_count": len(result_docs),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "top_k_scores": [r[1] for r in result_docs[:k]]
    }

def run_benchmark_suite(vectorstore, metrics: QueryMetrics) -> List[Dict]:
    queries = get_benchmark_queries()
    query_results = []
    
    print(f"Running {len(queries)} benchmark queries...")
    
    for i, query_data in enumerate(queries, 1):
        print(f"Query {i}/{len(queries)}: {query_data['query']}")
        
        query_result = None
        
        for _ in range(1):
            start_time = time.time()
            temp_result = run_single_query(vectorstore, query_data)
            query_time = time.time() - start_time
            
            if query_result is None:
                query_result = temp_result
                query_result["latency_seconds"] = query_time
                metrics.add_query_time(query_time)
        
        query_results.append(query_result)
    
    return query_results

def calculate_aggregate_metrics(query_results: List[Dict]) -> Dict[str, float]:
    if not query_results:
        return {}

    precision_scores = [r["precision_at_k"] for r in query_results]
    recall_scores = [r["recall_at_k"] for r in query_results]

    return {
        "avg_precision": statistics.mean(precision_scores),
        "avg_recall": statistics.mean(recall_scores),
        "median_precision": statistics.median(precision_scores),
        "median_recall": statistics.median(recall_scores)
    }

def count_lines_of_code() -> int:
    current_file = Path(__file__)
    with open(current_file, 'r') as f:
        lines = f.readlines()
    
    loc = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
            loc += 1
    
    return loc

def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    metrics = QueryMetrics()
    metrics.framework = "langchain"
    metrics.start_tracking()
    
    try:
        print("Connecting to Weaviate vectorstore...")
        vectorstore = setup_weaviate_vectorstore()
        
        start_time = time.time()
        query_results = run_benchmark_suite(vectorstore, metrics)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        performance_metrics = metrics.to_dict(total_time)
        quality_metrics = calculate_aggregate_metrics(query_results)
        loc = count_lines_of_code()
        
        final_metrics = {
            "framework": "langchain",
            "timestamp": time.time(),
            "performance": performance_metrics,
            "quality": quality_metrics,
            "lines_of_code": loc,
            "individual_queries": query_results
        }
        
        results_file = results_dir / "query_metrics_langchain.json"
        with open(results_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"QUERY BENCHMARK COMPLETED")
        print(f"{'='*50}")
        print(f"Framework: LangChain")
        print(f"Total queries: {metrics.total_queries}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average latency: {performance_metrics['latency_stats']['avg_latency_ms']:.1f} ms")
        print(f"Queries per second: {performance_metrics['queries_per_second']:.1f}")
        print(f"Average Precision@5: {quality_metrics['avg_precision']:.3f}")
        print(f"Average Recall@5: {quality_metrics['avg_recall']:.3f}")
        print(f"Peak memory: {performance_metrics['memory_usage_mb']['peak']:.1f} MB")
        print(f"Lines of code: {loc}")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()