#!/usr/bin/env python3
"""
Shared benchmarking utilities: QueryMetrics and run_queries function.
"""
import time
import json
import psutil
import statistics
from pathlib import Path
from typing import List, Dict, Any

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
        latency_stats = {
            "avg_latency_ms": statistics.mean(self.query_times) * 1000 if self.query_times else 0,
            "median_latency_ms": statistics.median(self.query_times) * 1000 if self.query_times else 0,
        }
        return {
            "total_queries": self.total_queries,
            "queries_per_second": qps,
            "latency_stats": latency_stats,
            "peak_memory_mb": self.peak_memory
        }


def run_queries(vectorstore, queries: List[Dict[str, Any]], top_k: int, framework: str, results_dir: Path):
    metrics = QueryMetrics()
    metrics.start_tracking()
    for q in queries:
        start = time.time()
        results = vectorstore.similarity_search_with_score(q["query"], k=top_k)
        duration = time.time() - start
        metrics.add_query_time(duration)
    total_time = sum(metrics.query_times)
    metrics_dict = metrics.to_dict(total_time)
    output_file = results_dir / f"query_metrics_{framework}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"QUERY ({framework.upper()}) COMPLETED in {total_time:.2f}s")
    print(f"Results saved to: {output_file}")
    return metrics_dict


def get_benchmark_queries() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "query": "How to start a startup?", "expected_topics": ["startup", "entrepreneur", "business", "founder"]},
        {"id": 2, "query": "What makes a good programmer?", "expected_topics": ["programming", "coding", "developer", "software"]},
        {"id": 3, "query": "Venture capital and funding", "expected_topics": ["vc", "investor", "funding", "money", "capital"]}
    ]