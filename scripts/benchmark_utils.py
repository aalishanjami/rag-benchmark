#!/usr/bin/env python3
"""
Shared benchmarking utilities: QueryMetrics and run_queries function.
"""
import time
import json
import psutil
import statistics
import math
from pathlib import Path
from typing import List, Dict, Any

class QueryMetrics:
    def __init__(self):
        self.query_times: List[float] = []
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.total_queries = 0
        self.sum_precision_at_k = 0.0
        self.sum_recall_at_k = 0.0

    def start_tracking(self):
        self.start_memory = self.process.memory_info().rss / 1024 / 1024

    def add_query_time(self, qtime: float):
        self.query_times.append(qtime)
        self.total_queries += 1
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)

    def to_dict(self, total_time: float) -> Dict[str, Any]:
        qps = self.total_queries / total_time if total_time > 0 else 0.0
        def _percentile(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            k = (len(s) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return s[int(k)]
            return s[f] + (s[c] - s[f]) * (k - f)

        latency_stats = {
            "avg_latency_ms": statistics.mean(self.query_times) * 1000 if self.query_times else 0,
            "median_latency_ms": statistics.median(self.query_times) * 1000 if self.query_times else 0,
            "p95_latency_ms": _percentile(self.query_times, 0.95) * 1000 if self.query_times else 0,
            "p99_latency_ms": _percentile(self.query_times, 0.99) * 1000 if self.query_times else 0,
            "min_latency_ms": min(self.query_times) * 1000 if self.query_times else 0,
            "max_latency_ms": max(self.query_times) * 1000 if self.query_times else 0,
        }
        avg_precision_at_k = (self.sum_precision_at_k / self.total_queries) if self.total_queries else 0.0
        avg_recall_at_k = (self.sum_recall_at_k / self.total_queries) if self.total_queries else 0.0
        return {
            "total_queries": self.total_queries,
            "queries_per_second": qps,
            "latency_stats": latency_stats,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": (self.peak_memory - (self.start_memory or 0)),
            "avg_precision_at_k": avg_precision_at_k,
            "avg_recall_at_k": avg_recall_at_k,
        }


def run_queries(vectorstore, queries: List[Dict[str, Any]], top_k: int, framework: str, results_dir: Path):
    metrics = QueryMetrics()
    metrics.start_tracking()

    def _evaluate_results(results, expected_topics: List[str], k: int):
        relevant_results = 0
        matched_topics = set()
        expected_lower = [t.lower() for t in expected_topics]
        for doc, _score in results[:k]:
            content = (getattr(doc, "page_content", "") or "").lower()
            hit = False
            for topic in expected_lower:
                if topic and topic in content:
                    matched_topics.add(topic)
                    hit = True
            if hit:
                relevant_results += 1
        precision = relevant_results / float(max(1, k))
        recall = len(matched_topics) / float(max(1, len(expected_lower)))
        return precision, recall

    for q in queries:
        start = time.time()
        results = vectorstore.similarity_search_with_score(q["query"], k=top_k)
        duration = time.time() - start
        metrics.add_query_time(duration)
        if "expected_topics" in q:
            p, r = _evaluate_results(results, q.get("expected_topics", []), top_k)
            metrics.sum_precision_at_k += p
            metrics.sum_recall_at_k += r
    total_time = sum(metrics.query_times)
    metrics_dict = metrics.to_dict(total_time)
    output_file = results_dir / f"query_metrics_{framework}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print("\n" + "="*50)
    print(f"QUERY ({framework.upper()}) COMPLETED")
    print("="*50)
    print(f"Total time: {total_time:.2f} s | Total queries: {metrics_dict['total_queries']} | QPS: {metrics_dict['queries_per_second']:.2f}")
    lat = metrics_dict["latency_stats"]
    print(
        f"Latency ms — avg: {lat['avg_latency_ms']:.1f}, median: {lat['median_latency_ms']:.1f}, p95: {lat['p95_latency_ms']:.1f}, p99: {lat['p99_latency_ms']:.1f}"
    )
    print(f"Precision@{top_k}: {metrics_dict['avg_precision_at_k']:.3f} | Recall@{top_k}: {metrics_dict['avg_recall_at_k']:.3f}")
    print(f"Peak memory: {metrics_dict['peak_memory_mb']:.1f} MB | Memory increase: {metrics_dict['memory_increase_mb']:.1f} MB")
    print(f"Results saved to: {output_file}")
    return metrics_dict


def get_benchmark_queries() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "query": "How to start a startup?", "expected_topics": ["startup", "entrepreneur", "business", "founder"]},
        {"id": 2, "query": "What makes a good programmer?", "expected_topics": ["programming", "coding", "developer", "software"]},
        {"id": 3, "query": "Venture capital and funding", "expected_topics": ["vc", "investor", "funding", "money", "capital"]}
    ]