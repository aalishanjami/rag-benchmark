#!/usr/bin/env python3
"""
Unified query benchmarking for LangChain or Haystack pipelines.
"""
import argparse
from pathlib import Path

from benchmark_utils import run_queries, get_benchmark_queries


def main():
    parser = argparse.ArgumentParser(description="Run query benchmarks across frameworks.")
    parser.add_argument(
        "--framework",
        choices=["langchain", "haystack"],
        required=True,
        help="Which framework to benchmark",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top results to retrieve",
    )
    args = parser.parse_args()

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.framework == "langchain":
        from ingest_langchain import setup_langchain_vectorstore
        vectorstore = setup_langchain_vectorstore()
    else:
        from ingest_haystack import setup_haystack_vectorstore
        vectorstore = setup_haystack_vectorstore()

    queries = get_benchmark_queries()
    run_queries(vectorstore, queries, args.top_k, args.framework, results_dir)


if __name__ == "__main__":
    main()