# RAG Benchmark Suite

## Overview
This project benchmarks RAG (Retrieval-Augmented Generation) using LangChain, Weaviate, and OpenAI.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Running the Benchmark
To run the benchmark, execute:
```bash
python run_benchmark.py
```
You can choose to install/update dependencies when prompted.

## Notes
- The benchmark currently uses a reduced set of 3 essays to minimize embedding costs.
- The metrics calculated include:
  - Index Build Time
  - Query Latency
  - Precision@K / Recall@K
  - Memory Footprint
  - Disk Usage
