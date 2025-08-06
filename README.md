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
4. (Optional) To run the Haystack pipeline, install additional dependencies:
   ```bash
   pip install farm-haystack[weaviate,preprocessing] sentence-transformers
   ```

## Running the Benchmark
To run the benchmark, execute:
```bash
python run_benchmark.py
```