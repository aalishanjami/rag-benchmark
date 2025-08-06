# Setup Notes - Python 3.11 Environment

## Issue Resolved ✅
Your original `requirements.txt` had dependency conflicts with Python 3.13. We successfully resolved this by:
1. **Created a Python 3.11 virtual environment**: `.venv-py311/`
2. **Installed modern, compatible versions** of your packages.

## How to Use
### Set Your OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file with: OPENAI_API_KEY=your-api-key-here
```

### Activate the Python 3.11 Environment
```bash
source .venv-py311/bin/activate
```

### Run Your Benchmark Scripts
```bash
# Data ingestion (LangChain)
python scripts/ingest_langchain.py

# Query benchmarking (LangChain)
python scripts/query_benchmark.py --framework langchain --top_k 3

# Data ingestion (Haystack)
python scripts/ingest_haystack.py

# Query benchmarking (Haystack)
python scripts/query_benchmark.py --framework haystack --top_k 3

# Complete end-to-end pipeline
python run_benchmark.py
```

### Deactivate When Done
```bash
deactivate
```
