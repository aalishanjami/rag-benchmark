#!/usr/bin/env python3
"""
LangChain-based ingestion script for RAG benchmarking.
Loads Paul Graham essays, chunks them, and indexes into Weaviate.
"""

import os
import time
import json
import psutil
import weaviate
import requests
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document

load_dotenv()

class IngestionMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.process = psutil.Process()
    
    def start_tracking(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
    
    def update_peak_memory(self):
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def finish_tracking(self, index_size_mb: float = 0):
        self.end_time = time.time()
        self.index_size_mb = index_size_mb
    
    def get_duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            "ingestion_time_seconds": self.get_duration(),
            "document_count": getattr(self, 'document_count', 0),
            "chunk_count": getattr(self, 'chunk_count', 0),
            "index_size_mb": getattr(self, 'index_size_mb', 0),
            "vector_bytes_estimate_mb": getattr(self, 'vector_bytes_estimate_mb', 0),
            "disk_usage_mb": getattr(self, 'disk_usage_mb', 0),
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - self.start_memory,
            "lines_of_code": count_lines_of_code(),
            "framework": "langchain"
        }

def setup_weaviate_client():
    client = weaviate.Client(
        url="http://localhost:8080",
        timeout_config=(5, 15)
    )
    
    if not client.is_ready():
        raise ConnectionError("Weaviate is not ready at localhost:8080")
    
    return client

def create_schema(client):
    class_name = "PaulGrahamEssay"
    
    try:
        client.schema.delete_class(class_name)
        print(f"Deleted existing class: {class_name}")
    except:
        pass
    
    class_obj = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "The content of the essay",
            },
        ],
    }
    client.schema.create_class(class_obj)
    print(f"Created schema for class: {class_name}")

def load_documents(essays_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        essays_dir,
        glob="*.md",
        loader_cls=TextLoader,
        show_progress=True
    )
    return loader.load()

def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def calculate_index_size(client, class_name: str) -> float:
    try:
        result = client.query.aggregate(class_name).with_meta_count().do()
        count = result['data']['Aggregate'][class_name][0]['meta']['count']
        return count * 0.001
    except:
        return 0.0

def ingest_with_langchain(essays_dir: str, metrics: IngestionMetrics):
    client = setup_weaviate_client()
    create_schema(client)
    
    print(f"Loading documents from: {essays_dir}")
    documents = load_documents(essays_dir)
    metrics.document_count = len(documents)
    print(f"Loaded {metrics.document_count} documents")
    
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    metrics.chunk_count = len(chunks)
    print(f"Created {metrics.chunk_count} chunks")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("Starting ingestion into Weaviate...")
    vectorstore = Weaviate(
        client=client,
        index_name="PaulGrahamEssay",
        text_key="content",
        embedding=embeddings
    )
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        metrics.update_peak_memory()
    
    index_size = calculate_index_size(client, "PaulGrahamEssay")
    # Estimate vector bytes based on object count
    try:
        agg = client.query.aggregate("PaulGrahamEssay").with_meta_count().do()
        count = agg['data']['Aggregate']["PaulGrahamEssay"][0]['meta']['count']
        vector_bytes_estimate_mb = (count * 1536 * 4) / (1024 * 1024)
    except Exception:
        vector_bytes_estimate_mb = 0.0
    metrics.vector_bytes_estimate_mb = vector_bytes_estimate_mb
    # Optional actual disk usage if WEAVIATE_DATA_DIR is provided
    data_dir = os.getenv("WEAVIATE_DATA_DIR")
    disk_usage_mb = 0.0
    if data_dir and Path(data_dir).exists():
        total = 0
        for root, _dirs, files in os.walk(data_dir):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except Exception:
                    pass
        disk_usage_mb = total / (1024 * 1024)
    metrics.disk_usage_mb = disk_usage_mb
    metrics.finish_tracking(index_size)
    
    print("Ingestion completed. Index size: {:.1f} MB".format(index_size))
    return index_size

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
    
    essays_dir = project_root / "data" / "essays"
    
    if not essays_dir.exists():
        print(f"Essays directory not found: {essays_dir}")
        return
    
    metrics = IngestionMetrics()
    metrics.start_tracking()
    
    try:
        index_size = ingest_with_langchain(str(essays_dir), metrics)
        
        print("=" * 50)
        print("INGESTION COMPLETED")
        print("=" * 50)
        print(f"Time: {metrics.get_duration():.2f} seconds")
        print(f"Documents: {metrics.document_count}")
        print(f"Chunks: {metrics.chunk_count}")
        print(f"Index size (approx): {index_size:.1f} MB | Vector bytes est.: {metrics.vector_bytes_estimate_mb:.1f} MB")
        if getattr(metrics, 'disk_usage_mb', 0) > 0:
            print(f"Disk usage (WEAVIATE_DATA_DIR): {metrics.disk_usage_mb:.1f} MB")
        print(f"Peak memory: {metrics.peak_memory:.1f} MB")
        print(f"Lines of code: {metrics.to_dict()['lines_of_code']}")
        
        results_file = results_dir / "ingestion_metrics_langchain.json"
        with open(results_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    main()

# Custom wrapper for LangChain to use nearVector in Weaviate queries
class LangchainWeaviateWrapper:
    def __init__(self, client, embeddings, index_name: str):
        self.client = client
        self.embeddings = embeddings
        self.index_name = index_name

    def similarity_search_with_score(self, query: str, k: int = 3):
        # Embed the query
        vector = self.embeddings.embed_query(query)
        # Build GraphQL query with nearVector
        qb = (
            self.client.query.get(self.index_name, ["content"])
            .with_near_vector({"vector": vector})
            .with_additional(["certainty"])
            .with_limit(k)
        )
        result = qb.do()
        objs = result.get("data", {}).get("Get", {}).get(self.index_name, []) or []
        results = []
        for obj in objs:
            content = obj.get("content", "")
            score = obj.get("_additional", {}).get("certainty", 0)
            # Use langchain Document for consistency
            results.append((Document(page_content=content, metadata={}), score))
        return results

def setup_langchain_vectorstore():
    client = setup_weaviate_client()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    # Return custom wrapper for query benchmarking
    return LangchainWeaviateWrapper(
        client=client,
        embeddings=embeddings,
        index_name="PaulGrahamEssay"
    )