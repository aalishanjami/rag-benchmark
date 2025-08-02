#!/usr/bin/env python3
"""
Complete RAG benchmarking pipeline for LangChain.
Runs both ingestion and query benchmarking in sequence.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    issues = []
    
    # Check if we're in the right directory
    if not Path("scripts/ingest_langchain.py").exists():
        issues.append("Please run this script from the rag-benchmark directory")
    
    # # Check for OpenAI API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     issues.append("OPENAI_API_KEY environment variable not set")
    
    # Check for essays
    essays_dir = Path("data/essays")
    if not essays_dir.exists() or not any(essays_dir.glob("*.md")):
        issues.append(f"No .md files found in {essays_dir}")
    
    return issues

def run_command(command, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=False)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed in {end_time - start_time:.1f} seconds")
        return True

def main():
    """Run the complete benchmarking pipeline."""
    
    print("🚀 RAG Benchmark Suite - LangChain Pipeline")
    print("=" * 60)
    
    # Pre-flight checks
    issues = check_requirements()
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the benchmark.")
        return 1
    
    print("✅ Pre-flight checks passed")
    
    # Step 1: Install dependencies (optional)
    response = input("\nInstall/update dependencies? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        if not run_command("pip install -r requirements.txt", "Installing dependencies"):
            return 1
    
    # Step 2: Run ingestion
    if not run_command("python scripts/ingest_langchain.py", "Data ingestion"):
        return 1
    
    # Step 3: Run queries
    if not run_command("python scripts/query_benchmark.py", "Query benchmarking"):
        return 1
    
    # Step 4: Show results summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("*.json"):
            print(f"📄 {result_file}")
    
    print("\n🎉 Benchmark pipeline completed successfully!")
    print("\nNext steps:")
    print("  - Review results in the results/ directory")
    print("  - Compare with other frameworks (coming soon)")
    print("  - Tune parameters for better performance")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())