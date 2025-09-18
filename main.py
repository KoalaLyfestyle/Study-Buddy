#!/usr/bin/env python3
"""
Study Buddy RAG Assistant
A lightweight offline study assistant using RAG with ChromaDB, HuggingFace embeddings, and Ollama.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from study_buddy.cli import cli

if __name__ == "__main__":
    cli()