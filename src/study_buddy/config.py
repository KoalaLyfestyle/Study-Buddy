"""Configuration management for Study Buddy RAG assistant."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration."""
    
    def __init__(self):
        # Paths
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.DOCUMENTS_DIR = self.DATA_DIR / "documents"
        self.VECTOR_STORE_DIR = self.DATA_DIR / "vector_store"
        
        # Embedding settings
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Vector store settings
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "study_documents")
        
        # Ollama settings
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistral")
        
        # CLI settings
        self.MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
        
        # Create necessary directories
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()