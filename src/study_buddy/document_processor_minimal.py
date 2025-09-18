"""Minimal document processor for testing without heavy dependencies."""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Document:
    """Simple document class."""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SimpleTextSplitter:
    """Simple text splitter."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": i}
                )
                result.append(new_doc)
        return result


class DocumentProcessor:
    """Handles document loading and text processing."""
    
    def __init__(self):
        from .config import config
        self.text_splitter = SimpleTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load a single document based on its file extension."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path)}
                    )
                    return [doc]
            elif file_extension == '.pdf':
                # For now, just return a placeholder
                logger.warning(f"PDF support requires PyPDF2. Skipping {file_path}")
                return []
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []
    
    def load_directory(self, directory_path: Path) -> List[Document]:
        """Load and process all supported documents from a directory."""
        all_documents = []
        supported_extensions = {'.txt'}  # Start with just .txt for minimal version
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
        
        if all_documents:
            processed_documents = self.process_documents(all_documents)
            logger.info(f"Processed {len(processed_documents)} total chunks from directory")
            return processed_documents
        else:
            logger.warning(f"No supported documents found in {directory_path}")
            return []
    
    def get_document_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a document file."""
        return {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
        }