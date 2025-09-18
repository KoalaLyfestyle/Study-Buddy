"""Vector store management using ChromaDB and HuggingFace embeddings."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import sqlite3

# Try to import full dependencies, fallback to minimal implementation
try:
    import chromadb
    from chromadb.config import Settings
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    FULL_FEATURES = True
except ImportError:
    FULL_FEATURES = False

from .config import config

logger = logging.getLogger(__name__)


class MinimalVectorStore:
    """Minimal vector store implementation for testing without dependencies."""
    
    def __init__(self):
        self.db_path = config.VECTOR_STORE_DIR / "minimal_store.db"
        self.documents = []
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for storing documents."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_documents(self, documents):
        """Add documents to the minimal store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for doc in documents:
            content = doc.page_content
            metadata = json.dumps(doc.metadata)
            cursor.execute(
                'INSERT INTO documents (content, metadata) VALUES (?, ?)',
                (content, metadata)
            )
        
        conn.commit()
        conn.close()
        logger.info(f"Added {len(documents)} documents to minimal store")
        return True
    
    def similarity_search(self, query: str, k: int = 5):
        """Simple text search (not semantic)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Simple keyword search
        query_lower = query.lower()
        cursor.execute('SELECT content, metadata FROM documents')
        rows = cursor.fetchall()
        
        results = []
        for content, metadata_str in rows:
            if query_lower in content.lower():
                from .document_processor_minimal import Document
                doc = Document(
                    page_content=content,
                    metadata=json.loads(metadata_str)
                )
                results.append(doc)
                if len(results) >= k:
                    break
        
        conn.close()
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Simple search with dummy scores."""
        docs = self.similarity_search(query, k)
        return [(doc, 0.5) for doc in docs]  # Dummy scores


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self):
        if FULL_FEATURES:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            self.vector_store = None
            self._initialize_store()
        else:
            logger.warning("Using minimal vector store. Install chromadb and sentence-transformers for full features.")
            self.vector_store = MinimalVectorStore()
    
    def _initialize_store(self):
        """Initialize the ChromaDB vector store."""
        try:
            # Create ChromaDB client with persistent storage
            chroma_client = chromadb.PersistentClient(
                path=str(config.VECTOR_STORE_DIR)
            )
            
            # Initialize Langchain Chroma wrapper
            self.vector_store = Chroma(
                client=chroma_client,
                collection_name=config.COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents) -> bool:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided for indexing")
            return False
        
        try:
            # Add documents to the vector store
            result = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = None):
        """Perform similarity search on the vector store."""
        if k is None:
            k = config.MAX_RESULTS
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None):
        """Perform similarity search with relevance scores."""
        if k is None:
            k = config.MAX_RESULTS
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search with scores: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            if FULL_FEATURES:
                # Get the underlying ChromaDB collection
                collection = self.vector_store._collection
                count = collection.count()
            else:
                # Count from minimal store
                import sqlite3
                conn = sqlite3.connect(self.vector_store.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM documents')
                count = cursor.fetchone()[0]
                conn.close()
            
            return {
                "collection_name": config.COLLECTION_NAME,
                "document_count": count,
                "embedding_model": config.EMBEDDING_MODEL,
                "vector_store_path": str(config.VECTOR_STORE_DIR),
                "full_features": FULL_FEATURES
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            if FULL_FEATURES:
                # Delete the collection and recreate it
                collection = self.vector_store._collection
                collection.delete()
                
                # Reinitialize the vector store
                self._initialize_store()
            else:
                # Clear minimal store
                import sqlite3
                conn = sqlite3.connect(self.vector_store.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM documents')
                conn.commit()
                conn.close()
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False