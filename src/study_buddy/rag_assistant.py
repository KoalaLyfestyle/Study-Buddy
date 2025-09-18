"""Main RAG Assistant class that orchestrates all components."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import config
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_manager import LLMManager

logger = logging.getLogger(__name__)


class RAGAssistant:
    """Main RAG Assistant that combines document processing, vector storage, and LLM."""
    
    def __init__(self, model_name: str = None):
        """Initialize the RAG Assistant."""
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager(model_name)
        
        logger.info("RAG Assistant initialized successfully")
    
    def ingest_document(self, file_path: Path) -> bool:
        """Ingest a single document into the knowledge base."""
        try:
            # Load and process the document
            documents = self.document_processor.load_document(file_path)
            if not documents:
                logger.error(f"Failed to load document: {file_path}")
                return False
            
            # Process documents into chunks
            processed_docs = self.document_processor.process_documents(documents)
            if not processed_docs:
                logger.error(f"Failed to process document: {file_path}")
                return False
            
            # Add to vector store
            success = self.vector_store.add_documents(processed_docs)
            if success:
                logger.info(f"Successfully ingested document: {file_path.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            return False
    
    def ingest_directory(self, directory_path: Path) -> Dict[str, Any]:
        """Ingest all supported documents from a directory."""
        results = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "failed_files": []
        }
        
        try:
            supported_extensions = {'.pdf', '.txt'}
            files = [f for f in directory_path.iterdir() 
                    if f.is_file() and f.suffix.lower() in supported_extensions]
            
            results["total_files"] = len(files)
            
            for file_path in files:
                if self.ingest_document(file_path):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["failed_files"].append(file_path.name)
            
            logger.info(f"Directory ingestion complete: {results['successful']}/{results['total_files']} successful")
            return results
            
        except Exception as e:
            logger.error(f"Error ingesting directory {directory_path}: {str(e)}")
            return results
    
    def query(self, question: str, max_results: int = None) -> Dict[str, Any]:
        """Query the knowledge base and get an AI-generated response."""
        if max_results is None:
            max_results = config.MAX_RESULTS
        
        try:
            # Check if Ollama is running
            if not self.llm_manager.check_ollama_connection():
                return {
                    "question": question,
                    "answer": "Error: Ollama server is not running. Please start Ollama first.",
                    "context_documents": [],
                    "error": "Ollama not available"
                }
            
            # Retrieve relevant documents
            context_docs = self.vector_store.similarity_search(question, k=max_results)
            
            if not context_docs:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant documents in the knowledge base for your question.",
                    "context_documents": [],
                    "error": "No relevant documents found"
                }
            
            # Generate response using LLM
            response_data = self.llm_manager.ask_question(question, context_docs)
            
            # Add context information
            response_data["context_documents"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "metadata": doc.metadata
                }
                for doc in context_docs
            ]
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                "question": question,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "context_documents": [],
                "error": str(e)
            }
    
    def search_documents(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search for documents without generating an AI response."""
        if max_results is None:
            max_results = config.MAX_RESULTS
        
        try:
            # Perform similarity search with scores
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=max_results)
            
            search_results = []
            for doc, score in results_with_scores:
                search_results.append({
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "Unknown")
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error during document search: {str(e)}")
            return []
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the current knowledge base."""
        try:
            collection_info = self.vector_store.get_collection_info()
            available_models = self.llm_manager.list_available_models()
            ollama_status = self.llm_manager.check_ollama_connection()
            
            return {
                "vector_store": collection_info,
                "llm": {
                    "current_model": self.llm_manager.model_name,
                    "available_models": available_models,
                    "ollama_running": ollama_status
                },
                "config": {
                    "embedding_model": config.EMBEDDING_MODEL,
                    "chunk_size": config.CHUNK_SIZE,
                    "max_results": config.MAX_RESULTS
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {str(e)}")
            return {"error": str(e)}
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        try:
            return self.vector_store.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            return False