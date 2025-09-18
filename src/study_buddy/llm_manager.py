"""LLM integration using Ollama for local inference."""

import logging
from typing import List, Dict, Any, Optional
import requests

# Try to import full dependencies
try:
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    FULL_FEATURES = True
except ImportError:
    FULL_FEATURES = False

from .config import config

logger = logging.getLogger(__name__)


class MockLLM:
    """Mock LLM for testing without Ollama."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def __call__(self, prompt: str) -> str:
        """Generate a mock response."""
        return (
            f"[Mock Response from {self.model_name}]\n"
            f"This is a simulated response to your query. "
            f"To get real AI responses, please install Ollama and the required models.\n"
            f"Your prompt was: {prompt[:100]}..."
        )


class LLMManager:
    """Manages LLM interactions using Ollama."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            if FULL_FEATURES and self.check_ollama_connection():
                self.llm = Ollama(
                    base_url=config.OLLAMA_BASE_URL,
                    model=self.model_name
                )
                logger.info(f"LLM initialized with model: {self.model_name}")
            else:
                logger.warning("Using mock LLM. Install Ollama and langchain for real AI responses.")
                self.llm = MockLLM(self.model_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.llm = MockLLM(self.model_name)
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_available_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def generate_response(self, query: str, context_documents) -> str:
        """Generate a response using the LLM with provided context."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        # Create context from documents
        context = self._format_context(context_documents)
        
        # Create prompt
        if FULL_FEATURES:
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""Use the following context to answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
            )
            formatted_prompt = prompt_template.format(context=context, question=query)
        else:
            formatted_prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response
            response = self.llm(formatted_prompt)
            logger.info("Generated response successfully")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _format_context(self, documents) -> str:
        """Format context documents for the prompt."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Get document content and metadata
            content = doc.page_content
            source = doc.metadata.get("source", "Unknown source")
            
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_summary(self, text: str) -> str:
        """Generate a summary of the provided text."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        if FULL_FEATURES:
            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="""Please provide a concise summary of the following text:

{text}

Summary:"""
            )
            formatted_prompt = prompt_template.format(text=text)
        else:
            formatted_prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""
        
        try:
            response = self.llm(formatted_prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def ask_question(self, question: str, context_documents = None) -> Dict[str, Any]:
        """Ask a question and get a structured response."""
        if context_documents is None:
            context_documents = []
        
        response = self.generate_response(question, context_documents)
        
        return {
            "question": question,
            "answer": response,
            "context_used": len(context_documents),
            "model": self.model_name,
            "mock_mode": not FULL_FEATURES or not self.check_ollama_connection()
        }