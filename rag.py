from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set embedding model once for everything
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Global variables for caching
_index = None
_retriever = None

def load_docs():
    """Load documents from the data directory."""
    return SimpleDirectoryReader("./data").load_data()

def build_index():
    """Build a vector store index from the loaded documents."""
    docs = load_docs()
    return VectorStoreIndex.from_documents(docs)

def get_retriever():
    """Get a retriever from the built index. Builds index only once per session."""
    global _index, _retriever
    
    # Only build the index if it hasn't been built yet
    if _index is None:
        print("Building index for the first time...")
        _index = build_index()
        print("Index built successfully!")
    
    # Only create retriever if it hasn't been created yet
    if _retriever is None:
        _retriever = _index.as_retriever()
    
    return _retriever

def reset_index():
    """Reset the cached index and retriever. Useful for reloading data."""
    global _index, _retriever
    _index = None
    _retriever = None