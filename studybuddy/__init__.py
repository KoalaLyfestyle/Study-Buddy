"""Study Buddy core: course import and retrieval with citations.

This package has no LangChain or LlamaIndex dependency. The agent, the
Streamlit app, and the CLI all call these functions.
"""

from .embeddings import HashingEmbedder, SentenceTransformerEmbedder, default_embedder
from .library import Citation, EmptyLibraryError, ImportReport, Library, SearchResult

__all__ = [
    "Citation",
    "EmptyLibraryError",
    "HashingEmbedder",
    "ImportReport",
    "Library",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "default_embedder",
]
