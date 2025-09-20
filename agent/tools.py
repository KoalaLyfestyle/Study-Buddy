from rag import get_retriever

def get_cached_retriever():
    """Get the cached retriever instance."""
    return get_retriever()

def search_notes(query: str) -> str:
    """Search your study notes with RAG."""
    retriever = get_cached_retriever()
    results = retriever.retrieve(query)
    return "\n".join([r.text for r in results])

def quiz_me(topic: str) -> str:
    """Generate simple quiz questions."""
    return f"Here are 3 practice questions about {topic}:\n1. ...?\n2. ...?\n3. ...?"

def summarize(text: str) -> str:
    """Summarize a chunk of text."""
    return f"Summary: {text[:200]}..."