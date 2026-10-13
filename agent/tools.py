from rag import EmptyLibraryError, search


def search_notes(query: str) -> str:
    """Search the course library. Each result names its source location."""
    try:
        results = search(query, k=4)
    except EmptyLibraryError as exc:
        return str(exc)
    if not results:
        return "No matching passages found in the course library."
    return "\n\n".join(f"[{r.citation.label()}]\n{r.text}" for r in results)


def quiz_me(topic: str) -> str:
    """Placeholder: returns a fixed template. It does not read your notes yet."""
    return f"Here are 3 practice questions about {topic}:\n1. ...?\n2. ...?\n3. ...?"


def summarize(text: str) -> str:
    """Placeholder: returns the first 200 characters. It does not summarise yet."""
    return f"Summary: {text[:200]}..."
