from rag import EmptyLibraryError, last_refresh_error, search


def search_notes(query: str) -> str:
    """Search the course library. Each result names its source location."""
    try:
        results = search(query, k=4)
    except EmptyLibraryError as exc:
        return str(exc)
    note = ""
    if last_refresh_error() is not None:
        note = (
            f"Note: could not refresh the course library ({last_refresh_error()}). "
            "These results come from the last successful import.\n\n"
        )
    if not results:
        return note + "No matching passages found in the course library."
    return note + "\n\n".join(f"[{r.citation.label()}]\n{r.text}" for r in results)


def quiz_me(topic: str) -> str:
    """Placeholder: returns a fixed template. It does not read your notes yet."""
    return f"Here are 3 practice questions about {topic}:\n1. ...?\n2. ...?\n3. ...?"


def summarize(text: str) -> str:
    """Placeholder: returns the first 200 characters. It does not summarise yet."""
    return f"Summary: {text[:200]}..."
