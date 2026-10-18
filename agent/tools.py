from rag import EmptyLibraryError, search_with_status
from studybuddy import api


def search_notes(query: str) -> str:
    """Search the course library. Each result names its source location."""
    errors = []

    def search(*args, **kwargs):
        results, error = search_with_status(*args, **kwargs)
        if error is not None:
            errors.append(error)
        return results

    try:
        results = api.search_materials(search, query, k=4, full_text=True)["results"]
    except (EmptyLibraryError, ValueError) as exc:
        return str(exc)
    note = ""
    if errors:
        note = (
            f"Note: could not refresh the course library ({errors[0]}). "
            "These results come from the last successful import.\n\n"
        )
    if not results:
        return note + "No matching passages found in the course library."
    return note + "\n\n".join(f"[{r['citation']['label']}]\n{r['text']}" for r in results)


def quiz_me(topic: str) -> str:
    """Placeholder: returns a fixed template. It does not read your notes yet."""
    return f"Here are 3 practice questions about {topic}:\n1. ...?\n2. ...?\n3. ...?"


def summarize(text: str) -> str:
    """Placeholder: returns the first 200 characters. It does not summarise yet."""
    return f"Summary: {text[:200]}..."
