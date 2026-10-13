"""Plain functions behind the CLI and the MCP server.

Each returns JSON-ready dicts with stable chunk IDs, short snippets, and
citations. Output is bounded: at most MAX_K search results, snippets of
SNIPPET_CHARS, and reads of at most MAX_READ_CHARS per call. None of these
calls a generation model; search only uses the local embedder.
"""

from __future__ import annotations

from typing import Callable

from .library import Library, SearchResult
from .sources import SUPPORTED_SUFFIXES

MAX_K = 10
SNIPPET_CHARS = 300
MAX_READ_CHARS = 4000
MAX_OUTLINE_ITEMS = 50
MAX_QUERY_CHARS = 500
MAX_COURSES = 100
MAX_FAILURES = 20
KINDS = sorted(set(SUPPORTED_SUFFIXES.values()))

SearchFn = Callable[..., list[SearchResult]]


def citation_dict(r: SearchResult) -> dict:
    c = r.citation
    return {
        "label": c.label(),
        "path": c.relpath,
        "course": c.course,
        "week": c.week,
        "kind": c.kind,
        "location": c.location,
        "stale": c.stale,
    }


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def list_courses(library: Library) -> dict:
    """Courses with their weeks and file counts, plus files that failed to import."""
    courses = library.courses()
    failures = sorted(library.failures().items())
    return {
        "courses": courses[:MAX_COURSES],
        "failed_files": {path: (error or "")[:200] for path, error in failures[:MAX_FAILURES]},
        "truncated": len(courses) > MAX_COURSES or len(failures) > MAX_FAILURES,
    }


def search_materials(
    search: SearchFn,
    query: str,
    k: int = 5,
    course: str | None = None,
    week: int | None = None,
    kind: str | None = None,
    full_text: bool = False,
) -> dict:
    """Ranked passages with snippets and citations. Filters apply before ranking.

    full_text adds each passage's whole text (the agent quotes it; the MCP
    server leaves it out and offers read_source instead).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    results = search(query, k=k, course=course, week=week, kind=kind)
    return {
        "query": query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "score": round(r.score, 4),
                "snippet": _snippet(r.text),
                "citation": citation_dict(r),
                **({"text": r.text[:MAX_READ_CHARS]} if full_text else {}),
            }
            for r in results
        ],
    }


def read_source(library: Library, chunk_id: str, offset: int = 0, max_chars: int = MAX_READ_CHARS) -> dict:
    """The full text of one passage, in pages of at most MAX_READ_CHARS."""
    if not 1 <= max_chars <= MAX_READ_CHARS:
        raise ValueError(f"max_chars must be between 1 and {MAX_READ_CHARS}")
    if offset < 0:
        raise ValueError("offset must not be negative")
    r = library.read_chunk(chunk_id)
    if r is None:
        raise ValueError(f"no passage with id {chunk_id!r}; use search_materials to find one")
    end = offset + max_chars
    return {
        "chunk_id": r.chunk_id,
        "text": r.text[offset:end],
        "offset": offset,
        "total_chars": len(r.text),
        "next_offset": end if end < len(r.text) else None,
        "citation": citation_dict(r),
    }


def get_week_context(library: Library, course: str, week: int | None) -> dict:
    """What one course week contains: each passage's ID, citation, and a snippet, in reading order."""
    items = library.outline(course, week)
    if not items:
        known = {c["course"]: c["weeks"] for c in library.courses()}
        if course not in known:
            if not known:
                raise ValueError("the library is empty; import a course folder first: python -m studybuddy import <folder>")
            raise ValueError(f"unknown course {course!r}; known: {', '.join(sorted(known))}")
        raise ValueError(f"course {course!r} has no material for week {week}; weeks: {known[course]}")
    return {
        "course": course,
        "week": week,
        "total_passages": len(items),
        "passages": [
            {"chunk_id": r.chunk_id, "citation": citation_dict(r), "snippet": _snippet(r.text, 160)}
            for r in items[:MAX_OUTLINE_ITEMS]
        ],
        "truncated": len(items) > MAX_OUTLINE_ITEMS,
    }
