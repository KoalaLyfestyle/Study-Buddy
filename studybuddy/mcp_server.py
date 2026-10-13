"""MCP server for the course library (stdio).

    STUDYBUDDY_DATA_DIR=sample_course python -m studybuddy.mcp_server

Tools: list_courses, search_materials, read_source, get_week_context.
They call the same functions as the CLI (studybuddy.api) through rag.py,
which refreshes the index from the course folder before each search.
No tool calls a generation model.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# rag.py lives at the repository root, next to this package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402

from . import api  # noqa: E402

mcp = FastMCP(
    "study-buddy",
    instructions=(
        "Search the user's course material (PDF, Markdown, notebooks). Cite each claim with the "
        "citation label of the passage it came from, for example 'notes.md, Week 1 > Stopping' "
        "or 'lecture.pdf, page 2'. Use read_source for the full passage."
    ),
)


def _library():
    library = rag.get_library()
    if library.chunk_count() == 0:
        rag.refresh()  # raises EmptyLibraryError with import advice
    return library


@mcp.tool()
def list_courses() -> dict:
    """List imported courses with their weeks and file counts."""
    return api.list_courses(_library())


@mcp.tool()
def search_materials(
    query: str, k: int = 5, course: str | None = None, week: int | None = None, kind: str | None = None
) -> dict:
    """Search course material. Returns up to 10 passages with chunk IDs, snippets, and citations.

    Optional filters: course name, week number, and kind (pdf, markdown, notebook).
    """
    return api.search_materials(rag.search, query, k=k, course=course, week=week, kind=kind)


@mcp.tool()
def read_source(chunk_id: str, offset: int = 0) -> dict:
    """Read the full text of one passage by chunk ID, 4000 characters at a time."""
    return api.read_source(_library(), chunk_id, offset=offset)


@mcp.tool()
def get_week_context(course: str, week: int) -> dict:
    """List every passage in one course week, in reading order, with citations and short snippets."""
    return api.get_week_context(_library(), course, week)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
