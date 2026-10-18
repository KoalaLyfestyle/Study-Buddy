"""MCP server for the course library (stdio).

    STUDYBUDDY_DATA_DIR=sample_course python -m studybuddy.mcp_server

Tools: list_courses, search_materials, read_source, get_week_context.
They call the same functions as the CLI (studybuddy.api) through rag.py,
which refreshes the index from the course folder before each search.
No tool calls a generation model.
"""

from __future__ import annotations

from anyio import to_thread
from mcp.server.fastmcp import FastMCP

# rag.py lives at the repository root: run from there, or put the
# repository root on PYTHONPATH (see the README).
import rag

from . import api

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


# Tools run in a worker thread: the first search may load the embedding
# model, and the server must keep answering pings and cancels meanwhile.


@mcp.tool()
async def list_courses() -> dict:
    """List imported courses with their weeks and file counts."""
    return await to_thread.run_sync(lambda: api.list_courses(_library()))


def _search(query, k, course, week, kind) -> dict:
    errors = []

    def search(*args, **kwargs):
        results, error = rag.search_with_status(*args, **kwargs)
        if error is not None:
            errors.append(error)
        return results

    found = api.search_materials(search, query, k=k, course=course, week=week, kind=kind)
    if errors:
        found["warning"] = (
            f"Could not refresh the course library ({errors[0]}). "
            "These results come from the last successful import."
        )
    return found


@mcp.tool()
async def search_materials(
    query: str, k: int = 5, course: str | None = None, week: int | None = None, kind: str | None = None
) -> dict:
    """Search course material. Returns up to 10 passages with chunk IDs, snippets, and citations.

    Optional filters: course name, week number, and kind (pdf, markdown, notebook).
    """
    return await to_thread.run_sync(lambda: _search(query, k, course, week, kind))


@mcp.tool()
async def read_source(chunk_id: str, offset: int = 0) -> dict:
    """Read the full text of one passage by chunk ID, 4000 characters at a time."""
    return await to_thread.run_sync(lambda: api.read_source(_library(), chunk_id, offset=offset))


@mcp.tool()
async def get_week_context(course: str, week: int | None = None) -> dict:
    """List every passage in one course week, in reading order, with citations and short snippets.

    Leave week out for files that are not in a week folder.
    """
    return await to_thread.run_sync(lambda: api.get_week_context(_library(), course, week))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
