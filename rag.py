"""Retrieval for the agent and the app, backed by the studybuddy package.

Nothing is loaded at import time. Each search first refreshes the index
from the course folder. That is cheap: unchanged files are only hashed,
and only new or changed files are parsed and embedded.

Settings:
- STUDYBUDDY_DATA_DIR: course folder (default ./data)
- STUDYBUDDY_INDEX_DIR: index folder (default ./.studybuddy)
- STUDYBUDDY_EMBEDDER: embedding model (default BAAI/bge-small-en), or
  "hashing" for an offline lexical stand-in
"""

import os
import threading

from studybuddy import EmptyLibraryError, Library, default_embedder

_lock = threading.Lock()
_library = None
_last_report = None
_last_refresh_error = None


def data_dir() -> str:
    return os.environ.get("STUDYBUDDY_DATA_DIR", "./data")


def get_library() -> Library:
    """Open the library once per process."""
    global _library
    with _lock:
        if _library is None:
            _library = Library(
                data_dir(),
                os.environ.get("STUDYBUDDY_INDEX_DIR", ".studybuddy"),
                default_embedder(),
            )
        return _library


def refresh():
    """Import new and changed files. Raises EmptyLibraryError with advice."""
    global _last_report
    _last_report = get_library().import_folder()
    return _last_report


def last_import_report():
    return _last_report


def last_refresh_error():
    """The error from the latest refresh, if search fell back to the old index."""
    return _last_refresh_error


def search(
    query: str, k: int = 4, course: str | None = None, week: int | None = None, kind: str | None = None
):
    """Refresh, then search. If the refresh fails but an index exists, search
    the last good index and record the error in last_refresh_error()."""
    global _last_refresh_error
    library = get_library()
    try:
        refresh()
        _last_refresh_error = None
    except Exception as exc:
        if library.chunk_count() == 0:
            raise
        _last_refresh_error = exc
    return library.search(query, k=k, course=course, week=week, kind=kind)


def reset_index() -> None:
    """Close the library so the next call reopens it with current settings."""
    global _library, _last_report, _last_refresh_error
    with _lock:
        if _library is not None:
            _library.close()
        _library = None
        _last_report = None
        _last_refresh_error = None


__all__ = [
    "EmptyLibraryError",
    "get_library",
    "last_import_report",
    "last_refresh_error",
    "refresh",
    "reset_index",
    "search",
]
