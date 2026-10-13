"""Retrieval for the agent and the app, backed by the studybuddy package.

Nothing is loaded at import time. The first search imports the course
folder (incrementally, so only new or changed files are processed) and
later searches reuse the index.

Settings:
- STUDYBUDDY_DATA_DIR: course folder (default ./data)
- STUDYBUDDY_INDEX_DIR: index folder (default ./.studybuddy)
- STUDYBUDDY_EMBEDDER: embedding model (default BAAI/bge-small-en), or
  "hashing" for an offline lexical stand-in
"""

import os

from studybuddy import EmptyLibraryError, Library, default_embedder

_library = None
_last_report = None


def data_dir() -> str:
    return os.environ.get("STUDYBUDDY_DATA_DIR", "./data")


def get_library() -> Library:
    """Open the library and bring it up to date once per process."""
    global _library, _last_report
    if _library is None:
        library = Library(
            data_dir(),
            os.environ.get("STUDYBUDDY_INDEX_DIR", ".studybuddy"),
            default_embedder(),
        )
        try:
            _last_report = library.import_folder()
        except Exception:
            library.close()
            raise
        _library = library
    return _library


def last_import_report():
    return _last_report


def search(query: str, k: int = 4, course: str | None = None, week: int | None = None):
    return get_library().search(query, k=k, course=course, week=week)


def reset_index() -> None:
    """Forget the open library so the next search re-checks the folder."""
    global _library, _last_report
    if _library is not None:
        _library.close()
    _library = None
    _last_report = None


__all__ = ["EmptyLibraryError", "get_library", "last_import_report", "reset_index", "search"]
