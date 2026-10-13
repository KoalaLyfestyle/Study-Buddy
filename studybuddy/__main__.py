"""Command line: `python -m studybuddy import <folder>` and `python -m studybuddy search <query>`."""

from __future__ import annotations

import argparse
import os
import sys

from .embeddings import default_embedder
from .library import EmptyLibraryError, Library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studybuddy")
    parser.add_argument(
        "--index-dir",
        default=os.environ.get("STUDYBUDDY_INDEX_DIR", ".studybuddy"),
        help="where the index is stored (default: .studybuddy)",
    )
    parser.add_argument(
        "--embedder",
        default=None,
        help="'hashing' for an offline lexical stand-in; default is BAAI/bge-small-en",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="import or refresh a course folder")
    imp.add_argument("folder")
    search = sub.add_parser("search", help="search the imported library")
    search.add_argument("query")
    search.add_argument("--folder", default=os.environ.get("STUDYBUDDY_DATA_DIR", "data"))
    search.add_argument("-k", type=int, default=4)
    search.add_argument("--course")
    search.add_argument("--week", type=int)
    args = parser.parse_args(argv)

    if args.embedder:
        os.environ["STUDYBUDDY_EMBEDDER"] = args.embedder
    embedder = default_embedder()

    folder = args.folder
    library = Library(folder, args.index_dir, embedder)
    try:
        if args.command == "import":
            report = library.import_folder()
            print(report.summary())
            return 1 if report.failed else 0
        results = library.search(args.query, k=args.k, course=args.course, week=args.week)
        for r in results:
            snippet = " ".join(r.text.split())[:200]
            print(f"[{r.score:.3f}] {r.citation.label()}  ({r.chunk_id})\n    {snippet}")
        return 0
    except EmptyLibraryError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        library.close()


if __name__ == "__main__":
    sys.exit(main())
