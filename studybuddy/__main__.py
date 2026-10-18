"""Command line: import, search, courses, read, and week.

`courses`, `read`, and `week` print the same JSON the MCP server returns
(studybuddy.api), so the two stay in step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import api
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
    search.add_argument("--folder", default=None, help="default: the last imported folder")
    search.add_argument("-k", type=int, default=4)
    search.add_argument("--course")
    search.add_argument("--week", type=int)
    search.add_argument("--kind", choices=api.KINDS)
    sub.add_parser("courses", help="list imported courses (JSON)")
    read = sub.add_parser("read", help="print one passage by chunk ID (JSON)")
    read.add_argument("chunk_id")
    read.add_argument("--offset", type=int, default=0)
    week = sub.add_parser("week", help="list the passages of one course week (JSON)")
    week.add_argument("course")
    week.add_argument("week", type=int)
    args = parser.parse_args(argv)

    if args.embedder:
        os.environ["STUDYBUDDY_EMBEDDER"] = args.embedder
    embedder = default_embedder()

    folder = getattr(args, "folder", None)
    library = Library(folder, args.index_dir, embedder)
    try:
        if args.command == "import":
            report = library.import_folder()
            print(report.summary())
            return 1 if report.failed else 0
        if args.command == "courses":
            print(json.dumps(api.list_courses(library), indent=2))
            return 0
        if args.command == "read":
            print(json.dumps(api.read_source(library, args.chunk_id, offset=args.offset), indent=2))
            return 0
        if args.command == "week":
            print(json.dumps(api.get_week_context(library, args.course, args.week), indent=2))
            return 0
        found = api.search_materials(
            library.search, args.query, k=max(1, min(args.k, api.MAX_K)), course=args.course, week=args.week, kind=args.kind
        )
        for r in found["results"]:
            print(f"[{r['score']:.3f}] {r['citation']['label']}  ({r['chunk_id']})\n    {r['snippet']}")
        return 0
    except EmptyLibraryError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        library.close()


if __name__ == "__main__":
    sys.exit(main())
