"""Command line: import, search, courses, read, week, and room.

`courses`, `read`, and `week` print the same JSON the MCP server returns
(studybuddy.api), so the two stay in step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import api
from .embeddings import default_embedder
from .library import EmptyLibraryError, Library
from .room import (
    RoomValidationError,
    build_room,
    generate_room,
    ollama_model,
    parse_room,
    room_dir_for,
    save_room,
    week_chunk_ids,
    week_texts,
)


def _room(library: Library, args) -> int:
    if args.from_json:
        try:
            with open(args.from_json, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise RoomValidationError([f"{args.from_json}: cannot read it: {exc}"]) from None
        room = parse_room(text)
        if (room["course"], room["week"]) != (args.course, args.week):
            raise RoomValidationError([f"/course, /week: the file is for {room['course']!r} week {room['week']}"])
        room = parse_room(text, week_chunk_ids(library, args.course, args.week))
    else:
        room = build_room(library, args.course, args.week)
        if args.generate:
            try:
                model = ollama_model(args.model)
            except ImportError as exc:
                print(f"--generate needs langchain-ollama ({exc}). Install it with: pip install -r requirements.lock", file=sys.stderr)
                return 2
            room, report = generate_room(room, week_texts(library, args.course, args.week), model, f"ollama:{args.model}")
            print(report.summary())
    out = room_dir_for(library.index_dir, args.course, args.week) if not args.out else Path(args.out)
    saved = save_room(library, room, out)
    print(
        f"Built {saved.sections} sections and {saved.questions} quiz questions; "
        f"{saved.uncovered} passages are not covered.\nOpen {saved.html}"
    )
    if saved.orphans:
        print(f"{len(saved.orphans)} saved progress or edit IDs no longer match a section; they were kept.")
    return 0


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
    room = sub.add_parser("room", help="build a study room page for one course week")
    room.add_argument("course")
    room.add_argument("week", type=int)
    source = room.add_mutually_exclusive_group()
    source.add_argument("--from-json", metavar="FILE", help="use this room JSON (for example model output) instead of building one")
    source.add_argument("--generate", action="store_true", help="add sections written by the local Ollama model")
    room.add_argument("--model", default="mistral", help="Ollama model for --generate (default: mistral)")
    room.add_argument("--out", help="output folder (default: <index-dir>/rooms/<course>/week-NN)")
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
        if args.command == "room":
            return _room(library, args)
        if args.command == "week":
            print(json.dumps(api.get_week_context(library, args.course, args.week), indent=2))
            return 0
        found = api.search_materials(
            library.search, args.query, k=max(1, min(args.k, api.MAX_K)), course=args.course, week=args.week, kind=args.kind
        )
        for r in found["results"]:
            print(f"[{r['score']:.3f}] {r['citation']['label']}  ({r['chunk_id']})\n    {r['snippet']}")
        return 0
    except RoomValidationError as exc:
        print(f"{exc}\nThe previous room, if any, was left as it was.", file=sys.stderr)
        return 2
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
