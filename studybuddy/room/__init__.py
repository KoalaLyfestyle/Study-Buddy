"""Study rooms: one course week as a navigable page with quizzes and sources.

    room = build_room(library, course, week)        # extractive, no model
    result = save_room(library, room, room_dir)    # validate, merge state, write

A room from a model goes through `parse_room(text, known_chunks)` first;
a malformed one raises RoomValidationError and leaves the last good room.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..library import Library
from .build import BUILDER, build_room
from .model import RoomValidationError, dumps, parse_room, validate_room, write_atomic
from .render import render_room, render_source_pages
from .state import RoomState, apply_state, load_state

__all__ = [
    "BUILDER",
    "RoomState",
    "RoomValidationError",
    "SavedRoom",
    "build_room",
    "parse_room",
    "render_room",
    "room_dir_for",
    "save_room",
    "validate_room",
    "week_chunk_ids",
]


def room_dir_for(index_dir: Path, course: str, week: int | None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", course).strip(".-") or "course"
    if safe != course:  # keep "a b" and "a-b" apart
        safe += "-" + hashlib.sha256(course.encode("utf-8")).hexdigest()[:6]
    return Path(index_dir) / "rooms" / safe / (f"week-{week:02d}" if week is not None else "other")


def week_chunk_ids(library: Library, course: str, week: int | None) -> set[str]:
    return {r.chunk_id for r in library.outline(course, week)}


@dataclass
class SavedRoom:
    room_json: Path
    html: Path
    sections: int
    questions: int
    uncovered: int
    orphans: list[str]


def save_room(library: Library, room: dict, room_dir: Path) -> SavedRoom:
    """Check the room against the library, apply saved edits, and write it.

    Everything is checked and rendered before anything is written, so a bad
    room or a bad edits.json keeps the previous room untouched.
    progress.json and edits.json are read, never rewritten.
    """
    validate_room(room, week_chunk_ids(library, room["course"], room["week"]))
    state = load_state(room_dir)
    merged, state = apply_state(room, state)
    html = render_room(merged, state, library)
    source_pages = render_source_pages(room, library)
    room_json, page = room_dir / "room.json", room_dir / "index.html"
    # Each file is replaced atomically. room.json goes last: it is what marks
    # a build as finished, so if an earlier write fails it still describes
    # the previous room.
    for relname, text in sorted(source_pages.items()):
        write_atomic(room_dir / relname, text)
    write_atomic(page, html)
    write_atomic(room_json, dumps(room))
    for stale in sorted((room_dir / "sources").glob("*.html")):
        if f"sources/{stale.name}" not in source_pages:
            stale.unlink()
    sections = [s for c in room["chapters"] for s in c["sections"]]
    return SavedRoom(
        room_json=room_json,
        html=page,
        sections=len(sections),
        questions=sum(len(s.get("quiz", [])) for s in sections),
        uncovered=sum(1 for ids in room["coverage"].values() if not ids),
        orphans=state.orphans,
    )
