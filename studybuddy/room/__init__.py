"""Study rooms: one course week as a navigable page with quizzes and sources.

    room = build_room(library, course, week)        # extractive, no model
    result = save_room(library, room, room_dir)    # validate, merge state, write

A room from a model goes through `parse_room(text, known_chunks)` first;
a malformed one raises RoomValidationError and leaves the last good room.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..library import Library
from .build import BUILDER, build_room
from .model import RoomValidationError, dumps, parse_room, validate_room, write_atomic
from .render import render_room
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

    Everything is checked and rendered before anything is written, and each
    file is replaced atomically, so a failure keeps the previous room.
    progress.json and edits.json are read, never rewritten.
    """
    validate_room(room, week_chunk_ids(library, room["course"], room["week"]))
    state = load_state(room_dir)
    merged, state = apply_state(room, state)
    html = render_room(merged, state, library)
    room_json, page = room_dir / "room.json", room_dir / "index.html"
    write_atomic(room_json, dumps(room))
    write_atomic(page, html)
    sections = [s for c in room["chapters"] for s in c["sections"]]
    return SavedRoom(
        room_json=room_json,
        html=page,
        sections=len(sections),
        questions=sum(len(s.get("quiz", [])) for s in sections),
        uncovered=sum(1 for ids in room["coverage"].values() if not ids),
        orphans=state.orphans,
    )
