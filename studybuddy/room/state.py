"""Reading progress and authored edits, kept apart from the built room.

Both files are keyed by section or question ID, so a rebuild (which
replaces room.json) never touches them. IDs that no longer exist in the
room are kept in the file and reported, so nothing is lost if a section
comes back.

progress.json: {"read": [section_id, ...], "answers": {question_id: choice_index}}
edits.json:    {section_id: {"title": str?, "blocks": [block, ...]?}}
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import RoomValidationError, validate_room


@dataclass
class RoomState:
    read: list[str] = field(default_factory=list)
    answers: dict[str, int] = field(default_factory=dict)
    edits: dict[str, dict] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)  # saved IDs the room no longer has


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RoomValidationError([f"{path.name}: not valid JSON: {exc.msg} at line {exc.lineno}"]) from None


def load_state(room_dir: Path) -> RoomState:
    progress = _load_json(room_dir / "progress.json", {})
    edits = _load_json(room_dir / "edits.json", {})
    problems = []
    if not isinstance(progress, dict):
        problems.append("progress.json: must be an object")
        progress = {}
    if not isinstance(edits, dict):
        problems.append("edits.json: must be an object")
        edits = {}
    read = progress.get("read", [])
    answers = progress.get("answers", {})
    if not (isinstance(read, list) and all(isinstance(x, str) for x in read)):
        problems.append("progress.json/read: must be a list of section IDs")
    if not (
        isinstance(answers, dict)
        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in answers.values())
    ):
        problems.append("progress.json/answers: must map question IDs to choice numbers (0 or more)")
    for sid, edit in edits.items():
        if not isinstance(edit, dict) or set(edit) - {"title", "blocks"}:
            problems.append(f"edits.json/{sid}: only 'title' and 'blocks' can be edited")
    if problems:
        raise RoomValidationError(problems)
    return RoomState(read=read, answers=answers, edits=edits)


def apply_state(room: dict, state: RoomState) -> tuple[dict, RoomState]:
    """Return the room with authored edits applied, and the state with orphans listed.

    An edited section is checked against the schema like the rest of the room,
    so a bad edit raises RoomValidationError and is not shown half-applied.
    """
    merged = copy.deepcopy(room)
    section_ids: set[str] = set()
    choices: dict[str, int] = {}
    for chapter in merged["chapters"]:
        for section in chapter["sections"]:
            section_ids.add(section["id"])
            choices.update({q["id"]: len(q["choices"]) for q in section.get("quiz", [])})
            edit = state.edits.get(section["id"])
            if edit:
                section.update({k: copy.deepcopy(v) for k, v in edit.items()})
                section["edited"] = True
    # "edited" is a render-time marker, not part of the stored format.
    check = copy.deepcopy(merged)
    for chapter in check["chapters"]:
        for section in chapter["sections"]:
            section.pop("edited", None)
    try:
        validate_room(check)
    except RoomValidationError as exc:
        raise RoomValidationError([f"edits.json: {p}" for p in exc.problems]) from None

    orphans = sorted(
        {sid for sid in state.read if sid not in section_ids}
        | {sid for sid in state.edits if sid not in section_ids}
        | {qid for qid in state.answers if qid not in choices}
    )
    # An answer past the last choice (the question changed) is treated like a missing one.
    answers = {q: a for q, a in state.answers.items() if q not in choices or a < choices[q]}
    return merged, RoomState(read=state.read, answers=answers, edits=state.edits, orphans=orphans)
