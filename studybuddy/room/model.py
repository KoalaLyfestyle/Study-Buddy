"""Load, check, and save the study room format (schema.json).

Every room, from the extractive builder or from a model, goes through
`validate_room` before it is saved or rendered. A room that fails raises
`RoomValidationError` with the JSON path of each problem, and nothing is
written, so the last good room stays in place.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("schema.json")


class RoomValidationError(ValueError):
    """The room is malformed. `problems` lists "path: message" strings."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        shown = "\n".join(f"- {p}" for p in problems[:20])
        more = f"\n- and {len(problems) - 20} more" if len(problems) > 20 else ""
        super().__init__(f"The study room is not valid ({len(problems)} problems):\n{shown}{more}")


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def location_key(location: dict) -> str:
    """Where a passage lives, without line numbers: page, cell ID, or heading path."""
    if "page" in location:
        return f"page:{location['page']}"
    if "cell_id" in location:
        return f"cell:{location['cell_id']}"
    return "h:" + " > ".join(location.get("headings", []))


def _pointer(path: Iterable) -> str:
    return "/" + "/".join(str(p) for p in path)


def validate_room(room: object, known_chunks: set[str] | None = None) -> dict:
    """Check the schema, then the rules a schema cannot state.

    known_chunks: if given, every cite must be one of these chunk IDs, and
    coverage must list exactly these chunks.
    """
    problems = [
        f"{_pointer(e.absolute_path)}: {e.message}"
        for e in sorted(_validator().iter_errors(room), key=lambda e: list(map(str, e.absolute_path)))
    ]
    if problems:
        raise RoomValidationError(problems)
    assert isinstance(room, dict)

    seen: set[str] = set()

    def claim(item_id: str, where: str) -> None:
        if item_id in seen:
            problems.append(f"{where}/id: duplicate id {item_id!r}")
        seen.add(item_id)

    def check_cites(cites: list[str], where: str) -> None:
        if known_chunks is None:
            return
        for i, chunk_id in enumerate(cites):
            if chunk_id not in known_chunks:
                problems.append(f"{where}/cites/{i}: unknown chunk {chunk_id!r}")

    section_ids: set[str] = set()
    for ci, chapter in enumerate(room["chapters"]):
        cwhere = f"/chapters/{ci}"
        claim(chapter["id"], cwhere)
        for si, section in enumerate(chapter["sections"]):
            swhere = f"{cwhere}/sections/{si}"
            claim(section["id"], swhere)
            section_ids.add(section["id"])
            check_cites(section["cites"], swhere)
            if section["kind"] == "quiz" and not section.get("quiz"):
                problems.append(f"{swhere}/quiz: a quiz section needs at least one question")
            for qi, question in enumerate(section.get("quiz", [])):
                qwhere = f"{swhere}/quiz/{qi}"
                claim(question["id"], qwhere)
                check_cites(question["cites"], qwhere)
                if question["answer_index"] >= len(question["choices"]):
                    problems.append(
                        f"{qwhere}/answer_index: {question['answer_index']} is past the last of "
                        f"{len(question['choices'])} choices"
                    )
                if len(set(question["choices"])) != len(question["choices"]):
                    problems.append(f"{qwhere}/choices: choices repeat")

    for chunk_id, ids in room["coverage"].items():
        for i, section_id in enumerate(ids):
            if section_id not in section_ids:
                problems.append(f"/coverage/{chunk_id}/{i}: unknown section {section_id!r}")
    if known_chunks is not None:
        missing = sorted(known_chunks - set(room["coverage"]))
        if missing:
            problems.append(f"/coverage: missing {len(missing)} chunks of this week, first {missing[0]!r}")
        extra = sorted(set(room["coverage"]) - known_chunks)
        if extra:
            problems.append(f"/coverage: {len(extra)} chunks are not in this week, first {extra[0]!r}")

    if problems:
        raise RoomValidationError(problems)
    return room


def parse_room(text: str, known_chunks: set[str] | None = None) -> dict:
    """Parse model or file output. Bad JSON is a RoomValidationError too."""
    try:
        room = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RoomValidationError([f"/: not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"]) from None
    return validate_room(room, known_chunks)


def dumps(room: dict) -> str:
    """Stable JSON: the same room always gives the same bytes."""
    return json.dumps(room, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a failed write keeps the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
