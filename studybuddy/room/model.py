"""Load, check, and save the study room format (schema.json).

Every room, from the extractive builder or from a model, goes through
`validate_room` before it is saved or rendered. A room that fails raises
`RoomValidationError` with the JSON path of each problem, and nothing is
written, so the last good room stays in place.
"""

from __future__ import annotations

import json
import os
import re
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


_MD_HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$")


def section_keys(parts: list[tuple[dict, str]]) -> list[str]:
    """Stable keys for one file's sections, given in reading order as (location, text).

    A key says where a section lives without line numbers: the page, the
    notebook cell ID, or the Markdown heading path. A key that repeats in
    the file (two "Example" headings under one parent, or two cells with
    the same ID) gets "#2", "#3" by order. A notebook cell with no ID of
    its own (older notebooks; sources.py falls back to "cell-<index>") is
    keyed by the nearest Markdown heading above it (a heading on the first
    line of a Markdown cell) and its order under that heading, so a cell
    added under another heading does not change it; one added above it
    under the same heading does.
    """
    keys: list[str] = []
    counts: dict[str, int] = {}
    heading = ""
    for location, text in parts:
        if "page" in location:
            base = f"page:{location['page']}"
        elif "cell_id" in location:
            if location.get("cell_type") == "markdown":
                first = text.strip().splitlines()[0] if text.strip() else ""
                match = _MD_HEADING.match(first)
                if match:
                    heading = match.group(1)
            if location["cell_id"] == f"cell-{location.get('cell_index')}":
                base = f"cell-under:{heading}"
            else:
                base = f"cell:{location['cell_id']}"
        else:
            base = "h:" + " > ".join(location.get("headings", []))
        n = counts.get(base, 0)
        counts[base] = n + 1
        keys.append(base if n == 0 else f"{base}#{n + 1}")
    return keys


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
                if not isinstance(question["answer_index"], int) or isinstance(question["answer_index"], bool):
                    problems.append(f"{qwhere}/answer_index: must be a whole number like 0, not {question['answer_index']!r}")
                elif question["answer_index"] >= len(question["choices"]):
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
