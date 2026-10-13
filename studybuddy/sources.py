"""Read course files into sections with source locations.

Parsing never executes anything. Notebooks are read as JSON and their
outputs are ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PARSER_VERSION = "1"

SUPPORTED_SUFFIXES = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".ipynb": "notebook",
}

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_WEEK = re.compile(r"^(?:week|wk|w)[-_ ]?(\d{1,2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    """A piece of one file with a location a reader can open.

    location keys by kind:
    - pdf: page (1-based)
    - markdown: headings (list, outermost first), line (1-based start line)
    - notebook: cell_id, cell_index (0-based), cell_type
    """

    text: str
    location: dict = field(default_factory=dict)


def file_kind(path: Path) -> str | None:
    return SUPPORTED_SUFFIXES.get(path.suffix.lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(relpath: str) -> str:
    """Stable document ID: derived from the path relative to the library root."""
    return hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:12]


def course_and_week(relpath: str, root_name: str) -> tuple[str, int | None]:
    """Course is the top-level folder; week comes from a `week-NN` style folder."""
    parts = Path(relpath).parts
    course = parts[0] if len(parts) > 1 else root_name
    week = None
    for part in parts[:-1]:
        match = _WEEK.match(part)
        if match:
            week = int(match.group(1))
    return course, week


def scan_folder(root: Path) -> list[Path]:
    """Supported files under root, skipping hidden files and folders."""
    found = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_file() and file_kind(path):
            found.append(path)
    return found


def parse_file(path: Path) -> list[Section]:
    kind = file_kind(path)
    if kind == "pdf":
        return parse_pdf(path)
    if kind == "markdown":
        return parse_markdown(path.read_text(encoding="utf-8"))
    if kind == "notebook":
        return parse_notebook(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def parse_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text=text, location={"page": number}))
    return sections


def parse_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    headings: list[tuple[int, str]] = []
    buffer: list[str] = []
    start_line = 1
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(
                Section(
                    text=body,
                    location={"headings": [h for _, h in headings], "line": start_line},
                )
            )

    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, match.group(2))]
            buffer = [line]
            start_line = number
        else:
            if not buffer:
                start_line = number
            buffer.append(line)
    flush()
    return sections


def parse_notebook(text: str) -> list[Section]:
    data = json.loads(text)
    sections = []
    for index, cell in enumerate(data.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type not in ("markdown", "code"):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip()
        if not source:
            continue
        sections.append(
            Section(
                text=source,
                location={
                    "cell_id": cell.get("id") or f"cell-{index}",
                    "cell_index": index,
                    "cell_type": cell_type,
                },
            )
        )
    return sections
