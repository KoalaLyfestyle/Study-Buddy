"""Build a study room from one imported course week, without a model.

The extractive builder only quotes the source. Every section it makes is
marked `provenance: source` and cites the chunks it quotes. A model-made
room (provenance `generated`) uses the same format and the same checks;
see model.parse_room.

IDs come from where a passage lives (file, page, heading path, or cell
ID), never from its text, so editing a paragraph and rebuilding keeps the
same section IDs, and with them reading progress and authored edits.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import PurePosixPath

from ..library import Library, SearchResult
from .model import validate_room

BUILDER = "extractive-v1"

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`(])")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([\w+-]*)\s*$")
_HEADING = re.compile(r"^ {0,3}#{1,6}\s+")
# `name = expression` in inline code, or "f(x) = expression" in prose.
_DEFINITION = re.compile(r"`([A-Za-z_][\w.]*(?:\([\w, ]*\))?)\s*=\s*([^`]+)`|\b([A-Za-z]\([a-z, ]+\))\s*=\s*([^,;.]+)")
_MISTAKE = re.compile(
    r"\btoo (?:large|small|big|high|low|many|few)\b|\bavoid\b|\bdiverg\w*|\bmistakes?\b|\bpitfalls?\b|"
    r"\bdo not\b|\bdon't\b|\bnever\b|\boscillat\w*",
    re.IGNORECASE,
)


def _hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE.split(flat) if s.strip()]


def _stable_key(result: SearchResult) -> str:
    loc = result.citation.location
    if "page" in loc:
        return f"page:{loc['page']}"
    if "cell_id" in loc:
        return f"cell:{loc['cell_id']}"
    return "h:" + " > ".join(loc.get("headings", []))


def _blocks(texts: list[str], is_code_cell: bool) -> list[dict]:
    """Paragraph and code blocks from a section's chunks, without overlap repeats."""
    if is_code_cell:
        code = texts[0]
        for extra in texts[1:]:
            # Chunks overlap; keep only the part a later chunk adds.
            overlap = next((n for n in range(min(len(code), len(extra)), 0, -1) if code.endswith(extra[:n])), 0)
            code += extra[overlap:]
        return [{"type": "code", "text": code, "language": "python"}]

    blocks: list[dict] = []
    seen: set[str] = set()
    for text in texts:
        lines = text.splitlines()
        para: list[str] = []
        code: list[str] | None = None
        language = ""

        def flush_para() -> None:
            body = " ".join(" ".join(para).split())
            para.clear()
            if body and body not in seen and not _HEADING.match(body):
                seen.add(body)
                blocks.append({"type": "paragraph", "text": body})

        for line in lines:
            fence = _FENCE.match(line)
            if code is not None:
                if fence:
                    body = "\n".join(code).strip("\n")
                    if body and body not in seen:
                        seen.add(body)
                        block = {"type": "code", "text": body}
                        if language:
                            block["language"] = language
                        blocks.append(block)
                    code = None
                else:
                    code.append(line)
            elif fence:
                flush_para()
                code, language = [], fence.group(2)
            elif not line.strip() or _HEADING.match(line):
                flush_para()
            else:
                para.append(line)
        flush_para()
        if code:  # a fence the chunker cut; keep what we have as code
            body = "\n".join(code).strip("\n")
            if body and body not in seen:
                seen.add(body)
                blocks.append({"type": "code", "text": body})
    return blocks


def _source_title(result: SearchResult, index: int, chapter_title: str) -> str:
    loc = result.citation.location
    if "page" in loc:
        return f"Page {loc['page']}"
    if "cell_id" in loc:
        first = result.text.strip().splitlines()[0] if result.text.strip() else ""
        if loc.get("cell_type") == "markdown" and _HEADING.match(first):
            return _HEADING.sub("", first).strip()
        kind = "Code" if loc.get("cell_type") == "code" else "Notes"
        return f"{kind}, cell {loc['cell_index'] + 1}"
    headings = loc.get("headings") or []
    return headings[-1] if headings else (chapter_title if index == 0 else f"Part {index + 1}")


def _chapter_title(relpath: str, results: list[SearchResult]) -> str:
    first = results[0]
    loc = first.citation.location
    if loc.get("headings"):
        return loc["headings"][0]
    if "cell_id" in loc:
        for r in results:
            line = r.text.strip().splitlines()[0] if r.text.strip() else ""
            if r.citation.location.get("cell_type") == "markdown" and _HEADING.match(line):
                return _HEADING.sub("", line).strip()
    stem = PurePosixPath(relpath).stem
    return stem.replace("-", " ").replace("_", " ").capitalize()


def _section(sid: str, kind: str, title: str, cites: list[str], blocks: list[dict], quiz=None) -> dict:
    section = {"id": sid, "kind": kind, "title": title, "provenance": "source", "cites": cites, "blocks": blocks}
    if quiz is not None:
        section["quiz"] = quiz
    return section


def _quiz(chapter_id: str, chapter_title: str, parts: list[dict]) -> list[dict]:
    """"Which part says this?" questions. The answer and the choices come from the source."""
    titles = list(OrderedDict.fromkeys(p["title"] for p in parts))
    if len(titles) < 2:
        return []
    questions = []
    for part in parts:
        quote = next(
            (s for b in part["blocks"] if b["type"] == "paragraph" for s in _sentences(b["text"]) if len(s) >= 40),
            None,
        )
        if quote is None:
            continue
        others = sorted((t for t in titles if t != part["title"]), key=lambda t: _hash(part["id"], t))[:3]
        choices = sorted([part["title"], *others], key=lambda t: _hash("order", part["id"], t))
        questions.append(
            {
                "id": "q-" + _hash(part["id"], "which-part"),
                "prompt": f"Which part of “{chapter_title}” says this: “{quote}”",
                "choices": choices,
                "answer_index": choices.index(part["title"]),
                "explanation": f"It is in “{part['title']}”.",
                "cites": part["cites"],
            }
        )
        if len(questions) == 5:
            break
    return questions


def build_chapter(relpath: str, results: list[SearchResult]) -> dict:
    doc_id = results[0].citation.doc_id
    chapter_id = f"ch-{doc_id}"
    title = _chapter_title(relpath, results)

    groups: OrderedDict[str, list[SearchResult]] = OrderedDict()
    for r in results:
        groups.setdefault(_stable_key(r), []).append(r)

    parts: list[dict] = []
    for index, (key, members) in enumerate(groups.items()):
        loc = members[0].citation.location
        is_code = loc.get("cell_type") == "code"
        blocks = _blocks([m.text for m in members], is_code)
        if not blocks:
            continue
        kind = "worked_example" if is_code or all(b["type"] == "code" for b in blocks) else "explanation"
        sid = f"sec-{_hash(doc_id, key)}"
        parts.append(_section(sid, kind, _source_title(members[0], index, title), [m.chunk_id for m in members], blocks))

    sections: list[dict] = []
    first_para = next(
        ((p, b) for p in parts for b in p["blocks"] if b["type"] == "paragraph" and not b["text"].lower().startswith("my own notes")),
        None,
    )
    if first_para:
        part, block = first_para
        sections.append(
            _section(f"setup-{_hash(doc_id, 'setup')}", "setup", "In plain words", part["cites"][:1], [block])
        )

    definitions: list[dict] = []
    def_cites: list[str] = []
    seen_terms: set[str] = set()
    mistakes: list[str] = []
    mistake_cites: list[str] = []
    for part in parts:
        for block in part["blocks"]:
            if block["type"] != "paragraph":
                continue
            for sentence in _sentences(block["text"]):
                for m in _DEFINITION.finditer(sentence):
                    term = (m.group(1) or m.group(3)).strip()
                    if term not in seen_terms:
                        seen_terms.add(term)
                        definitions.append({"type": "definition", "term": term, "meaning": sentence})
                        def_cites.extend(c for c in part["cites"] if c not in def_cites)
                if _MISTAKE.search(sentence) and sentence not in mistakes:
                    mistakes.append(sentence)
                    mistake_cites.extend(c for c in part["cites"] if c not in mistake_cites)
    if definitions:
        sections.append(_section(f"symbols-{_hash(doc_id, 'symbols')}", "symbols", "Symbols and definitions", def_cites, definitions))

    sections.extend(parts)

    if mistakes:
        sections.append(
            _section(f"mistakes-{_hash(doc_id, 'mistakes')}", "common_mistakes", "Common mistakes", mistake_cites, [{"type": "list", "items": mistakes}])
        )

    summary_items: list[str] = []
    summary_cites: list[str] = []
    for part in parts:
        if part["kind"] != "explanation":
            continue
        lead = next((s for b in part["blocks"] if b["type"] == "paragraph" for s in _sentences(b["text"])[:1]), None)
        if lead and lead not in summary_items:
            summary_items.append(lead)
            summary_cites.extend(c for c in part["cites"] if c not in summary_cites)
    if summary_items:
        sections.append(
            _section(f"summary-{_hash(doc_id, 'summary')}", "summary", "Summary", summary_cites, [{"type": "list", "items": summary_items}])
        )

    questions = _quiz(chapter_id, title, [p for p in parts if p["kind"] == "explanation"])
    if questions:
        quiz_cites = list(OrderedDict.fromkeys(c for q in questions for c in q["cites"]))
        sections.append(_section(f"quiz-{_hash(doc_id, 'quiz')}", "quiz", "Check yourself", quiz_cites, [], quiz=questions))

    return {
        "id": chapter_id,
        "title": title,
        "source": {"relpath": relpath, "kind": results[0].citation.kind},
        "sections": sections,
    }


def build_room(library: Library, course: str, week: int | None) -> dict:
    """Build and validate the room for one course week. Raises ValueError if the week is empty."""
    outline = library.outline(course, week)
    if not outline:
        where = f"week {week}" if week is not None else "the files outside week folders"
        raise ValueError(
            f"Course {course!r} has no imported material for {where}. "
            "List courses with: python -m studybuddy courses"
        )
    by_file: OrderedDict[str, list[SearchResult]] = OrderedDict()
    for r in outline:
        by_file.setdefault(r.citation.relpath, []).append(r)

    chapters = [build_chapter(relpath, results) for relpath, results in by_file.items()]
    chapters = [c for c in chapters if c["sections"]]
    coverage: dict[str, list[str]] = {r.chunk_id: [] for r in outline}
    for chapter in chapters:
        for section in chapter["sections"]:
            for chunk_id in section["cites"]:
                if section["id"] not in coverage[chunk_id]:
                    coverage[chunk_id].append(section["id"])

    room = {
        "schema_version": 1,
        "course": course,
        "week": week,
        "builder": BUILDER,
        "chapters": chapters,
        "coverage": coverage,
    }
    return validate_room(room, set(coverage))
