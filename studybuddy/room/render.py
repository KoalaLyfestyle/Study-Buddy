"""Render a validated study room to one self-contained HTML file.

Output is deterministic: the same room and state give the same bytes (no
timestamps, sorted data). Every piece of course text is HTML-escaped, and
notebook code is shown as inert text in <pre>; nothing from the source is
ever run. The page works without JavaScript; a short script only saves
reading progress and quiz answers in the browser and exports them as
progress.json.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

from .. import sources
from ..library import Library
from .build import group_sections
from .model import section_keys
from .state import RoomState

KIND_LABEL = {
    "setup": "Setup",
    "symbols": "Symbols",
    "explanation": "Explanation",
    "worked_example": "Worked example",
    "common_mistakes": "Common mistakes",
    "quiz": "Quiz",
    "summary": "Summary",
}

CSS = """
:root { --bg: #fdfcf9; --fg: #1d1d1b; --muted: #5d5b55; --line: #dedad0; --accent: #2f5d8a; --code: #f2efe7; --src: #2e6b3a; --gen: #8a5a14; }
@media (prefers-color-scheme: dark) { :root { --bg: #161614; --fg: #ecebe6; --muted: #a8a59c; --line: #34332f; --accent: #8db8e3; --code: #22211e; --src: #8fd19e; --gen: #e6b366; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 17px/1.6 system-ui, sans-serif; }
a { color: var(--accent); }
.skip { position: absolute; left: -999px; } .skip:focus { left: 16px; top: 8px; background: var(--bg); padding: 4px 8px; z-index: 2; }
.layout { display: grid; grid-template-columns: 17rem minmax(0, 1fr); max-width: 72rem; margin: 0 auto; }
nav { position: sticky; top: 0; align-self: start; max-height: 100vh; overflow-y: auto; padding: 24px 16px; border-right: 1px solid var(--line); font-size: 15px; }
nav ol { padding-left: 1.1rem; margin: 4px 0 12px; } nav li.read > a::after { content: " ✓"; color: var(--src); }
main { padding: 24px 32px 64px; min-width: 0; }
h1 { font-size: 1.7rem; margin: 0 0 4px; } h2 { margin-top: 2.5rem; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
article { margin: 1.5rem 0; } article h3 { margin-bottom: 4px; }
.meta { color: var(--muted); font-size: 14px; }
.badge { display: inline-block; font-size: 12px; padding: 0 6px; border: 1px solid currentColor; border-radius: 4px; margin-right: 6px; }
.badge.source { color: var(--src); } .badge.generated, .badge.edited { color: var(--gen); }
pre { background: var(--code); padding: 12px; overflow-x: auto; border-radius: 6px; font-size: 14px; }
dl dt { font-family: ui-monospace, monospace; font-weight: 600; } dl dd { margin: 0 0 8px 1rem; }
fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 12px 0; } legend { padding: 0 4px; }
.cites { font-size: 14px; color: var(--muted); } .cites ul { margin: 2px 0; }
label.done { font-size: 14px; color: var(--muted); }
.notice { border-left: 4px solid var(--gen); padding: 4px 12px; background: var(--code); }
button { font: inherit; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
@media (max-width: 760px) { .layout { display: block; } nav { position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--line); } main { padding: 16px; } }
"""

SCRIPT = """
(function () {
  var key = "studybuddy-room:" + document.body.dataset.room;
  var state = JSON.parse(document.getElementById("progress").textContent);
  // What this browser saved replaces progress.json (so an untick sticks); bad values are ignored.
  try {
    var saved = JSON.parse(localStorage.getItem(key) || "null");
    if (saved && Array.isArray(saved.read)) {
      state.read = saved.read.filter(function (id) { return typeof id === "string"; }).sort();
    }
    if (saved && saved.answers && typeof saved.answers === "object") {
      var answers = {};
      Object.keys(saved.answers).forEach(function (q) {
        var v = saved.answers[q];
        var known = document.querySelector('input[data-question="' + CSS.escape(q) + '"][value="' + v + '"]');
        if (Number.isInteger(v) && v >= 0 && (known || !document.querySelector('input[data-question="' + CSS.escape(q) + '"]'))) answers[q] = v;
      });
      state.answers = answers;
    }
  } catch (e) {}
  function save() { try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {} }
  function mark(id, on) {
    var link = document.querySelector('nav a[href="#' + id + '"]');
    if (link) link.parentElement.classList.toggle("read", on);
  }
  document.querySelectorAll("input[data-read]").forEach(function (box) {
    var id = box.dataset.read; box.checked = state.read.indexOf(id) >= 0; mark(id, box.checked);
    box.addEventListener("change", function () {
      state.read = state.read.filter(function (x) { return x !== id; });
      if (box.checked) state.read.push(id);
      state.read.sort(); mark(id, box.checked); save();
    });
  });
  document.querySelectorAll("input[data-question]").forEach(function (radio) {
    var q = radio.dataset.question;
    if (state.answers[q] === Number(radio.value)) radio.checked = true;
    radio.addEventListener("change", function () { state.answers[q] = Number(radio.value); save(); });
  });
  document.getElementById("export").addEventListener("click", function () {
    var blob = new Blob([JSON.stringify(state, null, 2) + "\\n"], { type: "application/json" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "progress.json"; a.click();
  });
})();
"""


def anchor_for(key: str) -> str:
    """The HTML id of a section in its source page, from its stable key (model.section_keys)."""
    return "loc-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _loc(location: dict) -> str:
    return json.dumps(location, sort_keys=True)


def citation_keys(library: Library, course: str, week: int | None) -> dict[tuple[str, str], str]:
    """(relpath, location) of every passage of the week -> its section's stable key."""
    by_file: dict[str, list] = {}
    for r in library.outline(course, week):
        by_file.setdefault(r.citation.relpath, []).append(r)
    keys = {}
    for relpath, results in by_file.items():
        for key, members in group_sections(results):
            keys[(relpath, _loc(members[0].citation.location))] = key
    return keys


def _source_href(library: Library | None, keys: dict, pages: set[str], c) -> str | None:
    """PDFs open at the page. Markdown and notebooks open the source view at the section or cell,
    when that view was rendered; otherwise there is no link."""
    if library is None or library.source_dir is None:
        return None
    if c.kind == "pdf":
        return f"{(Path(library.source_dir).resolve() / c.relpath).as_uri()}#page={c.location['page']}"
    page = f"sources/{c.doc_id}.html"
    key = keys.get((c.relpath, _loc(c.location)))
    if page not in pages or key is None:
        return None
    return f"{page}#{anchor_for(key)}"


def render_source_pages(room: dict, library: Library | None) -> dict[str, str]:
    """One read-only page per Markdown or notebook file, with an anchor per section or cell.

    Returns {relative output path: html}. A file that cannot be read now is
    skipped; its citations still show the label and chunk ID.
    """
    if library is None or library.source_dir is None:
        return {}
    pages: dict[str, str] = {}
    for chapter in room["chapters"]:
        relpath, kind = chapter["source"]["relpath"], chapter["source"]["kind"]
        if kind == "pdf":
            continue
        indexed = library.read_chunk(chapter["sections"][0]["cites"][0])
        path = Path(library.source_dir) / relpath
        try:
            # Only a file that still matches the index: otherwise anchors could land on the wrong passage.
            if indexed is None or sources.sha256_file(path) != indexed.citation.source_sha256:
                continue
            parsed = sources.parse_file(path)
        except (OSError, ValueError):
            continue
        doc_id = sources.doc_id_for(relpath)
        parts = []
        keys = section_keys([(section.location, section.text) for section in parsed])
        for section, key in zip(parsed, keys):
            if "cell_id" in section.location:
                where = f"Cell {section.location['cell_index'] + 1} ({section.location['cell_type']})"
            else:
                where = " > ".join(section.location.get("headings", [])) or "Start"
                where += f", line {section.location.get('line', 1)}"
            parts.append(
                f'<article id="{anchor_for(key)}" tabindex="-1"><p class="meta">{escape(where)}</p>'
                f"<pre><code>{escape(section.text)}</code></pre></article>"
            )
        pages[f"sources/{doc_id}.html"] = (
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(relpath)}</title><style>{CSS} article:target {{ outline: 3px solid var(--accent); }}</style></head>"
            f'<body><main id="main"><p><a href="../index.html">Back to the study room</a></p>'
            f"<h1>{escape(relpath)}</h1>{''.join(parts)}</main></body></html>\n"
        )
    return pages


def _cites_html(library: Library | None, keys: dict, pages: set[str], chunk_ids: list[str]) -> str:
    items = []
    for chunk_id in chunk_ids:
        found = library.read_chunk(chunk_id) if library is not None else None
        if found is None:
            items.append(f"<li><code>{escape(chunk_id)}</code></li>")
            continue
        label = escape(found.citation.label())
        href = _source_href(library, keys, pages, found.citation)
        link = f'<a href="{escape(href)}">{label}</a>' if href else label
        items.append(f'<li>{link} <span class="meta">({escape(chunk_id)})</span></li>')
    return '<div class="cites">Sources:<ul>' + "".join(items) + "</ul></div>"


def _block_html(block: dict) -> str:
    kind = block["type"]
    if kind == "paragraph":
        return f"<p>{escape(block['text'])}</p>"
    if kind == "code":
        lang = f' data-language="{escape(block["language"])}"' if block.get("language") else ""
        return f"<pre><code{lang}>{escape(block['text'])}</code></pre>"
    if kind == "list":
        return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in block["items"]) + "</ul>"
    if kind == "definition":
        return f"<dl><dt>{escape(block['term'])}</dt><dd>{escape(block['meaning'])}</dd></dl>"
    raise ValueError(f"unknown block type {kind!r}")


def _quiz_html(question: dict) -> str:
    qid = escape(question["id"])
    options = "".join(
        f'<div><label><input type="radio" name="{qid}" value="{i}" data-question="{qid}"> {escape(choice)}</label></div>'
        for i, choice in enumerate(question["choices"])
    )
    answer = escape(question["choices"][question["answer_index"]])
    return (
        f'<fieldset id="{qid}"><legend>{escape(question["prompt"])}</legend>{options}'
        f"<details><summary>Show the answer</summary><p><strong>{answer}</strong>. {escape(question['explanation'])}</p></details>"
        "</fieldset>"
    )


def render_room(
    room: dict, state: RoomState | None = None, library: Library | None = None, source_pages: set[str] | None = None
) -> str:
    """HTML for a room that has already been validated and had its edits applied.

    source_pages: the source-view pages written next to it (render_source_pages);
    citations link only to those.
    """
    source_pages = source_pages or set()
    state = state or RoomState()
    week = f"Week {room['week']}" if room["week"] is not None else "Other material"
    title = f"{room['course']}: {week}"
    room_key = f"{room['course']}/{room['week']}"
    keys = citation_keys(library, room["course"], room["week"]) if library is not None else {}

    nav, body = [], []
    for chapter in room["chapters"]:
        cid = escape(chapter["id"])
        links = "".join(
            f'<li><a href="#{escape(s["id"])}">{escape(s["title"])}</a></li>' for s in chapter["sections"]
        )
        nav.append(f'<li><a href="#{cid}">{escape(chapter["title"])}</a><ol>{links}</ol></li>')
        articles = []
        for s in chapter["sections"]:
            sid = escape(s["id"])
            if s.get("edited"):
                badges = '<span class="badge edited">Edited by you</span>'
            elif s["provenance"] == "source":
                badges = '<span class="badge source">From the source</span>'
            else:
                badges = '<span class="badge generated">Generated</span>'
            content = "".join(_block_html(b) for b in s["blocks"])
            content += "".join(_quiz_html(q) for q in s.get("quiz", []))
            articles.append(
                f'<article id="{sid}" aria-labelledby="{sid}-h"><h3 id="{sid}-h">{escape(s["title"])}</h3>'
                f'<div class="meta">{badges}{escape(KIND_LABEL[s["kind"]])}</div>{content}'
                f"{_cites_html(library, keys, source_pages, s['cites'])}"
                f'<label class="done"><input type="checkbox" data-read="{sid}"> I have read this</label></article>'
            )
        body.append(
            f'<section id="{cid}" aria-labelledby="{cid}-h"><h2 id="{cid}-h">{escape(chapter["title"])}</h2>'
            f'<p class="meta">{escape(chapter["source"]["relpath"])}</p>{"".join(articles)}</section>'
        )

    notice = ""
    if state.orphans:
        notice = (
            '<p class="notice">Saved progress or edits refer to '
            f"{len(state.orphans)} sections that this build no longer has. They are kept in the files.</p>"
        )
    progress = json.dumps({"read": sorted(state.read), "answers": dict(sorted(state.answers.items()))})
    progress = progress.replace("<", "\\u003c")
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head>"
        f'<body data-room="{escape(room_key)}"><a class="skip" href="#main">Skip to content</a>'
        '<div class="layout"><nav aria-label="Contents"><strong>Contents</strong>'
        f'<ol>{"".join(nav)}</ol><button id="export" type="button">Export progress</button></nav>'
        f'<main id="main" tabindex="-1"><h1>{escape(title)}</h1>'
        f'<p class="meta">Built by {escape(room["builder"])}. Each section links to the passage it comes from.</p>'
        f"{notice}{''.join(body)}</main></div>"
        f'<script type="application/json" id="progress">{progress}</script>'
        f"<script>{SCRIPT}</script></body></html>\n"
    )
