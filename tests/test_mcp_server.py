"""Tool-contract test: a real MCP client talks to the server over stdio.

Runs `python -m studybuddy.mcp_server` as a subprocess on a copy of the
sample course with the offline embedder, so it needs no model download.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _data(result):
    assert not result.isError, result.content
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def _session_run(tmp_path, steps):
    course = tmp_path / "course"
    shutil.copytree(ROOT / "sample_course", course)
    # A file outside any week folder.
    (course / "intro-to-optimisation" / "syllabus.md").write_text("# Syllabus\n\nFive weeks of optimisation.\n")
    env = {
        **os.environ,
        "STUDYBUDDY_DATA_DIR": str(course),
        "STUDYBUDDY_INDEX_DIR": str(tmp_path / "index"),
        "STUDYBUDDY_EMBEDDER": "hashing",
    }
    # Run from another folder with PYTHONPATH, as the README's client config does.
    env["PYTHONPATH"] = str(ROOT)
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "studybuddy.mcp_server"], env=env, cwd=str(tmp_path)
    )

    async def session_run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await steps(session)

    # Fail instead of hanging if the server stops answering.
    return await asyncio.wait_for(session_run(), timeout=60)


def test_client_discovers_tools_and_gets_cited_answers(tmp_path):
    async def steps(session):
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert set(tools) == {"list_courses", "search_materials", "read_source", "get_week_context"}
        assert "query" in tools["search_materials"].inputSchema["properties"]

        found = _data(await session.call_tool("search_materials", {"query": "why does convexity matter", "k": 3}))
        top = found["results"][0]
        assert top["citation"]["path"].endswith("convexity-lecture.pdf")
        assert top["citation"]["location"] == {"page": 2}
        assert top["citation"]["label"].endswith("convexity-lecture.pdf, page 2")
        assert len(found["results"]) <= 3 and all(len(r["snippet"]) <= 300 for r in found["results"])

        nb = _data(await session.call_tool("search_materials", {"query": "learning rate 0.1 converges", "kind": "notebook"}))
        assert nb["results"] and all(r["citation"]["kind"] == "notebook" for r in nb["results"])
        assert "cell_id" in nb["results"][0]["citation"]["location"]

        passage = _data(await session.call_tool("read_source", {"chunk_id": top["chunk_id"]}))
        assert "local minimum is a global minimum" in passage["text"]
        assert passage["next_offset"] is None

        courses = _data(await session.call_tool("list_courses", {}))
        assert courses["courses"] == [{"course": "intro-to-optimisation", "files": 4, "weeks": [1, 2]}]

        week = _data(await session.call_tool("get_week_context", {"course": "intro-to-optimisation", "week": 1}))
        assert week["total_passages"] >= 5 and not week["truncated"]
        assert {p["citation"]["path"].rsplit("/", 1)[1] for p in week["passages"]} == {
            "notes.md", "gradient-descent-lab.ipynb"
        }

        loose = _data(await session.call_tool("get_week_context", {"course": "intro-to-optimisation"}))
        assert [p["citation"]["path"] for p in loose["passages"]] == ["intro-to-optimisation/syllabus.md"]

        bad = await session.call_tool("read_source", {"chunk_id": "nope"})
        assert bad.isError and "no passage with id" in bad.content[0].text
        too_many = await session.call_tool("search_materials", {"query": "x", "k": 50})
        assert too_many.isError and "between 1 and 10" in too_many.content[0].text

    asyncio.run(_session_run(tmp_path, steps))
