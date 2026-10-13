"""The agent's search tool returns cited passages and a helpful empty message."""

import shutil
from pathlib import Path

import pytest

import rag
from agent.tools import search_notes

SAMPLE = Path(__file__).resolve().parent.parent / "sample_course"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDYBUDDY_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("STUDYBUDDY_EMBEDDER", "hashing")
    rag.reset_index()
    yield tmp_path
    rag.reset_index()


def test_search_notes_cites_sources(env, monkeypatch):
    shutil.copytree(SAMPLE, env / "data")
    monkeypatch.setenv("STUDYBUDDY_DATA_DIR", str(env / "data"))
    out = search_notes("every local minimum is a global minimum")
    assert out.startswith("[intro-to-optimisation/week-02/convexity-lecture.pdf, page 2]")
    assert rag.last_import_report().added


def test_search_notes_on_empty_folder_explains_what_to_do(env, monkeypatch):
    (env / "data").mkdir()
    monkeypatch.setenv("STUDYBUDDY_DATA_DIR", str(env / "data"))
    out = search_notes("anything")
    assert out.startswith("No course material found in")
    assert "python -m studybuddy import" in out


def test_search_notes_picks_up_new_files_and_works_across_threads(env, monkeypatch):
    import threading

    shutil.copytree(SAMPLE, env / "data")
    monkeypatch.setenv("STUDYBUDDY_DATA_DIR", str(env / "data"))
    assert "convexity-lecture.pdf" in search_notes("convex epigraph")

    (env / "data" / "intro-to-optimisation" / "week-03").mkdir()
    (env / "data" / "intro-to-optimisation" / "week-03" / "newton.md").write_text(
        "# Newton's method\n\nNewton's method uses the Hessian to take curvature-aware steps.\n"
    )
    outputs, errors = [], []

    def ask():  # Streamlit runs each rerun in a new thread
        try:
            outputs.append(search_notes("Newton's method curvature-aware steps"))
        except Exception as exc:  # pragma: no cover - the failure we guard against
            errors.append(exc)

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert all(o.startswith("[intro-to-optimisation/week-03/newton.md") for o in outputs)


def test_search_notes_falls_back_to_the_last_index_when_the_folder_empties(env, monkeypatch):
    data = env / "data"
    shutil.copytree(SAMPLE, data)
    monkeypatch.setenv("STUDYBUDDY_DATA_DIR", str(data))
    search_notes("convex")
    shutil.rmtree(data / "intro-to-optimisation")  # e.g. an unmounted drive

    out = search_notes("every local minimum is a global minimum")
    assert out.startswith("Note: could not refresh the course library (No course material found")
    assert "[intro-to-optimisation/week-02/convexity-lecture.pdf, page 2]" in out
