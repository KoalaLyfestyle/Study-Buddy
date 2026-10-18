"""The refresh error belongs to the call that hit it, even with concurrent searches."""

import threading
import time

import rag


class _FakeLibrary:
    def chunk_count(self):
        return 1

    def search(self, query, **kwargs):
        time.sleep(0.05)  # let the other thread's refresh run in between
        return [query]


def test_refresh_error_is_per_call(monkeypatch):
    monkeypatch.setattr(rag, "get_library", lambda: _FakeLibrary())

    def refresh():
        if threading.current_thread().name == "failing":
            raise OSError("course folder unmounted")

    monkeypatch.setattr(rag, "refresh", refresh)
    out = {}

    def run():
        out[threading.current_thread().name] = rag.search_with_status("q")

    threads = [threading.Thread(target=run, name=n) for n in ("failing", "healthy")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert isinstance(out["failing"][1], OSError)
    assert out["healthy"][1] is None
