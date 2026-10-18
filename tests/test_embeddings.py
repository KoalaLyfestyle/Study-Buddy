"""Model revision handling. No model is loaded or downloaded."""

import pytest

from studybuddy import Library, embeddings
from studybuddy.embeddings import SentenceTransformerEmbedder, cached_commit, default_embedder

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def hub(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(embeddings, "DEFAULT_REVISION", None)
    monkeypatch.setattr(embeddings, "remote_commit", lambda name, revision="main": None)  # no network in tests

    def put(model, ref, sha):
        refs = cache / f"models--{model.replace('/', '--')}" / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / ref).write_text(sha + "\n")

    return put


def test_the_revision_is_part_of_the_id(hub):
    assert SentenceTransformerEmbedder("BAAI/bge-small-en", SHA_A).id == f"st:BAAI/bge-small-en@{SHA_A}"
    assert SentenceTransformerEmbedder("BAAI/bge-small-en").id == "st:BAAI/bge-small-en@unpinned"


def test_an_unpinned_model_uses_and_records_the_cached_commit(hub):
    hub("BAAI/bge-small-en", "main", SHA_A)
    model = SentenceTransformerEmbedder()
    assert model.revision == SHA_A and model.id.endswith(f"@{SHA_A}")
    assert cached_commit("BAAI/bge-small-en", "main") == SHA_A


def test_the_pinned_default_wins(hub, monkeypatch):
    hub("BAAI/bge-small-en", "main", SHA_A)
    monkeypatch.setattr(embeddings, "DEFAULT_REVISION", SHA_B)
    assert SentenceTransformerEmbedder().id == f"st:BAAI/bge-small-en@{SHA_B}"


def test_env_accepts_name_at_revision(hub, monkeypatch):
    monkeypatch.setenv("STUDYBUDDY_EMBEDDER", f"BAAI/bge-small-en@{SHA_B}")
    model = default_embedder()
    assert (model.model_name, model.revision) == ("BAAI/bge-small-en", SHA_B)


def test_a_new_revision_changes_every_cache_key(hub, tmp_path):
    one = Library(tmp_path, tmp_path / "i1", SentenceTransformerEmbedder("BAAI/bge-small-en", SHA_A))
    two = Library(tmp_path, tmp_path / "i2", SentenceTransformerEmbedder("BAAI/bge-small-en", SHA_B))
    try:
        assert one._cache_key("f" * 64) != two._cache_key("f" * 64)
    finally:
        one.close()
        two.close()


def test_without_a_cache_the_hub_commit_is_recorded_and_loaded(hub, monkeypatch):
    import sys
    import types

    monkeypatch.setattr(embeddings, "remote_commit", lambda name, revision="main": SHA_B)
    loaded = {}

    class FakeModel:
        def __init__(self, name, revision=None):
            loaded.update(name=name, revision=revision)

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    model = SentenceTransformerEmbedder()
    assert model.id == f"st:BAAI/bge-small-en@{SHA_B}"
    model.embed_documents(["x"])
    assert loaded == {"name": "BAAI/bge-small-en", "revision": SHA_B}
