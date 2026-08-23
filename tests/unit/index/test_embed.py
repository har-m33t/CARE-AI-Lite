"""Unit tests for carelite.index.embed.

Pure-logic tests only here (prefix application, caching, dimension checks,
retry-on-failure) using a stubbed `_raw_embed` so nothing touches a real
Ollama daemon or the network — part of `make check`. Anything that calls a
live model is `@pytest.mark.inference`.
"""

from __future__ import annotations

import pytest

from carelite.index.embed import (
    DEFAULT_DOCUMENT_PREFIX,
    DEFAULT_QUERY_PREFIX,
    EmbedDimensionError,
    EmbedError,
    OllamaEmbedder,
    hash_text,
)


def _embedder(tmp_path, **overrides) -> OllamaEmbedder:
    defaults: dict[str, object] = {
        "model_tag": "fake-embedder",
        "host": "http://unused.invalid",
        "cache_path": tmp_path / "cache.jsonl",
        "expected_dim": 4,
    }
    defaults.update(overrides)
    e = OllamaEmbedder(**defaults)
    e._digest = "fake-digest"  # skip the live `ollama list` resolution
    return e


def test_hash_text_is_deterministic_sha256_hex():
    h1 = hash_text("hello world")
    h2 = hash_text("hello world")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_text_differs_for_different_content():
    assert hash_text("a") != hash_text("b")


def test_default_prefixes_are_both_empty_for_bge_m3():
    """Both defaults are "" — an empirical finding, not an oversight. Adding
    an instruction on the query side (the usual BGE-large/E5 convention)
    measurably *hurt* bge-m3's discrimination on this corpus: see embed.py's
    module docstring for the cosine-similarity measurements that overturned
    the original non-empty default. The mechanism for asymmetric prefixes
    stays real and independently configurable (see the tests below); the
    shipped default for this specific model is symmetric because that's
    what measured best."""
    assert DEFAULT_QUERY_PREFIX == ""
    assert DEFAULT_DOCUMENT_PREFIX == ""


def test_query_and_document_prefix_fields_are_independently_configurable():
    """The structural guarantee that actually matters: `embed_query` and
    `embed_document` are separate code paths whose prefixes can be set
    independently, so HyDE (document-style) and framework-query construction
    (query-style) can never be mixed up by accident even though today's
    defaults happen to coincide."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(OllamaEmbedder)}
    assert "query_prefix" in fields
    assert "document_prefix" in fields


def test_embed_queries_and_embed_documents_apply_distinct_prefixes(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False)
    seen: list[list[str]] = []

    def fake_raw_embed(batch):
        seen.append(list(batch))
        return [[0.1, 0.2, 0.3, 0.4] for _ in batch]

    monkeypatch.setattr(embedder, "_raw_embed", fake_raw_embed)

    embedder.embed_query("what is teach-back?")
    embedder.embed_document("Teach-back improves comprehension.")

    # Each path applies its own prefix field independently. Both default
    # prefixes happen to be "" today (see test_default_prefixes_are_both_empty_
    # for_bge_m3), so this mainly proves the two code paths are genuinely
    # separate — test_custom_prefixes_are_honoured proves they don't collapse
    # into one call site when the prefixes actually differ.
    assert seen[0] == [f"{DEFAULT_QUERY_PREFIX}what is teach-back?"]
    assert seen[1] == [f"{DEFAULT_DOCUMENT_PREFIX}Teach-back improves comprehension."]


def test_custom_prefixes_are_honoured(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False, query_prefix="Q: ", document_prefix="D: ")
    seen: list[str] = []
    monkeypatch.setattr(
        embedder, "_raw_embed", lambda batch: seen.extend(batch) or [[0.0] * 4 for _ in batch]
    )
    embedder.embed_query("x")
    embedder.embed_document("y")
    assert seen == ["Q: x", "D: y"]


def test_check_dim_raises_on_mismatch(tmp_path):
    embedder = _embedder(tmp_path, expected_dim=1024)
    with pytest.raises(EmbedDimensionError, match="1024"):
        embedder._check_dim([0.1, 0.2, 0.3])  # 3-dim, not 1024


def test_verify_dimension_raises_with_both_numbers_in_message(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False, expected_dim=1024)
    monkeypatch.setattr(embedder, "_raw_embed", lambda batch: [[0.0] * 7 for _ in batch])
    with pytest.raises(EmbedDimensionError) as exc_info:
        embedder.verify_dimension()
    msg = str(exc_info.value)
    assert "7" in msg
    assert "1024" in msg


def test_verify_dimension_passes_and_records_dim(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False, expected_dim=4)
    monkeypatch.setattr(embedder, "_raw_embed", lambda batch: [[0.0] * 4 for _ in batch])
    dim = embedder.verify_dimension()
    assert dim == 4
    assert embedder._verified_dim == 4


def test_embed_documents_caches_by_content_hash(tmp_path, monkeypatch):
    """Second call with the same text must not hit `_raw_embed` again."""
    embedder = _embedder(tmp_path, use_cache=True)
    calls: list[list[str]] = []

    def fake_raw_embed(batch):
        calls.append(list(batch))
        return [[0.1, 0.2, 0.3, 0.4] for _ in batch]

    monkeypatch.setattr(embedder, "_raw_embed", fake_raw_embed)

    v1 = embedder.embed_document("stable text")
    v2 = embedder.embed_document("stable text")
    assert v1 == v2
    assert len(calls) == 1  # cache hit avoided a second call
    embedder.close()


def test_cache_survives_a_fresh_embedder_instance_same_path(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"

    e1 = _embedder(tmp_path, use_cache=True, cache_path=cache_path)
    monkeypatch.setattr(e1, "_raw_embed", lambda batch: [[1.0, 2.0, 3.0, 4.0] for _ in batch])
    e1.embed_document("persisted text")
    e1.close()

    e2 = _embedder(tmp_path, use_cache=True, cache_path=cache_path)

    def fail_if_called(batch):
        raise AssertionError("should have been served from the on-disk cache")

    monkeypatch.setattr(e2, "_raw_embed", fail_if_called)
    vec = e2.embed_document("persisted text")
    assert vec == [1.0, 2.0, 3.0, 4.0]
    e2.close()


def test_cache_is_keyed_by_digest_so_a_model_change_invalidates(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"

    e1 = _embedder(tmp_path, use_cache=True, cache_path=cache_path)
    e1._digest = "digest-a"
    monkeypatch.setattr(e1, "_raw_embed", lambda batch: [[1.0, 1.0, 1.0, 1.0] for _ in batch])
    e1.embed_document("same text")
    e1.close()

    e2 = _embedder(tmp_path, use_cache=True, cache_path=cache_path)
    e2._digest = "digest-b"  # different model digest -> must be a cache miss
    calls: list[str] = []
    monkeypatch.setattr(
        e2, "_raw_embed", lambda batch: calls.extend(batch) or [[2.0, 2.0, 2.0, 2.0] for _ in batch]
    )
    vec = e2.embed_document("same text")
    assert calls  # was actually called, i.e. not served from e1's cache entry
    assert vec == [2.0, 2.0, 2.0, 2.0]
    e2.close()


def test_call_with_retry_recovers_from_transient_failures(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False, max_attempts=3)
    attempts = {"n": 0}

    def flaky(batch):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated transient failure")
        return [[0.0] * 4 for _ in batch]

    monkeypatch.setattr(embedder, "_raw_embed", flaky)
    monkeypatch.setattr("carelite.index.embed.wait_exponential", lambda **kw: lambda *a, **k: 0)
    vec = embedder._call_with_retry(["text"])
    assert vec == [[0.0] * 4]
    assert attempts["n"] == 3


def test_call_with_retry_gives_up_after_max_attempts(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False, max_attempts=2)
    monkeypatch.setattr(
        embedder, "_raw_embed", lambda batch: (_ for _ in ()).throw(ConnectionError("down"))
    )
    with pytest.raises(ConnectionError):
        embedder._call_with_retry(["text"])


def test_raw_embed_raises_embed_error_on_batch_size_mismatch(tmp_path, monkeypatch):
    embedder = _embedder(tmp_path, use_cache=False)

    class FakeResp:
        def __init__(self):
            self.embeddings = [[0.0, 0.0, 0.0, 0.0]]  # 1 vector for a 2-item batch

    class FakeClient:
        def __init__(self, host):
            pass

        def embed(self, model, input):
            return FakeResp()

    monkeypatch.setattr("ollama.Client", FakeClient)
    with pytest.raises(EmbedError):
        embedder._raw_embed(["a", "b"])
