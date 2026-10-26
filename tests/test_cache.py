"""Real tests for the semantic cache and the Redis JSON helpers."""
import pytest

from app.cache import redis_client, semantic_cache
from app.cache.semantic_cache import cosine_similarity


def _fake_embedding_factory():
    """Deterministic embeddings: 'hello*' prompts -> e1, anything else -> e2."""

    async def fake_embedding(text: str):
        return [1.0, 0.0] if text.startswith("hello") else [0.0, 1.0]

    return fake_embedding


# ---------- cosine similarity ----------

def test_cosine_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_cosine_zero_vector_does_not_crash():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)


# ---------- redis JSON helpers ----------

def test_set_and_get_json_round_trip(fake_redis, run):
    run(redis_client.set_json("k1", {"a": [1, 2, 3]}))
    assert run(redis_client.get_json("k1")) == {"a": [1, 2, 3]}


def test_get_json_missing_key_returns_none(fake_redis, run):
    assert run(redis_client.get_json("nope")) is None


def test_get_json_invalid_json_returns_raw_string(fake_redis, run):
    fake_redis.data["raw"] = "not-json"
    assert run(redis_client.get_json("raw")) == "not-json"


def test_incr_sets_expiry_only_when_counter_created(fake_redis, run):
    # Regression test: incr() used to refresh the TTL on every call,
    # so a steadily used key would never expire.
    assert run(redis_client.incr("counter", ttl_seconds=60)) == 1
    assert run(redis_client.incr("counter", ttl_seconds=60)) == 2
    assert run(redis_client.incr("counter", ttl_seconds=60)) == 3
    assert fake_redis.expire_calls == [("counter", 60)]


# ---------- semantic cache behaviour ----------

def test_cache_miss_on_empty_cache(fake_redis, run, monkeypatch):
    monkeypatch.setattr(semantic_cache, "get_embedding", _fake_embedding_factory())
    assert run(semantic_cache.get_cached_response("hello there")) is None


def test_cache_store_then_similar_prompt_hits(fake_redis, run, monkeypatch):
    monkeypatch.setattr(semantic_cache, "get_embedding", _fake_embedding_factory())
    response = {"id": "chatcmpl-1", "choices": [{"message": {"content": "Hi!"}}]}
    run(semantic_cache.set_cached_response("hello world", response))
    # A different prompt with the same embedding direction must hit.
    assert run(semantic_cache.get_cached_response("hello again")) == response


def test_cache_unrelated_prompt_misses(fake_redis, run, monkeypatch):
    monkeypatch.setattr(semantic_cache, "get_embedding", _fake_embedding_factory())
    run(semantic_cache.set_cached_response("hello world", {"id": "x"}))
    # Orthogonal embedding -> similarity 0 < threshold -> miss.
    assert run(semantic_cache.get_cached_response("goodbye world")) is None


def test_cache_lookup_survives_embedding_failure(fake_redis, run, monkeypatch):
    async def broken_embedding(text: str):
        raise RuntimeError("embedding API down")

    monkeypatch.setattr(semantic_cache, "get_embedding", broken_embedding)
    assert run(semantic_cache.get_cached_response("hello")) is None


def test_cache_store_survives_embedding_failure(fake_redis, run, monkeypatch):
    async def broken_embedding(text: str):
        raise RuntimeError("embedding API down")

    monkeypatch.setattr(semantic_cache, "get_embedding", broken_embedding)
    # Must not raise, and must not store anything.
    run(semantic_cache.set_cached_response("hello", {"id": "x"}))
    assert fake_redis.data == {}
