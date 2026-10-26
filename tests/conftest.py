"""Shared fixtures: an in-memory async Redis fake and path setup.

No real Redis, PostgreSQL, or provider API key is needed for this suite:
Redis is faked here, DB/provider calls are monkeypatched per-test.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache import redis_client  # noqa: E402


class FakeRedis:
    """Minimal async stand-in for redis.asyncio.Redis (decode_responses=True)."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None, **kwargs):
        self.data[key] = value
        if ex is not None:
            self.expire_calls.append((key, ex))
        return True

    async def incr(self, key):
        n = int(self.data.get(key, "0")) + 1
        self.data[key] = str(n)
        return n

    async def expire(self, key, seconds):
        self.expire_calls.append((key, seconds))
        return True

    async def close(self):
        return None


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(redis_client, "_client", fake)
    return fake


@pytest.fixture
def run():
    """Run a coroutine from a sync test: run(coro) -> result."""

    def _run(coro):
        return asyncio.run(coro)

    return _run
