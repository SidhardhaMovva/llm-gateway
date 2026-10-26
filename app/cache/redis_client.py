"""
Redis connection for semantic cache and rate limiting.
"""
import os
import json
from typing import Any, Optional
import redis.asyncio as redis

_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None


def get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


async def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(get_redis_url(), decode_responses=True)
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def get_json(key: str) -> Optional[Any]:
    r = await get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def set_json(key: str, value: Any, ttl_seconds: int = 3600) -> None:
    r = await get_redis()
    await r.set(key, json.dumps(value), ex=ttl_seconds)


async def incr(key: str, ttl_seconds: Optional[int] = None) -> int:
    r = await get_redis()
    n = await r.incr(key)
    # Set the expiry only when the counter is created. Previously this ran
    # on every increment, so a steadily used key never expired.
    if ttl_seconds is not None and n == 1:
        await r.expire(key, ttl_seconds)
    return n
