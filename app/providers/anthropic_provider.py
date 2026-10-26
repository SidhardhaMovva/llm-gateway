"""
Anthropic API client: chat completion and streaming.
Maps to OpenAI-style messages where possible.
"""
import os
import time
from typing import Any, AsyncIterator, Optional
from anthropic import AsyncAnthropic

_client: Optional[AsyncAnthropic] = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _client = AsyncAnthropic(api_key=key)
    return _client


def is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


# Approximate cost per 1K tokens (USD)
ANTHROPIC_COST = {
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
}


def estimate_cost_anthropic(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_cost, out_cost = ANTHROPIC_COST.get(model, (0.003, 0.015))
    return (prompt_tokens / 1000 * in_cost) + (completion_tokens / 1000 * out_cost)


def _messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system = ""
    anthropic_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            # Keep every system message; previously each one overwrote the
            # last, silently dropping earlier system instructions.
            system = f"{system}\n\n{content}" if system else content
        else:
            anthropic_messages.append({"role": role, "content": content})
    return system, anthropic_messages


async def chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> Any:
    client = get_client()
    system, anthropic_messages = _messages_to_anthropic(messages)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or 1024,
        "system": system or "",
        "messages": anthropic_messages,
        "timeout": timeout,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    start = time.perf_counter()
    resp = await client.messages.create(**kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)
    setattr(resp, "_latency_ms", latency_ms)
    return resp


def anthropic_response_to_openai_style(resp: Any) -> dict[str, Any]:
    """Convert Anthropic response to OpenAI-style for proxy response."""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    content = ""
    for b in getattr(resp, "content", []):
        if getattr(b, "type", None) == "text":
            content += getattr(b, "text", "")
    return {
        "id": getattr(resp, "id", ""),
        "object": "chat.completion",
        "created": 0,
        "model": getattr(resp, "model", ""),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }
