"""
Core proxy logic: auth, rate limit, cache, route to provider, audit.
"""
import time
from typing import Any, Optional

from fastapi import Request, HTTPException

from app.cache import semantic_cache
from app.rate_limiter import check_rate_limit
from app.fallback import get_fallback_model, get_provider
from app.security.injection_scanner import scan_messages
from app.security.permissions import scope_tools_for_tenant
from app.security.prompt_isolation import apply_system_prompt_isolation
from app.observability.audit_logger import log_request
from app import db
from app.providers import openai_provider, anthropic_provider


def _get_tenant_id(request: Request) -> str:
    return request.headers.get("X-Tenant-ID", "default")


def _prompt_from_messages(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


async def proxy_chat(request: Request, body: dict[str, Any]) -> dict[str, Any] | Any:
    tenant_id = _get_tenant_id(request)
    messages = body.get("messages", [])
    model = body.get("model", "gpt-4")
    stream = body.get("stream", False)
    if stream:
        raise HTTPException(status_code=501, detail="Streaming not yet supported; use stream=false")
    temperature = body.get("temperature")
    max_tokens = body.get("max_tokens")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    # 1. Rate limit
    tenant_config = await db.get_tenant_config(tenant_id)
    limit_rpm = (tenant_config or {}).get("rate_limit_rpm", 60)
    allowed, count = await check_rate_limit(tenant_id, limit_rpm)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # 2. Prompt injection scan
    safe, match = scan_messages(messages)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Prompt injection pattern detected: {match}")

    # 3. Permission scope + system prompt isolation
    tools, tool_choice = scope_tools_for_tenant(
        tenant_config,
        tools,
        tool_choice,
    )
    system_override = (tenant_config or {}).get("system_prompt_override")
    messages = apply_system_prompt_isolation(messages, system_override)

    prompt_text = _prompt_from_messages(messages)

    # 4. Semantic cache lookup (non-stream only)
    if not stream:
        cached = await semantic_cache.get_cached_response(prompt_text)
        if cached is not None:
            await log_request(
                tenant_id=tenant_id,
                user_id=None,
                provider="cache",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cache_hit=True,
            )
            return cached

    # 5. Route to provider with fallback
    provider = get_provider(model)
    provider_name = provider
    used_model = model
    start = time.perf_counter()
    error_msg = None
    resp = None

    try:
        if provider == "openai":
            resp = await openai_provider.chat_completion(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,
            )
        else:
            resp = await anthropic_provider.chat_completion(
                model=model,
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60.0,
            )
    except Exception as e:
        error_msg = str(e)
        fallback_model = get_fallback_model(provider, model)
        if fallback_model:
            used_model = fallback_model
            provider = get_provider(fallback_model)
            try:
                if provider == "openai":
                    resp = await openai_provider.chat_completion(
                        model=fallback_model,
                        messages=messages,
                        stream=stream,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=60.0,
                    )
                else:
                    resp = await anthropic_provider.chat_completion(
                        model=fallback_model,
                        messages=messages,
                        stream=stream,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=60.0,
                    )
                error_msg = None
                provider_name = provider
            except Exception as e2:
                error_msg = str(e2)
                raise HTTPException(status_code=502, detail=f"Provider error: {error_msg}")

    if resp is None:
        # Primary provider failed and no fallback model exists for it.
        # Previously this fell through and returned (and cached) a fabricated
        # empty response as if it were a success.
        raise HTTPException(status_code=502, detail=f"Provider error: {error_msg}")

    latency_ms = int((time.perf_counter() - start) * 1000)

    # 6. Normalize response and extract usage for audit
    if provider == "anthropic" and resp is not None:
        openai_style = anthropic_provider.anthropic_response_to_openai_style(resp)
        usage = openai_style.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = anthropic_provider.estimate_cost_anthropic(used_model, prompt_tokens, completion_tokens)
        if not stream:
            await log_request(
                tenant_id=tenant_id,
                user_id=None,
                provider=provider_name,
                model=used_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                cache_hit=False,
                error_message=error_msg,
            )
            if prompt_text:
                await semantic_cache.set_cached_response(prompt_text, openai_style)
            return openai_style
    else:
        # OpenAI response
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        cost = openai_provider.estimate_cost_openai(used_model, prompt_tokens, completion_tokens)
        if not stream:
            await log_request(
                tenant_id=tenant_id,
                user_id=None,
                provider=provider_name,
                model=used_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                cache_hit=False,
                error_message=error_msg,
            )
            if hasattr(resp, "model_dump"):
                resp_dict = resp.model_dump()
            elif hasattr(resp, "dict"):
                resp_dict = resp.dict()
            else:
                resp_dict = {"id": getattr(resp, "id", ""), "object": "chat.completion", "choices": getattr(resp, "choices", []), "usage": getattr(resp, "usage", None)}
            if isinstance(resp_dict, dict) and prompt_text:
                await semantic_cache.set_cached_response(prompt_text, resp_dict)
            return resp_dict

    return resp
