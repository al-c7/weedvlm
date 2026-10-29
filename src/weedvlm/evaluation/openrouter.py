"""
Minimal async OpenRouter chat-completions client (OpenAI-compatible API),
with every failure sorted into what the caller should do about it:

    transient -- worth retrying with backoff (429, 5xx, 408, timeouts,
                 dropped connections, garbled bodies)
    request   -- this request will never succeed as sent (400 bad
                 request / model doesn't take images, 403 moderation,
                 404 no such model, 413 too large); give up on it
    fatal     -- nothing will work until the user acts (401 bad key,
                 402 out of credits); stop the whole run
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

FailureKind = Literal["transient", "request", "fatal"]


class RequestFailure(Exception):
    def __init__(
        self,
        kind: FailureKind,
        message: str,
        *,
        http_status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.kind: FailureKind = kind
        self.message = message
        self.http_status = http_status
        self.retry_after = retry_after


@dataclass
class ChatReply:
    content: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    generation_id: str | None = None
    provider: str | None = None
    latency_seconds: float = 0.0


def _classify_status(status: int) -> FailureKind:
    if status in (401, 402):
        return "fatal"
    if status in (408, 429) or status >= 500:
        return "transient"
    return "request"


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP-date; not worth parsing, the backoff covers it


def _error_message(body: Any, fallback: str) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
        message = str(error.get("message") or fallback)
        # OpenRouter puts the upstream provider's own error here.
        raw = (error.get("metadata") or {}).get("raw")
        return f"{message} ({raw})" if raw else message
    return fallback


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content parts
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


class OpenRouterClient:
    def __init__(self, *, base_url: str, api_key: str, max_connections: int):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "weedvlm",
            },
            limits=httpx.Limits(max_connections=max_connections),
        )
        # Set when any request is rate limited; every worker waits it out
        # rather than each finding out about the 429 separately.
        self._paused_until = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _wait_out_cooldown(self) -> None:
        while (remaining := self._paused_until - time.monotonic()) > 0:
            await asyncio.sleep(remaining)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra_body: dict[str, Any],
        timeout: float,
    ) -> ChatReply:
        await self._wait_out_cooldown()
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},  # asks OpenRouter to report cost
            **extra_body,
        }

        started = time.monotonic()
        try:
            response = await self._http.post("chat/completions", json=body, timeout=timeout)
        except httpx.TimeoutException as error:
            raise RequestFailure("transient", f"timed out after {timeout:g}s") from error
        except httpx.TransportError as error:
            raise RequestFailure(
                "transient", f"{type(error).__name__}: {error}".rstrip(": ")
            ) from error
        latency = time.monotonic() - started

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code != 200:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            if response.status_code == 429:
                self._paused_until = max(self._paused_until, time.monotonic() + (retry_after or 1.0))
            raise RequestFailure(
                _classify_status(response.status_code),
                _error_message(payload, response.text[:300] or response.reason_phrase),
                http_status=response.status_code,
                retry_after=retry_after,
            )

        if not isinstance(payload, dict):
            raise RequestFailure("transient", f"response was not JSON: {response.text[:200]!r}")

        # OpenRouter can return HTTP 200 with the failure in the body,
        # either at the top level or on the choice.
        choice = (payload.get("choices") or [None])[0]
        error_holder = payload if "error" in payload else (choice if isinstance(choice, dict) and "error" in choice else None)
        if error_holder is not None:
            error = error_holder["error"]
            code = error.get("code") if isinstance(error, dict) else None
            kind = _classify_status(code) if isinstance(code, int) else "transient"
            raise RequestFailure(
                kind, _error_message(error_holder, "error in response body"),
                http_status=code if isinstance(code, int) else None,
            )
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise RequestFailure("transient", f"response had no choices: {response.text[:200]!r}")

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        cost = usage.get("cost") if usage else None
        return ChatReply(
            content=_content_text(choice["message"]),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
            generation_id=payload.get("id"),
            provider=payload.get("provider"),
            latency_seconds=latency,
        )
