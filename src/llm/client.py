"""LLM client — OpenAI-compatible interface with retry logic."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from src.config.settings import LLMConfig


class LLMClient:
    """Async LLM client with automatic retry and structured output."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_s,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool | None = None,
        stream_handler: Any | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request with timeout/retry/stream support."""
        kwargs = self._build_chat_kwargs(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        use_stream = self._config.enable_stream if stream is None else stream
        max_attempts = max(1, self._config.retry_count + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                if use_stream:
                    result = await self._chat_stream(kwargs, stream_handler=stream_handler)
                else:
                    result = await self._chat_once(kwargs)
                logger.debug(
                    "LLM: {}+{} tokens",
                    result["usage"]["prompt_tokens"],
                    result["usage"]["completion_tokens"],
                )
                return result
            except Exception as exc:
                if attempt >= max_attempts:
                    raise
                backoff = self._config.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "LLM call failed attempt {}/{}: {}. Retrying in {:.2f}s",
                    attempt,
                    max_attempts,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("Unreachable retry branch in LLMClient.chat")

    def _build_chat_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature or self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "timeout": self._config.request_timeout_s,
        }
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def _chat_once(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        response = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs),
            timeout=self._config.request_timeout_s + 1,
        )
        choice = response.choices[0]
        result: dict[str, Any] = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ]
        return result

    async def _chat_stream(
        self,
        kwargs: dict[str, Any],
        *,
        stream_handler: Any | None = None,
    ) -> dict[str, Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream = await asyncio.wait_for(
            self._client.chat.completions.create(**stream_kwargs),
            timeout=self._config.request_timeout_s + 1,
        )

        chunks: list[str] = []
        tool_call_acc: dict[int, dict[str, str]] = {}
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            text = delta.content or ""
            if text:
                chunks.append(text)
                if stream_handler is not None:
                    maybe_awaitable = stream_handler(text)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    index = int(getattr(tc, "index", 0))
                    item = tool_call_acc.setdefault(index, {"name": "", "arguments": ""})
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    if getattr(fn, "name", None):
                        item["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        item["arguments"] += fn.arguments

        tool_calls = [tool_call_acc[idx] for idx in sorted(tool_call_acc)]
        return {
            "content": "".join(chunks),
            "tool_calls": tool_calls,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        }

    async def decide_action(self, system_prompt: str, screen_context: str, task: str) -> str:
        """Shortcut: given screen context and task, return LLM's action decision."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nCurrent Screen:\n{screen_context}"},
        ]
        result = await self.chat(messages)
        return result["content"]
