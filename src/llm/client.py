"""LLM client — provider fallback + retry/timeout + tool-use parsing."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI

from src.llm.prompts import ACTION_TOOLS

if TYPE_CHECKING:
    from src.config.settings import LLMConfig


class LLMClient:
    """Async LLM client with provider fallback and structured decision extraction."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._provider_chain = self._build_provider_chain()
        self._provider_models: dict[str, str] = {}
        self._provider_clients: dict[str, AsyncOpenAI] = {}
        self._legacy_provider = self._provider_chain[0]

        # Keep `_client` for backwards-compat with existing tests/mocks.
        client, model = self._make_provider_runtime(self._legacy_provider)
        self._client = client
        self._provider_models[self._legacy_provider] = model
        self._provider_clients[self._legacy_provider] = client

        for provider in self._provider_chain[1:]:
            p_client, p_model = self._make_provider_runtime(provider)
            self._provider_models[provider] = p_model
            self._provider_clients[provider] = p_client

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool | None = None,
        stream_handler: Any | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request with retries and provider fallback."""
        use_stream = bool(self._config.enable_stream if stream is None else stream)
        max_attempts = max(1, int(self._config.retry_count) + 1)
        last_error: Exception | None = None

        for provider in self._provider_chain:
            client = self._provider_clients[provider]
            if provider == self._legacy_provider:
                # Tests may monkeypatch `_client`; honor it for primary provider.
                client = self._client

            model = self._provider_models[provider]
            kwargs = self._build_chat_kwargs(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )

            for attempt in range(1, max_attempts + 1):
                try:
                    if use_stream:
                        result = await self._chat_stream(
                            client,
                            kwargs,
                            stream_handler=stream_handler,
                        )
                    else:
                        result = await self._chat_once(client, kwargs)
                    result["provider"] = provider
                    logger.debug(
                        "LLM[{}]: {}+{} tokens",
                        provider,
                        result["usage"]["prompt_tokens"],
                        result["usage"]["completion_tokens"],
                    )
                    return result
                except Exception as exc:  # pragma: no cover - exception paths environment-dependent
                    last_error = exc
                    if attempt >= max_attempts:
                        break
                    backoff = float(self._config.retry_backoff_s) * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM[{}] failed attempt {}/{}: {}. Retrying in {:.2f}s",
                        provider,
                        attempt,
                        max_attempts,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

            logger.warning(
                "LLM provider `{}` exhausted, trying next fallback if available",
                provider,
            )

        if last_error is not None:
            raise last_error
        raise RuntimeError("No LLM provider available")

    async def decide_action_structured(
        self,
        system_prompt: str,
        screen_context: str,
        task: str,
    ) -> dict[str, Any]:
        """Return normalized structured action decision."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nCurrent Screen:\n{screen_context}"},
        ]

        use_tools = bool(getattr(self._config, "tool_use_enabled", False))
        result = await self.chat(
            messages,
            tools=ACTION_TOOLS if use_tools else None,
            tool_choice="auto" if use_tools else None,
        )

        decision: dict[str, Any] | None = None
        if use_tools:
            decision = self._decision_from_tool_calls(result.get("tool_calls", []))

        if decision is None:
            decision = self._decision_from_json_text(result.get("content", ""))

        if decision is None:
            return {
                "reasoning": result.get("content", "") or "Unstructured model output",
                "action": "wait",
                "params": {"ms": 1000},
            }

        if "reasoning" not in decision:
            decision["reasoning"] = result.get("content", "") or "tool_call"
        return decision

    async def decide_action(self, system_prompt: str, screen_context: str, task: str) -> str:
        """Compatibility wrapper returning JSON string decision."""
        decision = await self.decide_action_structured(system_prompt, screen_context, task)
        return json.dumps(decision, ensure_ascii=False)

    def _build_provider_chain(self) -> list[str]:
        primary = str(getattr(self._config, "provider", "legacy")).strip() or "legacy"
        fallbacks = [str(p).strip() for p in getattr(self._config, "fallback_providers", [])]
        chain: list[str] = []
        for provider in [primary, *fallbacks]:
            if provider and provider not in chain:
                chain.append(provider)
        if not chain:
            chain.append("legacy")
        return chain

    def _provider_config(self, provider: str) -> Any | None:
        providers = getattr(self._config, "providers", {})
        if isinstance(providers, dict):
            return providers.get(provider)
        return None

    def _make_provider_runtime(self, provider: str) -> tuple[AsyncOpenAI, str]:
        cfg = self._provider_config(provider)
        if cfg is None:
            model = str(getattr(self._config, "model", ""))
            base_url = str(getattr(self._config, "base_url", ""))
            api_key = str(getattr(self._config, "api_key", ""))
        else:
            model = str(getattr(cfg, "model", ""))
            base_url = str(getattr(cfg, "base_url", ""))
            api_key = str(getattr(cfg, "api_key", "")) or str(getattr(self._config, "api_key", ""))

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(self._config.request_timeout_s),
        )
        return client, model

    def _build_chat_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "timeout": self._config.request_timeout_s,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return kwargs

    async def _chat_once(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> dict[str, Any]:
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=float(self._config.request_timeout_s) + 1,
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
        client: AsyncOpenAI,
        kwargs: dict[str, Any],
        *,
        stream_handler: Any | None = None,
    ) -> dict[str, Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream = await asyncio.wait_for(
            client.chat.completions.create(**stream_kwargs),
            timeout=float(self._config.request_timeout_s) + 1,
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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _decision_from_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        for call in tool_calls:
            name = str(call.get("name", "")).strip()
            if not name:
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    params = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    continue
            elif isinstance(arguments, dict):
                params = arguments
            else:
                continue
            return {"action": name, "params": params}
        return None

    @staticmethod
    def _decision_from_json_text(text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if "action" not in payload:
            return None
        if "params" not in payload or not isinstance(payload.get("params"), dict):
            payload["params"] = {}
        return payload
