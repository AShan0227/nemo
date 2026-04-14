"""LLM client — provider fallback + logprobs entropy + tool-use parsing."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from typing import TYPE_CHECKING, Any

from loguru import logger
from openai import AsyncOpenAI

from src.llm.prompts import ACTION_TOOLS, build_decision_messages, resolve_system_prompt

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
        provider_override: str | None = None,
        logprobs: bool | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request with retries, fallback, and optional logprobs."""
        use_stream = bool(self._config.enable_stream if stream is None else stream)
        use_logprobs = bool(self._config.logprobs_enabled if logprobs is None else logprobs)
        max_attempts = max(1, int(self._config.retry_count) + 1)
        last_error: Exception | None = None

        provider_sequence = self._resolve_provider_sequence(provider_override)

        for provider in provider_sequence:
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
                logprobs=use_logprobs and not use_stream,
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
                        "LLM[{}]: {}+{} tokens, entropy={} ({})",
                        provider,
                        result["usage"]["prompt_tokens"],
                        result["usage"]["completion_tokens"],
                        result["metrics"].get("entropy"),
                        result["metrics"].get("entropy_source", "none"),
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
        *,
        history: list[dict[str, Any]] | None = None,
        prompt_version: str | None = None,
        provider_override: str | None = None,
    ) -> dict[str, Any]:
        """Return normalized structured action decision."""
        effective_prompt_version = prompt_version or getattr(
            self._config,
            "prompt_version",
            "reflect_fewshot_v1",
        )
        prompt_text = system_prompt or resolve_system_prompt(effective_prompt_version)
        messages = build_decision_messages(
            prompt_text,
            task,
            screen_context,
            history=history,
            prompt_version=effective_prompt_version,
        )

        use_tools = bool(getattr(self._config, "tool_use_enabled", False))
        result = await self.chat(
            messages,
            tools=ACTION_TOOLS if use_tools else None,
            tool_choice="auto" if use_tools else None,
            provider_override=provider_override,
            logprobs=getattr(self._config, "logprobs_enabled", True),
        )

        decision: dict[str, Any] | None = None
        if use_tools:
            decision = self._decision_from_tool_calls(result.get("tool_calls", []))

        if decision is None:
            decision = self._decision_from_json_text(result.get("content", ""))

        if decision is None:
            decision = {
                "reasoning": result.get("content", "") or "Unstructured model output",
                "action": "wait",
                "params": {"ms": 1000},
            }

        if "reasoning" not in decision:
            decision["reasoning"] = result.get("content", "") or "tool_call"

        decision["_meta"] = {
            "provider": result.get("provider"),
            "entropy": result.get("metrics", {}).get("entropy"),
            "entropy_source": result.get("metrics", {}).get("entropy_source"),
            "prompt_version": effective_prompt_version,
        }
        return decision

    async def decide_action(
        self,
        system_prompt: str,
        screen_context: str,
        task: str,
        *,
        history: list[dict[str, Any]] | None = None,
        prompt_version: str | None = None,
        provider_override: str | None = None,
    ) -> str:
        """Compatibility wrapper returning JSON string decision."""
        decision = await self.decide_action_structured(
            system_prompt,
            screen_context,
            task,
            history=history,
            prompt_version=prompt_version,
            provider_override=provider_override,
        )
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

    def _resolve_provider_sequence(self, provider_override: str | None) -> list[str]:
        if provider_override is None:
            return list(self._provider_chain)

        provider = provider_override.strip()
        if not provider:
            return list(self._provider_chain)

        if provider not in self._provider_clients:
            raise ValueError(f"Unknown provider override: {provider}")
        return [provider]

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
        logprobs: bool,
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
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = max(1, int(getattr(self._config, "top_logprobs", 5)))
        return kwargs

    async def _chat_once(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> dict[str, Any]:
        response = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=float(self._config.request_timeout_s) + 1,
        )
        choice = response.choices[0]
        entropy = self._extract_choice_entropy(choice)
        result: dict[str, Any] = {
            "content": choice.message.content or "",
            "tool_calls": [],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            "metrics": {
                "entropy": entropy,
                "entropy_source": "logprobs" if entropy is not None else "none",
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
            "metrics": {"entropy": None, "entropy_source": "stream_no_logprobs"},
        }

    def _extract_choice_entropy(self, choice: Any) -> float | None:
        logprobs_obj = getattr(choice, "logprobs", None)
        return self.shannon_entropy_from_logprobs(logprobs_obj)

    @staticmethod
    def shannon_entropy_from_logprobs(logprobs_obj: Any) -> float | None:
        """Estimate normalized Shannon entropy from token top-logprobs.

        Returns value in [0, 1] where 0=confident and 1=high uncertainty.
        """
        if logprobs_obj is None:
            return None

        token_items = getattr(logprobs_obj, "content", None)
        if not token_items:
            return None

        token_entropies: list[float] = []
        for token_item in token_items:
            candidates = []
            top_items = getattr(token_item, "top_logprobs", None) or []
            for top in top_items:
                raw = getattr(top, "logprob", None)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    candidates.append(value)

            # Ensure selected token probability contributes if provided.
            token_logprob = getattr(token_item, "logprob", None)
            if token_logprob is not None:
                try:
                    value = float(token_logprob)
                except (TypeError, ValueError):
                    value = math.nan
                if math.isfinite(value):
                    candidates.append(value)

            if len(candidates) < 2:
                continue

            max_lp = max(candidates)
            probs = [math.exp(lp - max_lp) for lp in candidates]
            total = sum(probs)
            if total <= 0:
                continue
            probs = [p / total for p in probs]

            entropy = 0.0
            for prob in probs:
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            max_entropy = math.log2(len(probs))
            if max_entropy > 0:
                token_entropies.append(entropy / max_entropy)

        if not token_entropies:
            return None
        return sum(token_entropies) / len(token_entropies)

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
