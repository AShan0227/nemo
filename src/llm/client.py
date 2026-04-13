"""LLM client — OpenAI-compatible interface with retry logic."""

from __future__ import annotations

from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from src.config.settings import LLMConfig


class LLMClient:
    """Async LLM client with automatic retry and structured output."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request."""
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature or self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
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

        logger.debug(
            f"LLM: {result['usage']['prompt_tokens']}+{result['usage']['completion_tokens']} tokens"
        )
        return result

    async def decide_action(self, system_prompt: str, screen_context: str, task: str) -> str:
        """Shortcut: given screen context and task, return LLM's action decision."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nCurrent Screen:\n{screen_context}"},
        ]
        result = await self.chat(messages)
        return result["content"]
