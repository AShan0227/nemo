"""Tests for LLM client retry and stream behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.llm.client import LLMClient


def _make_non_stream_response(content: str) -> Any:
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_stream_event(content: str) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


@dataclass
class FakeCompletions:
    responses: list[Any]

    async def create(self, **kwargs: Any) -> Any:
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


def make_client() -> LLMClient:
    config = SimpleNamespace(
        model="fake-model",
        api_key="fake-key",
        base_url="https://example.com/v1",
        temperature=0.1,
        max_tokens=128,
        retry_count=2,
        retry_backoff_s=0,
        request_timeout_s=5,
        enable_stream=False,
    )
    return LLMClient(config)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_chat_retries_then_succeeds():
    client = make_client()
    completions = FakeCompletions(
        responses=[
            RuntimeError("temporary"),
            _make_non_stream_response("ok"),
        ]
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = await client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert result["content"] == "ok"
    assert result["usage"]["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_chat_stream_collects_content_and_callbacks():
    client = make_client()
    stream = FakeStream([_make_stream_event("hel"), _make_stream_event("lo")])
    completions = FakeCompletions(responses=[stream])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    received: list[str] = []

    async def on_chunk(chunk: str) -> None:
        received.append(chunk)

    result = await client.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
        stream_handler=on_chunk,
    )

    assert result["content"] == "hello"
    assert received == ["hel", "lo"]


@pytest.mark.asyncio
async def test_decide_action_uses_chat_content():
    client = make_client()

    async def fake_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"content": '{"action":"done","params":{"summary":"ok"}}'}

    client.chat = fake_chat  # type: ignore[method-assign]
    result = await client.decide_action("sys", "screen", "task")
    assert '"action":"done"' in result
