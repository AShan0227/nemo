"""Tests for LLM client retry and stream behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from src.llm.client import LLMClient


def _make_non_stream_response(content: str) -> Any:
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, logprobs=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_response(name: str, arguments: str, content: str = "") -> Any:
    tool_call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
    message = SimpleNamespace(content=content, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_stream_event(content: str) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


@dataclass
class FakeCompletions:
    responses: list[Any]
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
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


def make_client(**overrides: Any) -> LLMClient:
    config = SimpleNamespace(
        provider="legacy",
        model="fake-model",
        api_key="fake-key",
        base_url="https://example.com/v1",
        providers={},
        fallback_providers=[],
        temperature=0.1,
        max_tokens=128,
        retry_count=2,
        retry_backoff_s=0,
        request_timeout_s=5,
        enable_stream=False,
        tool_use_enabled=False,
        logprobs_enabled=True,
        top_logprobs=5,
        prompt_version="reflect_fewshot_v1",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
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
    assert completions.requests[-1]["logprobs"] is True
    assert completions.requests[-1]["top_logprobs"] == 5


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
    assert result["metrics"]["entropy"] is None


@pytest.mark.asyncio
async def test_decide_action_uses_chat_content():
    client = make_client()

    async def fake_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"content": '{"action":"done","params":{"summary":"ok"}}'}

    client.chat = fake_chat  # type: ignore[method-assign]
    result = await client.decide_action("sys", "screen", "task")
    payload = json.loads(result)
    assert payload["action"] == "done"


@pytest.mark.asyncio
async def test_decide_action_structured_prefers_tool_call():
    client = make_client(tool_use_enabled=True)

    async def fake_chat(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "content": "Choosing tap",
            "tool_calls": [{"name": "tap", "arguments": "{\"index\": 2}"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    client.chat = fake_chat  # type: ignore[method-assign]
    decision = await client.decide_action_structured("sys", "screen", "task")
    assert decision["action"] == "tap"
    assert decision["params"]["index"] == 2
    assert "_meta" in decision


@pytest.mark.asyncio
async def test_provider_fallback_switches_to_secondary():
    providers = {
        "qwen": SimpleNamespace(model="qwen-plus", base_url="https://qwen", api_key="k1"),
        "gpt": SimpleNamespace(model="gpt-4o-mini", base_url="https://openai", api_key="k2"),
    }
    client = make_client(
        provider="qwen",
        providers=providers,
        fallback_providers=["gpt"],
        retry_count=0,
    )
    client._client = SimpleNamespace(  # primary provider client
        chat=SimpleNamespace(completions=FakeCompletions(responses=[RuntimeError("qwen down")]))
    )
    client._provider_clients["gpt"] = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(responses=[_make_non_stream_response("ok-gpt")]))
    )

    result = await client.chat([{"role": "user", "content": "hi"}], stream=False)
    assert result["content"] == "ok-gpt"
    assert result["provider"] == "gpt"


def test_entropy_from_logprobs_estimation():
    token_a = SimpleNamespace(
        logprob=-0.1,
        top_logprobs=[SimpleNamespace(logprob=-0.1), SimpleNamespace(logprob=-2.3)],
    )
    token_b = SimpleNamespace(
        logprob=-1.2,
        top_logprobs=[SimpleNamespace(logprob=-1.2), SimpleNamespace(logprob=-1.3)],
    )
    logprobs = SimpleNamespace(content=[token_a, token_b])
    entropy = LLMClient.shannon_entropy_from_logprobs(logprobs)
    assert entropy is not None
    assert 0.0 <= entropy <= 1.0


@pytest.mark.asyncio
async def test_chat_provider_override_uses_single_provider():
    providers = {
        "qwen": SimpleNamespace(model="qwen-plus", base_url="https://qwen", api_key="k1"),
        "gpt": SimpleNamespace(model="gpt-4o-mini", base_url="https://openai", api_key="k2"),
    }
    client = make_client(provider="qwen", providers=providers, fallback_providers=["gpt"])
    qwen_completions = FakeCompletions(responses=[_make_non_stream_response("qwen")])
    gpt_completions = FakeCompletions(responses=[_make_non_stream_response("gpt")])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=qwen_completions))
    client._provider_clients["gpt"] = SimpleNamespace(
        chat=SimpleNamespace(completions=gpt_completions)
    )

    result = await client.chat(
        [{"role": "user", "content": "hi"}],
        provider_override="gpt",
    )
    assert result["provider"] == "gpt"
    assert result["content"] == "gpt"
    assert not qwen_completions.requests
