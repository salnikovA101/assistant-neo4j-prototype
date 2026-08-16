"""Smoke test: mocked OpenAI stream → thinking / tool_call / tool_result / content / done."""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock

import pytest

from server.llm.providers.openai_provider import OpenAIProvider
from server.utils.config import OpenAIProfile


class _Delta:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model_extra = {
            k: v
            for k, v in kwargs.items()
            if k in ("reasoning_content", "reasoning", "thinking")
        }

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.__dict__.items() if k != "model_extra" and v is not None}


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, delta=None, usage=None):
        self.choices = [_Choice(delta)] if delta is not None else []
        self.usage = usage


async def _aiter(items) -> AsyncIterator[Any]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_generate_response_stream_tool_loop(monkeypatch):
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=True,
        think_effort="high",
        max_turns=2,
    )
    provider = OpenAIProvider(profile)

    turn0 = [
        _Chunk(_Delta(reasoning_content="plan ")),
        _Chunk(
            _Delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "ask_subgraph",
                            "arguments": '{"subquestions":["a"]}',
                        },
                    }
                ]
            )
        ),
    ]
    turn1 = [
        _Chunk(_Delta(reasoning_content="ok ")),
        _Chunk(_Delta(content="Final ")),
        _Chunk(_Delta(content="answer")),
    ]

    streams = [_aiter(turn0), _aiter(turn1)]

    async def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return streams.pop(0)

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    async def fake_tool(subquestions, effort="medium"):
        return "UNIT [1]\nevidence (source:1)"

    events: List[Any] = []
    async for ev in provider.generate_response_stream(
        user_text="q",
        prompt="sys",
        history=[],
        tools=[{"type": "function", "function": {"name": "ask_subgraph"}}],
        tool_map={"ask_subgraph": fake_tool},
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types[0] == "thinking"
    assert "tool_call" in types
    assert "tool_result" in types
    assert types.count("content") >= 2
    assert types[-1] == "done"
    assert events[-1].data["final_content"] == "Final answer"

    tool_call = next(e for e in events if e.type == "tool_call")
    assert tool_call.data["name"] == "ask_subgraph"
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.data["ok"] is True
    assert "UNIT [1]" in tool_result.data["result"]
    assert "preview" in tool_result.data

    done = events[-1]
    receipts = done.data.get("history_tool_messages") or []
    assert len(receipts) == 2
    assert receipts[0]["role"] == "assistant"
    assert receipts[0]["content"] is None
    assert "reasoning" not in receipts[0]
    assert receipts[0]["tool_calls"][0]["id"] == "call_1"
    assert '"subquestions":["a"]' in receipts[0]["tool_calls"][0]["function"]["arguments"]
    assert receipts[1]["role"] == "tool"
    assert receipts[1]["tool_call_id"] == "call_1"
    assert "UNIT" not in receipts[1]["content"]
    assert "Session sources 1" in receipts[1]["content"]
