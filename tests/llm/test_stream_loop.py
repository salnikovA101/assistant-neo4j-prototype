"""Smoke test: mocked OpenAI stream → thinking / tool_call / tool_result / content / done."""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock

import pytest

from server.llm.base import _tool_budget_footer
from server.llm.providers.openai_provider import OpenAIProvider
from server.utils.config import OpenAIProfile


def test_tool_budget_footer_asks_to_close_gaps():
    remaining = _tool_budget_footer(2, 10)
    assert "2/10 used" in remaining
    assert "8 calls left" in remaining
    assert "close remaining gaps" in remaining
    assert "no new search direction" in remaining
    assert "or answer now" not in remaining
    exhausted = _tool_budget_footer(10, 10)
    assert "10/10 exhausted" in exhausted
    assert "Do not call tools again" in exhausted


@pytest.mark.asyncio
async def test_quota_tools_skip_global_tool_budget_footer():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=False,
        max_turns=3,
    )
    provider = OpenAIProvider(profile)
    last_contents: List[str] = []

    async def fake_create(**kwargs):
        messages = kwargs.get("messages") or []
        last_contents.append(str((messages[-1] or {}).get("content") or "") if messages else "")
        n = len(last_contents)
        if n == 1:
            return _aiter([_Chunk(_Delta(tool_calls=[{
                "index": 0,
                "id": "call_ask",
                "type": "function",
                "function": {
                    "name": "ask_subgraph",
                    "arguments": '{"subquestions":["a"]}',
                },
            }]))])
        if n == 2:
            return _aiter([_Chunk(_Delta(tool_calls=[{
                "index": 0,
                "id": "call_guide",
                "type": "function",
                "function": {
                    "name": "get_service_guide",
                    "arguments": '{"section":"modes"}',
                },
            }]))])
        return _aiter([_Chunk(_Delta(content="Done."))])

    async def fake_ask(**_kwargs):
        return "UNIT evidence"

    async def fake_guide(**_kwargs):
        return "Guide text"

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)
    events = [
        event
        async for event in provider.generate_response_stream(
            user_text="q",
            prompt="sys",
            tools=[
                {"type": "function", "function": {"name": "ask_subgraph"}},
                {"type": "function", "function": {"name": "get_service_guide"}},
            ],
            tool_map={"ask_subgraph": fake_ask, "get_service_guide": fake_guide},
        )
    ]
    assert events[-1].type == "done"
    assert "UNIT evidence" in last_contents[1]
    assert "Tool budget:" not in last_contents[1]
    assert "Guide text" in last_contents[2]
    assert "Tool budget:" in last_contents[2]


@pytest.mark.asyncio
async def test_advance_research_new_sq_stops_sibling_tools():
    from server.llm.base import _prioritize_staged_approval_calls
    from server.llm.stream_events import AssembledToolCall

    query = AssembledToolCall(
        id="q",
        name="query_graph",
        arguments='{"cypher":"MATCH (n) RETURN n"}',
        index=0,
    )
    adv = AssembledToolCall(
        id="a",
        name="advance_research",
        arguments='{"new_subquestions":["Which starter cultures are used in kefir?"]}',
        index=1,
    )
    ordered = _prioritize_staged_approval_calls([query, adv])
    assert [item.name for item in ordered] == ["advance_research", "query_graph"]

    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=False,
        max_turns=2,
    )
    provider = OpenAIProvider(profile)
    called: list[str] = []

    async def fake_create(**_kwargs):
        return _aiter([_Chunk(_Delta(tool_calls=[
            {
                "index": 0,
                "id": "call_query",
                "type": "function",
                "function": {
                    "name": "query_graph",
                    "arguments": '{"cypher":"MATCH (n) RETURN n"}',
                },
            },
            {
                "index": 1,
                "id": "call_adv",
                "type": "function",
                "function": {
                    "name": "advance_research",
                    "arguments": '{"new_subquestions":["Which cultures?"]}',
                },
            },
        ]))])

    async def fake_query(**_kwargs):
        called.append("query_graph")
        return "rows"

    async def fake_adv(**_kwargs):
        called.append("advance_research")
        return "should not run"

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)
    events = [
        event
        async for event in provider.generate_response_stream(
            user_text="q",
            prompt="sys",
            tools=[
                {"type": "function", "function": {"name": "query_graph"}},
                {"type": "function", "function": {"name": "advance_research"}},
            ],
            tool_map={"query_graph": fake_query, "advance_research": fake_adv},
        )
    ]
    assert called == []
    names = [event.data.get("name") for event in events if event.type == "tool_call"]
    assert names == ["advance_research"]
    assert not any(event.type == "tool_result" for event in events)



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
        _Chunk(_Delta(content="Let me look.")),
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
    assert "content_rewind" in types
    assert "progress" in types
    assert types.index("progress") < types.index("content_rewind")
    assert types[-1] == "done"
    assert events[-1].data["final_content"] == "Final answer"

    visible = ""
    journal = ""
    for ev in events:
        if ev.type == "content":
            visible += ev.data.get("delta") or ""
        elif ev.type == "progress":
            journal += ev.data.get("delta") or ""
        elif ev.type == "content_rewind":
            text = ev.data.get("text") or ""
            if text and visible.endswith(text):
                visible = visible[: -len(text)]
    assert visible == "Final answer"
    assert journal == "Let me look."
    assert "Let me look." not in events[-1].data["final_content"]

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


@pytest.mark.asyncio
async def test_uncited_final_answer_is_streamed_without_grounding_gate():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    calls: List[dict[str, Any]] = []
    streams = [_aiter([_Chunk(_Delta(content="Неподтверждённое длинное утверждение без ссылки."))])]

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return streams.pop(0)

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)
    events = [event async for event in provider.generate_response_stream(user_text="q", prompt="sys")]

    assert len(calls) == 1
    assert any(event.type == "content" for event in events)
    assert events[-1].type == "done"
    assert events[-1].data["final_content"] == "Неподтверждённое длинное утверждение без ссылки."


@pytest.mark.asyncio
async def test_tool_call_after_budget_returns_error_without_second_execution():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    tool_delta = _Delta(tool_calls=[{
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "ask_subgraph", "arguments": '{"subquestions":["a"]}'},
    }])
    streams = [_aiter([_Chunk(tool_delta)]), _aiter([_Chunk(tool_delta)])]
    provider.client.chat.completions.create = AsyncMock(
        side_effect=lambda **_kwargs: streams.pop(0)
    )
    calls = {"n": 0}

    async def fake_tool(**_kwargs):
        calls["n"] += 1
        return "UNIT evidence"

    events = [
        event
        async for event in provider.generate_response_stream(
            user_text="q",
            prompt="sys",
            tools=[{"type": "function", "function": {"name": "ask_subgraph"}}],
            tool_map={"ask_subgraph": fake_tool},
        )
    ]
    assert calls["n"] == 1
    assert not any(event.type == "done" for event in events)
    assert events[-1].type == "error"
    assert events[-1].data["code"] == "tool_budget_exhausted"


@pytest.mark.asyncio
async def test_stream_override_uses_with_options():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="server-key",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    seen: dict[str, str] = {}

    def fake_with_options(**kwargs):
        seen["api_key"] = kwargs.get("api_key") or ""
        return provider.client

    provider.client.with_options = fake_with_options

    async def fake_create(**_k):
        return _aiter([_Chunk(_Delta(content="ok"))])

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    events: List[Any] = []
    async for ev in provider.generate_response_stream(
        user_text="q",
        prompt="sys",
        api_key="user-secret-key",
    ):
        events.append(ev)

    assert seen["api_key"] == "user-secret-key"
    assert events[-1].type == "done"


@pytest.mark.asyncio
async def test_stream_without_override_skips_with_options():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="server-key",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    called = {"n": 0}

    def fake_with_options(**_kwargs):
        called["n"] += 1
        return provider.client

    provider.client.with_options = fake_with_options

    async def fake_create(**_k):
        return _aiter([_Chunk(_Delta(content="ok"))])

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    async for _ev in provider.generate_response_stream(user_text="q", prompt="sys"):
        pass

    assert called["n"] == 0


@pytest.mark.asyncio
async def test_stream_maps_401_to_settings_hint():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="server-key",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)

    class AuthErr(Exception):
        status_code = 401

    async def boom(**_kwargs):
        raise AuthErr("unauthorized sk-secret")

    provider.client.chat.completions.create = AsyncMock(side_effect=boom)

    events: List[Any] = []
    async for ev in provider.generate_response_stream(user_text="q", prompt="sys"):
        events.append(ev)

    assert events[-1].type == "error"
    msg = events[-1].data["message"]
    assert "настройки" in msg.lower()
    assert "sk-secret" not in msg


@pytest.mark.asyncio
async def test_invalid_tool_json_does_not_call_tool():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        think=False,
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    called = {"n": 0}

    turn0 = [
        _Chunk(
            _Delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "ask_subgraph",
                            "arguments": "{not-json",
                        },
                    }
                ]
            )
        ),
    ]

    async def fake_create(**_kwargs):
        return _aiter(turn0)

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    async def fake_tool(**_kwargs):
        called["n"] += 1
        return "should not run"

    events: List[Any] = []
    async for ev in provider.generate_response_stream(
        user_text="q",
        prompt="sys",
        history=[],
        tools=[{"type": "function", "function": {"name": "ask_subgraph"}}],
        tool_map={"ask_subgraph": fake_tool},
    ):
        events.append(ev)

    assert called["n"] == 0
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.data["ok"] is False
    assert "invalid tool arguments" in str(tool_result.data.get("result") or "").lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_tool_call", [False, True])
async def test_autonomous_budget_allows_twelve_rounds_then_final_or_existing_error(extra_tool_call):
    # The per-request budget must override the old two-round model default.
    profile = OpenAIProfile(model="test", base_url="http://localhost", api_key="x", max_turns=2)
    provider = OpenAIProvider(profile)
    requests = []
    executed = []

    async def fake_create(**kwargs):
        requests.append(kwargs["tool_choice"])
        n = len(requests)
        if n == 13 and not extra_tool_call:
            return _aiter([_Chunk(_Delta(content="Confirmed result (source:1)."))])
        return _aiter([_Chunk(_Delta(tool_calls=[{
            "index": 0, "id": f"call_{n}", "type": "function",
            "function": {"name": "ask_subgraph", "arguments": f'{{"step":{n}}}'},
        }]))])

    async def fake_tool(step):
        executed.append(step)
        return f"Evidence {step} (source:1)"

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)
    events = [event async for event in provider.generate_response_stream(
        user_text="q", prompt="sys", max_tool_turns=12,
        tools=[{"type": "function", "function": {"name": "ask_subgraph"}}],
        tool_map={"ask_subgraph": fake_tool},
    )]
    assert executed == list(range(1, 13))
    assert requests == ["auto"] * 12 + ["none"]
    assert profile.max_turns == 2
    if extra_tool_call:
        assert events[-1].type == "error"
        assert events[-1].data["code"] == "tool_budget_exhausted"
        assert not any(event.type == "done" for event in events)
    else:
        assert events[-1].type == "done"
        assert events[-1].data["final_content"] == "Confirmed result (source:1)."
        assert len(events[-1].data["history_tool_messages"]) == 24
