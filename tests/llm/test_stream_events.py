"""Unit tests for stream delta normalization and tool-call assembly."""

from server.llm.stream_events import (
    ContentThinkSplitter,
    ToolCallAssembler,
    build_assistant_replay,
    normalize_chunk,
    preview_tool_result,
)
from server.utils.config import OpenAIProfile


def test_normalize_reasoning_content():
    profile = OpenAIProfile(think=True)
    splitter = ContentThinkSplitter(profile)
    delta = {"reasoning_content": "hmm ", "content": "hi"}
    norm = normalize_chunk(delta, splitter)
    assert norm.thinking == "hmm "
    assert norm.content == "hi"


def test_normalize_openrouter_reasoning_details():
    profile = OpenAIProfile(think=True)
    splitter = ContentThinkSplitter(profile)
    # Only reasoning_details present.
    delta = {
        "reasoning_details": [{"text": "step1"}, {"content": "step2"}],
        "content": "ans",
    }
    norm = normalize_chunk(delta, splitter)
    assert "step1" in norm.thinking and "step2" in norm.thinking
    assert norm.content == "ans"


def test_normalize_openrouter_no_double_reasoning():
    """OpenRouter often mirrors the same chunk in reasoning + reasoning_details."""
    profile = OpenAIProfile(think=True)
    splitter = ContentThinkSplitter(profile)
    delta = {
        "reasoning": "The",
        "reasoning_details": [{"text": "The", "type": "reasoning.text"}],
        "content": None,
    }
    norm = normalize_chunk(delta, splitter)
    assert norm.thinking == "The"


def test_content_think_tags():
    profile = OpenAIProfile(think=True)
    splitter = ContentThinkSplitter(profile)
    a = normalize_chunk({"content": "<think>secret"}, splitter)
    assert a.thinking == "secret" or a.content == ""
    b = normalize_chunk({"content": "</think>visible"}, splitter)
    assert "visible" in b.content
    assert "secret" in (a.thinking + b.thinking) or "secret" in a.thinking


def test_bare_think_token_not_swallow():
    profile = OpenAIProfile(think=True, think_token="<|think|>")
    splitter = ContentThinkSplitter(profile)
    norm = normalize_chunk({"content": "<|think|>hello"}, splitter)
    assert norm.content == "hello"
    assert norm.thinking == ""


def test_tool_call_assembler():
    asm = ToolCallAssembler()
    asm.push(
        [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "ask_subgraph", "arguments": '{"a"'},
            }
        ]
    )
    asm.push([{"index": 0, "function": {"arguments": ':1}'}}])
    calls = asm.finish()
    assert len(calls) == 1
    assert calls[0].name == "ask_subgraph"
    assert calls[0].arguments == '{"a":1}'


def test_build_assistant_replay_tool_turn():
    from server.llm.stream_events import AssembledToolCall

    profile = OpenAIProfile(think=True, think_token="<|think|>")
    calls = [
        AssembledToolCall(id="c1", name="ask_subgraph", arguments="{}", index=0)
    ]
    msg = build_assistant_replay(
        content="",
        tool_calls=calls,
        reasoning_parts=["why"],
        profile=profile,
    )
    assert msg["role"] == "assistant"
    assert msg["reasoning_content"] == "why"
    assert msg["tool_calls"][0]["function"]["name"] == "ask_subgraph"
    assert msg["content"] == "<|think|>"


def test_preview_truncates():
    assert preview_tool_result("abc", limit=10) == "abc"
    assert preview_tool_result("x" * 20, limit=10).endswith("…")
    assert len(preview_tool_result("x" * 20, limit=10)) == 10


def test_content_rewind_sse_serializes():
    from server.llm.stream_events import StreamEvent

    event = StreamEvent("content_rewind", {"text": "Let me look."})
    sse = event.to_sse()
    assert sse.startswith("event: content_rewind\n")
    assert '"text": "Let me look."' in sse
    assert sse.endswith("\n\n")


def test_graph_highlight_sse_serializes():
    from server.llm.stream_events import StreamEvent

    event = StreamEvent(
        "graph_highlight",
        {
            "graph_run_id": "gr_abc",
            "tokens": [{"id": "t1", "text": "kefir", "color": "#f59e0b"}],
            "note": "Смотри рёбра про kefir",
        },
    )
    sse = event.to_sse()
    assert sse.startswith("event: graph_highlight\n")
    assert '"graph_run_id": "gr_abc"' in sse
    assert '"text": "kefir"' in sse
    assert sse.endswith("\n\n")
