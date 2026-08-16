"""History stores compact ask_subgraph receipts, not UNIT blocks."""

from __future__ import annotations

from server.llm.history_manager import HistoryManager
from server.tools.source_registry import (
    format_source_id_list,
    session_source_ids_in_text,
    tool_history_stub,
)


def test_session_source_ids_and_ranges():
    assert session_source_ids_in_text("A (source:1; source:3). B (source:1).") == [1, 3]
    assert format_source_id_list([1, 2, 5, 6, 7, 8, 9]) == "1, 2, 5-9"
    assert format_source_id_list([3]) == "3"
    assert format_source_id_list([]) == ""


def test_tool_history_stub_with_ids():
    stub = tool_history_stub("UNIT\nfact (source:1; source:3)\nmore (source:2)")
    assert "UNIT" not in stub
    assert "Session sources 1-3 (stable ids)" in stub
    assert "Not an empty result" in stub
    assert "following assistant message" in stub


def test_tool_history_stub_no_ids():
    stub = tool_history_stub("no citations here")
    assert stub.startswith("Retrieved data. No source:N")
    assert "Not an empty result" in stub


def test_tool_history_stub_error():
    stub = tool_history_stub("Error: boom happened", ok=False)
    assert stub == "Tool error: boom happened"


def test_history_without_tools():
    hm = HistoryManager(max_len=6)
    hm.add_entry("hello", "hi there")
    assert hm.get_history() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_two_tool_pairs():
    hm = HistoryManager(max_len=6)
    tool_messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "ask_subgraph",
                        "arguments": '{"subquestions":["Anthocyanins indicate freshness."],"effort":"high"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": tool_history_stub("x (source:1)\ny (source:18)"),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "ask_subgraph",
                        "arguments": '{"subquestions":["Anthocyanin film dose."],"effort":"medium"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": tool_history_stub("z (source:21)"),
        },
    ]
    hm.add_entry("expand the table", "Table row (source:1).", tool_messages=tool_messages)
    hist = hm.get_history()
    assert hist[0] == {"role": "user", "content": "expand the table"}
    assert hist[1]["role"] == "assistant"
    assert hist[1]["tool_calls"][0]["id"] == "call_1"
    assert "Anthocyanins indicate freshness." in hist[1]["tool_calls"][0]["function"]["arguments"]
    assert "reasoning" not in hist[1]
    assert hist[2]["role"] == "tool"
    assert hist[2]["tool_call_id"] == "call_1"
    assert "UNIT" not in hist[2]["content"]
    assert "Session sources 1, 18" in hist[2]["content"]
    assert hist[3]["tool_calls"][0]["id"] == "call_2"
    assert hist[4]["tool_call_id"] == "call_2"
    assert "Session sources 21" in hist[4]["content"]
    assert hist[5] == {
        "role": "assistant",
        "content": "Table row (source:1).",
    }


def test_history_maxlen_is_user_turns():
    hm = HistoryManager(max_len=2)
    hm.add_entry("u1", "a1", tool_messages=[{"role": "tool", "tool_call_id": "x", "content": "stub"}])
    hm.add_entry("u2", "a2")
    hm.add_entry("u3", "a3")
    hist = hm.get_history()
    assert hist[0]["content"] == "u2"
    assert hist[-1]["content"] == "a3"
    assert all(m.get("content") != "u1" for m in hist)
