"""Qwen thinking-mode sampling is merged into the chat request."""

from __future__ import annotations

from server.llm.base import (
    _merge_request_kwargs,
    _reasoning_kwargs,
    _sampling_kwargs,
)
from server.utils.config import OpenAIProfile


def test_sampling_omitted_when_unset():
    kwargs = _sampling_kwargs(OpenAIProfile(temperature=0.1, max_output_tokens=1024))
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 1024
    assert "top_p" not in kwargs
    assert "extra_body" not in kwargs


def test_qwen_thinking_sampling_goes_to_request():
    profile = OpenAIProfile(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_output_tokens=32768,
        think=True,
        think_effort="high",
        preserve_thinking=True,
    )
    merged = _merge_request_kwargs(_sampling_kwargs(profile), _reasoning_kwargs(profile))
    assert merged["temperature"] == 1.0
    assert merged["top_p"] == 0.95
    assert merged["presence_penalty"] == 0.0
    assert merged["max_tokens"] == 32768
    assert merged["extra_body"]["top_k"] == 20
    assert merged["extra_body"]["min_p"] == 0.0
    assert merged["extra_body"]["repetition_penalty"] == 1.0
    assert merged["extra_body"]["reasoning"]["enabled"] is True
    assert merged["extra_body"]["preserve_thinking"] is True
    assert merged["extra_body"]["chat_template_kwargs"]["preserve_thinking"] is True
    assert merged["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_preserve_thinking_omitted_by_default():
    kwargs = _reasoning_kwargs(OpenAIProfile(think=True, think_effort="xhigh"))
    assert "preserve_thinking" not in kwargs["extra_body"]
    assert "preserve_thinking" not in kwargs["extra_body"]["chat_template_kwargs"]


def test_reasoning_effort_levels():
    for effort in ("low", "medium", "xhigh"):
        kwargs = _reasoning_kwargs(OpenAIProfile(think=True, think_effort=effort))
        assert kwargs["reasoning_effort"] == effort
        assert kwargs["extra_body"]["reasoning"]["effort"] == effort
        assert kwargs["extra_body"]["reasoning"]["enabled"] is True


def test_reasoning_effort_override():
    kwargs = _reasoning_kwargs(
        OpenAIProfile(think=True, think_effort="xhigh"),
        effort_override="low",
    )
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["extra_body"]["reasoning"]["effort"] == "low"


def test_parse_ui_think_effort():
    from server.llm.base import parse_ui_think_effort

    assert parse_ui_think_effort("xhigh") == "xhigh"
    assert parse_ui_think_effort(" Medium ") == "medium"
    assert parse_ui_think_effort("low") == "low"
    assert parse_ui_think_effort("high") is None
    assert parse_ui_think_effort("") is None
    assert parse_ui_think_effort(None) is None
