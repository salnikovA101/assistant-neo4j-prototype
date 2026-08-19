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
    for effort in ("low", "medium", "high", "xhigh"):
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


def test_reasoning_effort_off_sends_none():
    kwargs = _reasoning_kwargs(
        OpenAIProfile(think=True, think_effort="high"),
        effort_override="off",
    )
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["extra_body"]["reasoning"]["enabled"] is False
    assert kwargs["extra_body"]["reasoning"]["effort"] == "none"
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_parse_ui_think_effort():
    from server.llm.base import parse_ui_think_effort, profile_think_efforts

    allowed = ("high", "off")
    assert parse_ui_think_effort("high", allowed) == "high"
    assert parse_ui_think_effort(" Off ", allowed) == "off"
    assert parse_ui_think_effort("xhigh", allowed) is None
    assert parse_ui_think_effort("low", allowed) is None
    assert parse_ui_think_effort("", allowed) is None
    assert parse_ui_think_effort(None, allowed) is None

    qwen = ("low", "medium", "xhigh")
    assert parse_ui_think_effort("xhigh", qwen) == "xhigh"
    assert parse_ui_think_effort(" Medium ", qwen) == "medium"
    assert parse_ui_think_effort("high", qwen) is None

    profile = OpenAIProfile(think=True, think_effort="high", think_efforts=["high", "off"])
    assert profile_think_efforts(profile) == ("high", "off")
    fallback = OpenAIProfile(think=True, think_effort="xhigh")
    assert profile_think_efforts(fallback) == ("xhigh",)


def test_think_token_skipped_when_off():
    from server.llm.base import _tool_result_content, _with_think_token, thinking_is_on

    profile = OpenAIProfile(
        think=True,
        think_effort="high",
        think_token="<|think|>",
        think_efforts=["high", "off"],
    )
    assert thinking_is_on(profile, "high") is True
    assert thinking_is_on(profile, "off") is False
    assert _with_think_token("sys", profile, thinking=True).startswith("<|think|>")
    assert _with_think_token("sys", profile, thinking=False) == "sys"
    assert _tool_result_content("tool-out", profile, thinking=True).startswith("<|think|>")
    assert _tool_result_content("tool-out", profile, thinking=False) == "tool-out"


def test_ollama_yaml_think_efforts():
    from server.utils.config import load_config

    ollama = load_config().llm.profiles.ollama
    assert ollama.think_effort == "high"
    assert ollama.think_efforts == ["high", "off"]


def test_public_llm_error_message_auth_and_quota():
    from server.llm.base import public_llm_error_message

    class AuthErr(Exception):
        status_code = 401

    class ForbiddenErr(Exception):
        status_code = 403

    class QuotaErr(Exception):
        status_code = 429

    assert "настройки" in public_llm_error_message(AuthErr("nope")).lower()
    assert "настройки" in public_llm_error_message(ForbiddenErr("nope")).lower()
    assert "лимит" in public_llm_error_message(QuotaErr("nope")).lower()
    assert "secret-key" not in public_llm_error_message(RuntimeError("secret-key"))
