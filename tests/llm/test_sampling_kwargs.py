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
    assert merged["extra_body"]["enable_thinking"] is True


def test_preserve_thinking_omitted_by_default():
    kwargs = _reasoning_kwargs(OpenAIProfile(think=True, think_effort="xhigh"))
    assert "preserve_thinking" not in kwargs["extra_body"]
    assert "preserve_thinking" not in kwargs["extra_body"]["chat_template_kwargs"]


def test_reasoning_effort_levels():
    for effort in ("low", "medium", "high", "xhigh"):
        kwargs = _reasoning_kwargs(OpenAIProfile(think=True, think_effort=effort))
        assert kwargs["reasoning_effort"] == effort
        assert kwargs["extra_body"]["reasoning_effort"] == effort
        assert kwargs["extra_body"]["enable_thinking"] is True
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
    assert kwargs["extra_body"]["enable_thinking"] is False
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


def test_ollama_gptoss_yaml_profile():
    from server.llm.base import parse_ui_think_effort, profile_think_efforts
    from server.utils.config import load_config

    cfg = load_config()
    gptoss = cfg.llm.profiles.ollama_gptoss
    ollama = cfg.llm.profiles.ollama
    assert gptoss.model == "gpt-oss:120b-cloud"
    assert gptoss.display_name == "GPT-OSS 120B"
    assert gptoss.temperature == 1.0
    assert gptoss.top_p == 1.0
    assert gptoss.top_k is None
    assert gptoss.think_effort == "medium"
    assert gptoss.think_efforts == ["low", "medium", "high"]
    assert gptoss.think_token == ""
    assert gptoss.preserve_thinking is False
    allowed = profile_think_efforts(gptoss)
    assert parse_ui_think_effort("off", allowed) is None
    assert parse_ui_think_effort("medium", allowed) == "medium"
    assert parse_ui_think_effort("high", allowed) == "high"
    if (ollama.api_key or "").strip():
        assert gptoss.api_key == ollama.api_key
    if (ollama.base_url or "").strip():
        assert gptoss.base_url == ollama.base_url


def test_ollama_gptoss_inherits_empty_credentials():
    from server.utils.config import AppConfig, inherit_ollama_cloud_credentials

    cfg = AppConfig()
    cfg.llm.profiles.ollama.api_key = "shared-ollama-key"
    cfg.llm.profiles.ollama.base_url = "https://ollama.com/v1"
    cfg.llm.profiles.ollama_gptoss.api_key = ""
    cfg.llm.profiles.ollama_gptoss.base_url = ""
    inherit_ollama_cloud_credentials(cfg)
    assert cfg.llm.profiles.ollama_gptoss.api_key == "shared-ollama-key"
    assert cfg.llm.profiles.ollama_gptoss.base_url == "https://ollama.com/v1"


def test_qwen_cloud_yaml_profile():
    from server.llm.base import parse_ui_think_effort, profile_think_efforts
    from server.utils.config import load_config

    qwen = load_config().llm.profiles.qwen_cloud
    assert qwen.model == "qwen3.8-27b"
    assert qwen.display_name == "Qwen 3.8 27B"
    assert "dashscope" in (qwen.base_url or "")
    assert qwen.temperature == 0.7
    assert qwen.top_p == 0.8
    assert qwen.top_k == 20
    assert qwen.think_effort == "xhigh"
    assert qwen.think_efforts == ["low", "medium", "xhigh"]
    assert qwen.preserve_thinking is True
    allowed = profile_think_efforts(qwen)
    assert parse_ui_think_effort("off", allowed) is None
    assert parse_ui_think_effort("xhigh", allowed) == "xhigh"
    assert parse_ui_think_effort("medium", allowed) == "medium"


def test_resolve_request_profile_ui_allowlist():
    from server.utils.config import load_config, resolve_request_profile

    llm = load_config().llm
    assert resolve_request_profile(llm, "ollama_gptoss") == "ollama_gptoss"
    assert resolve_request_profile(llm, "qwen_cloud") == "qwen_cloud"
    assert resolve_request_profile(llm, "other") == llm.current_profile
    assert resolve_request_profile(llm, None) == llm.current_profile
    assert resolve_request_profile(llm, "nope") == llm.current_profile


def test_provider_for_caches_ui_profiles():
    from server.llm.manager import LLMManager
    from server.utils.config import load_config

    mgr = LLMManager(load_config())
    gptoss = mgr.provider_for("ollama_gptoss")
    assert gptoss is mgr.provider_for("ollama_gptoss")
    assert gptoss.profile.model == "gpt-oss:120b-cloud"
    qwen = mgr.provider_for("qwen_cloud")
    assert qwen is mgr.provider_for("qwen_cloud")
    assert qwen.profile.model == "qwen3.8-27b"
    rejected = mgr.provider_for("other")
    assert rejected is mgr.provider_for(mgr.config.current_profile)
    assert rejected.profile.model == mgr.model.profile.model


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
