"""graph_embeddings yaml is gone; V6 embed client uses OpenRouter defaults."""

from __future__ import annotations

import pytest

from server.algorithm.embed_client import _resolve_embed_settings
from server.utils.config import AppConfig, load_config
from server.utils.constants import EmbeddingBackend


def test_app_config_has_no_graph_embeddings_field():
    assert "graph_embeddings" not in AppConfig.model_fields
    cfg = load_config()
    assert not hasattr(cfg, "graph_embeddings")


def test_resolve_embed_defaults_to_openrouter_nemotron(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    backend, model, url, headers = _resolve_embed_settings()
    assert backend == EmbeddingBackend.OPENROUTER
    assert model == "nvidia/nemotron-3-embed-1b:free"
    assert url.endswith("/embeddings")
    assert headers["Authorization"] == "Bearer sk-test"


def test_resolve_embed_requires_openrouter_key(monkeypatch):
    monkeypatch.setattr(
        "server.algorithm.embed_client._ensure_env_loaded", lambda: None
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM__PROFILES__OTHER__API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OpenRouter API key"):
        _resolve_embed_settings()
