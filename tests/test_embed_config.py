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


def test_post_embeddings_raises_after_retries(monkeypatch):
    import asyncio

    import httpx

    from server.algorithm.embed_client import EmbeddingError, _post_embeddings_once
    from server.utils.constants import EmbeddingBackend

    monkeypatch.setattr("server.algorithm.embed_client._MAX_RETRIES", 2)
    monkeypatch.setattr("server.algorithm.embed_client._RETRY_BASE_SEC", 0)

    class FakeResp:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", "http://x"), response=self
            )

        def json(self):
            return {}

    class FakeClient:
        async def post(self, *_a, **_k):
            return FakeResp()

    async def _run():
        await _post_embeddings_once(
            FakeClient(),  # type: ignore[arg-type]
            embeddings_url="http://x/embeddings",
            headers={},
            payload={"model": "m", "input": ["t"]},
            resolved_backend=EmbeddingBackend.OPENROUTER,
        )

    with pytest.raises(EmbeddingError, match="retries"):
        asyncio.run(_run())


def test_embed_texts_does_not_pad_empty_vectors(monkeypatch):
    import asyncio

    from server.algorithm.embed import embed_texts
    from server.algorithm.embed_client import EmbeddingError

    async def boom(*_a, **_k):
        raise EmbeddingError("backend down")

    monkeypatch.setattr("server.algorithm.embed.get_embeddings_batch", boom)

    with pytest.raises(EmbeddingError, match="backend down"):
        asyncio.run(embed_texts(["hello"], {}))
