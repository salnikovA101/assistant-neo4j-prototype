"""V6 embed client uses local Ollama defaults (not OpenRouter)."""

from __future__ import annotations

import pytest

from server.algorithm.embed_client import (
    DEFAULT_EMBED_API_KEY,
    DEFAULT_EMBED_MODEL,
    _ensure_env_loaded,
    _resolve_embed_settings,
    default_embed_base_url,
)
from server.utils.config import AppConfig, load_config
from server.utils.constants import EmbeddingBackend


def test_ensure_env_loaded_does_not_read_ragas_file():
    import inspect

    src = inspect.getsource(_ensure_env_loaded)
    assert ".env.ragas-testing" not in src
    assert 'root / ".env"' in src


def test_app_config_has_no_graph_embeddings_field():
    assert "graph_embeddings" not in AppConfig.model_fields
    cfg = load_config()
    assert not hasattr(cfg, "graph_embeddings")


def test_resolve_embed_defaults_to_local_ollama(monkeypatch):
    monkeypatch.setattr(
        "server.algorithm.embed_client._ensure_env_loaded", lambda: None
    )
    monkeypatch.delenv("EMBED__BASE_URL", raising=False)
    monkeypatch.delenv("EMBED__MODEL", raising=False)
    monkeypatch.delenv("EMBED__API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM__PROFILES__OTHER__API_KEY", raising=False)
    backend, model, url, headers = _resolve_embed_settings()
    assert backend == EmbeddingBackend.OLLAMA
    assert model == DEFAULT_EMBED_MODEL
    assert model == "embeddinggemma:300m-qat-q8_0"
    assert url.endswith("/embeddings")
    assert "11434" in url
    assert url.startswith(default_embed_base_url().rstrip("/"))
    assert headers["Authorization"] == f"Bearer {DEFAULT_EMBED_API_KEY}"


def test_resolve_embed_env_overrides(monkeypatch):
    monkeypatch.setenv("EMBED__BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("EMBED__MODEL", "embeddinggemma:300m-qat-q8_0")
    monkeypatch.setenv("EMBED__API_KEY", "ollama")
    backend, model, url, headers = _resolve_embed_settings()
    assert backend == EmbeddingBackend.OLLAMA
    assert model == "embeddinggemma:300m-qat-q8_0"
    assert url == "http://127.0.0.1:11434/v1/embeddings"
    assert headers["Authorization"] == "Bearer ollama"


def test_resolve_embed_does_not_require_openrouter_key(monkeypatch):
    monkeypatch.setattr(
        "server.algorithm.embed_client._ensure_env_loaded", lambda: None
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM__PROFILES__OTHER__API_KEY", raising=False)
    monkeypatch.delenv("EMBED__API_KEY", raising=False)
    monkeypatch.delenv("EMBED__BASE_URL", raising=False)
    monkeypatch.delenv("EMBED__MODEL", raising=False)
    _backend, model, _url, headers = _resolve_embed_settings()
    assert model == DEFAULT_EMBED_MODEL
    assert "sk-" not in headers["Authorization"]


def test_post_embeddings_raises_after_retries(monkeypatch):
    import asyncio

    import httpx

    from server.algorithm.embed_client import EmbeddingError, _post_embeddings_once

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
            resolved_backend=EmbeddingBackend.OLLAMA,
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


def test_embed_texts_prefixes_queries(monkeypatch):
    import asyncio

    from server.algorithm.embed import QUERY_PREFIX, embed_texts

    seen: list[str] = []

    async def fake_batch(texts, **_k):
        seen.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("server.algorithm.embed.get_embeddings_batch", fake_batch)
    asyncio.run(embed_texts(["starter culture"], {}))
    assert len(seen) == 1
    assert seen[0].startswith(QUERY_PREFIX)
    assert seen[0].endswith("starter culture")


def test_embed_texts_prefixes_documents(monkeypatch):
    import asyncio

    from server.algorithm.embed import DOCUMENT_PREFIX, embed_texts

    seen: list[str] = []

    async def fake_batch(texts, **_k):
        seen.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("server.algorithm.embed.get_embeddings_batch", fake_batch)
    asyncio.run(embed_texts(["lactate in the vat"], {}, as_query=False))
    assert len(seen) == 1
    assert seen[0].startswith(DOCUMENT_PREFIX)
    assert seen[0].endswith("lactate in the vat")
