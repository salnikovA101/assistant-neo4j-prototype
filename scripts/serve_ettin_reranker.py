#!/usr/bin/env python3
"""TEI-compatible /rerank API for cross-encoder/ettin-reranker-150m-v1 on Apple Silicon.

Native Hugging Face TEI cannot load Ettin's modular CrossEncoder head
(ModernBertModel + Pooling/Dense). This FastAPI shim exposes the same
request/response shape as TEI so existing clients (e.g. stage5 on :7997) work.

Default is 150m (near-400m MTEB quality, faster on MPS). Override with --model-id.

Usage:
  # Local (same machine as the client):
  .venv/bin/python scripts/serve_ettin_reranker.py --port 7997

  # Reachable from Docker app (compose sets RERANK_URL=host.docker.internal:7997):
  .venv/bin/python scripts/serve_ettin_reranker.py --host 0.0.0.0 --port 7997
"""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

logger = logging.getLogger("serve_ettin_reranker")

DEFAULT_MODEL = "cross-encoder/ettin-reranker-150m-v1"

_model: CrossEncoder | None = None
_device: str = "cpu"
_model_id: str = DEFAULT_MODEL


def _pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def _load_model(model_id: str, device: str) -> CrossEncoder:
    # float32 is the stable default on Apple MPS; bf16/FA2 are CUDA-oriented.
    kwargs: dict[str, Any] = {}
    if device == "cuda":
        kwargs["model_kwargs"] = {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    logger.info("Loading %s on %s ...", model_id, device)
    model = CrossEncoder(model_id, device=device, **kwargs)
    logger.info("Model ready.")
    return model


class RerankRequest(BaseModel):
    query: str
    texts: list[str] = Field(default_factory=list)
    # Kept for TEI client compat; scores are always raw CrossEncoder logits (no sigmoid).
    raw_scores: bool = True
    truncate: bool = True
    truncation_direction: str = "right"
    return_text: bool = False


def create_app(model_id: str, device: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _model, _device, _model_id
        _model_id = model_id
        _device = _pick_device(device)
        _model = _load_model(model_id, _device)
        yield
        _model = None

    app = FastAPI(title="Ettin reranker (TEI-compatible)", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        if _model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "model_id": _model_id,
            "device": _device,
            "backend": "sentence-transformers CrossEncoder",
            "tei_compatible": True,
            "scores": "raw_logits",
        }

    @app.post("/rerank")
    def rerank(req: RerankRequest) -> list[dict[str, Any]]:
        if _model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if not req.texts:
            return []

        pairs = [(req.query, text) for text in req.texts]
        scores = _model.predict(pairs, convert_to_numpy=True)
        results: list[dict[str, Any]] = []
        for i, raw in enumerate(scores):
            item: dict[str, Any] = {"index": i, "score": float(raw)}
            if req.return_text:
                item["text"] = req.texts[i]
            results.append(item)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    # 0.0.0.0 so Docker containers can reach the host via host.docker.internal.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7997)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(args.model_id, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
