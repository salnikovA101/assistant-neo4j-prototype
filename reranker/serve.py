#!/usr/bin/env python3
"""HTTP wrapper: CrossEncoder behind POST /rerank."""

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

logger = logging.getLogger("reranker")

DEFAULT_MODEL = "cross-encoder/ettin-reranker-150m-v1"

_model: CrossEncoder | None = None
_device: str = "cpu"
_model_id: str = DEFAULT_MODEL


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(model_id: str, device: str) -> CrossEncoder:
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

    app = FastAPI(title="reranker", lifespan=lifespan)

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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7997)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        create_app(args.model_id, args.device),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
