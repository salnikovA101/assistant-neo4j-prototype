#!/usr/bin/env python3
"""Speed-test local Ollama embeddings without writing to Neo4j.

Same HTTP client and batch size as scripts/vectorize_edges.py. Fake evidence
texts (no DB). Use this on the VM before a full --force re-embed.

  .venv/bin/python scripts/bench_embed.py
  .venv/bin/python scripts/bench_embed.py --n 32 --batch 8 --chars 400

From the app image (Ollama on the host):

  docker compose run --rm --no-deps \\
    -v "$PWD/scripts:/app/scripts:ro" \\
    --entrypoint python3 app scripts/bench_embed.py --n 24
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.algorithm.embed import format_document  # noqa: E402
from server.algorithm.embed_client import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    get_embeddings_batch,
    _resolve_embed_settings,
)

_SEED = (
    "Lactic acid bacteria acidify milk during cottage cheese production. "
    "Starter cultures secrete lactate and lower pH in the vat. "
)


def fake_evidence(index: int, chars: int) -> str:
    """Unique quote-like text so the embedder cannot collapse identical inputs."""
    if chars < 1:
        raise SystemExit("--chars must be >= 1")
    unit = f"[{index}] {_SEED}"
    if len(unit) >= chars:
        return unit[:chars]
    reps = (chars // len(unit)) + 1
    return (unit * reps)[:chars]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60.0:.1f} h"


def summarize(
    n: int,
    elapsed_s: float,
    *,
    dim: int,
    estimate_n: int,
) -> list[str]:
    if elapsed_s <= 0 or n <= 0:
        raise SystemExit("benchmark produced no timed texts")
    per_sec = n / elapsed_s
    ms_each = (elapsed_s / n) * 1000.0
    eta = estimate_n / per_sec
    return [
        f"timed: n={n} dim={dim} elapsed={format_duration(elapsed_s)}",
        f"speed: {per_sec:.2f} texts/s  ({ms_each:.0f} ms/text)",
        f"estimate {estimate_n} edges: {format_duration(eta)}",
    ]


async def _embed_chunks(
    texts: list[str],
    *,
    model: str,
    batch: int,
) -> tuple[int, float]:
    dim = 0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        vecs = await get_embeddings_batch(chunk, model_id=model)
        if len(vecs) != len(chunk) or not vecs[0]:
            raise SystemExit(
                f"embedding count mismatch: got {len(vecs)} want {len(chunk)}"
            )
        dim = len(vecs[0])
        done = min(i + batch, len(texts))
        print(f"  {done}/{len(texts)} dim={dim}")
    return dim, time.perf_counter() - t0


async def _run(args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env", override=False)
    _backend, model, embeddings_url, _headers = _resolve_embed_settings()
    model = args.model or model
    print(f"model={model}")
    print(f"url={embeddings_url}")
    print(f"n={args.n} batch={args.batch} chars={args.chars} warmup={args.warmup}")

    warmup_texts = [
        format_document(fake_evidence(i, args.chars)) for i in range(args.warmup)
    ]
    timed_texts = [
        format_document(fake_evidence(args.warmup + i, args.chars))
        for i in range(args.n)
    ]

    if warmup_texts:
        print("warmup (excluded from speed)...")
        try:
            wdim, wsec = await _embed_chunks(
                warmup_texts, model=model, batch=args.batch
            )
        except EmbeddingError as exc:
            raise SystemExit(f"Ollama embed failed: {exc}") from exc
        print(f"  warmup {len(warmup_texts)} texts in {format_duration(wsec)} dim={wdim}")

    print("timed...")
    try:
        dim, elapsed = await _embed_chunks(
            timed_texts, model=model, batch=args.batch
        )
    except EmbeddingError as exc:
        raise SystemExit(f"Ollama embed failed: {exc}") from exc
    for line in summarize(
        args.n, elapsed, dim=dim, estimate_n=args.estimate
    ):
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local Ollama embeddings (no Neo4j writes). "
            "Same client as vectorize_edges.py."
        )
    )
    parser.add_argument("--n", type=int, default=24, help="Timed texts (default 24)")
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size, same default as vectorize_edges.py",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=400,
        help="Characters per fake evidence string",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Texts to embed first (model load); not counted",
    )
    parser.add_argument(
        "--estimate",
        type=int,
        default=10_000,
        help="Extrapolate wall time for this many edges",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override EMBED__MODEL (default {DEFAULT_EMBED_MODEL})",
    )
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.estimate < 1:
        raise SystemExit("--estimate must be >= 1")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
