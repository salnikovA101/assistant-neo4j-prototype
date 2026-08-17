#!/usr/bin/env python3
"""Benchmark TEI-compatible ettin reranker API (latency + pairs/s).

Requires the server from scripts/serve_ettin_reranker.py:

  .venv/bin/python scripts/serve_ettin_reranker.py --port 7997

Then:

  .venv/bin/python scripts/bench_ettin_reranker.py --url http://127.0.0.1:7997 --k 20,50,100
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import httpx

CHEMISTRY_QUERY = (
    "Which catalyst and reaction conditions improve selectivity for "
    "asymmetric hydrogenation of alpha,beta-unsaturated ketones?"
)

# Mix of short abstracts and longer passages (rough token lengths).
SEED_DOCS: list[str] = [
    "Rhodium–DuPhos complexes catalyze asymmetric hydrogenation of enones with high ee under mild H2 pressure.",
    "Palladium on carbon is a common heterogeneous catalyst for alkene hydrogenation but offers little enantiocontrol.",
    "Organocatalysis with proline derivatives enables aldol reactions; it is not typically used for ketone hydrogenation.",
    "Iridium–P,N ligand systems achieve high enantioselectivity in asymmetric hydrogenation of unsaturated carbonyls.",
    "Microwave-assisted synthesis shortens reaction times for amide couplings without changing stereochemical outcomes.",
    "Solvent choice (MeOH vs THF) and H2 pressure strongly affect conversion and ee in Rh-catalyzed hydrogenations.",
    "Computational DFT studies of transition states help rationalize facial selectivity in asymmetric hydrogenation.",
    "Baker's yeast reductions of ketones proceed via dehydrogenase enzymes rather than transition-metal catalysis.",
    "Noyori-type Ru–BINAP/diamine catalysts are classic tools for asymmetric hydrogenation of ketones.",
    "Temperature elevation from 25 C to 60 C can erode enantioselectivity even when conversion improves.",
]


def _make_docs(k: int) -> list[str]:
    docs: list[str] = []
    while len(docs) < k:
        base = SEED_DOCS[len(docs) % len(SEED_DOCS)]
        # Stretch some docs to exercise longer sequences.
        pad = " Additional experimental detail on workup, chromatography, and NMR assignment." * (
            1 + (len(docs) % 4)
        )
        docs.append(f"[{len(docs)}] {base}{pad}")
    return docs


def _wait_healthy(client: httpx.Client, url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = client.get(f"{url}/health")
            if r.status_code == 200:
                info = client.get(f"{url}/info")
                return info.json() if info.status_code == 200 else {"status": "ok"}
        except Exception as e:  # noqa: BLE001 — bench waits for server
            last_err = e
        time.sleep(0.5)
    raise SystemExit(f"Server not healthy at {url}/health: {last_err}")


def _rerank(
    client: httpx.Client,
    url: str,
    query: str,
    texts: list[str],
    raw_scores: bool,
) -> list[dict[str, Any]]:
    payload = {"query": query, "texts": texts, "raw_scores": raw_scores}
    r = client.post(f"{url}/rerank", json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


def _bench_k(
    client: httpx.Client,
    url: str,
    k: int,
    runs: int,
    raw_scores: bool,
) -> dict[str, float]:
    texts = _make_docs(k)
    # Warmup (not timed)
    _rerank(client, url, CHEMISTRY_QUERY, texts, raw_scores)

    latencies: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _rerank(client, url, CHEMISTRY_QUERY, texts, raw_scores)
        latencies.append(time.perf_counter() - t0)

    mean_s = statistics.mean(latencies)
    return {
        "k": float(k),
        "runs": float(runs),
        "mean_s": mean_s,
        "p50_s": statistics.median(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "pairs_per_s": k / mean_s if mean_s > 0 else 0.0,
        "ms_per_query": mean_s * 1000.0,
        "ms_per_pair": (mean_s * 1000.0) / k if k else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7997")
    parser.add_argument("--k", default="20,50,100", help="Comma-separated candidate counts")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs per K after warmup")
    parser.add_argument("--raw-scores", action="store_true", help="Request unnormalized logits")
    parser.add_argument("--wait", type=float, default=600.0, help="Seconds to wait for /health")
    parser.add_argument("--smoke", action="store_true", help="One tiny rerank then exit")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    if not ks:
        raise SystemExit("No K values parsed from --k")

    url = args.url.rstrip("/")
    with httpx.Client() as client:
        info = _wait_healthy(client, url, args.wait)
        print(f"server: {url}")
        print(f"info:   {info}")

        if args.smoke:
            out = _rerank(
                client,
                url,
                "What is deep learning?",
                [
                    "Deep learning is a subset of ML.",
                    "Baking bread needs yeast.",
                ],
                raw_scores=args.raw_scores,
            )
            print("smoke:", out)
            return

        print(f"query:  {CHEMISTRY_QUERY[:72]}...")
        print(f"runs:   {args.runs} timed (+1 warmup) per K\n")
        print(
            f"{'K':>5}  {'mean_ms':>10}  {'p50_ms':>10}  {'pairs/s':>10}  {'ms/pair':>10}"
        )
        print("-" * 52)
        for k in ks:
            m = _bench_k(client, url, k, args.runs, args.raw_scores)
            print(
                f"{int(m['k']):>5}  "
                f"{m['ms_per_query']:10.1f}  "
                f"{m['p50_s'] * 1000:10.1f}  "
                f"{m['pairs_per_s']:10.1f}  "
                f"{m['ms_per_pair']:10.2f}"
            )


if __name__ == "__main__":
    try:
        main()
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
