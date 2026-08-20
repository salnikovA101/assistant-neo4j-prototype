"""Unit tests for scripts/bench_embed.py helpers (no live Ollama)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_embed.py"


def _mod():
    spec = importlib.util.spec_from_file_location("bench_embed", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bench():
    return _mod()


def test_fake_evidence_unique_and_length(bench):
    a = bench.fake_evidence(0, 400)
    b = bench.fake_evidence(1, 400)
    assert len(a) == 400
    assert len(b) == 400
    assert a != b
    assert a.startswith("[0]")


def test_summarize_speed(bench):
    lines = bench.summarize(20, 10.0, dim=1024, estimate_n=1000)
    joined = "\n".join(lines)
    assert "dim=1024" in joined
    assert "2.00 texts/s" in joined
    assert "500 ms/text" in joined
    assert "estimate 1000 edges" in joined


def test_summarize_rejects_empty(bench):
    with pytest.raises(SystemExit):
        bench.summarize(0, 1.0, dim=1024, estimate_n=10)
    with pytest.raises(SystemExit):
        bench.summarize(10, 0.0, dim=1024, estimate_n=10)
