"""Shared loader for the user-facing service guide.

The guide deliberately lives outside the system prompt. Both the frontend and
the optional LLM tool read this same Markdown file on demand.
"""

from __future__ import annotations

from pathlib import Path


SERVICE_GUIDE_FILENAME = "service_guide.md"


def load_service_guide(prompt_folder: str | Path) -> str:
    """Read and validate the single Markdown source used by UI and LLM tool."""
    path = Path(prompt_folder) / SERVICE_GUIDE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Service guide not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Service guide is empty: {path}")
    return text
