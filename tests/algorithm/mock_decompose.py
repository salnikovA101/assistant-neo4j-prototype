"""Mock assistant: isolate the SUBQUESTIONS module from assistant_logic.md."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from server.utils.config import load_config

logger = logging.getLogger(__name__)


def _resolve_slm_base_url(raw: str | None) -> str:
    url = (raw or "").strip().rstrip("/")
    if not url:
        url = "http://127.0.0.1:1234/v1"
    if "host.docker.internal" in url:
        url = url.replace("host.docker.internal", "127.0.0.1")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _resolve_tool_llm_profile(config: Any) -> Any:
    profile_name = config.llm.tool_profile
    llm_profile = getattr(config.llm.profiles, profile_name, None)
    if llm_profile is None:
        raise RuntimeError(f"unknown tool_profile {profile_name!r}")
    return llm_profile

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSISTANT_LOGIC_PATH = _REPO_ROOT / "prompts" / "assistant_logic.md"

_QUESTION_START_RE = re.compile(
    r"^(what|which|who|whom|whose|where|when|why|how|do|does|did|is|are|was|were|"
    r"can|could|should|would|will|may|might)\b",
    re.IGNORECASE,
)

_DECOMPOSE_TEST_FOOTER = """
# Test mode (eval harness only)

Сейчас проверяется только модуль SUBQUESTIONS. Не вызывай ask_subgraph, не
выбирай effort, не пиши научную заметку и не заполняй GAPS.
На вопрос пользователя верни исключительно JSON без markdown-ограждения:

{"subquestions":[{"id":"sq1","text":"..."},{"id":"sq2","text":"..."}]}

1–6 элементов. Каждый text — готовый sq по правилам модуля SUBQUESTIONS.
""".strip()


def _load_decompose_prompt() -> str:
    logic = _ASSISTANT_LOGIC_PATH.read_text(encoding="utf-8").strip()
    return logic + "\n\n" + _DECOMPOSE_TEST_FOOTER


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if "?" in t:
        return True
    return bool(_QUESTION_START_RE.match(t))


def _parse_sq(content: str) -> list[dict[str, str]]:
    content = (content or "").strip()
    m = re.search(r"```(?:json)?\n?(.*?)\n?```", content, re.DOTALL)
    text = m.group(1).strip() if m else content
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return []
    raw = data.get("subquestions") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        t = str(item.get("text") or "").strip()
        if not t or _looks_like_question(t):
            continue
        sid = str(item.get("id") or f"sq{i+1}")
        out.append({"id": sid, "text": t})
    return out


def _fallback_statements(question: str) -> list[dict[str, str]]:
    """Deterministic declarative fallbacks when the SLM fails or returns questions."""
    q = question.strip().rstrip("?")
    return [
        {
            "id": "sq1",
            "text": (
                "produces Proteolytic lactic acid bacteria produce antimicrobial "
                f"peptides related to: {q}."
            ),
        },
        {
            "id": "sq2",
            "text": (
                "inhibits Casein-derived antimicrobial peptides inhibit "
                "bacterial and fungal pathogens."
            ),
        },
        {
            "id": "sq3",
            "text": (
                "requires Hydrolysis conditions and medium support release of "
                "antimicrobial peptides from casein."
            ),
        },
    ]


async def mock_decompose(question: str) -> list[dict[str, str]]:
    if not question.strip():
        return []
    try:
        config = load_config()
        llm_profile = _resolve_tool_llm_profile(config)
        client = AsyncOpenAI(
            api_key=llm_profile.api_key or "EMPTY",
            base_url=_resolve_slm_base_url(llm_profile.base_url),
        )
        params: dict[str, Any] = {
            "model": llm_profile.model,
            "temperature": 0.0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": _load_decompose_prompt()},
                {"role": "user", "content": question},
            ],
        }
        if hasattr(llm_profile, "think") and not llm_profile.think:
            params["reasoning_effort"] = "none"
        resp = await client.chat.completions.create(**params)
        raw = resp.choices[0].message.content or ""
        sqs = _parse_sq(raw)
        if not sqs:
            raise RuntimeError("mock_decompose: SLM returned no usable subquestions")
        return sqs
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"mock_decompose failed: {e}") from e
