"""Mock assistant: decompose question into retrieval statements (eval only)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from server.algorithm.slm_utils import (
    resolve_slm_base_url,
    resolve_tool_llm_profile,
)
from server.utils.config import load_config

logger = logging.getLogger(__name__)

_QUESTION_START_RE = re.compile(
    r"^(what|which|who|whom|whose|where|when|why|how|do|does|did|is|are|was|were|"
    r"can|could|should|would|will|may|might)\b",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """
You decompose a scientific user question into 3–6 English RETRIEVAL STATEMENTS
for dense vector search over GraphRAG *evidence* sentences.

CRITICAL — statements, NOT questions:
- Each text MUST be a declarative phrase (HyDE / hypothetical evidence style).
- NO question marks. NO interrogatives (What/Which/How/Does/…).
- Write as authors write results: processes, lists, inhibitory effects, mappings.
- Prefer domain keywords and entity names when plausible (peptides, pathogens,
  casein fractions, proteases) so embeddings match paper evidence.
- Do NOT invent long fake abstracts; keep each statement one dense sentence
  (or short noun-phrase + clause), English only.

Coverage (adapt to the question; skip irrelevant blocks):
1) Antimicrobial peptides from casein (alpha/beta/kappa) hydrolysis
2) Enzymes / LAB / pathways releasing those peptides
3) Antibacterial spectrum (Gram+/Gram− pathogens)
4) Antifungal activity only if the question asks about fungi
5) Peptide↔pathogen mapping (activity / MIC style)

GOOD examples:
- "Antimicrobial peptides derived from enzymatic hydrolysis of alpha-s1, beta, and kappa caseins, including isracidin, casocidin, and kappacin."
- "Inhibitory effects of casein-derived antimicrobial peptides against Gram-positive and Gram-negative bacterial pathogens."
- "Correlation between specific casein-derived peptides and susceptible pathogens."

BAD examples (forbidden):
- "What antimicrobial peptides are formed during casein hydrolysis?"
- "Which pathogens are inhibited by casein-derived peptides?"

ONLY JSON:
{"subquestions":[{"id":"sq1","text":"..."},{"id":"sq2","text":"..."}]}
""".strip()


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
                "Antimicrobial peptides formed during enzymatic hydrolysis of "
                f"caseins related to: {q}."
            ),
        },
        {
            "id": "sq2",
            "text": (
                "Inhibitory spectrum of casein-derived antimicrobial peptides "
                "against bacterial and fungal pathogens."
            ),
        },
        {
            "id": "sq3",
            "text": (
                "Mapping of specific casein-derived peptides to susceptible "
                "pathogens and reported antimicrobial activity."
            ),
        },
    ]


async def mock_decompose(question: str) -> list[dict[str, str]]:
    if not question.strip():
        return []
    try:
        config = load_config()
        llm_profile = resolve_tool_llm_profile(config)
        client = AsyncOpenAI(
            api_key=llm_profile.api_key or "EMPTY",
            base_url=resolve_slm_base_url(llm_profile.base_url),
        )
        params: dict[str, Any] = {
            "model": llm_profile.model,
            "temperature": 0.0,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": question},
            ],
        }
        if hasattr(llm_profile, "think") and not llm_profile.think:
            params["reasoning_effort"] = "none"
        resp = await client.chat.completions.create(**params)
        raw = resp.choices[0].message.content or ""
        sqs = _parse_sq(raw)
        if sqs:
            return sqs
        logger.warning("mock_decompose parse/filter fail; using declarative fallbacks")
    except Exception as e:
        logger.warning("mock_decompose failed: %s", e)
    return _fallback_statements(question)
