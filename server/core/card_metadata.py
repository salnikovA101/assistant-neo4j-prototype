"""User-facing authorship, including cards saved before this label existed."""

from typing import Any


def card_authorship(revision: int, provenance: dict[str, Any], origin: dict[str, Any]) -> str:
    refs = [ref for value in provenance.values() for ref in (value if isinstance(value, list) else [value]) if isinstance(ref, dict)]
    if revision > 1 or any(ref.get("verification") == "user-edited" for ref in refs):
        return "Изменено пользователем"
    if origin.get("kind") == "imported" or any(ref.get("source_document") == "user import" for ref in refs):
        return "Создано пользователем"
    return "Создано ассистентом"
