"""Small, deterministic JSON-Schema subset used by card templates."""

from __future__ import annotations

from typing import Any


_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _types(schema: dict[str, Any]) -> set[str]:
    raw = schema.get("type")
    values = raw if isinstance(raw, list) else [raw]
    out = {str(value) for value in values if value is not None}
    if not out:
        out = {"object"} if "properties" in schema else set()
    return out


def validate_template_schema(schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]
    types = _types(schema)
    unknown = types - _TYPES
    if unknown:
        errors.append(f"{path}: unsupported types {sorted(unknown)}")
    if path == "$" and "object" not in types:
        errors.append("$: root type must be object")
    properties = schema.get("properties", {})
    if properties is not None and not isinstance(properties, dict):
        errors.append(f"{path}.properties must be an object")
    elif isinstance(properties, dict):
        for key, child in properties.items():
            errors.extend(validate_template_schema(child, f"{path}.{key}"))
    items = schema.get("items")
    if items is not None:
        errors.extend(validate_template_schema(items, f"{path}[]"))
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(v, str) for v in required):
        errors.append(f"{path}.required must be a string array")
    return errors


def _matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def validate_card_data(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    types = _types(schema)
    if types and not any(_matches(data, expected) for expected in types):
        return [f"{path}: expected {'|'.join(sorted(types))}"]
    if data is None:
        return []
    errors: list[str] = []
    if isinstance(data, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in data:
                errors.append(f"{path}.{key}: required")
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in properties:
                    errors.append(f"{path}.{key}: additional property")
        for key, value in data.items():
            child = properties.get(key)
            if isinstance(child, dict):
                errors.extend(validate_card_data(value, child, f"{path}.{key}"))
    elif isinstance(data, list) and isinstance(schema.get("items"), dict):
        for index, value in enumerate(data):
            errors.extend(validate_card_data(value, schema["items"], f"{path}[{index}]"))
    return errors


def blank_card(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, child in (schema.get("properties") or {}).items():
        types = _types(child)
        if "array" in types:
            out[key] = []
        elif "object" in types and "null" not in types:
            out[key] = blank_card(child)
        else:
            out[key] = None
    return out
