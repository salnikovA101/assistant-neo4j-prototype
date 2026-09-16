"""Compile model Cypher into a corpus-scoped read query.

The model writes Cypher; this module tokenizes it (so strings/comments cannot
hide writes), rejects mutating or uncheckable constructs, injects r.run_id onto
every relationship and `$__run_id IN n.run_ids` onto nodes that are not
incident to such a relationship, and applies RETURN LIMIT (honor the query's
LIMIT up to 100; default 20 if omitted). Neo4j is not consulted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_PATH_LENGTH = 4
DEFAULT_MAX_ROWS = 20
MAX_MAX_ROWS = 100
QUERY_TIMEOUT_SEC = 20.0
MAX_RESULT_CHARS = 6000
MAX_FIELD_CHARS = 600
HISTORY_CHARS = 1500

SERVER_RUN_ID_PARAM = "__run_id"
SERVER_FT_NODES_PARAM = "__ft_nodes"
SERVER_FT_RELS_PARAM = "__ft_rels"
FT_NODE_INDEX = "query_graph_node_name"
FT_REL_INDEX = "query_graph_rel_evidence"

FORBIDDEN_CLAUSE = frozenset({
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "REMOVE",
    "DROP",
    "FOREACH",
    "INSERT",
    "LOAD",
    "DETACH",
    "SHOW",
    "GRANT",
    "DENY",
    "REVOKE",
    "START",
    "USING",
    "PERIODIC",
    "COMMIT",
    "CONSTRAINT",
    "SEARCH",
    "CYPHER",
    "USE",
    "EXPLAIN",
    "PROFILE",
    "FINISH",
    "SHORTESTPATH",
    "ALLSHORTESTPATHS",
})
FORBIDDEN_FUNCS = frozenset({
    "PROPERTIES",
    "ELEMENTID",
    "ID",
    "KEYS",
    "SHORTESTPATH",
    "ALLSHORTESTPATHS",
    "APOC",
})
AGG_FUNCS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX", "COLLECT", "STDEV", "STDEVP"})
CLAUSE_START = frozenset({
    "MATCH",
    "OPTIONAL",
    "WITH",
    "RETURN",
    "UNWIND",
    "CALL",
    "UNION",
    "WHERE",
    "ORDER",
    "SKIP",
    "LIMIT",
})


class QueryCompileError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_tool_text(self) -> str:
        return f"QUERY_ERROR {self.code}\n{self.message}"


@dataclass
class Tok:
    kind: str
    value: str
    raw: str
    pos: int


@dataclass
class CompiledQuery:
    cypher: str
    server_params: dict[str, Any]
    fetch_limit: int | None
    output_limit: int
    is_pure_aggregate: bool
    notes: list[str] = field(default_factory=list)


def clamp_max_rows(value: object) -> int:
    try:
        n = int(value) if value is not None else DEFAULT_MAX_ROWS
    except (TypeError, ValueError):
        n = DEFAULT_MAX_ROWS
    return max(1, min(MAX_MAX_ROWS, n))


def resolve_output_limit(*, user_limit: int | None, max_rows: object) -> int:
    """Cypher LIMIT wins (capped at 100). Else explicit max_rows. Else 20."""
    if user_limit is not None:
        try:
            n = int(user_limit)
        except (TypeError, ValueError):
            n = DEFAULT_MAX_ROWS
        return max(1, min(MAX_MAX_ROWS, n))
    if max_rows is not None:
        return clamp_max_rows(max_rows)
    return DEFAULT_MAX_ROWS


def compile_query(
    cypher: str,
    *,
    parameters: dict[str, Any] | None = None,
    max_rows: object = None,
) -> CompiledQuery:
    raw = (cypher or "").strip()
    if not raw:
        raise QueryCompileError(
            "syntax",
            "Broken here: the Cypher text is empty. Fix the Cypher; do not restate the user question.",
        )
    if raw.startswith(("```", "cypher")):
        raise QueryCompileError(
            "syntax",
            "Broken here: pass raw Cypher, not a markdown fence. Fix the Cypher; do not restate the user question.",
        )
    user_params = dict(parameters or {})
    for key in user_params:
        name = str(key)
        if name.startswith("__"):
            raise QueryCompileError(
                "scope",
                f"Broken here: parameter ${name} is reserved by the server. Do not pass run_id or __* names.",
            )
    tokens = tokenize(raw)
    _reject_forbidden(tokens)
    tokens = expand_anonymous_rels(tokens)
    tokens = rewrite_call_fulltext(tokens)
    tokens = rewrite_relationship_maps(tokens)
    tokens = rewrite_run_id_predicates(tokens)
    tokens = apply_node_run_ids(tokens)
    enforce_scope(tokens)
    tokens, fetch_limit, is_pure_aggregate, output_limit = apply_return_limits(
        tokens, max_rows
    )
    compiled = emit(tokens)
    return CompiledQuery(
        cypher=compiled,
        server_params={
            SERVER_RUN_ID_PARAM: None,  # filled by the executor
            SERVER_FT_NODES_PARAM: FT_NODE_INDEX,
            SERVER_FT_RELS_PARAM: FT_REL_INDEX,
        },
        fetch_limit=fetch_limit,
        output_limit=output_limit,
        is_pure_aggregate=is_pure_aggregate,
    )


def tokenize(text: str) -> list[Tok]:
    tokens: list[Tok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: unterminated block comment. Fix the Cypher; do not restate the user question.",
                )
            i = end + 2
            continue
        if ch in "'\"":
            raw, nxt = _read_string(text, i)
            tokens.append(Tok("string", raw, raw, i))
            i = nxt
            continue
        if ch == "$":
            j = i + 1
            if j < n and (text[j].isalpha() or text[j] == "_"):
                j += 1
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
            raw = text[i:j]
            if len(raw) == 1:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: '$' is not a valid parameter. Fix the Cypher; do not restate the user question.",
                )
            tokens.append(Tok("param", raw[1:], raw, i))
            i = j
            continue
        if ch == "`":
            j = i + 1
            while j < n:
                if text[j] == "`":
                    if j + 1 < n and text[j + 1] == "`":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: unterminated quoted identifier. Fix the Cypher; do not restate the user question.",
                )
            raw = text[i:j]
            tokens.append(Tok("ident", raw.strip("`").replace("``", "`"), raw, i))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            if ch == ".":
                j += 1
                while j < n and text[j].isdigit():
                    j += 1
            else:
                while j < n and text[j].isdigit():
                    j += 1
                if j < n and text[j] == "." and not text.startswith("..", j):
                    j += 1
                    while j < n and text[j].isdigit():
                        j += 1
            raw = text[i:j]
            tokens.append(Tok("number", raw, raw, i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            raw = text[i:j]
            upper = raw.upper()
            kind = "keyword" if upper in _KEYWORD_SET else "ident"
            tokens.append(Tok(kind, upper if kind == "keyword" else raw, raw, i))
            i = j
            continue
        # multi-char operators
        if text.startswith("=~", i) or text.startswith("<>", i) or text.startswith("<=", i) or text.startswith(">=", i) or text.startswith("..", i):
            raw = text[i : i + 2]
            tokens.append(Tok("symbol", raw, raw, i))
            i += 2
            continue
        tokens.append(Tok("symbol", ch, ch, i))
        i += 1
    return tokens


_KEYWORD_SET = FORBIDDEN_CLAUSE | CLAUSE_START | AGG_FUNCS | frozenset({
    "AND",
    "OR",
    "NOT",
    "XOR",
    "AS",
    "IN",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "DISTINCT",
    "BY",
    "ASC",
    "DESC",
    "ASCENDING",
    "DESCENDING",
    "YIELD",
    "EXISTS",
    "ALL",
    "ANY",
    "NONE",
    "SINGLE",
    "UNIQUE",
    "WHERE",
    "INDEX",
    "CSV",
    "STARTS",
    "ENDS",
    "CONTAINS",
    "FOR",
    "EACH",
})


def _read_string(text: str, i: int) -> tuple[str, int]:
    quote = text[i]
    j = i + 1
    out = [quote]
    while j < len(text):
        ch = text[j]
        out.append(ch)
        if ch == "\\" and j + 1 < len(text):
            out.append(text[j + 1])
            j += 2
            continue
        if ch == quote:
            return "".join(out), j + 1
        j += 1
    raise QueryCompileError(
        "syntax",
        "Broken here: unterminated string literal. Fix the Cypher; do not restate the user question.",
    )


def _reject_forbidden(tokens: list[Tok]) -> None:
    if any(t.kind == "symbol" and t.value == ";" for t in tokens):
        raise QueryCompileError(
            "unsupported",
            "This construct is unavailable: multiple statements. Write one MATCH/WHERE/RETURN query or a path *1..4.",
        )
    for i, tok in enumerate(tokens):
        if tok.kind == "keyword" and tok.value in FORBIDDEN_CLAUSE:
            raise QueryCompileError(
                "unsupported",
                f"This construct is unavailable: {tok.raw}. Write a read-only MATCH/WHERE/RETURN query or a path *1..4.",
            )
        if tok.kind in {"ident", "keyword"} and tok.value.upper() in FORBIDDEN_FUNCS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and nxt.raw == "(":
                raise QueryCompileError(
                    "unsupported",
                    f"This construct is unavailable: {tok.raw}(). Do not return embeddings, properties(), or elementId.",
                )
        if tok.kind == "ident" and _is_embedding_name(tok.value):
            prev = tokens[i - 1] if i else None
            if prev is not None and prev.raw == ".":
                raise QueryCompileError(
                    "unsupported",
                    "This construct is unavailable: embedding properties. Vector SEARCH belongs to ask_subgraph.",
                )
        if tok.kind in {"ident", "keyword"} and tok.value.upper() == "CALL":
            if not _is_allowed_fulltext_call(tokens, i):
                raise QueryCompileError(
                    "unsupported",
                    "This construct is unavailable: CALL. Only db.index.fulltext.queryNodes / queryRelationships are allowed.",
                )


def _is_embedding_name(name: str) -> bool:
    lower = name.lower()
    return lower == "embedding" or lower.endswith("_embedding")


def _dotted_name(tokens: list[Tok], start: int, count: int) -> str | None:
    parts: list[str] = []
    i = start
    for k in range(count):
        if i >= len(tokens):
            return None
        if k > 0:
            if tokens[i].raw != ".":
                return None
            i += 1
            if i >= len(tokens):
                return None
        if tokens[i].kind not in {"ident", "keyword"}:
            return None
        parts.append(tokens[i].value.lower() if tokens[i].kind == "keyword" else tokens[i].value.lower())
        i += 1
    return ".".join(parts)


def _is_allowed_fulltext_call(tokens: list[Tok], call_i: int) -> bool:
    name = _dotted_name(tokens, call_i + 1, 4)
    return name in {
        "db.index.fulltext.querynodes",
        "db.index.fulltext.queryrelationships",
    }


def expand_anonymous_rels(tokens: list[Tok]) -> list[Tok]:
    """Turn `--`, `-->`, `<--`, `<-->` into explicit `-[]-` so maps can be injected."""
    empty = [
        Tok("symbol", "[", "[", 0),
        Tok("symbol", "]", "]", 0),
    ]
    out: list[Tok] = []
    i = 0
    while i < len(tokens):
        four = _raws(tokens, i, 4)
        three = _raws(tokens, i, 3)
        two = _raws(tokens, i, 2)
        if four == ["<", "-", "-", ">"]:
            out.extend([Tok("symbol", "<", "<", tokens[i].pos), Tok("symbol", "-", "-", 0)])
            out.extend(empty)
            out.extend([Tok("symbol", "-", "-", 0), Tok("symbol", ">", ">", 0)])
            i += 4
            continue
        if three == ["<", "-", "-"] and not _raws(tokens, i + 3, 1) == ["["]:
            out.extend([Tok("symbol", "<", "<", tokens[i].pos), Tok("symbol", "-", "-", 0)])
            out.extend(empty)
            out.append(Tok("symbol", "-", "-", 0))
            i += 3
            continue
        if three == ["-", "-", ">"] and (i == 0 or tokens[i - 1].raw != "<"):
            out.append(Tok("symbol", "-", "-", tokens[i].pos))
            out.extend(empty)
            out.extend([Tok("symbol", "-", "-", 0), Tok("symbol", ">", ">", 0)])
            i += 3
            continue
        if two == ["-", "-"] and (i == 0 or tokens[i - 1].raw != "<") and _raws(tokens, i + 2, 1) != [">"]:
            # `--(`  or `--[` already handled; `--[` should not match because second - then [
            if _raws(tokens, i + 1, 1) == ["["]:
                out.append(tokens[i])
                i += 1
                continue
            out.append(Tok("symbol", "-", "-", tokens[i].pos))
            out.extend(empty)
            out.append(Tok("symbol", "-", "-", 0))
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


def _raws(tokens: list[Tok], i: int, k: int) -> list[str]:
    return [tokens[j].raw for j in range(i, min(len(tokens), i + k))]


def rewrite_call_fulltext(tokens: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(tokens):
        if tokens[i].kind == "keyword" and tokens[i].value == "CALL" and _is_allowed_fulltext_call(tokens, i):
            name = _dotted_name(tokens, i + 1, 4) or ""
            # CALL + 4 dotted idents = tokens: CALL db . index . fulltext . queryXxx
            j = i + 1
            hops = 0
            while hops < 4 and j < len(tokens):
                if hops > 0:
                    j += 1  # dot
                j += 1
                hops += 1
            if j >= len(tokens) or tokens[j].raw != "(":
                raise QueryCompileError(
                    "syntax",
                    "Broken here: fulltext CALL is missing arguments. Fix the Cypher; do not restate the user question.",
                )
            close = _matching(tokens, j, "(", ")")
            inner = tokens[j + 1 : close]
            args = _split_top(inner, ",")
            if len(args) != 2:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: fulltext CALL needs (index, query). The server binds the index name — do not invent it.",
                )
            param_name = (
                SERVER_FT_RELS_PARAM
                if name.endswith("queryrelationships")
                else SERVER_FT_NODES_PARAM
            )
            new_inner = [Tok("param", param_name, f"${param_name}", 0), Tok("symbol", ",", ",", 0)]
            new_inner.extend(args[1])
            chunk = tokens[i : j + 1] + new_inner + [tokens[close]]
            i = close + 1
            if i < len(tokens) and tokens[i].kind == "keyword" and tokens[i].value == "YIELD":
                y_end = i + 1
                while y_end < len(tokens):
                    yt = tokens[y_end]
                    if yt.kind == "keyword" and yt.value in {
                        "MATCH", "OPTIONAL", "WITH", "RETURN", "UNWIND", "CALL",
                        "UNION", "ORDER", "SKIP", "LIMIT", "WHERE",
                    }:
                        break
                    y_end += 1
                chunk.extend(tokens[i:y_end])
                binding = _yield_first_binding(tokens, i + 1, y_end)
                i = y_end
                if binding:
                    if name.endswith("queryrelationships"):
                        pred = tokenize(
                            f"WHERE {binding}.run_id = ${SERVER_RUN_ID_PARAM}"
                        )
                        pred, i = _merge_where(pred, tokens, i)
                        chunk.extend(pred)
                    elif name.endswith("querynodes"):
                        if not _yielded_node_has_rel_hop(tokens, i, {binding}):
                            pred = tokenize(
                                f"WHERE ${SERVER_RUN_ID_PARAM} IN {binding}.run_ids"
                            )
                            pred, i = _merge_where(pred, tokens, i)
                            chunk.extend(pred)
            out.extend(chunk)
            continue
        out.append(tokens[i])
        i += 1
    return out


def _merge_where(
    where_clause: list[Tok], tokens: list[Tok], i: int
) -> tuple[list[Tok], int]:
    """If the next token is WHERE, fold its predicate with AND; else keep WHERE …."""
    if i < len(tokens) and tokens[i].kind == "keyword" and tokens[i].value == "WHERE":
        end = _clause_end(tokens, i + 1)
        extra = tokens[i + 1 : end]
        i = end
        return (
            where_clause
            + [Tok("keyword", "AND", "AND", 0)]
            + extra,
            i,
        )
    return where_clause, i


def _yield_first_binding(tokens: list[Tok], start: int, end: int) -> str:
    """In-scope name of the first YIELD item (`node AS n` → `n`)."""
    parts = _split_top(tokens[start:end], ",")
    if not parts or not parts[0]:
        return ""
    part = parts[0]
    for k, tok in enumerate(part):
        if tok.kind == "keyword" and tok.value == "AS" and k + 1 < len(part):
            nxt = part[k + 1]
            if nxt.kind in {"ident", "keyword"}:
                return nxt.raw
    if part[0].kind in {"ident", "keyword"}:
        return part[0].raw
    return ""


def rewrite_relationship_maps(tokens: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(tokens):
        if _is_rel_open(tokens, i):
            # tokens[i] is `[`
            close = _matching(tokens, i, "[", "]")
            inner = tokens[i + 1 : close]
            _reject_unbounded_or_long_path(inner)
            new_inner = _inject_run_id_map(inner)
            out.append(tokens[i])
            out.extend(new_inner)
            out.append(tokens[close])
            _require_node_after_rel(tokens, close)
            i = close + 1
            continue
        out.append(tokens[i])
        i += 1
    return out


def _require_node_after_rel(tokens: list[Tok], close: int) -> None:
    j = close + 1
    if j < len(tokens) and tokens[j].raw == "-":
        j += 1
        if j < len(tokens) and tokens[j].raw == ">":
            j += 1
    if j >= len(tokens) or tokens[j].raw != "(":
        raise QueryCompileError(
            "syntax",
            "Broken here: relationship pattern is missing a node after the arrow. Fix the Cypher; do not restate the user question.",
        )


def _is_rel_open(tokens: list[Tok], i: int) -> bool:
    if tokens[i].raw != "[":
        return False
    if i == 0:
        return False
    return tokens[i - 1].raw == "-"


def _reject_unbounded_or_long_path(inner: list[Tok]) -> None:
    for j, tok in enumerate(inner):
        if tok.raw != "*":
            continue
        rest = inner[j + 1 :]
        if not rest or rest[0].kind != "number":
            raise QueryCompileError(
                "unsupported",
                "This construct is unavailable: unbounded *. Use a path *1..4.",
            )
        lo = int(float(rest[0].value))
        hi = lo
        if len(rest) >= 3 and rest[1].raw == ".." and rest[2].kind == "number":
            hi = int(float(rest[2].value))
        elif len(rest) >= 2 and rest[1].raw == "..":
            raise QueryCompileError(
                "unsupported",
                "This construct is unavailable: unbounded *. Use a path *1..4.",
            )
        if lo < 1 or hi > MAX_PATH_LENGTH or lo > hi:
            raise QueryCompileError(
                "unsupported",
                f"This construct is unavailable: path length {lo}..{hi}. Use a path *1..4.",
            )


def _inject_run_id_map(inner: list[Tok]) -> list[Tok]:
    # Strip a user run_id key if present, then ensure {run_id: $__run_id}.
    map_start = _find_top_symbol(inner, "{")
    if map_start is not None:
        map_end = _matching(inner, map_start, "{", "}")
        body = inner[map_start + 1 : map_end]
        body = _strip_map_key(body, "run_id")
        injected = [Tok("ident", "run_id", "run_id", 0), Tok("symbol", ":", ":", 0),
                    Tok("param", SERVER_RUN_ID_PARAM, f"${SERVER_RUN_ID_PARAM}", 0)]
        if body:
            injected.append(Tok("symbol", ",", ",", 0))
            injected.extend(body)
        return inner[: map_start] + [inner[map_start]] + injected + [inner[map_end]] + inner[map_end + 1 :]
    # Insert before *range if any, else at end.
    star = next((j for j, t in enumerate(inner) if t.raw == "*"), len(inner))
    injected = [
        Tok("symbol", "{", "{", 0),
        Tok("ident", "run_id", "run_id", 0),
        Tok("symbol", ":", ":", 0),
        Tok("param", SERVER_RUN_ID_PARAM, f"${SERVER_RUN_ID_PARAM}", 0),
        Tok("symbol", "}", "}", 0),
    ]
    return inner[:star] + injected + inner[star:]


def _strip_map_key(body: list[Tok], key: str) -> list[Tok]:
    parts = _split_top(body, ",")
    kept: list[list[Tok]] = []
    for part in parts:
        if not part:
            continue
        name = part[0].value.lower() if part[0].kind in {"ident", "keyword", "string"} else ""
        if name.strip("'`\"") == key:
            continue
        kept.append(part)
    return _join_parts(kept, Tok("symbol", ",", ",", 0))


def rewrite_run_id_predicates(tokens: list[Tok]) -> list[Tok]:
    """Force `.run_id` / `IN n.run_ids` comparisons to the server parameter."""
    out: list[Tok] = []
    i = 0
    while i < len(tokens):
        if (
            i + 4 < len(tokens)
            and tokens[i].kind in {"ident", "keyword", "string", "param", "number"}
            and tokens[i + 1].kind == "keyword"
            and tokens[i + 1].value == "IN"
            and tokens[i + 2].kind in {"ident", "keyword"}
            and tokens[i + 3].raw == "."
            and tokens[i + 4].value.lower() == "run_ids"
        ):
            out.append(Tok("param", SERVER_RUN_ID_PARAM, f"${SERVER_RUN_ID_PARAM}", 0))
            out.append(tokens[i + 1])
            out.extend(tokens[i + 2 : i + 5])
            i += 5
            continue
        if (
            i + 3 < len(tokens)
            and tokens[i].kind in {"ident", "keyword"}
            and tokens[i + 1].raw == "."
            and tokens[i + 2].value.lower() == "run_id"
            and tokens[i + 3].raw in {"=", "<>", "IN", "IS"}
        ):
            out.extend(tokens[i : i + 3])
            out.append(tokens[i + 3])
            if tokens[i + 3].value == "IN":
                j = i + 4
                if j < len(tokens) and tokens[j].raw == "[":
                    close = _matching(tokens, j, "[", "]")
                    out.extend([
                        Tok("symbol", "[", "[", 0),
                        Tok("param", SERVER_RUN_ID_PARAM, f"${SERVER_RUN_ID_PARAM}", 0),
                        Tok("symbol", "]", "]", 0),
                    ])
                    i = close + 1
                    continue
            if tokens[i + 3].value == "IS":
                out.append(tokens[i + 4] if i + 4 < len(tokens) else Tok("keyword", "NULL", "NULL", 0))
                i += 5
                continue
            j = i + 4
            if j < len(tokens) and tokens[j].raw in {"(", "["}:
                close = _matching(tokens, j, tokens[j].raw, ")" if tokens[j].raw == "(" else "]")
                i = close + 1
            else:
                i = j + 1
            out.append(Tok("param", SERVER_RUN_ID_PARAM, f"${SERVER_RUN_ID_PARAM}", 0))
            continue
        out.append(tokens[i])
        i += 1
    return out


def enforce_scope(tokens: list[Tok]) -> None:
    if not any(t.kind == "keyword" and t.value in {"MATCH", "CALL"} for t in tokens):
        raise QueryCompileError(
            "scope",
            "The query cannot be isolated to this corpus: add a MATCH of relationships in this corpus.",
        )
    if not any(t.kind == "keyword" and t.value == "RETURN" for t in tokens):
        raise QueryCompileError(
            "syntax",
            "Broken here: missing RETURN. Fix the Cypher; do not restate the user question.",
        )
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "keyword" and tok.value == "CALL" and _is_allowed_fulltext_call(tokens, i):
            y_i = _find_keyword_from(tokens, i, "YIELD")
            if y_i is None:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: fulltext CALL must YIELD a node or relationship. Fix the Cypher; do not restate the user question.",
                )
            names = _yield_names(tokens, y_i + 1)
            if not names:
                raise QueryCompileError(
                    "syntax",
                    "Broken here: YIELD is empty. Fix the Cypher; do not restate the user question.",
                )
            i = y_i + 1
            continue
        if tok.kind == "keyword" and tok.value == "MATCH":
            optional = i > 0 and tokens[i - 1].kind == "keyword" and tokens[i - 1].value == "OPTIONAL"
            end = _clause_end(tokens, i + 1)
            rels, nodes = _pattern_stats(tokens[i + 1 : end])
            if rels == 0:
                if optional:
                    raise QueryCompileError(
                        "scope",
                        "The query cannot be isolated to this corpus: OPTIONAL MATCH must include a relationship. Add an edge pattern, not a node-only MATCH.",
                    )
                if not nodes:
                    raise QueryCompileError(
                        "scope",
                        "The query cannot be isolated to this corpus: MATCH only by node. Add an explicit relationship or a named node.",
                    )
            i = end
            continue
        i += 1


def apply_node_run_ids(tokens: list[Tok]) -> list[Tok]:
    """`$__run_id IN n.run_ids` on named nodes that are not incident to a relationship."""
    match_at = [
        i
        for i, tok in enumerate(tokens)
        if tok.kind == "keyword" and tok.value == "MATCH"
    ]
    for match_i in reversed(match_at):
        optional = match_i > 0 and tokens[match_i - 1].kind == "keyword" and tokens[match_i - 1].value == "OPTIONAL"
        end = _match_end(tokens, match_i)
        body = tokens[match_i + 1 : end]
        pattern, _where_at = _split_pattern_where(body)
        if optional and _pattern_rel_count(pattern) == 0:
            raise QueryCompileError(
                "scope",
                "The query cannot be isolated to this corpus: OPTIONAL MATCH must include a relationship. Add an edge pattern, not a node-only MATCH.",
            )
        floating = _floating_node_vars(pattern)
        preds: list[Tok] = []
        for var in floating:
            if _already_has_run_ids(body, var):
                continue
            if preds:
                preds.append(Tok("keyword", "AND", "AND", 0))
            preds.extend(_run_ids_pred(var))
        if not preds:
            continue
        if _where_at is not None:
            injected = [Tok("keyword", "AND", "AND", 0)] + preds
            tokens[match_i:end] = tokens[match_i:end] + injected
        else:
            injected = [Tok("keyword", "WHERE", "WHERE", 0)] + preds
            tokens[match_i:end] = [tokens[match_i]] + body + injected
    return tokens


def _run_ids_pred(var: str) -> list[Tok]:
    return tokenize(f"${SERVER_RUN_ID_PARAM} IN {var}.run_ids")


def _already_has_run_ids(body: list[Tok], var: str) -> bool:
    needle = emit(_run_ids_pred(var)).replace(" ", "")
    return needle.lower() in emit(body).replace(" ", "").lower()


def _split_pattern_where(body: list[Tok]) -> tuple[list[Tok], int | None]:
    depth_paren = depth_brack = depth_brace = 0
    for i, tok in enumerate(body):
        if tok.raw == "(":
            depth_paren += 1
        elif tok.raw == ")":
            depth_paren -= 1
        elif tok.raw == "[":
            depth_brack += 1
        elif tok.raw == "]":
            depth_brack -= 1
        elif tok.raw == "{":
            depth_brace += 1
        elif tok.raw == "}":
            depth_brace -= 1
        elif (
            depth_paren == depth_brack == depth_brace == 0
            and tok.kind == "keyword"
            and tok.value == "WHERE"
        ):
            return body[:i], i
    return body, None


def _pattern_rel_count(pattern: list[Tok]) -> int:
    rels, _ = _pattern_stats(pattern)
    return rels


def _match_end(tokens: list[Tok], match_i: int) -> int:
    """End index of this MATCH clause, stopping at `}` that closes an enclosing EXISTS."""
    base_brace = _brace_depth(tokens, match_i)
    depth_paren = depth_brack = 0
    depth_brace = 0
    i = match_i + 1
    while i < len(tokens):
        t = tokens[i]
        if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            if t.raw == "}" and base_brace > 0:
                return i
            if t.kind == "keyword" and t.value in {
                "MATCH",
                "OPTIONAL",
                "WITH",
                "RETURN",
                "UNWIND",
                "CALL",
                "UNION",
                "ORDER",
                "SKIP",
                "LIMIT",
            }:
                return i
        if t.raw == "(":
            depth_paren += 1
        elif t.raw == ")":
            depth_paren -= 1
        elif t.raw == "[":
            depth_brack += 1
        elif t.raw == "]":
            depth_brack -= 1
        elif t.raw == "{":
            depth_brace += 1
        elif t.raw == "}":
            depth_brace -= 1
        i += 1
    return len(tokens)


def _brace_depth(tokens: list[Tok], idx: int) -> int:
    depth = 0
    for tok in tokens[:idx]:
        if tok.raw == "{":
            depth += 1
        elif tok.raw == "}":
            depth -= 1
    return depth


def _floating_node_vars(pattern: list[Tok]) -> list[str]:
    """Named nodes in a MATCH pattern that do not touch a relationship."""
    incident: set[str] = set()
    named: list[str] = []
    seen: set[str] = set()
    prev: str | None = None
    pending_after_rel = False
    depth_paren = depth_brack = depth_brace = 0
    i = 0
    while i < len(pattern):
        tok = pattern[i]
        if tok.raw == "(" and depth_brack == 0 and depth_brace == 0 and depth_paren == 0:
            close = _matching(pattern, i, "(", ")")
            inner = pattern[i + 1 : close]
            name = None
            if inner and inner[0].kind in {"ident", "keyword"} and inner[0].raw not in {":"}:
                name = inner[0].raw
                key = name.lower()
                if key not in seen:
                    named.append(name)
                    seen.add(key)
            if pending_after_rel and name:
                incident.add(name.lower())
            pending_after_rel = False
            prev = name
            i = close + 1
            continue
        if _is_rel_open(pattern, i) and depth_paren == 0 and depth_brace == 0:
            if prev:
                incident.add(prev.lower())
            pending_after_rel = True
            close = _matching(pattern, i, "[", "]")
            i = close + 1
            continue
        if tok.raw == "," and depth_paren == depth_brack == depth_brace == 0:
            prev = None
            pending_after_rel = False
        if tok.raw == "(":
            depth_paren += 1
        elif tok.raw == ")":
            depth_paren -= 1
        elif tok.raw == "[":
            depth_brack += 1
        elif tok.raw == "]":
            depth_brack -= 1
        elif tok.raw == "{":
            depth_brace += 1
        elif tok.raw == "}":
            depth_brace -= 1
        i += 1
    return [name for name in named if name.lower() not in incident]


def _yielded_node_has_rel_hop(tokens: list[Tok], start: int, live: set[str]) -> bool:
    """True if a later MATCH uses the yielded node (or a WITH alias) on a relationship."""
    live_l = {name.lower() for name in live}
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "keyword" and tok.value == "WHERE":
            i = _clause_end(tokens, i + 1)
            continue
        if tok.kind == "keyword" and tok.value == "WITH":
            end = _clause_end(tokens, i + 1)
            live_l = _with_live(tokens[i + 1 : end], live_l)
            i = end
            continue
        if tok.kind == "keyword" and tok.value == "OPTIONAL":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and nxt.kind == "keyword" and nxt.value == "MATCH":
                i += 1
                continue
        if tok.kind == "keyword" and tok.value == "MATCH":
            end = _match_end(tokens, i)
            body = tokens[i + 1 : end]
            pattern, _ = _split_pattern_where(body)
            rels, nodes = _pattern_stats(pattern)
            if rels > 0:
                floating = {name.lower() for name in _floating_node_vars(pattern)}
                incident = {name.lower() for name in nodes if name.lower() not in floating}
                if live_l & incident:
                    return True
            i = end
            continue
        if tok.kind == "keyword" and tok.value in {"RETURN", "UNION"}:
            break
        i += 1
    return False


def _with_live(body: list[Tok], live: set[str]) -> set[str]:
    """Names that still refer to the yielded node after this WITH."""
    items_part, _ = _split_pattern_where(body)
    new_live: set[str] = set()
    for part in _split_top(items_part, ","):
        part = [tok for tok in part if not (tok.kind == "keyword" and tok.value == "DISTINCT")]
        if not part:
            continue
        if len(part) == 1 and part[0].raw == "*":
            new_live |= live
            continue
        alias = None
        expr = part
        for k, tok in enumerate(part):
            if tok.kind == "keyword" and tok.value == "AS" and _depth_zero_at(part, k):
                expr = part[:k]
                if k + 1 < len(part) and part[k + 1].kind in {"ident", "keyword"}:
                    alias = part[k + 1].raw.lower()
                break
        if len(expr) == 1 and expr[0].kind in {"ident", "keyword"}:
            src = expr[0].raw.lower()
            if src in live:
                new_live.add(alias or src)
    return new_live


def _yield_names(tokens: list[Tok], start: int) -> list[str]:
    names: list[str] = []
    i = start
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "keyword" and tok.value in {
            "MATCH",
            "OPTIONAL",
            "WITH",
            "RETURN",
            "UNWIND",
            "CALL",
            "UNION",
            "ORDER",
            "SKIP",
            "LIMIT",
            "WHERE",
        }:
            break
        if tok.kind in {"ident", "keyword"} and tok.value not in {"AS"}:
            names.append(tok.raw)
            if i + 2 < len(tokens) and tokens[i + 1].value == "AS":
                i += 3
                continue
        i += 1
    return names


def _pattern_stats(tokens: list[Tok]) -> tuple[int, list[str]]:
    """Count relationship brackets and collect node variables in a MATCH body (until WHERE)."""
    rels = 0
    nodes: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i].kind == "keyword" and tokens[i].value == "WHERE":
            break
        if _is_rel_open(tokens, i):
            rels += 1
            close = _matching(tokens, i, "[", "]")
            i = close + 1
            continue
        if tokens[i].raw == "(":
            close = _matching(tokens, i, "(", ")")
            inner = tokens[i + 1 : close]
            if inner and inner[0].kind in {"ident", "keyword"} and inner[0].raw not in {":"}:
                nodes.append(inner[0].raw)
            i = close + 1
            continue
        i += 1
    return rels, nodes


def _clause_end(tokens: list[Tok], start: int) -> int:
    depth_paren = depth_brack = depth_brace = 0
    i = start
    while i < len(tokens):
        t = tokens[i]
        if t.raw == "(":
            depth_paren += 1
        elif t.raw == ")":
            depth_paren -= 1
        elif t.raw == "[":
            depth_brack += 1
        elif t.raw == "]":
            depth_brack -= 1
        elif t.raw == "{":
            depth_brace += 1
        elif t.raw == "}":
            depth_brace -= 1
        elif depth_paren == depth_brack == depth_brace == 0:
            if t.kind == "keyword" and t.value in {
                "MATCH",
                "OPTIONAL",
                "WITH",
                "RETURN",
                "UNWIND",
                "CALL",
                "UNION",
                "ORDER",
                "SKIP",
                "LIMIT",
            }:
                # OPTIONAL is part of OPTIONAL MATCH — if we started after MATCH, OPTIONAL later is new
                return i
        i += 1
    return len(tokens)


def apply_return_limits(
    tokens: list[Tok], max_rows: object = None
) -> tuple[list[Tok], int | None, bool, int]:
    branches = _split_top(tokens, "UNION")
    # UNION ALL is UNION + ALL — _split_top on keyword UNION leaves ALL on the next branch.
    rebuilt: list[list[Tok]] = []
    fetch_limit: int | None = None
    output_limit = DEFAULT_MAX_ROWS
    any_non_agg = False
    for branch in branches:
        # drop leading ALL from UNION ALL
        b = list(branch)
        if b and b[0].kind == "keyword" and b[0].value == "ALL":
            b = b[1:]
        if not b:
            continue
        b, branch_fetch, pure, branch_cap = _limit_one_return(b, max_rows)
        if not pure:
            if not any_non_agg:
                output_limit = branch_cap
            else:
                output_limit = max(output_limit, branch_cap)
            any_non_agg = True
            fetch_limit = max(fetch_limit or 0, branch_fetch or 0) or branch_fetch
        rebuilt.append(b)
    if len(rebuilt) == 1:
        out = rebuilt[0]
    else:
        out = []
        for n, b in enumerate(rebuilt):
            if n:
                out.append(Tok("keyword", "UNION", "UNION", 0))
                # preserve ALL if original had it — simplified UNION
            out.extend(b)
        # Recover UNION ALL: if original tokens had ALL after UNION, keep it
        out = _restore_union_all(tokens, rebuilt)
    is_pure = not any_non_agg
    if is_pure:
        output_limit = resolve_output_limit(user_limit=None, max_rows=max_rows)
        return out, None, True, output_limit
    return out, (fetch_limit or output_limit + 1), False, output_limit


def _restore_union_all(original: list[Tok], branches: list[list[Tok]]) -> list[Tok]:
    alls: list[bool] = []
    i = 0
    depth = 0
    while i < len(original):
        t = original[i]
        if t.raw in "([{":
            depth += 1
        elif t.raw in ")]}":
            depth -= 1
        elif depth == 0 and t.kind == "keyword" and t.value == "UNION":
            nxt = original[i + 1] if i + 1 < len(original) else None
            alls.append(bool(nxt and nxt.kind == "keyword" and nxt.value == "ALL"))
        i += 1
    out: list[Tok] = []
    for n, b in enumerate(branches):
        if n:
            out.append(Tok("keyword", "UNION", "UNION", 0))
            if n - 1 < len(alls) and alls[n - 1]:
                out.append(Tok("keyword", "ALL", "ALL", 0))
        out.extend(b)
    return out


def _limit_one_return(
    tokens: list[Tok], max_rows: object = None
) -> tuple[list[Tok], int | None, bool, int]:
    r_i = None
    for i, t in enumerate(tokens):
        if t.kind == "keyword" and t.value == "RETURN":
            r_i = i
    if r_i is None:
        raise QueryCompileError(
            "syntax",
            "Broken here: missing RETURN. Fix the Cypher; do not restate the user question.",
        )
    tail = tokens[r_i + 1 :]
    # projection until ORDER/SKIP/LIMIT
    lim_i = None
    skip_i = None
    order_i = None
    depth = 0
    for j, t in enumerate(tail):
        if t.raw in "([{":
            depth += 1
        elif t.raw in ")]}":
            depth -= 1
        elif depth == 0 and t.kind == "keyword":
            if t.value == "ORDER" and order_i is None:
                order_i = j
            elif t.value == "SKIP" and skip_i is None:
                skip_i = j
            elif t.value == "LIMIT" and lim_i is None:
                lim_i = j
    proj_end = len(tail)
    for marker in (order_i, skip_i, lim_i):
        if marker is not None:
            proj_end = min(proj_end, marker)
    projection = tail[:proj_end]
    pure = _is_pure_aggregate(projection)
    user_limit = None
    if lim_i is not None and lim_i + 1 < len(tail) and tail[lim_i + 1].kind == "number":
        try:
            user_limit = int(float(tail[lim_i + 1].value))
        except ValueError:
            user_limit = None
    cap = resolve_output_limit(user_limit=user_limit, max_rows=max_rows)
    if pure:
        # keep user LIMIT if any but do not add a wrapping one
        return tokens, None, True, cap
    fetch = cap + 1
    new_tail: list[Tok]
    if lim_i is not None:
        # replace number
        new_tail = tail[: lim_i + 1] + [Tok("number", str(fetch), str(fetch), 0)] + tail[lim_i + 2 :]
    else:
        new_tail = tail + [
            Tok("keyword", "LIMIT", "LIMIT", 0),
            Tok("number", str(fetch), str(fetch), 0),
        ]
    return tokens[: r_i + 1] + new_tail, fetch, False, cap


def _is_pure_aggregate(projection: list[Tok]) -> bool:
    items = _split_top(projection, ",")
    if not items:
        return False
    for item in items:
        # drop AS alias
        expr = item
        for k, t in enumerate(item):
            if t.kind == "keyword" and t.value == "AS" and _depth_zero_at(item, k):
                expr = item[:k]
                break
        if not _item_is_aggregate(expr):
            return False
    return True


def _depth_zero_at(tokens: list[Tok], idx: int) -> bool:
    depth = 0
    for j, t in enumerate(tokens):
        if j == idx:
            return depth == 0
        if t.raw in "([{":
            depth += 1
        elif t.raw in ")]}":
            depth -= 1
    return False


def _item_is_aggregate(expr: list[Tok]) -> bool:
    if not expr:
        return False
    # count(*), sum(x), count(DISTINCT x)
    i = 0
    if expr[0].kind in {"ident", "keyword"} and expr[0].value in AGG_FUNCS:
        return True
    # DISTINCT count(...)
    if expr[0].kind == "keyword" and expr[0].value == "DISTINCT" and len(expr) > 1:
        return expr[1].kind in {"ident", "keyword"} and expr[1].value in AGG_FUNCS
    # nested: only treat as aggregate if every ident-call is agg and there is at least one
    found = False
    i = 0
    while i < len(expr):
        t = expr[i]
        nxt = expr[i + 1] if i + 1 < len(expr) else None
        if t.kind in {"ident", "keyword"} and nxt is not None and nxt.raw == "(":
            if t.value not in AGG_FUNCS and t.value not in {"TOLOWER", "TRIM", "COALESCE", "TOSTRING", "SIZE", "TYPE", "LABELS"}:
                return False
            if t.value in AGG_FUNCS:
                found = True
        i += 1
    return found


def _find_keyword_from(tokens: list[Tok], start: int, word: str) -> int | None:
    for i in range(start, len(tokens)):
        if tokens[i].kind == "keyword" and tokens[i].value == word:
            return i
    return None


def _find_top_symbol(tokens: list[Tok], symbol: str) -> int | None:
    depth = 0
    for i, t in enumerate(tokens):
        if t.raw in "([{":
            if t.raw == symbol and depth == 0:
                return i
            depth += 1
        elif t.raw in ")]}":
            depth -= 1
        elif t.raw == symbol and depth == 0:
            return i
    return None


def _matching(tokens: list[Tok], open_i: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(open_i, len(tokens)):
        if tokens[i].raw == open_ch:
            depth += 1
        elif tokens[i].raw == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise QueryCompileError(
        "syntax",
        f"Broken here: unmatched '{open_ch}'. Fix the Cypher; do not restate the user question.",
    )


def _split_top(tokens: list[Tok], sep: str) -> list[list[Tok]]:
    parts: list[list[Tok]] = []
    cur: list[Tok] = []
    depth_paren = depth_brack = depth_brace = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.raw == "(":
            depth_paren += 1
        elif t.raw == ")":
            depth_paren -= 1
        elif t.raw == "[":
            depth_brack += 1
        elif t.raw == "]":
            depth_brack -= 1
        elif t.raw == "{":
            depth_brace += 1
        elif t.raw == "}":
            depth_brace -= 1
        is_sep = False
        if depth_paren == depth_brack == depth_brace == 0:
            if sep == "UNION" and t.kind == "keyword" and t.value == "UNION":
                is_sep = True
            elif sep != "UNION" and t.raw == sep:
                is_sep = True
        if is_sep:
            parts.append(cur)
            cur = []
            i += 1
            continue
        cur.append(t)
        i += 1
    parts.append(cur)
    return parts


def _join_parts(parts: list[list[Tok]], sep: Tok) -> list[Tok]:
    out: list[Tok] = []
    for n, part in enumerate(parts):
        if n:
            out.append(sep)
        out.extend(part)
    return out


def emit(tokens: list[Tok]) -> str:
    if not tokens:
        return ""
    parts: list[str] = [tokens[0].raw]
    tight_after = set("([.")
    tight_before = set(".,;:)]}")
    for prev, tok in zip(tokens, tokens[1:]):
        if tok.raw in tight_before or prev.raw in tight_after:
            parts.append(tok.raw)
        elif prev.raw == ":" and tok.kind == "param":
            parts.append(" " + tok.raw)
        else:
            parts.append(" " + tok.raw)
    return "".join(parts)


def merge_params(compiled: CompiledQuery, user_params: dict[str, Any], run_id: str) -> dict[str, Any]:
    rid = (run_id or "").strip()
    if not rid:
        raise QueryCompileError(
            "scope",
            "The query cannot be isolated to this corpus: missing run_id in the server turn context.",
        )
    merged = dict(user_params)
    merged[SERVER_RUN_ID_PARAM] = rid
    merged[SERVER_FT_NODES_PARAM] = FT_NODE_INDEX
    merged[SERVER_FT_RELS_PARAM] = FT_REL_INDEX
    return merged
