"""Persistent accounts and conversation history backed by one SQLite database."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
SESSION_TOKEN_BYTES = 32
_PASSWORD_HASHER = PasswordHasher(memory_cost=19456, time_cost=2, parallelism=1)
_DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash("not-a-real-account-password")


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_username(value: str) -> str:
    username = (value or "").strip().lower()
    if not 3 <= len(username) <= 64:
        raise ValueError("Логин должен содержать от 3 до 64 символов")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    if any(ch not in allowed for ch in username):
        raise ValueError("Логин может содержать a-z, 0-9, точку, дефис и подчёркивание")
    return username


def validate_password(password: str) -> None:
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise ValueError(
            f"Пароль должен содержать от {PASSWORD_MIN_LENGTH} до {PASSWORD_MAX_LENGTH} символов"
        )


def hash_password(password: str) -> str:
    validate_password(password)
    return _PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return bool(_PASSWORD_HASHER.verify(password_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class AccountUser:
    id: str
    username: str
    is_active: bool = True


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    revoked_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_token ON auth_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id, expires_at);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'Новый чат',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations(user_id, updated_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    text TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'done',
    payload_json TEXT NOT NULL DEFAULT '{}',
    raw_text TEXT NOT NULL DEFAULT '',
    tool_messages_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE(conversation_id, turn_id, role),
    UNIQUE(conversation_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, ordinal);

CREATE TABLE IF NOT EXISTS conversation_sources (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL,
    source_file TEXT NOT NULL,
    PRIMARY KEY(conversation_id, source_id),
    UNIQUE(conversation_id, source_file)
);

CREATE TABLE IF NOT EXISTS graph_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    chains_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_runs_conversation ON graph_runs(conversation_id);
INSERT OR IGNORE INTO schema_migrations(version, applied_at)
VALUES(1, CAST(strftime('%s','now') AS INTEGER) * 1000);
"""


class AppStore:
    """Single-connection async repository. All ownership checks live here."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).expanduser())
        self.db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA foreign_keys=ON")
        await self.db.execute("PRAGMA busy_timeout=5000")
        await self.db.executescript(SCHEMA)
        await self.db.execute(
            "UPDATE messages SET status='aborted', updated_at=? WHERE status='streaming'",
            (now_ms(),),
        )
        await self.db.execute(
            "DELETE FROM auth_sessions WHERE expires_at<? OR (revoked_at IS NOT NULL AND revoked_at<?)",
            (now_ms(), now_ms() - 30 * 86400000),
        )
        await self.db.commit()

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    def _conn(self) -> aiosqlite.Connection:
        if self.db is None:
            raise RuntimeError("AppStore is not open")
        return self.db

    async def active_user_count(self) -> int:
        row = await (await self._conn().execute(
            "SELECT COUNT(*) AS n FROM users WHERE is_active=1"
        )).fetchone()
        return int(row["n"] if row else 0)

    async def user_count(self) -> int:
        row = await (await self._conn().execute("SELECT COUNT(*) AS n FROM users")).fetchone()
        return int(row["n"] if row else 0)

    async def create_user(self, username: str, password: str) -> AccountUser:
        username = normalize_username(username)
        password_hash = hash_password(password)
        user = AccountUser(str(uuid.uuid4()), username)
        ts = now_ms()
        async with self._write_lock:
            try:
                await self._conn().execute(
                    "INSERT INTO users(id,username,password_hash,is_active,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (user.id, user.username, password_hash, 1, ts, ts),
                )
                await self._conn().commit()
            except aiosqlite.IntegrityError as exc:
                await self._conn().rollback()
                raise ValueError("Пользователь с таким логином уже существует") from exc
        return user

    async def list_users(self) -> list[dict[str, Any]]:
        rows = await (await self._conn().execute(
            "SELECT id,username,is_active,created_at,updated_at FROM users ORDER BY username"
        )).fetchall()
        return [dict(row) for row in rows]

    async def set_user_active(self, username: str, active: bool) -> bool:
        username = normalize_username(username)
        ts = now_ms()
        async with self._write_lock:
            cur = await self._conn().execute(
                "UPDATE users SET is_active=?,updated_at=? WHERE username=? COLLATE NOCASE",
                (1 if active else 0, ts, username),
            )
            if cur.rowcount and not active:
                await self._conn().execute(
                    "UPDATE auth_sessions SET revoked_at=? WHERE user_id=(SELECT id FROM users WHERE username=? COLLATE NOCASE) AND revoked_at IS NULL",
                    (ts, username),
                )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def reset_password(self, username: str, password: str) -> bool:
        username = normalize_username(username)
        password_hash = hash_password(password)
        ts = now_ms()
        async with self._write_lock:
            cur = await self._conn().execute(
                "UPDATE users SET password_hash=?,updated_at=? WHERE username=? COLLATE NOCASE",
                (password_hash, ts, username),
            )
            if cur.rowcount:
                await self._conn().execute(
                    "UPDATE auth_sessions SET revoked_at=? WHERE user_id=(SELECT id FROM users WHERE username=? COLLATE NOCASE) AND revoked_at IS NULL",
                    (ts, username),
                )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def revoke_user_sessions(self, username: str) -> bool:
        username = normalize_username(username)
        async with self._write_lock:
            cur = await self._conn().execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE user_id=(SELECT id FROM users WHERE username=? COLLATE NOCASE) AND revoked_at IS NULL",
                (now_ms(), username),
            )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def authenticate(self, username: str, password: str) -> AccountUser | None:
        try:
            username = normalize_username(username)
        except ValueError:
            return None
        row = await (await self._conn().execute(
            "SELECT id,username,password_hash,is_active FROM users WHERE username=? COLLATE NOCASE",
            (username,),
        )).fetchone()
        candidate_hash = str(row["password_hash"]) if row is not None else _DUMMY_PASSWORD_HASH
        valid = await asyncio.to_thread(verify_password, candidate_hash, password)
        if not valid or row is None:
            return None
        if not bool(row["is_active"]):
            return None
        return AccountUser(str(row["id"]), str(row["username"]), True)

    async def create_session(self, user_id: str, lifetime_days: int = 30) -> str:
        raw = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute(
                "INSERT INTO auth_sessions(id,user_id,token_hash,created_at,last_seen_at,expires_at) VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), user_id, token_digest(raw), ts, ts, ts + lifetime_days * 86400000),
            )
            await self._conn().commit()
        return raw

    async def user_for_session(self, raw_token: str | None) -> AccountUser | None:
        if not raw_token:
            return None
        ts = now_ms()
        row = await (await self._conn().execute(
            """SELECT u.id,u.username,u.is_active,s.id AS session_id,s.last_seen_at
               FROM auth_sessions s JOIN users u ON u.id=s.user_id
               WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND u.is_active=1""",
            (token_digest(raw_token), ts),
        )).fetchone()
        if row is None:
            return None
        if ts - int(row["last_seen_at"]) > 3600000:
            async with self._write_lock:
                await self._conn().execute(
                    "UPDATE auth_sessions SET last_seen_at=? WHERE id=?", (ts, row["session_id"])
                )
                await self._conn().commit()
        return AccountUser(str(row["id"]), str(row["username"]), True)

    async def revoke_session(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        async with self._write_lock:
            await self._conn().execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (now_ms(), token_digest(raw_token)),
            )
            await self._conn().commit()

    async def create_conversation(self, user_id: str, conversation_id: str | None = None) -> dict[str, Any]:
        cid = conversation_id or str(uuid.uuid4())
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute(
                "INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
                (cid, user_id, "Новый чат", ts, ts),
            )
            await self._conn().commit()
        return {"id": cid, "title": "Новый чат", "createdAt": ts, "updatedAt": ts}

    async def conversation_owned(self, user_id: str, conversation_id: str) -> bool:
        row = await (await self._conn().execute(
            "SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
        )).fetchone()
        return row is not None

    async def list_conversations(self, user_id: str, limit: int = 50, before: str | None = None) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        args: list[Any] = [user_id]
        where = "user_id=?"
        if before:
            try:
                cursor_time, cursor_id = before.split(":", 1)
                cursor_ts = int(cursor_time)
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid conversation cursor") from exc
            where += " AND (updated_at<? OR (updated_at=? AND id<?))"
            args.extend([cursor_ts, cursor_ts, cursor_id])
        args.append(limit + 1)
        rows = await (await self._conn().execute(
            f"SELECT id,title,created_at,updated_at FROM conversations WHERE {where} ORDER BY updated_at DESC,id DESC LIMIT ?",
            args,
        )).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [
            {"id": str(r["id"]), "title": str(r["title"]), "createdAt": int(r["created_at"]), "updatedAt": int(r["updated_at"])}
            for r in rows
        ]
        return {
            "items": items,
            "nextCursor": (
                f'{items[-1]["updatedAt"]}:{items[-1]["id"]}'
                if has_more and items else None
            ),
        }

    async def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        conv = await (await self._conn().execute(
            "SELECT id,title,created_at,updated_at FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id),
        )).fetchone()
        if conv is None:
            return None
        rows = await (await self._conn().execute(
            "SELECT id,role,text,status,payload_json,created_at,updated_at FROM messages WHERE conversation_id=? ORDER BY ordinal",
            (conversation_id,),
        )).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            payload = _loads(row["payload_json"], {})
            payload.update({"id": str(row["id"]), "role": str(row["role"]), "text": str(row["text"]), "status": str(row["status"])})
            messages.append(payload)
        return {
            "id": str(conv["id"]), "title": str(conv["title"]),
            "createdAt": int(conv["created_at"]), "updatedAt": int(conv["updated_at"]),
            "messages": messages,
        }

    async def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> bool:
        clean = " ".join((title or "").strip().split())[:80] or "Новый чат"
        async with self._write_lock:
            cur = await self._conn().execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE id=? AND user_id=?",
                (clean, now_ms(), conversation_id, user_id),
            )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        async with self._write_lock:
            cur = await self._conn().execute(
                "DELETE FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
            )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def clear_conversation(self, user_id: str, conversation_id: str) -> bool:
        if not await self.conversation_owned(user_id, conversation_id):
            return False
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
                await self._conn().execute("DELETE FROM conversation_sources WHERE conversation_id=?", (conversation_id,))
                await self._conn().execute(
                    "UPDATE conversations SET title='Новый чат',updated_at=? WHERE id=?", (now_ms(), conversation_id)
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return True

    async def begin_turn(self, user_id: str, conversation_id: str, turn_id: str, text: str) -> tuple[str, str]:
        if not await self.conversation_owned(user_id, conversation_id):
            raise KeyError(conversation_id)
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                existing = await (await self._conn().execute(
                    "SELECT 1 FROM messages WHERE conversation_id=? AND turn_id=? LIMIT 1", (conversation_id, turn_id)
                )).fetchone()
                if existing is not None:
                    raise ValueError("duplicate turn_id")
                row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 AS n FROM messages WHERE conversation_id=?", (conversation_id,)
                )).fetchone()
                ordinal = int(row["n"])
                user_mid, assistant_mid = str(uuid.uuid4()), str(uuid.uuid4())
                ts = now_ms()
                title_row = await (await self._conn().execute(
                    "SELECT title FROM conversations WHERE id=?", (conversation_id,)
                )).fetchone()
                title = str(title_row["title"]) if title_row else "Новый чат"
                next_title = " ".join(text.strip().splitlines()[0].split())[:42] or "Новый чат"
                await self._conn().execute(
                    "INSERT INTO messages(id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (user_mid, conversation_id, turn_id, ordinal, "user", text, "done", "{}", ts, ts),
                )
                await self._conn().execute(
                    "INSERT INTO messages(id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (assistant_mid, conversation_id, turn_id, ordinal + 1, "assistant", "", "streaming", "{}", ts, ts),
                )
                await self._conn().execute(
                    "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
                    (next_title if title == "Новый чат" else title, ts, conversation_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return user_mid, assistant_mid

    async def finish_turn(
        self, conversation_id: str, assistant_message_id: str, *, text: str,
        status: str, payload: dict[str, Any], raw_text: str = "",
        tool_messages: list[dict[str, Any]] | None = None,
        sources: Iterable[tuple[int, str]] = (), graph_run_id: str = "",
        graph_chains: list[dict[str, Any]] | None = None,
    ) -> None:
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    "UPDATE messages SET text=?,status=?,payload_json=?,raw_text=?,tool_messages_json=?,updated_at=? WHERE id=? AND conversation_id=?",
                    (text, status, _json(payload), raw_text, _json(tool_messages or []), ts, assistant_message_id, conversation_id),
                )
                await self._conn().execute("DELETE FROM conversation_sources WHERE conversation_id=?", (conversation_id,))
                await self._conn().executemany(
                    "INSERT INTO conversation_sources(conversation_id,source_id,source_file) VALUES(?,?,?)",
                    [(conversation_id, int(sid), str(path)) for sid, path in sources],
                )
                if graph_run_id and graph_chains:
                    await self._conn().execute(
                        "INSERT OR REPLACE INTO graph_runs(id,conversation_id,message_id,chains_json,created_at) VALUES(?,?,?,?,?)",
                        (graph_run_id, conversation_id, assistant_message_id, _json(graph_chains), ts),
                    )
                await self._conn().execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conversation_id))
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise

    async def load_model_context(self, user_id: str, conversation_id: str, limit: int) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
        if not await self.conversation_owned(user_id, conversation_id):
            raise KeyError(conversation_id)
        rows = await (await self._conn().execute(
            """SELECT u.text AS user_text,a.raw_text,a.tool_messages_json
               FROM messages u JOIN messages a
                 ON a.conversation_id=u.conversation_id AND a.turn_id=u.turn_id AND a.role='assistant'
               WHERE u.conversation_id=? AND u.role='user' AND a.status='done' AND a.raw_text!=''
               ORDER BY u.ordinal DESC LIMIT ?""",
            (conversation_id, max(1, int(limit))),
        )).fetchall()
        turns = [
            {"user": str(r["user_text"]), "assistant": str(r["raw_text"]), "tool_messages": _loads(r["tool_messages_json"], [])}
            for r in reversed(rows)
        ]
        source_rows = await (await self._conn().execute(
            "SELECT source_id,source_file FROM conversation_sources WHERE conversation_id=? ORDER BY source_id",
            (conversation_id,),
        )).fetchall()
        return turns, [(int(r["source_id"]), str(r["source_file"])) for r in source_rows]

    async def get_graph_run(self, user_id: str, run_id: str) -> list[dict[str, Any]] | None:
        row = await (await self._conn().execute(
            """SELECT g.chains_json FROM graph_runs g JOIN conversations c ON c.id=g.conversation_id
               WHERE g.id=? AND c.user_id=?""", (run_id, user_id)
        )).fetchone()
        return _loads(row["chains_json"], []) if row is not None else None

    async def backup(self, destination: str) -> None:
        dest = str(Path(destination).expanduser())
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        target = await aiosqlite.connect(dest)
        try:
            async with self._write_lock:
                await self._conn().backup(target)
                await target.commit()
        finally:
            await target.close()
