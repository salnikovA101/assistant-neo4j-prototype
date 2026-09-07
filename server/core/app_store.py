"""Persistent accounts and conversation history backed by one SQLite database."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from server.utils.constants import LEGACY_RETRIEVAL_STATE_VERSIONS, RETRIEVAL_STATE_VERSION
from server.tools.source_registry import present_source_aliases_in_value
from server.core.sq_status import SQ_STATUS_USER_NOTICE


PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
SESSION_TOKEN_BYTES = 32
_PASSWORD_HASHER = PasswordHasher(memory_cost=19456, time_cost=2, parallelism=1)
_DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash("not-a-real-account-password")
_SUBQUESTION_REF_RE = re.compile(r"^subquestion:([1-9][0-9]*)$")
RUN_ID_MAX_LENGTH = 128


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


def normalize_run_id(value: str) -> str:
    """Return one exact, non-empty corpus id safe for parameterized queries."""
    run_id = str(value or "").strip()
    if not run_id:
        raise ValueError("run_id не может быть пустым")
    if len(run_id) > RUN_ID_MAX_LENGTH:
        raise ValueError(f"run_id не может быть длиннее {RUN_ID_MAX_LENGTH} символов")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in run_id):
        raise ValueError("run_id не может содержать управляющие символы")
    return run_id


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


def _model_card_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Keep internal field provenance out of card data shown to the model."""
    payload = dict(value)
    payload.pop("provenance", None)
    return payload


class CorruptStoreError(ValueError):
    """Stored JSON blob cannot be parsed."""


class ConversationRunMismatchError(Exception):
    """A conversation is pinned to a corpus other than the account corpus."""

    def __init__(self, conversation_run_id: str, account_run_id: str) -> None:
        super().__init__("conversation_run_mismatch")
        self.conversation_run_id = conversation_run_id
        self.account_run_id = account_run_id


def _loads(value: str | None, fallback: Any) -> Any:
    if value is None or value == "":
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise CorruptStoreError("corrupt JSON in app store") from exc


@dataclass(frozen=True)
class AccountUser:
    id: str
    username: str
    is_active: bool = True
    run_id: str = ""


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
    run_id TEXT NOT NULL DEFAULT '',
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
    run_id TEXT NOT NULL DEFAULT '',
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


STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'main',
    mode TEXT NOT NULL DEFAULT 'auto',
    created_from_checkpoint_id TEXT,
    head_checkpoint_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_branches_conversation
    ON branches(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    branch_id TEXT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES checkpoints(id),
    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_branch
    ON checkpoints(branch_id, created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_parent ON checkpoints(parent_id);

CREATE TABLE IF NOT EXISTS subquestions (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    display_no INTEGER NOT NULL,
    text TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(conversation_id, canonical_text),
    UNIQUE(conversation_id, display_no)
);

CREATE TABLE IF NOT EXISTS graph_snapshots (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sq_id TEXT NOT NULL REFERENCES subquestions(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(conversation_id, sq_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS retrieval_snapshots (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES retrieval_snapshots(id),
    state_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoint_subquestions (
    checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    sq_id TEXT NOT NULL REFERENCES subquestions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'not_closed',
    status_origin TEXT NOT NULL DEFAULT 'legacy',
    status_reason TEXT NOT NULL DEFAULT '',
    status_source_refs_json TEXT NOT NULL DEFAULT '[]',
    status_message_id TEXT,
    position INTEGER NOT NULL DEFAULT 0,
    question_count INTEGER NOT NULL DEFAULT 0,
    unit_count INTEGER NOT NULL DEFAULT 0,
    graph_snapshot_id TEXT REFERENCES graph_snapshots(id),
    agenda_visible INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(checkpoint_id, sq_id)
);

CREATE TABLE IF NOT EXISTS evidence_units (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    created_checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
    unit_no INTEGER NOT NULL,
    sq_id TEXT REFERENCES subquestions(id) ON DELETE SET NULL,
    signature TEXT NOT NULL,
    chain_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(conversation_id, signature)
);
CREATE INDEX IF NOT EXISTS idx_units_conversation
    ON evidence_units(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS checkpoint_units (
    checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    unit_id TEXT NOT NULL REFERENCES evidence_units(id) ON DELETE CASCADE,
    unit_no INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(checkpoint_id, unit_id)
);

CREATE TABLE IF NOT EXISTS pending_approvals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    branch_id TEXT NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    assistant_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    base_checkpoint_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    revision INTEGER NOT NULL DEFAULT 1,
    tool_call_json TEXT NOT NULL,
    resume_json TEXT NOT NULL DEFAULT '{}',
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_branch
    ON pending_approvals(branch_id, status, updated_at);

CREATE TABLE IF NOT EXISTS card_templates (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    archived_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_card_templates_owner
    ON card_templates(owner_user_id, archived_at, updated_at);

CREATE TABLE IF NOT EXISTS card_template_versions (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL REFERENCES card_templates(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    schema_json TEXT NOT NULL,
    ui_json TEXT NOT NULL DEFAULT '{}',
    instructions TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    UNIQUE(template_id, version)
);

CREATE TABLE IF NOT EXISTS card_template_hides (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    template_id TEXT NOT NULL REFERENCES card_templates(id) ON DELETE CASCADE,
    hidden_at INTEGER NOT NULL,
    PRIMARY KEY(user_id, template_id)
);

CREATE TABLE IF NOT EXISTS card_drafts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    origin_checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
    template_version_id TEXT NOT NULL REFERENCES card_template_versions(id),
    data_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    gaps_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    template_version_id TEXT NOT NULL REFERENCES card_template_versions(id),
    title TEXT NOT NULL,
    archived_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_user
    ON cards(user_id, archived_at, updated_at DESC);

CREATE TABLE IF NOT EXISTS card_revisions (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    gaps_json TEXT NOT NULL DEFAULT '[]',
    origin_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL,
    UNIQUE(card_id, revision)
);

CREATE TABLE IF NOT EXISTS checkpoint_card_attachments (
    checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id) ON DELETE CASCADE,
    card_revision_id TEXT NOT NULL REFERENCES card_revisions(id) ON DELETE CASCADE,
    PRIMARY KEY(checkpoint_id, card_revision_id)
);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_conversation
    ON retrieval_events(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS llm_key_state (
    key_fp TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_model_bans (
    key_fp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    banned_at INTEGER NOT NULL,
    PRIMARY KEY(key_fp, profile_id)
);
CREATE INDEX IF NOT EXISTS idx_llm_model_bans_key ON llm_model_bans(key_fp);
"""


EXPERIMENT_TEMPLATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"], "title": "Название"},
        "objective": {"type": ["string", "null"], "title": "Цель"},
        "product_or_matrix": {"type": ["string", "null"], "title": "Продукт или матрица"},
        "culture_or_material": {"type": ["string", "null"], "title": "Культура или материал"},
        "conditions": {"type": ["string", "null"], "title": "Условия"},
        "procedure": {"type": ["string", "null"], "title": "Процедура"},
        "controls": {"type": ["string", "null"], "title": "Контроль"},
        "measurements": {"type": ["string", "null"], "title": "Измерения"},
        "expected_result": {"type": ["string", "null"], "title": "Ожидаемый результат"},
    },
    "required": ["title", "objective", "product_or_matrix"],
    "additionalProperties": False,
}


class AppStore:
    """Single-connection async repository. All ownership checks live here."""

    def __init__(self, path: str, *, default_run_id: str | None = None) -> None:
        if default_run_id is None:
            # Keep direct test/maintenance construction compatible while the
            # server and CLIs pass the already-loaded config explicitly.
            from server.utils.config import load_config

            default_run_id = load_config().run_id
        self.path = str(Path(path).expanduser())
        self.default_run_id = normalize_run_id(default_run_id)
        self.db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA foreign_keys=ON")
        await self.db.execute("PRAGMA busy_timeout=5000")
        await self.db.execute("PRAGMA journal_mode=WAL")
        await self.db.executescript(SCHEMA)
        await self._migrate_state_schema()
        leftover = await (await self.db.execute(
            """SELECT conversation_id,id FROM messages
               WHERE role='assistant' AND status='streaming'"""
        )).fetchall()
        await self.db.execute(
            "DELETE FROM auth_sessions WHERE expires_at<? OR (revoked_at IS NOT NULL AND revoked_at<?)",
            (now_ms(), now_ms() - 30 * 86400000),
        )
        await self.db.commit()
        for row in leftover:
            await self.rollback_turn(
                str(row["conversation_id"]),
                str(row["id"]),
                reason="aborted",
                message="Поток оборвался",
            )

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    def _conn(self) -> aiosqlite.Connection:
        if self.db is None:
            raise RuntimeError("AppStore is not open")
        return self.db

    async def _table_columns(self, table: str) -> set[str]:
        rows = await (await self._conn().execute(f"PRAGMA table_info({table})")).fetchall()
        return {str(row["name"]) for row in rows}

    async def _migrate_state_schema(self) -> None:
        """Add branch/checkpoint columns and backfill a main branch for linear transcripts."""
        conn = self._conn()
        await conn.executescript(STATE_SCHEMA)
        user_columns = await self._table_columns("users")
        if "run_id" not in user_columns:
            await conn.execute("ALTER TABLE users ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")
        await conn.execute(
            "UPDATE users SET run_id=? WHERE trim(COALESCE(run_id,''))=''",
            (self.default_run_id,),
        )
        conversation_columns = await self._table_columns("conversations")
        if "run_id" not in conversation_columns:
            await conn.execute(
                "ALTER TABLE conversations ADD COLUMN run_id TEXT NOT NULL DEFAULT ''"
            )
        await conn.execute(
            """UPDATE conversations
               SET run_id=COALESCE((
                   SELECT u.run_id FROM users u WHERE u.id=conversations.user_id
               ), ?)
               WHERE trim(COALESCE(run_id,''))=''""",
            (self.default_run_id,),
        )
        columns = await self._table_columns("messages")
        additions = {
            "branch_id": "TEXT",
            "parent_message_id": "TEXT",
            "checkpoint_id": "TEXT",
            "mode": "TEXT NOT NULL DEFAULT 'auto'",
        }
        for name, ddl in additions.items():
            if name not in columns:
                await conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {ddl}")
        branch_columns = await self._table_columns("branches")
        if "mode" not in branch_columns:
            await conn.execute("ALTER TABLE branches ADD COLUMN mode TEXT NOT NULL DEFAULT 'auto'")
            await conn.execute(
                """UPDATE branches SET mode=COALESCE((
                       SELECT CASE WHEN m.mode='staged' THEN 'staged' ELSE 'auto' END
                       FROM messages m WHERE m.branch_id=branches.id
                       ORDER BY m.ordinal DESC LIMIT 1
                   ),'auto')"""
            )
        agenda_columns = await self._table_columns("checkpoint_subquestions")
        if "agenda_visible" not in agenda_columns:
            # Old rows mix Auto-internal decomposition with user-facing staged
            # agenda. Their origin cannot be reconstructed reliably, so keep
            # retrieval/evidence state and hide the old menu entries.
            await conn.execute(
                "ALTER TABLE checkpoint_subquestions ADD COLUMN agenda_visible INTEGER NOT NULL DEFAULT 0"
            )
        agenda_columns = await self._table_columns("checkpoint_subquestions")
        agenda_additions = {
            "status_origin": "TEXT NOT NULL DEFAULT 'legacy'",
            "status_reason": "TEXT NOT NULL DEFAULT ''",
            "status_source_refs_json": "TEXT NOT NULL DEFAULT '[]'",
            "status_message_id": "TEXT",
        }
        for name, ddl in agenda_additions.items():
            if name not in agenda_columns:
                await conn.execute(f"ALTER TABLE checkpoint_subquestions ADD COLUMN {name} {ddl}")
        await conn.execute(
            "UPDATE checkpoint_subquestions SET status='not_closed' WHERE status='open'"
        )
        subquestion_columns = await self._table_columns("subquestions")
        if "display_no" not in subquestion_columns:
            await conn.execute(
                "ALTER TABLE subquestions ADD COLUMN display_no INTEGER NOT NULL DEFAULT 0"
            )
        # Legacy rows did not have a public-safe alias. Populate one
        # deterministically per conversation and never renumber it again.
        legacy_rows = await (await conn.execute(
            "SELECT id,conversation_id,display_no FROM subquestions "
            "ORDER BY conversation_id,created_at,id"
        )).fetchall()
        next_display_no: dict[str, int] = {}
        for row in legacy_rows:
            conversation_id = str(row["conversation_id"])
            current = int(row["display_no"] or 0)
            expected = next_display_no.get(conversation_id, 1)
            if current < 1:
                await conn.execute(
                    "UPDATE subquestions SET display_no=? WHERE id=?",
                    (expected, str(row["id"])),
                )
                current = expected
            next_display_no[conversation_id] = max(expected, current + 1)
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_subquestions_conversation_display_no "
            "ON subquestions(conversation_id,display_no)"
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(2,?)",
            (now_ms(),),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(3,?)",
            (now_ms(),),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(4,?)",
            (now_ms(),),
        )
        await self._seed_system_templates()
        rows = await (await conn.execute("SELECT id FROM conversations ORDER BY created_at")).fetchall()
        for row in rows:
            await self._ensure_conversation_tree(str(row["id"]))
        # Older linear conversations persisted the accepted UNITs but had no
        # creation checkpoint. Recover the first assistant checkpoint that
        # contained each UNIT so the graph can mark historical additions.
        await conn.execute(
            """UPDATE evidence_units AS u
               SET created_checkpoint_id=(
                   SELECT cp.id
                   FROM checkpoint_units cu
                   JOIN checkpoints cp ON cp.id=cu.checkpoint_id
                   WHERE cu.unit_id=u.id AND cp.kind='assistant'
                   ORDER BY cp.created_at,cp.id
                   LIMIT 1
               )
               WHERE u.created_checkpoint_id IS NULL
                 AND EXISTS(
                   SELECT 1 FROM checkpoint_units cu
                   JOIN checkpoints cp ON cp.id=cu.checkpoint_id
                   WHERE cu.unit_id=u.id AND cp.kind='assistant'
                 )"""
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(5,?)",
            (now_ms(),),
        )
        await self._migrate_branch_unit_numbers()
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(7,?)",
            (now_ms(),),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(8,?)",
            (now_ms(),),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(9,?)",
            (now_ms(),),
        )
        await conn.commit()

    async def _migrate_branch_unit_numbers(self) -> None:
        """Move display UNIT numbers onto checkpoint_units; drop conversation-wide uniqueness."""
        conn = self._conn()
        unit_columns = await self._table_columns("checkpoint_units")
        if "unit_no" not in unit_columns:
            await conn.execute(
                "ALTER TABLE checkpoint_units ADD COLUMN unit_no INTEGER NOT NULL DEFAULT 0"
            )
        await conn.execute(
            """UPDATE checkpoint_units
               SET unit_no=COALESCE((
                   SELECT u.unit_no FROM evidence_units u WHERE u.id=checkpoint_units.unit_id
               ),0)
               WHERE unit_no=0"""
        )
        schema_row = await (await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='evidence_units'"
        )).fetchone()
        schema_sql = str(schema_row["sql"] or "") if schema_row is not None else ""
        compact = "".join(schema_sql.split())
        if "UNIQUE(conversation_id,unit_no)" in compact:
            await conn.execute("PRAGMA foreign_keys=OFF")
            await conn.execute(
                """CREATE TABLE evidence_units_v6 (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    created_checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE SET NULL,
                    unit_no INTEGER NOT NULL,
                    sq_id TEXT REFERENCES subquestions(id) ON DELETE SET NULL,
                    signature TEXT NOT NULL,
                    chain_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(conversation_id, signature)
                )"""
            )
            await conn.execute(
                """INSERT INTO evidence_units_v6
                   (id,conversation_id,created_checkpoint_id,unit_no,sq_id,signature,chain_json,created_at)
                   SELECT id,conversation_id,created_checkpoint_id,unit_no,sq_id,signature,chain_json,created_at
                   FROM evidence_units"""
            )
            await conn.execute("DROP TABLE evidence_units")
            await conn.execute("ALTER TABLE evidence_units_v6 RENAME TO evidence_units")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_units_conversation ON evidence_units(conversation_id, created_at)"
            )
            await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(6,?)",
            (now_ms(),),
        )

    async def _seed_system_templates(self) -> None:
        conn = self._conn()
        template_id = "system-experiment"
        ts = now_ms()
        await conn.execute(
            """INSERT OR IGNORE INTO card_templates
               (id,owner_user_id,name,description,created_at,updated_at)
               VALUES(?,NULL,?,?,?,?)""",
            (
                template_id,
                "Эксперимент",
                "Структурированная карточка плана и условий эксперимента",
                ts,
                ts,
            ),
        )
        await conn.execute(
            """INSERT OR IGNORE INTO card_template_versions
               (id,template_id,version,schema_json,ui_json,instructions,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                "system-experiment-v2",
                template_id,
                2,
                _json(EXPERIMENT_TEMPLATE_SCHEMA),
                _json({"order": list(EXPERIMENT_TEMPLATE_SCHEMA["properties"].keys())}),
                "Заполняй по текущему диалогу и evidence UNIT checkpoint. Неизвестное оставляй null.",
                ts,
            ),
        )

    @staticmethod
    def _empty_checkpoint_state() -> dict[str, Any]:
        return {
            "retrievalSnapshotId": None,
            "unitIds": [],
            "cardRevisionIds": [],
            "turnConfig": {},
        }

    @staticmethod
    def _chain_signature(chain: dict[str, Any]) -> str:
        raw = chain.get("spine_evidence_seq") or chain.get("edge_keys") or []
        if not raw:
            raw = [
                str(item.get("edge_key") or item.get("element_id") or "")
                for item in (chain.get("walk") or chain.get("edges") or [])
                if isinstance(item, dict)
            ]
        blob = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    async def _ensure_conversation_tree(self, conversation_id: str, mode: str = "auto") -> str:
        """Create main branch and checkpoint existing linear messages once."""
        conn = self._conn()
        existing = await (await conn.execute(
            "SELECT id FROM branches WHERE conversation_id=? ORDER BY created_at LIMIT 1",
            (conversation_id,),
        )).fetchone()
        if existing is not None:
            return str(existing["id"])

        branch_id = str(uuid.uuid4())
        ts = now_ms()
        branch_mode = "staged" if mode == "staged" else "auto"
        await conn.execute(
            "INSERT INTO branches(id,conversation_id,name,mode,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (branch_id, conversation_id, "main", branch_mode, ts, ts),
        )
        rows = await (await conn.execute(
            "SELECT id,role,created_at FROM messages WHERE conversation_id=? ORDER BY ordinal",
            (conversation_id,),
        )).fetchall()
        parent_checkpoint: str | None = None
        parent_message: str | None = None
        state = self._empty_checkpoint_state()
        next_unit = 1
        unit_numbers: dict[str, int] = {}
        for row in rows:
            message_id = str(row["id"])
            checkpoint_id = str(uuid.uuid4())
            graph_rows = await (await conn.execute(
                "SELECT chains_json FROM graph_runs WHERE conversation_id=? AND message_id=? ORDER BY created_at",
                (conversation_id, message_id),
            )).fetchall()
            if str(row["role"]) == "assistant":
                for graph_row in graph_rows:
                    for chain in _loads(graph_row["chains_json"], []):
                        if not isinstance(chain, dict):
                            continue
                        signature = self._chain_signature(chain)
                        prev = await (await conn.execute(
                            "SELECT id,unit_no FROM evidence_units WHERE conversation_id=? AND signature=?",
                            (conversation_id, signature),
                        )).fetchone()
                        if prev is None:
                            unit_id = str(uuid.uuid4())
                            await conn.execute(
                                """INSERT INTO evidence_units
                                   (id,conversation_id,created_checkpoint_id,unit_no,sq_id,signature,chain_json,created_at)
                                   VALUES(?,?,?,?,NULL,?,?,?)""",
                                (unit_id, conversation_id, None, next_unit, signature, _json(chain), int(row["created_at"])),
                            )
                            unit_numbers[unit_id] = next_unit
                            next_unit += 1
                        else:
                            unit_id = str(prev["id"])
                            unit_numbers.setdefault(unit_id, int(prev["unit_no"] or next_unit))
                        if unit_id not in state["unitIds"]:
                            state["unitIds"].append(unit_id)
            await conn.execute(
                """INSERT INTO checkpoints
                   (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    checkpoint_id,
                    conversation_id,
                    branch_id,
                    parent_checkpoint,
                    message_id,
                    str(row["role"]),
                    _json(state),
                    int(row["created_at"]),
                ),
            )
            for unit_id in state["unitIds"]:
                await conn.execute(
                    "INSERT OR IGNORE INTO checkpoint_units(checkpoint_id,unit_id,unit_no) VALUES(?,?,?)",
                    (checkpoint_id, unit_id, unit_numbers.get(unit_id, 0)),
                )
            await conn.execute(
                "UPDATE messages SET branch_id=?,parent_message_id=?,checkpoint_id=? WHERE id=?",
                (branch_id, parent_message, checkpoint_id, message_id),
            )
            parent_checkpoint = checkpoint_id
            parent_message = message_id
        await conn.execute(
            "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
            (parent_checkpoint, ts, branch_id),
        )
        return branch_id

    async def active_user_count(self) -> int:
        row = await (await self._conn().execute(
            "SELECT COUNT(*) AS n FROM users WHERE is_active=1"
        )).fetchone()
        return int(row["n"] if row else 0)

    async def user_count(self) -> int:
        row = await (await self._conn().execute("SELECT COUNT(*) AS n FROM users")).fetchone()
        return int(row["n"] if row else 0)

    async def banned_llm_models(self, key_fp: str) -> set[str]:
        fp = (key_fp or "").strip()
        if not fp:
            return set()
        rows = await (await self._conn().execute(
            "SELECT profile_id FROM llm_model_bans WHERE key_fp=?",
            (fp,),
        )).fetchall()
        return {str(row["profile_id"]) for row in rows}

    async def ban_llm_model(self, key_fp: str, profile_id: str, reason: str) -> None:
        fp = (key_fp or "").strip()
        name = (profile_id or "").strip()
        if not fp or not name:
            return
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO llm_model_bans(key_fp,profile_id,reason,banned_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(key_fp,profile_id) DO UPDATE SET reason=excluded.reason,banned_at=excluded.banned_at""",
                (fp, name, (reason or "quota_exhausted").strip() or "quota_exhausted", ts),
            )
            await self._conn().commit()

    async def mark_llm_key_dead(self, key_fp: str) -> None:
        fp = (key_fp or "").strip()
        if not fp:
            return
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO llm_key_state(key_fp,status,updated_at) VALUES(?,?,?)
                   ON CONFLICT(key_fp) DO UPDATE SET status='dead',updated_at=excluded.updated_at""",
                (fp, "dead", ts),
            )
            await self._conn().commit()

    async def llm_key_is_dead(self, key_fp: str) -> bool:
        fp = (key_fp or "").strip()
        if not fp:
            return False
        row = await (await self._conn().execute(
            "SELECT status FROM llm_key_state WHERE key_fp=?",
            (fp,),
        )).fetchone()
        return bool(row) and str(row["status"]) == "dead"

    async def create_user(
        self, username: str, password: str, *, run_id: str | None = None
    ) -> AccountUser:
        username = normalize_username(username)
        password_hash = hash_password(password)
        corpus_run_id = normalize_run_id(
            self.default_run_id if run_id is None else run_id
        )
        user = AccountUser(str(uuid.uuid4()), username, True, corpus_run_id)
        ts = now_ms()
        async with self._write_lock:
            try:
                await self._conn().execute(
                    """INSERT INTO users
                       (id,username,password_hash,is_active,run_id,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (user.id, user.username, password_hash, 1, user.run_id, ts, ts),
                )
                await self._conn().commit()
            except aiosqlite.IntegrityError as exc:
                await self._conn().rollback()
                raise ValueError("Пользователь с таким логином уже существует") from exc
        return user

    async def list_users(self) -> list[dict[str, Any]]:
        rows = await (await self._conn().execute(
            "SELECT id,username,is_active,run_id,created_at,updated_at FROM users ORDER BY username"
        )).fetchall()
        return [dict(row) for row in rows]

    async def set_user_run_id(self, username: str, run_id: str) -> dict[str, Any] | None:
        username = normalize_username(username)
        next_run_id = normalize_run_id(run_id)
        ts = now_ms()
        async with self._write_lock:
            row = await (await self._conn().execute(
                "SELECT id,run_id FROM users WHERE username=? COLLATE NOCASE",
                (username,),
            )).fetchone()
            if row is None:
                return None
            user_id = str(row["id"])
            previous = str(row["run_id"] or "")
            await self._conn().execute(
                "UPDATE users SET run_id=?,updated_at=? WHERE id=?",
                (next_run_id, ts, user_id),
            )
            stale_row = await (await self._conn().execute(
                "SELECT COUNT(*) AS n FROM conversations WHERE user_id=? AND run_id<>?",
                (user_id, next_run_id),
            )).fetchone()
            await self._conn().commit()
        return {
            "username": username,
            "previousRunId": previous,
            "runId": next_run_id,
            "changed": previous != next_run_id,
            "readOnlyConversations": int(stale_row["n"] if stale_row else 0),
        }

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
            "SELECT id,username,password_hash,is_active,run_id FROM users WHERE username=? COLLATE NOCASE",
            (username,),
        )).fetchone()
        candidate_hash = str(row["password_hash"]) if row is not None else _DUMMY_PASSWORD_HASH
        valid = await asyncio.to_thread(verify_password, candidate_hash, password)
        if not valid or row is None:
            return None
        if not bool(row["is_active"]):
            return None
        return AccountUser(
            str(row["id"]), str(row["username"]), True, str(row["run_id"] or "")
        )

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
            """SELECT u.id,u.username,u.is_active,u.run_id,s.id AS session_id,s.last_seen_at
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
        return AccountUser(
            str(row["id"]), str(row["username"]), True, str(row["run_id"] or "")
        )

    async def revoke_session(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        async with self._write_lock:
            await self._conn().execute(
                "UPDATE auth_sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (now_ms(), token_digest(raw_token)),
            )
            await self._conn().commit()

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None = None,
        *,
        mode: str = "auto",
    ) -> dict[str, Any]:
        cid = conversation_id or str(uuid.uuid4())
        branch_mode = "staged" if mode == "staged" else "auto"
        ts = now_ms()
        async with self._write_lock:
            user_row = await (await self._conn().execute(
                "SELECT run_id FROM users WHERE id=? AND is_active=1",
                (user_id,),
            )).fetchone()
            if user_row is None:
                raise KeyError(user_id)
            run_id = normalize_run_id(str(user_row["run_id"] or ""))
            await self._conn().execute(
                """INSERT INTO conversations
                   (id,user_id,title,run_id,created_at,updated_at)
                   VALUES(?,?,?,?,?,?)""",
                (cid, user_id, "Новый чат", run_id, ts, ts),
            )
            branch_id = await self._ensure_conversation_tree(cid, branch_mode)
            await self._conn().commit()
        return {
            "id": cid,
            "title": "Новый чат",
            "createdAt": ts,
            "updatedAt": ts,
            "activeBranchId": branch_id,
            "mode": branch_mode,
            "runId": run_id,
            "accountRunId": run_id,
            "readOnly": False,
            "readOnlyReason": None,
        }

    async def conversation_owned(self, user_id: str, conversation_id: str) -> bool:
        row = await (await self._conn().execute(
            "SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)
        )).fetchone()
        return row is not None

    async def conversation_run_access(
        self, user_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT c.run_id AS conversation_run_id,u.run_id AS account_run_id
               FROM conversations c JOIN users u ON u.id=c.user_id
               WHERE c.id=? AND c.user_id=?""",
            (conversation_id, user_id),
        )).fetchone()
        if row is None:
            return None
        conversation_run_id = str(row["conversation_run_id"] or "")
        account_run_id = str(row["account_run_id"] or "")
        return {
            "conversationRunId": conversation_run_id,
            "accountRunId": account_run_id,
            "readOnly": conversation_run_id != account_run_id,
        }

    async def require_conversation_writable(
        self, user_id: str, conversation_id: str
    ) -> str:
        access = await self.conversation_run_access(user_id, conversation_id)
        if access is None:
            raise KeyError(conversation_id)
        if access["readOnly"]:
            raise ConversationRunMismatchError(
                str(access["conversationRunId"]), str(access["accountRunId"])
            )
        return str(access["conversationRunId"])

    async def require_branch_writable(self, user_id: str, branch_id: str) -> str:
        row = await (await self._conn().execute(
            """SELECT b.conversation_id FROM branches b
               JOIN conversations c ON c.id=b.conversation_id
               WHERE b.id=? AND c.user_id=?""",
            (branch_id, user_id),
        )).fetchone()
        if row is None:
            raise KeyError(branch_id)
        return await self.require_conversation_writable(
            user_id, str(row["conversation_id"])
        )

    async def require_checkpoint_writable(
        self, user_id: str, checkpoint_id: str
    ) -> str:
        row = await (await self._conn().execute(
            """SELECT cp.conversation_id FROM checkpoints cp
               JOIN conversations c ON c.id=cp.conversation_id
               WHERE cp.id=? AND c.user_id=?""",
            (checkpoint_id, user_id),
        )).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        return await self.require_conversation_writable(
            user_id, str(row["conversation_id"])
        )

    async def main_branch_id(self, conversation_id: str) -> str:
        return await self._ensure_conversation_tree(conversation_id)

    async def branch_owned(self, user_id: str, branch_id: str) -> bool:
        row = await (await self._conn().execute(
            """SELECT 1 FROM branches b JOIN conversations c ON c.id=b.conversation_id
               WHERE b.id=? AND c.user_id=?""",
            (branch_id, user_id),
        )).fetchone()
        return row is not None

    async def list_branches(self, user_id: str, conversation_id: str) -> list[dict[str, Any]]:
        if not await self.conversation_owned(user_id, conversation_id):
            raise KeyError(conversation_id)
        rows = await (await self._conn().execute(
            """SELECT id,name,mode,created_from_checkpoint_id,head_checkpoint_id,created_at,updated_at
               FROM branches WHERE conversation_id=? ORDER BY created_at,id""",
            (conversation_id,),
        )).fetchall()
        return [
            {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "mode": str(row["mode"]),
                "createdFromCheckpointId": row["created_from_checkpoint_id"],
                "headCheckpointId": row["head_checkpoint_id"],
                "createdAt": int(row["created_at"]),
                "updatedAt": int(row["updated_at"]),
            }
            for row in rows
        ]

    async def branch_detail(self, user_id: str, branch_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT b.id,b.conversation_id,b.name,b.mode,b.created_from_checkpoint_id,
                      b.head_checkpoint_id,b.created_at,b.updated_at
               FROM branches b JOIN conversations c ON c.id=b.conversation_id
               WHERE b.id=? AND c.user_id=?""",
            (branch_id, user_id),
        )).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "conversationId": str(row["conversation_id"]),
            "name": str(row["name"]),
            "mode": str(row["mode"]),
            "createdFromCheckpointId": row["created_from_checkpoint_id"],
            "headCheckpointId": row["head_checkpoint_id"],
            "createdAt": int(row["created_at"]),
            "updatedAt": int(row["updated_at"]),
        }

    async def rename_branch(self, user_id: str, branch_id: str, name: str) -> dict[str, Any]:
        clean = " ".join((name or "").strip().split())[:64]
        if not clean:
            raise ValueError("Название версии не может быть пустым")
        if not await self.branch_owned(user_id, branch_id):
            raise KeyError(branch_id)
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute(
                "UPDATE branches SET name=?,updated_at=? WHERE id=?",
                (clean, ts, branch_id),
            )
            await self._conn().commit()
        detail = await self.branch_detail(user_id, branch_id)
        if detail is None:
            raise KeyError(branch_id)
        return detail

    async def checkpoint_owned(self, user_id: str, checkpoint_id: str) -> bool:
        row = await (await self._conn().execute(
            """SELECT 1 FROM checkpoints cp
               JOIN conversations c ON c.id=cp.conversation_id
               WHERE cp.id=? AND c.user_id=?""",
            (checkpoint_id, user_id),
        )).fetchone()
        return row is not None

    async def checkpoint_in_lineage(self, head_checkpoint_id: str | None, checkpoint_id: str) -> bool:
        if not head_checkpoint_id or not checkpoint_id:
            return False
        row = await (await self._conn().execute(
            """WITH RECURSIVE lineage(id,parent_id) AS (
                   SELECT id,parent_id FROM checkpoints WHERE id=?
                   UNION ALL
                   SELECT cp.id,cp.parent_id FROM checkpoints cp
                   JOIN lineage ON cp.id=lineage.parent_id
               )
               SELECT 1 FROM lineage WHERE id=? LIMIT 1""",
            (head_checkpoint_id, checkpoint_id),
        )).fetchone()
        return row is not None

    async def checkpoint_state(self, user_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT cp.id,cp.conversation_id,cp.branch_id,cp.parent_id,cp.message_id,
                      cp.kind,cp.state_json,cp.created_at,b.mode AS branch_mode,
                      c.run_id AS conversation_run_id
               FROM checkpoints cp JOIN conversations c ON c.id=cp.conversation_id
               JOIN branches b ON b.id=cp.branch_id
               WHERE cp.id=? AND c.user_id=?""",
            (checkpoint_id, user_id),
        )).fetchone()
        if row is None:
            return None
        agenda = await self._agenda_for_checkpoint(checkpoint_id)
        return {
            "id": str(row["id"]),
            "conversationId": str(row["conversation_id"]),
            "branchId": str(row["branch_id"]),
            "parentId": row["parent_id"],
            "messageId": row["message_id"],
            "kind": str(row["kind"]),
            "mode": str(row["branch_mode"] or "auto"),
            "runId": str(row["conversation_run_id"] or ""),
            "state": _loads(row["state_json"], self._empty_checkpoint_state()),
            "agenda": agenda,
            "createdAt": int(row["created_at"]),
        }

    async def create_fork(
        self,
        user_id: str,
        conversation_id: str,
        checkpoint_id: str,
        name: str = "",
        mode: str | None = None,
        source_branch_id: str | None = None,
    ) -> dict[str, Any]:
        await self.require_conversation_writable(user_id, conversation_id)
        cp = await (await self._conn().execute(
            """SELECT cp.id,b.mode FROM checkpoints cp
               JOIN branches b ON b.id=cp.branch_id
               WHERE cp.id=? AND cp.conversation_id=?""",
            (checkpoint_id, conversation_id),
        )).fetchone()
        if cp is None:
            raise KeyError(checkpoint_id)
        source_mode = str(cp["mode"] or "auto")
        if source_branch_id:
            source_branch = await (await self._conn().execute(
                """SELECT mode,head_checkpoint_id FROM branches
                   WHERE id=? AND conversation_id=?""",
                (source_branch_id, conversation_id),
            )).fetchone()
            if source_branch is None:
                raise KeyError(source_branch_id)
            reachable = await (await self._conn().execute(
                """WITH RECURSIVE lineage(id,parent_id) AS (
                       SELECT id,parent_id FROM checkpoints WHERE id=?
                       UNION ALL
                       SELECT cp.id,cp.parent_id FROM checkpoints cp
                       JOIN lineage ON cp.id=lineage.parent_id
                   )
                   SELECT 1 FROM lineage WHERE id=? LIMIT 1""",
                (source_branch["head_checkpoint_id"], checkpoint_id),
            )).fetchone()
            if reachable is None:
                raise KeyError(checkpoint_id)
            source_mode = str(source_branch["mode"] or "auto")
        target_mode = source_mode if mode is None else ("staged" if mode == "staged" else "auto")
        if source_mode == "auto" and target_mode == "staged":
            raise ValueError("Auto branch cannot be converted to staged")
        branch_id = str(uuid.uuid4())
        ts = now_ms()
        clean = " ".join((name or "").strip().split())[:64]
        if not clean:
            count = await (await self._conn().execute(
                "SELECT COUNT(*) AS n FROM branches WHERE conversation_id=?",
                (conversation_id,),
            )).fetchone()
            clean = f"Ветка {int(count['n']) + 1 if count else 2}"
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO branches
                   (id,conversation_id,name,mode,created_from_checkpoint_id,head_checkpoint_id,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (branch_id, conversation_id, clean, target_mode, checkpoint_id, checkpoint_id, ts, ts),
            )
            await self._conn().commit()
        return {
            "id": branch_id,
            "conversationId": conversation_id,
            "name": clean,
            "mode": target_mode,
            "createdFromCheckpointId": checkpoint_id,
            "headCheckpointId": checkpoint_id,
            "createdAt": ts,
            "updatedAt": ts,
        }

    async def _checkpoint_row(self, checkpoint_id: str | None) -> aiosqlite.Row | None:
        if not checkpoint_id:
            return None
        return await (await self._conn().execute(
            "SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)
        )).fetchone()

    async def _copy_checkpoint_links(self, source_id: str | None, target_id: str) -> None:
        if not source_id:
            return
        await self._conn().execute(
            """INSERT OR IGNORE INTO checkpoint_units(checkpoint_id,unit_id,unit_no)
               SELECT ?,unit_id,unit_no FROM checkpoint_units WHERE checkpoint_id=?""",
            (target_id, source_id),
        )
        await self._conn().execute(
            """INSERT OR IGNORE INTO checkpoint_card_attachments(checkpoint_id,card_revision_id)
               SELECT ?,card_revision_id FROM checkpoint_card_attachments WHERE checkpoint_id=?""",
            (target_id, source_id),
        )
        await self._conn().execute(
            """INSERT OR IGNORE INTO checkpoint_subquestions
               (checkpoint_id,sq_id,status,status_origin,status_reason,status_source_refs_json,
                status_message_id,position,question_count,unit_count,graph_snapshot_id,agenda_visible)
               SELECT ?,sq_id,status,status_origin,status_reason,status_source_refs_json,
                      status_message_id,position,question_count,unit_count,graph_snapshot_id,agenda_visible
               FROM checkpoint_subquestions WHERE checkpoint_id=?""",
            (target_id, source_id),
        )

    async def _next_branch_unit_no(self, checkpoint_id: str) -> int:
        row = await (await self._conn().execute(
            "SELECT COALESCE(MAX(unit_no),0)+1 AS n FROM checkpoint_units WHERE checkpoint_id=?",
            (checkpoint_id,),
        )).fetchone()
        return int(row["n"] if row is not None else 1)

    async def _attach_checkpoint_unit(self, checkpoint_id: str, unit_id: str) -> int:
        """Keep an existing branch number or allocate the next one on this checkpoint."""
        row = await (await self._conn().execute(
            "SELECT unit_no FROM checkpoint_units WHERE checkpoint_id=? AND unit_id=?",
            (checkpoint_id, unit_id),
        )).fetchone()
        if row is not None:
            return int(row["unit_no"])
        unit_no = await self._next_branch_unit_no(checkpoint_id)
        await self._conn().execute(
            "INSERT INTO checkpoint_units(checkpoint_id,unit_id,unit_no) VALUES(?,?,?)",
            (checkpoint_id, unit_id, unit_no),
        )
        return unit_no

    async def _merge_source_files(
        self, conversation_id: str, source_files: Iterable[str]
    ) -> list[tuple[int, str]]:
        """Insert-only source:N map. Caller must already be in a write transaction."""
        rows = await (await self._conn().execute(
            """SELECT source_id,source_file FROM conversation_sources
               WHERE conversation_id=? ORDER BY source_id""",
            (conversation_id,),
        )).fetchall()
        known = {str(row["source_file"]): int(row["source_id"]) for row in rows}
        known_basenames = {
            str(row["source_file"]).replace("\\", "/").rsplit("/", 1)[-1]
            for row in rows
        }
        next_id = max(known.values(), default=0) + 1
        for source_file in source_files:
            path = str(source_file or "").strip()
            if not path:
                continue
            basename = path.replace("\\", "/").rsplit("/", 1)[-1]
            if path in known or basename in known_basenames:
                continue
            await self._conn().execute(
                """INSERT INTO conversation_sources
                   (conversation_id,source_id,source_file) VALUES(?,?,?)""",
                (conversation_id, next_id, path),
            )
            known[path] = next_id
            known_basenames.add(basename)
            next_id += 1
        refreshed = await (await self._conn().execute(
            """SELECT source_id,source_file FROM conversation_sources
               WHERE conversation_id=? ORDER BY source_id""",
            (conversation_id,),
        )).fetchall()
        return [(int(row["source_id"]), str(row["source_file"])) for row in refreshed]

    @staticmethod
    def subquestion_ref(display_no: int) -> str:
        return f"subquestion:{int(display_no)}"

    @staticmethod
    def parse_subquestion_ref(value: str) -> int | None:
        match = _SUBQUESTION_REF_RE.fullmatch(str(value or "").strip())
        return int(match.group(1)) if match else None

    async def _next_subquestion_display_no(self, conversation_id: str) -> int:
        row = await (await self._conn().execute(
            "SELECT COALESCE(MAX(display_no),0)+1 AS n FROM subquestions WHERE conversation_id=?",
            (conversation_id,),
        )).fetchone()
        return int(row["n"] if row is not None else 1)

    async def _subquestion_ref_map(self, conversation_id: str) -> dict[str, str]:
        rows = await (await self._conn().execute(
            "SELECT id,display_no FROM subquestions WHERE conversation_id=?",
            (conversation_id,),
        )).fetchall()
        return {
            str(row["id"]): self.subquestion_ref(int(row["display_no"]))
            for row in rows
        }

    @staticmethod
    def _present_subquestion_value(value: Any, id_to_ref: dict[str, str]) -> Any:
        """Remove SQ UUIDs from model/UI presentation without touching audit data."""
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                present_key = {
                    "open_sq_ids": "open_sq_refs",
                    "sq_id": "sq_ref",
                    "ordered_ids": "ordered_refs",
                }.get(str(key), str(key))
                result[present_key] = AppStore._present_subquestion_value(item, id_to_ref)
            return result
        if isinstance(value, list):
            return [AppStore._present_subquestion_value(item, id_to_ref) for item in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return json.dumps(
                    AppStore._present_subquestion_value(parsed, id_to_ref),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            presented = value
            for sq_id, ref in id_to_ref.items():
                presented = presented.replace(sq_id, ref)
            return presented
        return value

    async def _agenda_for_checkpoint(
        self,
        checkpoint_id: str,
        *,
        include_internal: bool = False,
        include_ids: bool = False,
    ) -> list[dict[str, Any]]:
        visibility = "" if include_internal else "AND cs.agenda_visible=1"
        rows = await (await self._conn().execute(
            f"""SELECT s.id,s.display_no,s.text,cs.status,cs.status_origin,cs.status_reason,
                      cs.status_source_refs_json,cs.status_message_id,
                      cs.position,cs.question_count,cs.unit_count,
                      cs.graph_snapshot_id
               FROM checkpoint_subquestions cs JOIN subquestions s ON s.id=cs.sq_id
               WHERE cs.checkpoint_id=? {visibility}
               ORDER BY cs.position,s.created_at""",
            (checkpoint_id,),
        )).fetchall()
        agenda: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {
                "ref": self.subquestion_ref(int(row["display_no"])),
                "text": str(row["text"]),
                "status": str(row["status"]),
                "statusOrigin": str(row["status_origin"]),
                "statusReason": str(row["status_reason"] or ""),
                "statusSourceRefs": _loads(row["status_source_refs_json"], []),
                "statusMessageId": row["status_message_id"],
                "position": int(row["position"]),
                "questionCount": int(row["question_count"]),
                "unitCount": int(row["unit_count"]),
                "graphSnapshotId": row["graph_snapshot_id"],
                "reviewRecommended": int(row["unit_count"]) >= 2,
            }
            if include_ids:
                item["id"] = str(row["id"])
            agenda.append(item)
        return agenda

    async def list_conversations(self, user_id: str, limit: int = 50, before: str | None = None) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        args: list[Any] = [user_id]
        where = "c.user_id=?"
        if before:
            try:
                cursor_time, cursor_id = before.split(":", 1)
                cursor_ts = int(cursor_time)
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid conversation cursor") from exc
            where += " AND (c.updated_at<? OR (c.updated_at=? AND c.id<?))"
            args.extend([cursor_ts, cursor_ts, cursor_id])
        args.append(limit + 1)
        rows = await (await self._conn().execute(
            f"""SELECT c.id,c.title,c.run_id,c.created_at,c.updated_at,
                       u.run_id AS account_run_id
                FROM conversations c JOIN users u ON u.id=c.user_id
                WHERE {where}
                ORDER BY c.updated_at DESC,c.id DESC LIMIT ?""",
            args,
        )).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [
            {
                "id": str(r["id"]),
                "title": str(r["title"]),
                "createdAt": int(r["created_at"]),
                "updatedAt": int(r["updated_at"]),
                "runId": str(r["run_id"] or ""),
                "accountRunId": str(r["account_run_id"] or ""),
                "readOnly": str(r["run_id"] or "") != str(r["account_run_id"] or ""),
                "readOnlyReason": (
                    "run_id_changed"
                    if str(r["run_id"] or "") != str(r["account_run_id"] or "")
                    else None
                ),
            }
            for r in rows
        ]
        for item in items:
            branch = await (await self._conn().execute(
                """SELECT id,mode,head_checkpoint_id FROM branches WHERE conversation_id=?
                   ORDER BY updated_at DESC,created_at LIMIT 1""",
                (item["id"],),
            )).fetchone()
            if branch is not None:
                item["activeBranchId"] = str(branch["id"])
                item["mode"] = str(branch["mode"] or "auto")
                item["headCheckpointId"] = branch["head_checkpoint_id"]
        return {
            "items": items,
            "nextCursor": (
                f'{items[-1]["updatedAt"]}:{items[-1]["id"]}'
                if has_more and items else None
            ),
        }

    async def get_conversation(
        self,
        user_id: str,
        conversation_id: str,
        branch_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        conv = await (await self._conn().execute(
            """SELECT c.id,c.title,c.run_id,c.created_at,c.updated_at,
                      u.run_id AS account_run_id
               FROM conversations c JOIN users u ON u.id=c.user_id
               WHERE c.id=? AND c.user_id=?""",
            (conversation_id, user_id),
        )).fetchone()
        if conv is None:
            return None
        id_to_ref = await self._subquestion_ref_map(str(conv["id"]))
        branches = await self.list_branches(user_id, conversation_id)
        if not branches:
            branch_id = await self._ensure_conversation_tree(conversation_id)
            branches = await self.list_branches(user_id, conversation_id)
        selected = next((item for item in branches if item["id"] == branch_id), None)
        if selected is None:
            selected = max(branches, key=lambda item: item["updatedAt"])
        branch_id = str(selected["id"])
        head = selected.get("headCheckpointId")
        view_checkpoint_id = str(checkpoint_id or head or "")
        if checkpoint_id and not await self.checkpoint_in_lineage(
            str(head or ""), str(checkpoint_id)
        ):
            raise ValueError("Checkpoint does not belong to the selected version")
        if view_checkpoint_id:
            rows = await (await self._conn().execute(
                """WITH RECURSIVE lineage(id,parent_id,message_id,branch_id,depth) AS (
                       SELECT id,parent_id,message_id,branch_id,0 FROM checkpoints WHERE id=?
                       UNION ALL
                       SELECT cp.id,cp.parent_id,cp.message_id,cp.branch_id,lineage.depth+1
                       FROM checkpoints cp JOIN lineage ON cp.id=lineage.parent_id
                   )
                   SELECT m.id,m.role,m.text,m.status,m.payload_json,m.created_at,m.updated_at,
                          lineage.id AS checkpoint_id,lineage.branch_id,lineage.depth
                   FROM lineage JOIN messages m ON m.id=lineage.message_id
                   ORDER BY lineage.depth DESC""",
                (view_checkpoint_id,),
            )).fetchall()
        else:
            rows = []
        source_rows = await (await self._conn().execute(
            """SELECT source_id,source_file FROM conversation_sources
               WHERE conversation_id=? ORDER BY source_id""",
            (str(conv["id"]),),
        )).fetchall()
        source_snapshot = [
            (int(row["source_id"]), str(row["source_file"])) for row in source_rows
        ]
        messages: list[dict[str, Any]] = []
        for row in rows:
            payload = self._present_subquestion_value(
                _loads(row["payload_json"], {}), id_to_ref
            )
            payload = present_source_aliases_in_value(payload, source_snapshot)
            payload.update({
                "id": str(row["id"]),
                "role": str(row["role"]),
                "text": str(self._present_subquestion_value(str(row["text"]), id_to_ref)),
                "status": str(row["status"]),
                "checkpointId": str(row["checkpoint_id"]),
                "branchId": str(row["branch_id"]),
            })
            messages.append(payload)
        agenda = await self._agenda_for_checkpoint(view_checkpoint_id) if view_checkpoint_id else []
        at_branch_head = bool(view_checkpoint_id and view_checkpoint_id == str(head or ""))
        pending = await self.pending_for_branch(user_id, branch_id) if at_branch_head else None
        if pending and not any(item["id"] == pending["assistantMessageId"] for item in messages):
            waiting_row = await (await self._conn().execute(
                """SELECT id,role,text,status,payload_json FROM messages
                   WHERE id=? AND conversation_id=? AND branch_id=?""",
                (pending["assistantMessageId"], str(conv["id"]), branch_id),
            )).fetchone()
            if waiting_row is not None:
                waiting_payload = self._present_subquestion_value(
                    _loads(waiting_row["payload_json"], {}), id_to_ref
                )
                waiting_payload = present_source_aliases_in_value(
                    waiting_payload, source_snapshot
                )
                waiting_payload.update({
                    "id": str(waiting_row["id"]),
                    "role": str(waiting_row["role"]),
                    "text": str(self._present_subquestion_value(str(waiting_row["text"]), id_to_ref)),
                    "status": str(waiting_row["status"]),
                    "branchId": branch_id,
                })
                messages.append(waiting_payload)
        return {
            "id": str(conv["id"]), "title": str(conv["title"]),
            "createdAt": int(conv["created_at"]), "updatedAt": int(conv["updated_at"]),
            "runId": str(conv["run_id"] or ""),
            "accountRunId": str(conv["account_run_id"] or ""),
            "readOnly": str(conv["run_id"] or "") != str(conv["account_run_id"] or ""),
            "readOnlyReason": (
                "run_id_changed"
                if str(conv["run_id"] or "") != str(conv["account_run_id"] or "")
                else None
            ),
            "messages": messages,
            "branches": branches,
            "activeBranchId": branch_id,
            "headCheckpointId": head,
            "branchHeadCheckpointId": head,
            "viewCheckpointId": view_checkpoint_id or None,
            "atBranchHead": at_branch_head,
            "agenda": agenda,
            "pendingApproval": pending,
            "turnFailures": await self.list_turn_failures(user_id, str(conv["id"])),
        }

    async def research_map(
        self,
        user_id: str,
        conversation_id: str,
        branch_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a presentation-only tree of question/answer turns.

        Checkpoints remain the source of truth, but UI nodes are paired turns.
        Agenda/card attachment checkpoints are collapsed into the surrounding
        lineage and never become visual nodes of their own.
        """
        if not await self.conversation_owned(user_id, conversation_id):
            return None
        branches = await self.list_branches(user_id, conversation_id)
        if not branches:
            await self._ensure_conversation_tree(conversation_id)
            branches = await self.list_branches(user_id, conversation_id)
        selected = next((item for item in branches if item["id"] == branch_id), None)
        if selected is None and branches:
            selected = max(branches, key=lambda item: item["updatedAt"])

        message_rows = await (await self._conn().execute(
            """SELECT id,turn_id,ordinal,role,text,status,branch_id,checkpoint_id,
                      created_at,updated_at
               FROM messages WHERE conversation_id=? ORDER BY ordinal,id""",
            (conversation_id,),
        )).fetchall()
        checkpoint_rows = await (await self._conn().execute(
            """SELECT id,parent_id,message_id,branch_id,kind,created_at
               FROM checkpoints WHERE conversation_id=? ORDER BY created_at,id""",
            (conversation_id,),
        )).fetchall()
        unit_rows = await (await self._conn().execute(
            """SELECT u.created_checkpoint_id,cu.unit_no
               FROM checkpoint_units cu
               JOIN evidence_units u ON u.id=cu.unit_id
               JOIN checkpoints cp ON cp.id=cu.checkpoint_id
               WHERE cp.conversation_id=? AND cp.kind='assistant'
                 AND u.created_checkpoint_id=cu.checkpoint_id
               ORDER BY cu.unit_no""",
            (conversation_id,),
        )).fetchall()
        checkpoint_unit_rows = await (await self._conn().execute(
            """SELECT cu.checkpoint_id,COUNT(*) AS n
               FROM checkpoint_units cu JOIN checkpoints cp ON cp.id=cu.checkpoint_id
               WHERE cp.conversation_id=? GROUP BY cu.checkpoint_id""",
            (conversation_id,),
        )).fetchall()

        messages_by_id = {str(row["id"]): row for row in message_rows}
        checkpoints = {str(row["id"]): row for row in checkpoint_rows}
        turn_rows: dict[str, dict[str, aiosqlite.Row]] = {}
        turn_ordinals: dict[str, int] = {}
        for row in message_rows:
            turn_id = str(row["turn_id"] or "")
            if not turn_id or turn_id.startswith("cardref:"):
                continue
            turn_rows.setdefault(turn_id, {})[str(row["role"])] = row
            turn_ordinals[turn_id] = min(
                turn_ordinals.get(turn_id, int(row["ordinal"])), int(row["ordinal"])
            )

        ordered_turn_ids = sorted(turn_rows, key=lambda item: (turn_ordinals[item], item))
        display_numbers = {turn_id: index for index, turn_id in enumerate(ordered_turn_ids, 1)}
        unit_by_checkpoint: dict[str, list[int]] = {}
        for row in unit_rows:
            origin = str(row["created_checkpoint_id"] or "")
            if origin:
                unit_by_checkpoint.setdefault(origin, []).append(int(row["unit_no"]))
        checkpoint_unit_counts = {
            str(row["checkpoint_id"]): int(row["n"]) for row in checkpoint_unit_rows
        }

        def preview(value: Any, limit: int = 180) -> str:
            clean = " ".join(str(value or "").split())
            return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"

        def step_for_checkpoint(checkpoint_id_value: str | None) -> str | None:
            current = str(checkpoint_id_value or "")
            visited: set[str] = set()
            while current and current not in visited:
                visited.add(current)
                checkpoint = checkpoints.get(current)
                if checkpoint is None:
                    return None
                message = messages_by_id.get(str(checkpoint["message_id"] or ""))
                if message is not None:
                    turn_id = str(message["turn_id"] or "")
                    if turn_id in turn_rows:
                        return turn_id
                current = str(checkpoint["parent_id"] or "")
            return None

        steps: list[dict[str, Any]] = []
        for turn_id in ordered_turn_ids:
            pair = turn_rows[turn_id]
            user_row = pair.get("user")
            if user_row is None:
                continue
            assistant_row = pair.get("assistant")
            user_checkpoint_id = str(user_row["checkpoint_id"] or "")
            answer_checkpoint_id = (
                str(assistant_row["checkpoint_id"] or "") if assistant_row is not None else ""
            )
            user_checkpoint = checkpoints.get(user_checkpoint_id)
            parent_step_id = step_for_checkpoint(
                str(user_checkpoint["parent_id"] or "") if user_checkpoint is not None else ""
            )
            graph_checkpoint_id = answer_checkpoint_id or user_checkpoint_id
            created_units = unit_by_checkpoint.get(answer_checkpoint_id, [])
            steps.append({
                "id": turn_id,
                "displayNo": display_numbers[turn_id],
                "parentStepId": parent_step_id,
                "branchId": str(user_row["branch_id"] or ""),
                "question": {
                    "messageId": str(user_row["id"]),
                    "preview": preview(user_row["text"]),
                },
                "answer": ({
                    "messageId": str(assistant_row["id"]),
                    "preview": preview(assistant_row["text"]),
                    "status": str(assistant_row["status"]),
                } if assistant_row is not None else None),
                "userCheckpointId": user_checkpoint_id or None,
                "answerCheckpointId": answer_checkpoint_id or None,
                "graphCheckpointId": graph_checkpoint_id or None,
                "resumeCheckpointId": answer_checkpoint_id or user_checkpoint_id or None,
                "unitNos": created_units,
                "graphUnitCount": checkpoint_unit_counts.get(graph_checkpoint_id, 0),
                "createdAt": int(user_row["created_at"]),
            })

        branch_payload: list[dict[str, Any]] = []
        for branch in branches:
            head_checkpoint_id = str(branch.get("headCheckpointId") or "")
            branch_payload.append({
                **branch,
                "originStepId": step_for_checkpoint(
                    str(branch.get("createdFromCheckpointId") or "")
                ),
                "headStepId": step_for_checkpoint(head_checkpoint_id),
                "unitCount": checkpoint_unit_counts.get(head_checkpoint_id, 0),
            })
        return {
            "conversationId": conversation_id,
            "activeBranchId": str(selected["id"]) if selected else "",
            "branches": branch_payload,
            "steps": steps,
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

    async def begin_turn(self, user_id: str, conversation_id: str, turn_id: str, text: str) -> tuple[str, str]:
        branch_id = await self.main_branch_id(conversation_id)
        result = await self.begin_branch_turn(
            user_id,
            conversation_id,
            branch_id,
            turn_id,
            text,
        )
        return str(result["userMessageId"]), str(result["assistantMessageId"])

    async def begin_branch_turn(
        self,
        user_id: str,
        conversation_id: str,
        branch_id: str,
        turn_id: str,
        text: str,
        *,
        base_checkpoint_id: str | None = None,
        fork_if_needed: bool = False,
        mode: str = "auto",
        turn_config: dict[str, Any] | None = None,
        user_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self.require_conversation_writable(user_id, conversation_id)
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                branch = await (await self._conn().execute(
                    "SELECT head_checkpoint_id,mode FROM branches WHERE id=? AND conversation_id=?",
                    (branch_id, conversation_id),
                )).fetchone()
                if branch is None:
                    raise KeyError(branch_id)
                branch_mode = str(branch["mode"] or "auto")
                head = str(branch["head_checkpoint_id"]) if branch["head_checkpoint_id"] else None
                created_branch: dict[str, Any] | None = None
                if base_checkpoint_id is not None and base_checkpoint_id != head:
                    if not fork_if_needed:
                        raise RuntimeError("stale_checkpoint")
                    checkpoint = await (await self._conn().execute(
                        """SELECT id,kind FROM checkpoints
                           WHERE id=? AND conversation_id=?""",
                        (base_checkpoint_id, conversation_id),
                    )).fetchone()
                    if checkpoint is None:
                        raise RuntimeError("stale_checkpoint")
                    if str(checkpoint["kind"]) != "assistant":
                        raise RuntimeError("invalid_fork_checkpoint")
                    reachable = await (await self._conn().execute(
                        """WITH RECURSIVE lineage(id,parent_id) AS (
                               SELECT id,parent_id FROM checkpoints WHERE id=?
                               UNION ALL
                               SELECT cp.id,cp.parent_id FROM checkpoints cp
                               JOIN lineage ON cp.id=lineage.parent_id
                           )
                           SELECT 1 FROM lineage WHERE id=? LIMIT 1""",
                        (head, base_checkpoint_id),
                    )).fetchone()
                    if reachable is None:
                        raise RuntimeError("stale_checkpoint")
                    if branch_mode == "auto" and mode == "staged":
                        raise RuntimeError("mode_mismatch")
                    target_mode = "staged" if mode == "staged" else "auto"
                    next_number_row = await (await self._conn().execute(
                        "SELECT COUNT(*)+1 AS n FROM branches WHERE conversation_id=?",
                        (conversation_id,),
                    )).fetchone()
                    branch_id = str(uuid.uuid4())
                    branch_name = f"Версия {int(next_number_row['n'] if next_number_row else 2)}"
                    fork_ts = now_ms()
                    await self._conn().execute(
                        """INSERT INTO branches
                           (id,conversation_id,name,mode,created_from_checkpoint_id,
                            head_checkpoint_id,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            branch_id,
                            conversation_id,
                            branch_name,
                            target_mode,
                            base_checkpoint_id,
                            base_checkpoint_id,
                            fork_ts,
                            fork_ts,
                        ),
                    )
                    branch_mode = target_mode
                    head = base_checkpoint_id
                    created_branch = {
                        "id": branch_id,
                        "conversationId": conversation_id,
                        "name": branch_name,
                        "mode": target_mode,
                        "createdFromCheckpointId": base_checkpoint_id,
                        "headCheckpointId": base_checkpoint_id,
                        "createdAt": fork_ts,
                        "updatedAt": fork_ts,
                    }
                elif mode != branch_mode:
                    raise RuntimeError("mode_mismatch")
                existing = await (await self._conn().execute(
                    "SELECT 1 FROM messages WHERE conversation_id=? AND turn_id=? LIMIT 1", (conversation_id, turn_id)
                )).fetchone()
                if existing is not None:
                    raise ValueError("duplicate turn_id")
                active = await (await self._conn().execute(
                    """SELECT 1 FROM messages
                       WHERE conversation_id=? AND branch_id=? AND role='assistant'
                         AND status IN ('streaming','waiting_approval') LIMIT 1""",
                    (conversation_id, branch_id),
                )).fetchone()
                if active is not None:
                    raise RuntimeError("active_turn")
                row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 AS n FROM messages WHERE conversation_id=?", (conversation_id,)
                )).fetchone()
                ordinal = int(row["n"])
                user_mid, assistant_mid = str(uuid.uuid4()), str(uuid.uuid4())
                user_checkpoint_id = str(uuid.uuid4())
                ts = now_ms()
                parent_cp = await self._checkpoint_row(head)
                parent_message_id = str(parent_cp["message_id"]) if parent_cp and parent_cp["message_id"] else None
                state = (
                    _loads(parent_cp["state_json"], self._empty_checkpoint_state())
                    if parent_cp is not None
                    else self._empty_checkpoint_state()
                )
                state = dict(state)
                state["turnConfig"] = dict(turn_config or {}) | {"mode": mode}
                title_row = await (await self._conn().execute(
                    "SELECT title FROM conversations WHERE id=?", (conversation_id,)
                )).fetchone()
                title = str(title_row["title"]) if title_row else "Новый чат"
                next_title = " ".join(text.strip().splitlines()[0].split())[:42] or "Новый чат"
                await self._conn().execute(
                    """INSERT INTO messages
                       (id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at,
                        branch_id,parent_message_id,checkpoint_id,mode)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (user_mid, conversation_id, turn_id, ordinal, "user", text, "done", _json(user_payload or {}), ts, ts,
                     branch_id, parent_message_id, user_checkpoint_id, mode),
                )
                await self._conn().execute(
                    """INSERT INTO messages
                       (id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at,
                        branch_id,parent_message_id,checkpoint_id,mode)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
                    (assistant_mid, conversation_id, turn_id, ordinal + 1, "assistant", "", "streaming", "{}", ts, ts,
                     branch_id, user_mid, mode),
                )
                await self._conn().execute(
                    """INSERT INTO checkpoints
                       (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (user_checkpoint_id, conversation_id, branch_id, head, user_mid, "user", _json(state), ts),
                )
                await self._copy_checkpoint_links(head, user_checkpoint_id)
                await self._conn().execute(
                    "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
                    (next_title if title == "Новый чат" else title, ts, conversation_id),
                )
                await self._conn().execute(
                    "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                    (user_checkpoint_id, ts, branch_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "userMessageId": user_mid,
            "assistantMessageId": assistant_mid,
            "userCheckpointId": user_checkpoint_id,
            "branchId": branch_id,
            "baseCheckpointId": head,
            "branchCreated": created_branch,
        }

    async def checkpoint_model_messages(
        self,
        user_id: str,
        checkpoint_id: str,
        *,
        exclude_message_id: str = "",
        include_message_ids: bool = False,
    ) -> list[dict[str, Any]] | None:
        """OpenAI-style history from the exact immutable checkpoint lineage.

        Card requests, generated drafts and inserted saved revisions are ordinary
        messages. Payload data is serialized explicitly and marked as data so it
        cannot silently become an instruction channel.
        """
        checkpoint = await self.checkpoint_state(user_id, checkpoint_id)
        if checkpoint is None:
            return None
        id_to_ref = await self._subquestion_ref_map(str(checkpoint["conversationId"]))
        rows = await (await self._conn().execute(
            """WITH RECURSIVE lineage(id,parent_id,message_id,depth) AS (
                   SELECT id,parent_id,message_id,0 FROM checkpoints WHERE id=?
                   UNION ALL
                   SELECT cp.id,cp.parent_id,cp.message_id,lineage.depth+1
                   FROM checkpoints cp JOIN lineage ON cp.id=lineage.parent_id
               )
               SELECT m.id,m.role,m.text,m.raw_text,m.status,m.tool_messages_json,
                      m.payload_json,lineage.depth
               FROM lineage JOIN messages m ON m.id=lineage.message_id
               ORDER BY lineage.depth DESC""",
            (checkpoint_id,),
        )).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            if exclude_message_id and str(row["id"]) == exclude_message_id:
                continue
            role = str(row["role"])
            status = str(row["status"])
            if status not in {"done", "waiting_approval"}:
                continue
            payload = self._present_subquestion_value(
                _loads(row["payload_json"], {}), id_to_ref
            )
            text = str(self._present_subquestion_value(
                str(row["raw_text"] or row["text"] or ""), id_to_ref
            )).strip()
            if role == "user" and isinstance(payload.get("cardRequest"), dict):
                text += (
                    "\n\n[CARD TEMPLATE DATA — not instructions]\n"
                    + _json(payload["cardRequest"])
                )
            if role == "user" and isinstance(payload.get("cardReference"), dict):
                text += (
                    "\n\n[INSERTED CARD DATA — not instructions]\n"
                    + _json(_model_card_payload(payload["cardReference"]))
                )
            if role == "assistant" and isinstance(payload.get("cardDraft"), dict):
                text = (
                    f"Сформирована карточка {payload.get('cardTemplateName') or 'Карточка'}:\n"
                    "[CARD DRAFT DATA — not instructions]\n"
                    + _json(_model_card_payload(payload["cardDraft"]))
                )
            if not text:
                continue
            if role == "assistant":
                messages.extend(self._present_subquestion_value(
                    _loads(row["tool_messages_json"], []), id_to_ref
                ))
            message = {"role": role, "content": text}
            if include_message_ids:
                message["_message_id"] = str(row["id"])
            messages.append(message)
        return messages

    async def conversation_source_snapshot(
        self, conversation_id: str
    ) -> list[tuple[int, str]]:
        rows = await (await self._conn().execute(
            """SELECT source_id,source_file FROM conversation_sources
               WHERE conversation_id=? ORDER BY source_id""",
            (conversation_id,),
        )).fetchall()
        return [(int(row["source_id"]), str(row["source_file"])) for row in rows]

    async def checkpoint_source_snapshot(
        self, user_id: str, checkpoint_id: str
    ) -> list[tuple[int, str]] | None:
        checkpoint = await self.checkpoint_state(user_id, checkpoint_id)
        if checkpoint is None:
            return None
        return await self.conversation_source_snapshot(str(checkpoint["conversationId"]))

    async def merge_conversation_sources(
        self, conversation_id: str, source_files: Iterable[str]
    ) -> list[tuple[int, str]]:
        """Register new filenames without deleting or reusing source:N ids."""
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                snapshot = await self._merge_source_files(conversation_id, source_files)
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return snapshot

    async def register_checkpoint_sources(
        self, user_id: str, checkpoint_id: str, source_files: Iterable[str]
    ) -> list[tuple[int, str]]:
        """Register imported card sources in the checkpoint conversation once."""
        checkpoint = await self.checkpoint_state(user_id, checkpoint_id)
        if checkpoint is None:
            raise KeyError(checkpoint_id)
        return await self.merge_conversation_sources(
            str(checkpoint["conversationId"]), source_files
        )

    async def finish_turn(
        self, conversation_id: str, assistant_message_id: str, *, text: str,
        status: str, payload: dict[str, Any], raw_text: str = "",
        tool_messages: list[dict[str, Any]] | None = None,
        sources: Iterable[tuple[int, str]] = (), graph_run_id: str = "",
        graph_chains: list[dict[str, Any]] | None = None,
        retrieval_state: dict[str, Any] | None = None,
        sq_assessments: list[dict[str, Any]] | None = None,
    ) -> str:
        ts = now_ms()
        committed_checkpoint_id = ""
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    "UPDATE messages SET text=?,status=?,payload_json=?,raw_text=?,tool_messages_json=?,updated_at=? WHERE id=? AND conversation_id=?",
                    (text, status, _json(payload), raw_text, _json(tool_messages or []), ts, assistant_message_id, conversation_id),
                )
                await self._merge_source_files(
                    conversation_id,
                    [str(path) for _sid, path in sources],
                )
                if graph_run_id and graph_chains:
                    await self._conn().execute(
                        "INSERT OR REPLACE INTO graph_runs(id,conversation_id,message_id,chains_json,created_at) VALUES(?,?,?,?,?)",
                        (graph_run_id, conversation_id, assistant_message_id, _json(graph_chains), ts),
                    )
                message_row = await (await self._conn().execute(
                    """SELECT branch_id,parent_message_id,checkpoint_id FROM messages
                       WHERE id=? AND conversation_id=?""",
                    (assistant_message_id, conversation_id),
                )).fetchone()
                if message_row is not None and message_row["branch_id"]:
                    branch_id = str(message_row["branch_id"])
                    user_row = await (await self._conn().execute(
                        "SELECT checkpoint_id FROM messages WHERE id=?",
                        (message_row["parent_message_id"],),
                    )).fetchone()
                    parent_checkpoint_id = (
                        str(user_row["checkpoint_id"])
                        if user_row is not None and user_row["checkpoint_id"]
                        else None
                    )
                    assistant_checkpoint_id = (
                        str(message_row["checkpoint_id"])
                        if message_row["checkpoint_id"]
                        else str(uuid.uuid4())
                    )
                    committed_checkpoint_id = assistant_checkpoint_id
                    parent_cp = await self._checkpoint_row(parent_checkpoint_id)
                    state = (
                        _loads(parent_cp["state_json"], self._empty_checkpoint_state())
                        if parent_cp is not None
                        else self._empty_checkpoint_state()
                    )
                    state = dict(state)
                    if retrieval_state is not None:
                        snapshot_id = str(uuid.uuid4())
                        parent_snapshot = state.get("retrievalSnapshotId")
                        await self._conn().execute(
                            """INSERT INTO retrieval_snapshots(id,conversation_id,parent_id,state_json,created_at)
                               VALUES(?,?,?,?,?)""",
                            (snapshot_id, conversation_id, parent_snapshot, _json(retrieval_state), ts),
                        )
                        state["retrievalSnapshotId"] = snapshot_id
                    await self._conn().execute(
                        """INSERT OR IGNORE INTO checkpoints
                           (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            assistant_checkpoint_id,
                            conversation_id,
                            branch_id,
                            parent_checkpoint_id,
                            assistant_message_id,
                            "assistant",
                            _json(state),
                            ts,
                        ),
                    )
                    await self._copy_checkpoint_links(parent_checkpoint_id, assistant_checkpoint_id)
                    if sq_assessments:
                        applied = await self._apply_sq_assessments(
                            conversation_id,
                            assistant_checkpoint_id,
                            assistant_message_id,
                            sq_assessments,
                        )
                        if not applied:
                            payload = dict(payload)
                            payload["sqStatusWarning"] = SQ_STATUS_USER_NOTICE
                            await self._conn().execute(
                                "UPDATE messages SET payload_json=? WHERE id=? AND conversation_id=?",
                                (_json(payload), assistant_message_id, conversation_id),
                            )
                    if parent_checkpoint_id:
                        await self._conn().execute(
                            """UPDATE evidence_units SET created_checkpoint_id=?
                               WHERE conversation_id=? AND created_checkpoint_id=?""",
                            (assistant_checkpoint_id, conversation_id, parent_checkpoint_id),
                        )
                    if retrieval_state is not None:
                        graphs = dict((retrieval_state.get("s3Bundle") or {}).get("graphs") or {})
                        for sq_id, bundle in graphs.items():
                            if not isinstance(bundle, dict):
                                continue
                            sq_row = await (await self._conn().execute(
                                "SELECT 1 FROM subquestions WHERE id=? AND conversation_id=?",
                                (str(sq_id), conversation_id),
                            )).fetchone()
                            if sq_row is None:
                                continue
                            bundle_json = _json(bundle)
                            fingerprint = hashlib.sha256(bundle_json.encode("utf-8")).hexdigest()
                            snapshot_row = await (await self._conn().execute(
                                """SELECT id FROM graph_snapshots
                                   WHERE conversation_id=? AND sq_id=? AND fingerprint=?""",
                                (conversation_id, str(sq_id), fingerprint),
                            )).fetchone()
                            graph_snapshot_id = (
                                str(snapshot_row["id"])
                                if snapshot_row is not None
                                else str(uuid.uuid4())
                            )
                            if snapshot_row is None:
                                await self._conn().execute(
                                    """INSERT INTO graph_snapshots
                                       (id,conversation_id,sq_id,fingerprint,bundle_json,created_at)
                                       VALUES(?,?,?,?,?,?)""",
                                    (graph_snapshot_id, conversation_id, str(sq_id), fingerprint, bundle_json, ts),
                                )
                            await self._conn().execute(
                                """UPDATE checkpoint_subquestions SET graph_snapshot_id=?
                                   WHERE checkpoint_id=? AND sq_id=?""",
                                (graph_snapshot_id, assistant_checkpoint_id, str(sq_id)),
                            )
                        await self._conn().execute(
                            """INSERT INTO retrieval_events
                               (id,conversation_id,checkpoint_id,event_type,payload_json,created_at)
                               VALUES(?,?,?,?,?,?)""",
                            (
                                str(uuid.uuid4()),
                                conversation_id,
                                assistant_checkpoint_id,
                                "retrieval_committed",
                                _json({
                                    "algorithmVersion": retrieval_state.get("algorithmVersion"),
                                    "mode": retrieval_state.get("lastMode"),
                                    "depth": retrieval_state.get("lastDepth"),
                                    "subquestionIds": retrieval_state.get("lastSubquestionIds") or [],
                                    "trace": retrieval_state.get("lastTrace") or {},
                                    "pBefore": retrieval_state.get("pBefore") or {},
                                    "pAfter": (retrieval_state.get("carousel") or {}).get("p_store") or {},
                                }),
                                ts,
                            ),
                        )
                    unit_ids = list(state.get("unitIds") or [])
                    for chain in graph_chains or []:
                        if not isinstance(chain, dict):
                            continue
                        signature = self._chain_signature(chain)
                        existing_unit = await (await self._conn().execute(
                            "SELECT id FROM evidence_units WHERE conversation_id=? AND signature=?",
                            (conversation_id, signature),
                        )).fetchone()
                        if existing_unit is None:
                            unit_id = str(uuid.uuid4())
                            source_graph = str(chain.get("source_graph") or "")
                            sq_row = await (await self._conn().execute(
                                """SELECT s.id FROM checkpoint_subquestions cs
                                   JOIN subquestions s ON s.id=cs.sq_id
                                   WHERE cs.checkpoint_id=? AND (s.id=? OR s.canonical_text=?) LIMIT 1""",
                                (parent_checkpoint_id, source_graph, source_graph.casefold()),
                            )).fetchone() if parent_checkpoint_id else None
                            sq_id = str(sq_row["id"]) if sq_row is not None else None
                            unit_no = await self._next_branch_unit_no(assistant_checkpoint_id)
                            await self._conn().execute(
                                """INSERT INTO evidence_units
                                   (id,conversation_id,created_checkpoint_id,unit_no,sq_id,signature,chain_json,created_at)
                                   VALUES(?,?,?,?,?,?,?,?)""",
                                (unit_id, conversation_id, assistant_checkpoint_id, unit_no, sq_id, signature, _json(chain), ts),
                            )
                        else:
                            unit_id = str(existing_unit["id"])
                        await self._attach_checkpoint_unit(assistant_checkpoint_id, unit_id)
                        if unit_id not in unit_ids:
                            unit_ids.append(unit_id)
                    state["unitIds"] = unit_ids
                    await self._conn().execute(
                        "UPDATE checkpoints SET state_json=? WHERE id=?",
                        (_json(state), assistant_checkpoint_id),
                    )
                    await self._conn().execute(
                        "UPDATE messages SET checkpoint_id=? WHERE id=?",
                        (assistant_checkpoint_id, assistant_message_id),
                    )
                    await self._conn().execute(
                        "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                        (assistant_checkpoint_id, ts, branch_id),
                    )
                await self._conn().execute("UPDATE conversations SET updated_at=? WHERE id=?", (ts, conversation_id))
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return committed_checkpoint_id

    async def _apply_sq_assessments(
        self,
        conversation_id: str,
        checkpoint_id: str,
        assistant_message_id: str,
        assessments: list[dict[str, Any]],
    ) -> bool:
        """Apply matching assistant assessments inside finish_turn's transaction."""
        if not assessments:
            return False
        refs = [str(item.get("ref") or "").strip() for item in assessments]
        numbers = [self.parse_subquestion_ref(ref) for ref in refs]
        if any(number is None for number in numbers) or len(set(refs)) != len(refs):
            return False
        rows = await (await self._conn().execute(
            """SELECT s.id,s.display_no FROM checkpoint_subquestions cs
                JOIN subquestions s ON s.id=cs.sq_id
                WHERE cs.checkpoint_id=? AND s.conversation_id=?
                  AND cs.agenda_visible=1 AND cs.status!='closed'""",
            (checkpoint_id, conversation_id),
        )).fetchall()
        by_ref = {
            self.subquestion_ref(int(row["display_no"])): str(row["id"])
            for row in rows
        }
        if any(
            str(item.get("status") or "") not in {"closed", "partial", "not_closed"}
            or not str(item.get("reason") or "").strip()
            or not isinstance(item.get("source_refs"), list)
            for item in assessments
            if str(item.get("ref") or "").strip() in by_ref
        ):
            return False
        applied = False
        for item in assessments:
            ref = str(item["ref"])
            if ref not in by_ref:
                continue
            status = str(item.get("status") or "")
            reason = str(item.get("reason") or "")
            source_refs = item.get("source_refs") or []
            await self._conn().execute(
                """UPDATE checkpoint_subquestions
                   SET status=?,status_origin='assistant',status_reason=?,
                       status_source_refs_json=?,status_message_id=?
                   WHERE checkpoint_id=? AND sq_id=?""",
                (
                    status,
                    reason,
                    _json(source_refs),
                    assistant_message_id,
                    checkpoint_id,
                    by_ref[ref],
                ),
            )
            applied = True
        return applied

    async def list_turn_failures(
        self, user_id: str, conversation_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        if not await self.conversation_owned(user_id, conversation_id):
            return []
        rows = await (await self._conn().execute(
            """SELECT payload_json,created_at FROM retrieval_events
               WHERE conversation_id=? AND event_type='turn_failed'
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, max(1, int(limit))),
        )).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = _loads(row["payload_json"], {})
            if not isinstance(payload, dict):
                payload = {}
            out.append({
                "reason": str(payload.get("reason") or "error"),
                "message": str(payload.get("message") or ""),
                "text": str(payload.get("text") or ""),
                "createdAt": int(row["created_at"]),
            })
        return out

    async def rollback_turn(
        self,
        conversation_id: str,
        assistant_message_id: str,
        *,
        reason: str,
        message: str = "",
    ) -> dict[str, Any]:
        """Drop a failed turn from lineage and restore the previous branch head."""
        ts = now_ms()
        payload = {"text": "", "reason": reason, "message": message}
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                assistant = await (await self._conn().execute(
                    """SELECT id,branch_id,parent_message_id,checkpoint_id,status
                       FROM messages WHERE id=? AND conversation_id=? AND role='assistant'""",
                    (assistant_message_id, conversation_id),
                )).fetchone()
                if assistant is None:
                    await self._conn().commit()
                    return payload
                user_mid = str(assistant["parent_message_id"] or "")
                user_row = await (await self._conn().execute(
                    """SELECT id,text,checkpoint_id,branch_id FROM messages
                       WHERE id=? AND conversation_id=? AND role='user'""",
                    (user_mid, conversation_id),
                )).fetchone() if user_mid else None
                user_text = str(user_row["text"] or "") if user_row is not None else ""
                payload["text"] = user_text
                user_checkpoint_id = (
                    str(user_row["checkpoint_id"] or "") if user_row is not None else ""
                )
                assistant_checkpoint_id = str(assistant["checkpoint_id"] or "")
                branch_id = str(assistant["branch_id"] or "")
                parent_checkpoint_id = None
                if user_checkpoint_id:
                    user_cp = await self._checkpoint_row(user_checkpoint_id)
                    if user_cp is not None:
                        parent_checkpoint_id = (
                            str(user_cp["parent_id"]) if user_cp["parent_id"] else None
                        )
                await self._conn().execute(
                    """DELETE FROM pending_approvals
                       WHERE conversation_id=? AND (
                           assistant_message_id=? OR user_message_id=?
                       )""",
                    (conversation_id, assistant_message_id, user_mid),
                )
                drop_origins = [
                    checkpoint_id
                    for checkpoint_id in (user_checkpoint_id, assistant_checkpoint_id)
                    if checkpoint_id
                ]
                if drop_origins:
                    placeholders = ",".join("?" * len(drop_origins))
                    await self._conn().execute(
                        f"""DELETE FROM evidence_units
                            WHERE conversation_id=? AND created_checkpoint_id IN ({placeholders})
                              AND id NOT IN (
                                  SELECT unit_id FROM checkpoint_units
                                  WHERE checkpoint_id NOT IN ({placeholders})
                              )""",
                        (conversation_id, *drop_origins, *drop_origins),
                    )
                if assistant_checkpoint_id:
                    await self._conn().execute(
                        "DELETE FROM checkpoints WHERE id=?", (assistant_checkpoint_id,)
                    )
                if user_checkpoint_id:
                    await self._conn().execute(
                        "DELETE FROM checkpoints WHERE id=?", (user_checkpoint_id,)
                    )
                if user_mid:
                    await self._conn().execute("DELETE FROM messages WHERE id=?", (user_mid,))
                await self._conn().execute(
                    "DELETE FROM messages WHERE id=?", (assistant_message_id,)
                )
                if branch_id:
                    await self._conn().execute(
                        "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                        (parent_checkpoint_id, ts, branch_id),
                    )
                await self._conn().execute(
                    """INSERT INTO retrieval_events
                       (id,conversation_id,checkpoint_id,event_type,payload_json,created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        conversation_id,
                        parent_checkpoint_id,
                        "turn_failed",
                        _json({
                            "reason": reason,
                            "message": message,
                            "text": user_text,
                        }),
                        ts,
                    ),
                )
                await self._conn().execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (ts, conversation_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return payload

    async def load_model_context(
        self,
        user_id: str,
        conversation_id: str,
        limit: int,
        *,
        branch_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
        if not await self.conversation_owned(user_id, conversation_id):
            raise KeyError(conversation_id)
        id_to_ref = await self._subquestion_ref_map(conversation_id)
        selected_branch = branch_id or await self.main_branch_id(conversation_id)
        branch = await (await self._conn().execute(
            "SELECT head_checkpoint_id FROM branches WHERE id=? AND conversation_id=?",
            (selected_branch, conversation_id),
        )).fetchone()
        head = str(branch["head_checkpoint_id"] or "") if branch is not None else ""
        if head:
            rows = await (await self._conn().execute(
                """WITH RECURSIVE lineage(id,parent_id,message_id,depth) AS (
                       SELECT id,parent_id,message_id,0 FROM checkpoints WHERE id=?
                       UNION ALL
                       SELECT cp.id,cp.parent_id,cp.message_id,lineage.depth+1
                       FROM checkpoints cp JOIN lineage ON cp.id=lineage.parent_id
                   )
                   SELECT m.role,m.text,m.raw_text,m.status,m.tool_messages_json,lineage.depth
                   FROM lineage JOIN messages m ON m.id=lineage.message_id
                   ORDER BY lineage.depth DESC""",
                (head,),
            )).fetchall()
        else:
            rows = []
        turns: list[dict[str, Any]] = []
        pending_user = ""
        for row in rows:
            if str(row["role"]) == "user":
                pending_user = str(self._present_subquestion_value(str(row["text"]), id_to_ref))
            elif (
                pending_user
                and str(row["role"]) == "assistant"
                and str(row["status"]) == "done"
                and str(row["raw_text"] or "")
            ):
                turns.append({
                    "user": pending_user,
                    "assistant": str(self._present_subquestion_value(str(row["raw_text"]), id_to_ref)),
                    "tool_messages": self._present_subquestion_value(
                        _loads(row["tool_messages_json"], []), id_to_ref
                    ),
                })
                pending_user = ""
        turns = turns[-max(1, int(limit)):]
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

    async def graph_run_corpus_id(self, user_id: str, graph_run_id: str) -> str | None:
        row = await (await self._conn().execute(
            """SELECT c.run_id FROM graph_runs g
               JOIN conversations c ON c.id=g.conversation_id
               WHERE g.id=? AND c.user_id=?""",
            (graph_run_id, user_id),
        )).fetchone()
        return str(row["run_id"] or "") if row is not None else None

    @staticmethod
    def canonical_subquestion(text: str) -> str:
        return " ".join((text or "").strip().casefold().rstrip("?.!").split())

    async def apply_agenda_event(
        self,
        user_id: str,
        branch_id: str,
        *,
        base_checkpoint_id: str,
        action: str,
        sq_ref: str = "",
        text: str = "",
        ordered_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        await self.require_branch_writable(user_id, branch_id)
        branch = await self.branch_detail(user_id, branch_id)
        if branch is None:
            raise KeyError(branch_id)
        if branch["mode"] != "staged":
            raise RuntimeError("agenda_unavailable")
        if branch["headCheckpointId"] != base_checkpoint_id:
            raise RuntimeError("stale_checkpoint")
        checkpoint_id = str(uuid.uuid4())
        ts = now_ms()
        parent = await self._checkpoint_row(base_checkpoint_id)
        state = (
            _loads(parent["state_json"], self._empty_checkpoint_state())
            if parent is not None
            else self._empty_checkpoint_state()
        )
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    """INSERT INTO checkpoints
                       (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                       VALUES(?,?,?,?,NULL,'agenda',?,?)""",
                    (checkpoint_id, branch["conversationId"], branch_id, base_checkpoint_id, _json(state), ts),
                )
                await self._copy_checkpoint_links(base_checkpoint_id, checkpoint_id)
                sq_id = ""
                if sq_ref:
                    # Agenda mutations may deliberately target a closed SQ
                    # (for example, reopening it). Retrieval keeps the stricter
                    # open-only resolver below.
                    resolved = await self.open_agenda_subquestions(
                        checkpoint_id, [sq_ref], open_only=False
                    )
                    if len(resolved) != 1:
                        raise KeyError(sq_ref)
                    sq_id = str(resolved[0]["id"])
                if action in {"add", "edit"}:
                    clean = " ".join((text or "").strip().split())
                    canonical = self.canonical_subquestion(clean)
                    if not clean or not canonical:
                        raise ValueError("SQ не может быть пустым")
                    found = await (await self._conn().execute(
                        "SELECT id FROM subquestions WHERE conversation_id=? AND canonical_text=?",
                        (branch["conversationId"], canonical),
                    )).fetchone()
                    new_sq_id = str(found["id"]) if found is not None else str(uuid.uuid4())
                    if found is None:
                        display_no = await self._next_subquestion_display_no(branch["conversationId"])
                        await self._conn().execute(
                            "INSERT INTO subquestions(id,conversation_id,display_no,text,canonical_text,created_at) VALUES(?,?,?,?,?,?)",
                            (new_sq_id, branch["conversationId"], display_no, clean, canonical, ts),
                        )
                    if action == "edit":
                        old = await (await self._conn().execute(
                            """SELECT status,position,question_count,unit_count FROM checkpoint_subquestions
                               WHERE checkpoint_id=? AND sq_id=?""",
                            (checkpoint_id, sq_id),
                        )).fetchone()
                        if old is None:
                            raise KeyError(sq_id)
                        await self._conn().execute(
                            "DELETE FROM checkpoint_subquestions WHERE checkpoint_id=? AND sq_id=?",
                            (checkpoint_id, sq_id),
                        )
                        values = (
                            str(old["status"]), int(old["position"]), int(old["question_count"]), int(old["unit_count"])
                        )
                    else:
                        pos = await (await self._conn().execute(
                            "SELECT COALESCE(MAX(position),-1)+1 AS n FROM checkpoint_subquestions WHERE checkpoint_id=?",
                            (checkpoint_id,),
                        )).fetchone()
                        values = ("not_closed", int(pos["n"] if pos else 0), 0, 0)
                    await self._conn().execute(
                            """INSERT OR REPLACE INTO checkpoint_subquestions
                           (checkpoint_id,sq_id,status,position,question_count,unit_count,agenda_visible)
                           VALUES(?,?,?,?,?,?,1)""",
                        (checkpoint_id, new_sq_id, *values),
                    )
                elif action in {"close", "reopen", "set_status"}:
                    target_status = (
                        "closed" if action == "close"
                        else "not_closed" if action == "reopen"
                        else str(text or "").strip()
                    )
                    if target_status not in {"closed", "partial", "not_closed"}:
                        raise ValueError("Unsupported SQ status")
                    cur = await self._conn().execute(
                        """UPDATE checkpoint_subquestions
                           SET status=?,status_origin='user',status_reason='',
                               status_source_refs_json='[]',status_message_id=NULL
                           WHERE checkpoint_id=? AND sq_id=? AND agenda_visible=1""",
                        (target_status, checkpoint_id, sq_id),
                    )
                    if not cur.rowcount:
                        raise KeyError(sq_id)
                elif action == "reorder":
                    ordered = await self.open_agenda_subquestions(
                        checkpoint_id, ordered_refs or [], open_only=False
                    )
                    if len(ordered) != len(list(dict.fromkeys(ordered_refs or []))):
                        raise KeyError("unknown SQ ref")
                    for position, item in enumerate(ordered):
                        await self._conn().execute(
                            """UPDATE checkpoint_subquestions SET position=?
                               WHERE checkpoint_id=? AND sq_id=? AND agenda_visible=1""",
                            (position, checkpoint_id, item["id"]),
                        )
                else:
                    raise ValueError("Unsupported agenda action")
                await self._conn().execute(
                    "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                    (checkpoint_id, ts, branch_id),
                )
                await self._conn().execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (ts, branch["conversationId"]),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "checkpointId": checkpoint_id,
            "agenda": await self._agenda_for_checkpoint(checkpoint_id),
        }

    async def upsert_turn_subquestions(
        self,
        conversation_id: str,
        checkpoint_id: str,
        texts: list[str],
        *,
        increment: bool = True,
        agenda_visible: bool = False,
    ) -> list[dict[str, Any]]:
        """Attach approved/model SQs to a user checkpoint."""
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                pos_row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(position),-1)+1 AS n FROM checkpoint_subquestions WHERE checkpoint_id=?",
                    (checkpoint_id,),
                )).fetchone()
                next_pos = int(pos_row["n"] if pos_row else 0)
                touched: list[str] = []
                for raw in texts:
                    clean = " ".join(str(raw).strip().split())
                    canonical = self.canonical_subquestion(clean)
                    if not canonical:
                        continue
                    row = await (await self._conn().execute(
                        "SELECT id FROM subquestions WHERE conversation_id=? AND canonical_text=?",
                        (conversation_id, canonical),
                    )).fetchone()
                    sq_id = str(row["id"]) if row is not None else str(uuid.uuid4())
                    if row is None:
                        display_no = await self._next_subquestion_display_no(conversation_id)
                        await self._conn().execute(
                            "INSERT INTO subquestions(id,conversation_id,display_no,text,canonical_text,created_at) VALUES(?,?,?,?,?,?)",
                            (sq_id, conversation_id, display_no, clean, canonical, ts),
                        )
                    current = await (await self._conn().execute(
                        "SELECT question_count,status FROM checkpoint_subquestions WHERE checkpoint_id=? AND sq_id=?",
                        (checkpoint_id, sq_id),
                    )).fetchone()
                    if current is None:
                        await self._conn().execute(
                            """INSERT INTO checkpoint_subquestions
                               (checkpoint_id,sq_id,status,position,question_count,unit_count,agenda_visible)
                               VALUES(?,?, 'not_closed', ?, ?, 0, ?)""",
                            (checkpoint_id, sq_id, next_pos, 1 if increment else 0, 1 if agenda_visible else 0),
                        )
                        next_pos += 1
                    else:
                        if agenda_visible:
                            await self._conn().execute(
                                """UPDATE checkpoint_subquestions SET agenda_visible=1
                                   WHERE checkpoint_id=? AND sq_id=?""",
                                (checkpoint_id, sq_id),
                            )
                    if current is not None and increment and str(current["status"]) != "closed":
                        parent_count_row = await (await self._conn().execute(
                            """SELECT COALESCE(pcs.question_count,0) AS n
                               FROM checkpoints cp
                               LEFT JOIN checkpoint_subquestions pcs
                                 ON pcs.checkpoint_id=cp.parent_id AND pcs.sq_id=?
                               WHERE cp.id=?""",
                            (sq_id, checkpoint_id),
                        )).fetchone()
                        target_count = int(parent_count_row["n"] if parent_count_row else 0) + 1
                        await self._conn().execute(
                            """UPDATE checkpoint_subquestions
                               SET question_count=MAX(question_count,?)
                               WHERE checkpoint_id=? AND sq_id=?""",
                            (target_count, checkpoint_id, sq_id),
                        )
                    touched.append(sq_id)
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return await self._agenda_for_checkpoint(
            checkpoint_id, include_internal=not agenda_visible, include_ids=True
        )

    async def open_agenda_subquestions(
        self,
        checkpoint_id: str,
        sq_refs: list[str],
        *,
        open_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Resolve staged refs in caller order within one checkpoint.

        Retrieval uses the default open-only view. Agenda mutations can opt into
        visible closed SQs, while refs from another branch/checkpoint remain
        invalid in either case.
        """
        if not sq_refs:
            return []
        unique_refs = list(dict.fromkeys(str(value).strip() for value in sq_refs if str(value).strip()))
        numbers = [self.parse_subquestion_ref(value) for value in unique_refs]
        if any(number is None for number in numbers):
            return []
        placeholders = ",".join("?" for _ in numbers)
        status = "AND cs.status!='closed'" if open_only else ""
        rows = await (await self._conn().execute(
            f"""SELECT s.id,s.display_no,s.text FROM checkpoint_subquestions cs
                JOIN subquestions s ON s.id=cs.sq_id
                WHERE cs.checkpoint_id=? AND cs.agenda_visible=1 {status}
                  AND s.display_no IN ({placeholders})""",
            [checkpoint_id, *numbers],
        )).fetchall()
        by_ref = {
            self.subquestion_ref(int(row["display_no"])): {
                "id": str(row["id"]),
                "ref": self.subquestion_ref(int(row["display_no"])),
                "text": str(row["text"]),
            }
            for row in rows
        }
        return [by_ref[item_ref] for item_ref in unique_refs if item_ref in by_ref]

    async def agenda_refs_for_ids(
        self,
        checkpoint_id: str,
        sq_ids: list[str],
        *,
        open_only: bool = False,
    ) -> list[str]:
        """Compatibility resolver for legacy browser/pending payloads only."""
        unique_ids = list(dict.fromkeys(str(value).strip() for value in sq_ids if str(value).strip()))
        if not unique_ids:
            return []
        placeholders = ",".join("?" for _ in unique_ids)
        status = "AND cs.status!='closed'" if open_only else ""
        rows = await (await self._conn().execute(
            f"""SELECT s.id,s.display_no FROM checkpoint_subquestions cs
                JOIN subquestions s ON s.id=cs.sq_id
                WHERE cs.checkpoint_id=? AND cs.agenda_visible=1 {status}
                  AND s.id IN ({placeholders})""",
            [checkpoint_id, *unique_ids],
        )).fetchall()
        by_id = {
            str(row["id"]): self.subquestion_ref(int(row["display_no"]))
            for row in rows
        }
        return [by_id[sq_id] for sq_id in unique_ids if sq_id in by_id]

    async def checkpoint_chains(
        self,
        user_id: str,
        checkpoint_id: str,
        *,
        scope: str = "context",
        unit_id: str = "",
    ) -> list[dict[str, Any]] | None:
        state = await self.checkpoint_state(user_id, checkpoint_id)
        if state is None:
            return None
        select_columns = """u.id,cu.unit_no,u.chain_json,u.created_checkpoint_id,
                   question_message.turn_id AS origin_step_id,
                   question_message.text AS origin_question,
                   origin_branch.id AS origin_branch_id,
                   origin_branch.name AS origin_branch_name,
                   (SELECT COUNT(*) FROM messages numbered_question
                    WHERE numbered_question.conversation_id=u.conversation_id
                      AND numbered_question.role='user'
                      AND numbered_question.turn_id NOT LIKE 'cardref:%'
                      AND numbered_question.ordinal<=question_message.ordinal) AS origin_step_no"""
        origin_joins = """LEFT JOIN checkpoints origin_checkpoint
                     ON origin_checkpoint.id=u.created_checkpoint_id
                   LEFT JOIN checkpoints question_checkpoint
                     ON question_checkpoint.id=origin_checkpoint.parent_id
                   LEFT JOIN messages question_message
                     ON question_message.id=question_checkpoint.message_id
                   LEFT JOIN branches origin_branch
                     ON origin_branch.id=origin_checkpoint.branch_id"""
        if scope == "all_branches":
            rows = await (await self._conn().execute(
                f"""SELECT u.id,u.unit_no,u.chain_json,u.created_checkpoint_id,
                           question_message.turn_id AS origin_step_id,
                           question_message.text AS origin_question,
                           origin_branch.id AS origin_branch_id,
                           origin_branch.name AS origin_branch_name,
                           (SELECT COUNT(*) FROM messages numbered_question
                            WHERE numbered_question.conversation_id=u.conversation_id
                              AND numbered_question.role='user'
                              AND numbered_question.turn_id NOT LIKE 'cardref:%'
                              AND numbered_question.ordinal<=question_message.ordinal) AS origin_step_no
                    FROM evidence_units u {origin_joins}
                    WHERE u.conversation_id=? ORDER BY u.unit_no""",
                (state["conversationId"],),
            )).fetchall()
        else:
            params: list[Any] = [checkpoint_id]
            where = "cu.checkpoint_id=?"
            if scope == "unit" and unit_id:
                where += " AND u.id=?"
                params.append(unit_id)
            rows = await (await self._conn().execute(
                f"""SELECT {select_columns}
                    FROM checkpoint_units cu JOIN evidence_units u ON u.id=cu.unit_id
                    {origin_joins}
                    WHERE {where} ORDER BY cu.unit_no""",
                params,
            )).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            if scope == "new_in_answer" and str(row["created_checkpoint_id"] or "") != checkpoint_id:
                continue
            chain = _loads(row["chain_json"], {})
            if not isinstance(chain, dict):
                continue
            chain = dict(chain)
            chain["unit_id"] = str(row["id"])
            chain["unit_no"] = int(row["unit_no"])
            chain["chain_id"] = f"u{int(row['unit_no'])}"
            chain["is_new"] = str(row["created_checkpoint_id"] or "") == checkpoint_id
            chain["origin"] = {
                "step_id": str(row["origin_step_id"] or "") or None,
                "step_no": int(row["origin_step_no"] or 0) or None,
                "question": " ".join(str(row["origin_question"] or "").split())[:240],
                "branch_id": str(row["origin_branch_id"] or "") or None,
                "branch_name": str(row["origin_branch_name"] or "") or None,
                "answer_checkpoint_id": str(row["created_checkpoint_id"] or "") or None,
            }
            out.append(chain)
        return out

    async def load_retrieval_state(self, user_id: str, checkpoint_id: str) -> dict[str, Any]:
        cp = await self.checkpoint_state(user_id, checkpoint_id)
        if cp is None:
            raise KeyError(checkpoint_id)
        snapshot_id = cp["state"].get("retrievalSnapshotId")
        if not snapshot_id:
            return {
                "algorithmVersion": RETRIEVAL_STATE_VERSION,
                "carousel": {},
                "s3Bundle": {},
                "priorSignatures": [],
            }
        row = await (await self._conn().execute(
            "SELECT state_json FROM retrieval_snapshots WHERE id=? AND conversation_id=?",
            (snapshot_id, cp["conversationId"]),
        )).fetchone()
        state = _loads(row["state_json"], {}) if row is not None else {}
        if not isinstance(state, dict):
            raise CorruptStoreError("retrieval snapshot is not a JSON object")
        version = str(state.get("algorithmVersion") or "")
        if version in ("", *LEGACY_RETRIEVAL_STATE_VERSIONS):
            state = {**state, "algorithmVersion": RETRIEVAL_STATE_VERSION}
        return state

    async def record_units(
        self,
        conversation_id: str,
        checkpoint_id: str,
        chains: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Atomically allocate branch-local UNIT numbers on this checkpoint."""
        cp = await self._checkpoint_row(checkpoint_id)
        if cp is None or str(cp["conversation_id"]) != conversation_id:
            raise KeyError(checkpoint_id)
        state = _loads(cp["state_json"], self._empty_checkpoint_state())
        unit_ids = list(state.get("unitIds") or [])
        recorded: list[dict[str, Any]] = []
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                for source in chains:
                    chain = dict(source)
                    signature = self._chain_signature(chain)
                    existing = await (await self._conn().execute(
                        "SELECT id,sq_id,chain_json FROM evidence_units WHERE conversation_id=? AND signature=?",
                        (conversation_id, signature),
                    )).fetchone()
                    if existing is None:
                        unit_id = str(uuid.uuid4())
                        source_graph = str(chain.get("source_graph") or "")
                        sq = await (await self._conn().execute(
                            "SELECT sq_id FROM checkpoint_subquestions WHERE checkpoint_id=? AND sq_id=?",
                            (checkpoint_id, source_graph),
                        )).fetchone()
                        sq_id = str(sq["sq_id"]) if sq is not None else None
                        unit_no = await self._next_branch_unit_no(checkpoint_id)
                        await self._conn().execute(
                            """INSERT INTO evidence_units
                               (id,conversation_id,created_checkpoint_id,unit_no,sq_id,signature,chain_json,created_at)
                               VALUES(?,?,?,?,?,?,?,?)""",
                            (unit_id, conversation_id, checkpoint_id, unit_no, sq_id, signature, _json(chain), now_ms()),
                        )
                        if sq_id:
                            await self._conn().execute(
                                """UPDATE checkpoint_subquestions SET unit_count=unit_count+1
                                   WHERE checkpoint_id=? AND sq_id=?""",
                                (checkpoint_id, sq_id),
                            )
                    else:
                        unit_id = str(existing["id"])
                    unit_no = await self._attach_checkpoint_unit(checkpoint_id, unit_id)
                    if unit_id not in unit_ids:
                        unit_ids.append(unit_id)
                    chain["unit_id"] = unit_id
                    chain["unit_no"] = unit_no
                    chain["chain_id"] = f"u{unit_no}"
                    recorded.append(chain)
                state["unitIds"] = unit_ids
                await self._conn().execute(
                    "UPDATE checkpoints SET state_json=? WHERE id=?",
                    (_json(state), checkpoint_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return recorded

    async def audit_export(self, user_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        cp = await self.checkpoint_state(user_id, checkpoint_id)
        if cp is None:
            return None
        snapshot_id = cp["state"].get("retrievalSnapshotId")
        retrieval: dict[str, Any] | None = None
        if snapshot_id:
            row = await (await self._conn().execute(
                "SELECT id,parent_id,state_json,created_at FROM retrieval_snapshots WHERE id=?",
                (snapshot_id,),
            )).fetchone()
            if row is not None:
                retrieval = {
                    "id": str(row["id"]),
                    "parentId": row["parent_id"],
                    "state": _loads(row["state_json"], {}),
                    "createdAt": int(row["created_at"]),
                }
        events = await (await self._conn().execute(
            "SELECT event_type,payload_json,created_at FROM retrieval_events WHERE checkpoint_id=? ORDER BY created_at",
            (checkpoint_id,),
        )).fetchall()
        return {
            "checkpoint": cp,
            "retrieval": retrieval,
            "units": await self.checkpoint_chains(user_id, checkpoint_id) or [],
            "sources": [
                {"id": source_id, "file": source_file}
                for source_id, source_file in (
                    await self.checkpoint_source_snapshot(user_id, checkpoint_id) or []
                )
            ],
            "events": [
                {"type": str(row["event_type"]), "payload": _loads(row["payload_json"], {}), "createdAt": int(row["created_at"])}
                for row in events
            ],
        }

    async def list_card_templates(self, user_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        archived = "" if include_archived else "AND t.archived_at IS NULL"
        rows = await (await self._conn().execute(
            f"""SELECT t.id,t.owner_user_id,t.name,t.description,t.archived_at,t.created_at,t.updated_at,
                       v.id AS version_id,v.version,v.schema_json,v.ui_json,v.instructions
                FROM card_templates t
                JOIN card_template_versions v ON v.template_id=t.id
                LEFT JOIN card_template_hides h ON h.template_id=t.id AND h.user_id=?
                WHERE (t.owner_user_id IS NULL OR t.owner_user_id=?) {archived}
                  AND h.template_id IS NULL
                  AND v.version=(SELECT MAX(v2.version) FROM card_template_versions v2 WHERE v2.template_id=t.id)
                ORDER BY t.owner_user_id IS NOT NULL,t.name""",
            (user_id, user_id),
        )).fetchall()
        return [
            {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "system": row["owner_user_id"] is None,
                "archived": row["archived_at"] is not None,
                "createdAt": int(row["created_at"]),
                "updatedAt": int(row["updated_at"]),
                "latestVersion": {
                    "id": str(row["version_id"]),
                    "version": int(row["version"]),
                    "schema": _loads(row["schema_json"], {}),
                    "ui": _loads(row["ui_json"], {}),
                    "instructions": str(row["instructions"]),
                },
            }
            for row in rows
        ]

    async def create_card_template(
        self,
        user_id: str,
        *,
        name: str,
        description: str,
        schema: dict[str, Any],
        ui: dict[str, Any] | None = None,
        instructions: str = "",
    ) -> dict[str, Any]:
        clean = " ".join((name or "").strip().split())[:80]
        if not clean:
            raise ValueError("Название шаблона обязательно")
        template_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
        ts = now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    """INSERT INTO card_templates
                       (id,owner_user_id,name,description,created_at,updated_at)
                       VALUES(?,?,?,?,?,?)""",
                    (template_id, user_id, clean, (description or "").strip()[:400], ts, ts),
                )
                await self._conn().execute(
                    """INSERT INTO card_template_versions
                       (id,template_id,version,schema_json,ui_json,instructions,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (version_id, template_id, 1, _json(schema), _json(ui or {}), (instructions or "").strip()[:4000], ts),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "id": template_id,
            "name": clean,
            "description": (description or "").strip()[:400],
            "system": False,
            "archived": False,
            "createdAt": ts,
            "updatedAt": ts,
            "latestVersion": {"id": version_id, "version": 1, "schema": schema, "ui": ui or {}, "instructions": instructions},
        }

    async def add_card_template_version(
        self,
        user_id: str,
        template_id: str,
        *,
        schema: dict[str, Any],
        ui: dict[str, Any] | None = None,
        instructions: str = "",
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        row = await (await self._conn().execute(
            "SELECT owner_user_id,name,description FROM card_templates WHERE id=? AND archived_at IS NULL",
            (template_id,),
        )).fetchone()
        if row is None or str(row["owner_user_id"] or "") != user_id:
            raise KeyError(template_id)
        version_row = await (await self._conn().execute(
            "SELECT COALESCE(MAX(version),0)+1 AS n FROM card_template_versions WHERE template_id=?",
            (template_id,),
        )).fetchone()
        version = int(version_row["n"] if version_row else 1)
        version_id, ts = str(uuid.uuid4()), now_ms()
        clean_name = " ".join((name if name is not None else str(row["name"])).strip().split())[:80]
        clean_desc = (description if description is not None else str(row["description"])).strip()[:400]
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO card_template_versions
                   (id,template_id,version,schema_json,ui_json,instructions,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (version_id, template_id, version, _json(schema), _json(ui or {}), (instructions or "").strip()[:4000], ts),
            )
            await self._conn().execute(
                "UPDATE card_templates SET name=?,description=?,updated_at=? WHERE id=?",
                (clean_name, clean_desc, ts, template_id),
            )
            await self._conn().commit()
        return {
            "id": version_id,
            "templateId": template_id,
            "version": version,
            "schema": schema,
            "ui": ui or {},
            "instructions": instructions,
        }

    async def archive_card_template(self, user_id: str, template_id: str) -> bool:
        async with self._write_lock:
            row = await (await self._conn().execute(
                "SELECT owner_user_id FROM card_templates WHERE id=? AND archived_at IS NULL",
                (template_id,),
            )).fetchone()
            if row is None:
                return False
            ts = now_ms()
            if row["owner_user_id"] is None:
                cur = await self._conn().execute(
                    "INSERT OR IGNORE INTO card_template_hides(user_id,template_id,hidden_at) VALUES(?,?,?)",
                    (user_id, template_id, ts),
                )
            elif str(row["owner_user_id"]) == user_id:
                cur = await self._conn().execute(
                    "UPDATE card_templates SET archived_at=?,updated_at=? WHERE id=? AND owner_user_id=?",
                    (ts, ts, template_id, user_id),
                )
            else:
                return False
            await self._conn().commit()
        return bool(cur.rowcount)

    async def template_version_for_user(self, user_id: str, version_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT v.id,v.template_id,v.version,v.schema_json,v.ui_json,v.instructions,t.name,t.owner_user_id
               FROM card_template_versions v JOIN card_templates t ON t.id=v.template_id
               WHERE v.id=? AND t.archived_at IS NULL AND (t.owner_user_id IS NULL OR t.owner_user_id=?)""",
            (version_id, user_id),
        )).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "templateId": str(row["template_id"]),
            "templateName": str(row["name"]),
            "version": int(row["version"]),
            "schema": _loads(row["schema_json"], {}),
            "ui": _loads(row["ui_json"], {}),
            "instructions": str(row["instructions"]),
            "system": row["owner_user_id"] is None,
        }

    async def create_card_draft(
        self,
        user_id: str,
        *,
        checkpoint_id: str | None,
        template_version_id: str,
        data: dict[str, Any],
        provenance: dict[str, Any] | None = None,
        gaps: list[Any] | None = None,
    ) -> dict[str, Any]:
        if await self.template_version_for_user(user_id, template_version_id) is None:
            raise KeyError(template_version_id)
        if checkpoint_id:
            await self.require_checkpoint_writable(user_id, checkpoint_id)
        draft_id, ts = str(uuid.uuid4()), now_ms()
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO card_drafts
                   (id,user_id,origin_checkpoint_id,template_version_id,data_json,provenance_json,gaps_json,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,'draft',?,?)""",
                (draft_id, user_id, checkpoint_id, template_version_id, _json(data), _json(provenance or {}), _json(gaps or []), ts, ts),
            )
            await self._conn().commit()
        return {
            "id": draft_id,
            "originCheckpointId": checkpoint_id,
            "templateVersionId": template_version_id,
            "data": data,
            "provenance": provenance or {},
            "gaps": gaps or [],
            "status": "draft",
            "createdAt": ts,
            "updatedAt": ts,
        }

    async def append_card_draft_message(
        self,
        user_id: str,
        checkpoint_id: str,
        draft: dict[str, Any],
        *,
        template_name: str,
    ) -> dict[str, Any]:
        """Persist a generated draft as an assistant message and immutable checkpoint."""
        await self.require_checkpoint_writable(user_id, checkpoint_id)
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                checkpoint = await (await self._conn().execute(
                    """SELECT cp.*,b.head_checkpoint_id,c.user_id
                       FROM checkpoints cp
                       JOIN branches b ON b.id=cp.branch_id
                       JOIN conversations c ON c.id=cp.conversation_id
                       WHERE cp.id=?""",
                    (checkpoint_id,),
                )).fetchone()
                if checkpoint is None or str(checkpoint["user_id"]) != user_id:
                    raise KeyError(checkpoint_id)
                if str(checkpoint["head_checkpoint_id"] or "") != checkpoint_id:
                    raise RuntimeError("stale_checkpoint")
                conversation_id = str(checkpoint["conversation_id"])
                branch_id = str(checkpoint["branch_id"])
                row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 AS n FROM messages WHERE conversation_id=?",
                    (conversation_id,),
                )).fetchone()
                message_id = str(uuid.uuid4())
                next_checkpoint_id = str(uuid.uuid4())
                ts = now_ms()
                payload = {
                    "cardDraft": draft,
                    "cardTemplateName": template_name,
                }
                await self._conn().execute(
                    """INSERT INTO messages
                       (id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at,
                        branch_id,parent_message_id,checkpoint_id,mode)
                       VALUES(?,?,?,?,?,'','done',?,?,?,?,?,?,?)""",
                    (
                        message_id,
                        conversation_id,
                        f"card:{draft['id']}",
                        int(row["n"] if row else 0),
                        "assistant",
                        _json(payload),
                        ts,
                        ts,
                        branch_id,
                        checkpoint["message_id"],
                        next_checkpoint_id,
                        "card",
                    ),
                )
                await self._conn().execute(
                    """INSERT INTO checkpoints
                       (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                       VALUES(?,?,?,?,?,'card_draft',?,?)""",
                    (
                        next_checkpoint_id,
                        conversation_id,
                        branch_id,
                        checkpoint_id,
                        message_id,
                        checkpoint["state_json"],
                        ts,
                    ),
                )
                await self._copy_checkpoint_links(checkpoint_id, next_checkpoint_id)
                await self._conn().execute(
                    "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                    (next_checkpoint_id, ts, branch_id),
                )
                await self._conn().execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (ts, conversation_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            **payload,
            "id": message_id,
            "role": "assistant",
            "text": "",
            "status": "done",
            "checkpointId": next_checkpoint_id,
        }

    async def checkpoint_text_context(
        self, user_id: str, checkpoint_id: str
    ) -> list[dict[str, str]] | None:
        if not await self.checkpoint_owned(user_id, checkpoint_id):
            return None
        rows = await (await self._conn().execute(
            """WITH RECURSIVE lineage(id,parent_id,message_id,depth) AS (
                   SELECT id,parent_id,message_id,0 FROM checkpoints WHERE id=?
                   UNION ALL
                   SELECT cp.id,cp.parent_id,cp.message_id,lineage.depth+1
                   FROM checkpoints cp JOIN lineage ON cp.id=lineage.parent_id
               )
               SELECT m.role,m.text,lineage.depth
               FROM lineage JOIN messages m ON m.id=lineage.message_id
               WHERE m.text <> ''
               ORDER BY lineage.depth DESC""",
            (checkpoint_id,),
        )).fetchall()
        return [{"role": str(row["role"]), "text": str(row["text"])} for row in rows]

    async def update_card_draft(
        self, user_id: str, draft_id: str, *, data: dict[str, Any], provenance: dict[str, Any], gaps: list[Any]
    ) -> bool:
        draft = await self.draft_for_user(user_id, draft_id)
        if draft is None:
            return False
        origin_checkpoint_id = str(draft.get("originCheckpointId") or "")
        if origin_checkpoint_id:
            await self.require_checkpoint_writable(user_id, origin_checkpoint_id)
        async with self._write_lock:
            ts = now_ms()
            cur = await self._conn().execute(
                """UPDATE card_drafts SET data_json=?,provenance_json=?,gaps_json=?,updated_at=?
                   WHERE id=? AND user_id=? AND status='draft'""",
                (_json(data), _json(provenance), _json(gaps), ts, draft_id, user_id),
            )
            if cur.rowcount:
                message_rows = await (await self._conn().execute(
                    """SELECT m.id,m.payload_json FROM messages m
                       JOIN conversations c ON c.id=m.conversation_id
                       WHERE c.user_id=? AND m.payload_json LIKE ?""",
                    (user_id, f'%"id":"{draft_id}"%'),
                )).fetchall()
                for message_row in message_rows:
                    payload = _loads(message_row["payload_json"], {})
                    card_draft = payload.get("cardDraft")
                    if not isinstance(card_draft, dict) or str(card_draft.get("id") or "") != draft_id:
                        continue
                    payload["cardDraft"] = {
                        **card_draft,
                        "data": data,
                        "provenance": provenance,
                        "gaps": gaps,
                        "updatedAt": ts,
                    }
                    await self._conn().execute(
                        "UPDATE messages SET payload_json=?,updated_at=? WHERE id=?",
                        (_json(payload), ts, str(message_row["id"])),
                    )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def save_card_draft(self, user_id: str, draft_id: str, *, title: str = "") -> dict[str, Any]:
        row = await (await self._conn().execute(
            """SELECT * FROM card_drafts WHERE id=? AND user_id=? AND status='draft'""",
            (draft_id, user_id),
        )).fetchone()
        if row is None:
            raise KeyError(draft_id)
        checkpoint_id = str(row["origin_checkpoint_id"] or "")
        if checkpoint_id:
            await self.require_checkpoint_writable(user_id, checkpoint_id)
        origin = await self.audit_export(user_id, checkpoint_id) if checkpoint_id else {"kind": "imported"}
        data = _loads(row["data_json"], {})
        clean_title = " ".join((title or str(data.get("title") or "Карточка")).strip().split())[:120] or "Карточка"
        card_id, revision_id, ts = str(uuid.uuid4()), str(uuid.uuid4()), now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    """INSERT INTO cards(id,user_id,template_version_id,title,created_at,updated_at)
                       VALUES(?,?,?,?,?,?)""",
                    (card_id, user_id, str(row["template_version_id"]), clean_title, ts, ts),
                )
                await self._conn().execute(
                    """INSERT INTO card_revisions
                       (id,card_id,revision,data_json,provenance_json,gaps_json,origin_snapshot_json,created_at)
                       VALUES(?,?,1,?,?,?,?,?)""",
                    (revision_id, card_id, row["data_json"], row["provenance_json"], row["gaps_json"], _json(origin or {}), ts),
                )
                await self._conn().execute(
                    "UPDATE card_drafts SET status='saved',updated_at=? WHERE id=?",
                    (ts, draft_id),
                )
                message_rows = await (await self._conn().execute(
                    """SELECT m.id,m.payload_json FROM messages m
                       JOIN conversations c ON c.id=m.conversation_id
                       WHERE c.user_id=? AND m.payload_json LIKE ?""",
                    (user_id, f'%"id":"{draft_id}"%'),
                )).fetchall()
                for message_row in message_rows:
                    payload = _loads(message_row["payload_json"], {})
                    card_draft = payload.get("cardDraft")
                    if not isinstance(card_draft, dict) or str(card_draft.get("id") or "") != draft_id:
                        continue
                    payload["cardDraft"] = {
                        **card_draft,
                        "status": "saved",
                        "savedCardId": card_id,
                        "savedRevisionId": revision_id,
                    }
                    await self._conn().execute(
                        "UPDATE messages SET payload_json=?,updated_at=? WHERE id=?",
                        (_json(payload), ts, str(message_row["id"])),
                    )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "id": card_id,
            "title": clean_title,
            "templateVersionId": str(row["template_version_id"]),
            "latestRevision": {
                "id": revision_id,
                "revision": 1,
                "data": data,
                "provenance": _loads(row["provenance_json"], {}),
                "gaps": _loads(row["gaps_json"], []),
            },
            "createdAt": ts,
            "updatedAt": ts,
        }

    async def list_cards(self, user_id: str) -> list[dict[str, Any]]:
        rows = await (await self._conn().execute(
            """SELECT c.id,c.title,c.template_version_id,c.created_at,c.updated_at,
                      r.id AS revision_id,r.revision,r.data_json,r.provenance_json,r.gaps_json,
                      r.origin_snapshot_json,v.version,v.schema_json,v.ui_json,t.name AS template_name
               FROM cards c JOIN card_revisions r ON r.card_id=c.id
               JOIN card_template_versions v ON v.id=c.template_version_id
               JOIN card_templates t ON t.id=v.template_id
               WHERE c.user_id=? AND c.archived_at IS NULL
                 AND r.revision=(SELECT MAX(r2.revision) FROM card_revisions r2 WHERE r2.card_id=c.id)
               ORDER BY c.updated_at DESC,c.id""",
            (user_id,),
        )).fetchall()
        cards: list[dict[str, Any]] = []
        for row in rows:
            origin = _loads(row["origin_snapshot_json"], {})
            source_snapshot = [
                (int(item["id"]), str(item["file"]))
                for item in origin.get("sources", [])
                if isinstance(item, dict) and item.get("id") and item.get("file")
            ]
            if not source_snapshot:
                checkpoint_id = str(
                    ((origin.get("checkpoint") or {}).get("id") or "")
                    if isinstance(origin, dict)
                    else ""
                )
                if checkpoint_id:
                    source_snapshot = (
                        await self.checkpoint_source_snapshot(user_id, checkpoint_id) or []
                    )
            cards.append({
                "id": str(row["id"]),
                "title": str(row["title"]),
                "templateVersionId": str(row["template_version_id"]),
                "template": {
                    "name": str(row["template_name"]),
                    "version": int(row["version"]),
                    "schema": _loads(row["schema_json"], {}),
                    "ui": _loads(row["ui_json"], {}),
                },
                "latestRevision": {
                    "id": str(row["revision_id"]),
                    "revision": int(row["revision"]),
                    "data": present_source_aliases_in_value(
                        _loads(row["data_json"], {}), source_snapshot
                    ),
                    "provenance": present_source_aliases_in_value(
                        _loads(row["provenance_json"], {}), source_snapshot
                    ),
                    "gaps": _loads(row["gaps_json"], []),
                },
                "createdAt": int(row["created_at"]),
                "updatedAt": int(row["updated_at"]),
            })
        return cards

    async def card_for_edit(self, user_id: str, card_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT c.id,c.title,c.template_version_id,
                      r.id AS revision_id,r.revision,r.data_json,r.provenance_json,
                      r.gaps_json,r.origin_snapshot_json
               FROM cards c JOIN card_revisions r ON r.card_id=c.id
               WHERE c.id=? AND c.user_id=? AND c.archived_at IS NULL
                 AND r.revision=(SELECT MAX(r2.revision) FROM card_revisions r2 WHERE r2.card_id=c.id)""",
            (card_id, user_id),
        )).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "templateVersionId": str(row["template_version_id"]),
            "revisionId": str(row["revision_id"]),
            "revision": int(row["revision"]),
            "data": _loads(row["data_json"], {}),
            "provenance": _loads(row["provenance_json"], {}),
            "gaps": _loads(row["gaps_json"], []),
            "originSnapshot": _loads(row["origin_snapshot_json"], {}),
        }

    async def add_card_revision(
        self,
        user_id: str,
        card_id: str,
        *,
        title: str,
        data: dict[str, Any],
        provenance: dict[str, Any],
        gaps: list[Any],
        origin_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        clean_title = " ".join(title.strip().split())[:120]
        if not clean_title:
            raise ValueError("Card title is required")
        revision_id, ts = str(uuid.uuid4()), now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                card = await (await self._conn().execute(
                    "SELECT template_version_id FROM cards WHERE id=? AND user_id=? AND archived_at IS NULL",
                    (card_id, user_id),
                )).fetchone()
                if card is None:
                    raise KeyError(card_id)
                next_row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(revision),0)+1 AS n FROM card_revisions WHERE card_id=?",
                    (card_id,),
                )).fetchone()
                revision = int(next_row["n"] if next_row else 1)
                await self._conn().execute(
                    """INSERT INTO card_revisions
                       (id,card_id,revision,data_json,provenance_json,gaps_json,origin_snapshot_json,created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (revision_id, card_id, revision, _json(data), _json(provenance), _json(gaps), _json(origin_snapshot), ts),
                )
                await self._conn().execute(
                    "UPDATE cards SET title=?,updated_at=? WHERE id=?",
                    (clean_title, ts, card_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "id": card_id,
            "title": clean_title,
            "templateVersionId": str(card["template_version_id"]),
            "latestRevision": {
                "id": revision_id,
                "revision": revision,
                "data": data,
                "provenance": provenance,
                "gaps": gaps,
            },
            "updatedAt": ts,
        }

    async def checkpoint_card_context(
        self, user_id: str, checkpoint_id: str
    ) -> list[dict[str, Any]] | None:
        if not await self.checkpoint_owned(user_id, checkpoint_id):
            return None
        rows = await (await self._conn().execute(
            """SELECT c.id AS card_id,c.title,r.id AS revision_id,r.revision,
                      r.data_json,r.provenance_json,r.gaps_json
               FROM checkpoint_card_attachments a
               JOIN card_revisions r ON r.id=a.card_revision_id
               JOIN cards c ON c.id=r.card_id
               WHERE a.checkpoint_id=? AND c.user_id=? AND c.archived_at IS NULL
               ORDER BY c.title,c.id""",
            (checkpoint_id, user_id),
        )).fetchall()
        return [
            {
                "id": str(row["card_id"]),
                "title": str(row["title"]),
                "revisionId": str(row["revision_id"]),
                "revision": int(row["revision"]),
                "data": _loads(row["data_json"], {}),
                "provenance": _loads(row["provenance_json"], {}),
                "gaps": _loads(row["gaps_json"], []),
            }
            for row in rows
        ]

    async def draft_for_user(self, user_id: str, draft_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            "SELECT * FROM card_drafts WHERE id=? AND user_id=?",
            (draft_id, user_id),
        )).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "originCheckpointId": row["origin_checkpoint_id"],
            "templateVersionId": str(row["template_version_id"]),
            "data": _loads(row["data_json"], {}),
            "provenance": _loads(row["provenance_json"], {}),
            "gaps": _loads(row["gaps_json"], []),
            "status": str(row["status"]),
            "createdAt": int(row["created_at"]),
            "updatedAt": int(row["updated_at"]),
        }

    async def archive_card(self, user_id: str, card_id: str) -> bool:
        ts = now_ms()
        async with self._write_lock:
            cur = await self._conn().execute(
                "UPDATE cards SET archived_at=?,updated_at=? WHERE id=? AND user_id=?",
                (ts, ts, card_id, user_id),
            )
            await self._conn().commit()
        return bool(cur.rowcount)

    async def attach_card_revision(
        self,
        user_id: str,
        branch_id: str,
        *,
        base_checkpoint_id: str,
        card_revision_id: str,
        attached: bool,
    ) -> dict[str, Any]:
        await self.require_branch_writable(user_id, branch_id)
        branch = await self.branch_detail(user_id, branch_id)
        if branch is None:
            raise KeyError(branch_id)
        if branch["headCheckpointId"] != base_checkpoint_id:
            raise RuntimeError("stale_checkpoint")
        owned = await (await self._conn().execute(
            """SELECT 1 FROM card_revisions r JOIN cards c ON c.id=r.card_id
               WHERE r.id=? AND c.user_id=? AND c.archived_at IS NULL""",
            (card_revision_id, user_id),
        )).fetchone()
        if owned is None:
            raise KeyError(card_revision_id)
        parent = await self._checkpoint_row(base_checkpoint_id)
        state = _loads(parent["state_json"], self._empty_checkpoint_state()) if parent else self._empty_checkpoint_state()
        checkpoint_id, ts = str(uuid.uuid4()), now_ms()
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                await self._conn().execute(
                    """INSERT INTO checkpoints
                       (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                       VALUES(?,?,?,?,NULL,'card_attachment',?,?)""",
                    (checkpoint_id, branch["conversationId"], branch_id, base_checkpoint_id, _json(state), ts),
                )
                await self._copy_checkpoint_links(base_checkpoint_id, checkpoint_id)
                if attached:
                    await self._conn().execute(
                        "INSERT OR IGNORE INTO checkpoint_card_attachments(checkpoint_id,card_revision_id) VALUES(?,?)",
                        (checkpoint_id, card_revision_id),
                    )
                else:
                    await self._conn().execute(
                        "DELETE FROM checkpoint_card_attachments WHERE checkpoint_id=? AND card_revision_id=?",
                        (checkpoint_id, card_revision_id),
                    )
                revision_rows = await (await self._conn().execute(
                    "SELECT card_revision_id FROM checkpoint_card_attachments WHERE checkpoint_id=? ORDER BY card_revision_id",
                    (checkpoint_id,),
                )).fetchall()
                state["cardRevisionIds"] = [str(item["card_revision_id"]) for item in revision_rows]
                await self._conn().execute(
                    "UPDATE checkpoints SET state_json=? WHERE id=?",
                    (_json(state), checkpoint_id),
                )
                await self._conn().execute(
                    "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                    (checkpoint_id, ts, branch_id),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {"checkpointId": checkpoint_id, "attached": attached}

    async def append_card_reference_message(
        self,
        user_id: str,
        branch_id: str,
        *,
        base_checkpoint_id: str,
        card_revision_id: str,
    ) -> dict[str, Any]:
        """Insert a pinned saved-card revision as an ordinary user message."""
        await self.require_branch_writable(user_id, branch_id)
        branch = await self.branch_detail(user_id, branch_id)
        if branch is None:
            raise KeyError(branch_id)
        if branch["headCheckpointId"] != base_checkpoint_id:
            raise RuntimeError("stale_checkpoint")
        revision = await (await self._conn().execute(
            """SELECT c.title,c.template_version_id,r.id,r.revision,r.data_json,r.provenance_json
               FROM card_revisions r JOIN cards c ON c.id=r.card_id
               WHERE r.id=? AND c.user_id=? AND c.archived_at IS NULL""",
            (card_revision_id, user_id),
        )).fetchone()
        if revision is None:
            raise KeyError(card_revision_id)
        parent = await self._checkpoint_row(base_checkpoint_id)
        if parent is None:
            raise KeyError(base_checkpoint_id)
        payload = {
            "cardReference": {
                "revisionId": str(revision["id"]),
                "templateVersionId": str(revision["template_version_id"]),
                "title": str(revision["title"]),
                "revision": int(revision["revision"]),
                "data": _loads(revision["data_json"], {}),
                "provenance": _loads(revision["provenance_json"], {}),
            }
        }
        message_id, checkpoint_id, ts = str(uuid.uuid4()), str(uuid.uuid4()), now_ms()
        text = f"Добавляю карточку «{revision['title']}» в контекст этой версии."
        state = _loads(parent["state_json"], self._empty_checkpoint_state())
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                ordinal_row = await (await self._conn().execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 AS n FROM messages WHERE conversation_id=?",
                    (branch["conversationId"],),
                )).fetchone()
                await self._conn().execute(
                    """INSERT INTO messages
                       (id,conversation_id,turn_id,ordinal,role,text,status,payload_json,created_at,updated_at,
                        branch_id,parent_message_id,checkpoint_id,mode)
                       VALUES(?,?,?,?,?,?,'done',?,?,?,?,?,?,?)""",
                    (
                        message_id,
                        branch["conversationId"],
                        f"cardref:{message_id}",
                        int(ordinal_row["n"] if ordinal_row else 0),
                        "user",
                        text,
                        _json(payload),
                        ts,
                        ts,
                        branch_id,
                        parent["message_id"],
                        checkpoint_id,
                        "card",
                    ),
                )
                await self._conn().execute(
                    """INSERT INTO checkpoints
                       (id,conversation_id,branch_id,parent_id,message_id,kind,state_json,created_at)
                       VALUES(?,?,?,?,?,'card_reference',?,?)""",
                    (checkpoint_id, branch["conversationId"], branch_id, base_checkpoint_id, message_id, _json(state), ts),
                )
                await self._copy_checkpoint_links(base_checkpoint_id, checkpoint_id)
                await self._conn().execute(
                    "UPDATE branches SET head_checkpoint_id=?,updated_at=? WHERE id=?",
                    (checkpoint_id, ts, branch_id),
                )
                await self._conn().execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?",
                    (ts, branch["conversationId"]),
                )
                await self._conn().commit()
            except Exception:
                await self._conn().rollback()
                raise
        return {
            "checkpointId": checkpoint_id,
            "message": {
                "id": message_id,
                "role": "user",
                "text": text,
                "status": "done",
                "checkpointId": checkpoint_id,
                **payload,
            },
        }

    async def create_pending_approval(
        self,
        user_id: str,
        *,
        conversation_id: str,
        branch_id: str,
        user_message_id: str,
        assistant_message_id: str,
        base_checkpoint_id: str,
        tool_call: dict[str, Any],
        resume: dict[str, Any],
        settings: dict[str, Any],
        approval_id: str | None = None,
        revision: int = 1,
    ) -> dict[str, Any]:
        await self.require_conversation_writable(user_id, conversation_id)
        if not await self.branch_owned(user_id, branch_id):
            raise KeyError(branch_id)
        aid, ts = approval_id or str(uuid.uuid4()), now_ms()
        async with self._write_lock:
            await self._conn().execute(
                """INSERT INTO pending_approvals
                   (id,user_id,conversation_id,branch_id,user_message_id,assistant_message_id,
                    base_checkpoint_id,status,revision,tool_call_json,resume_json,settings_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,'pending',?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET status='pending',revision=excluded.revision,
                       tool_call_json=excluded.tool_call_json,resume_json=excluded.resume_json,
                       settings_json=excluded.settings_json,updated_at=excluded.updated_at""",
                (
                    aid,
                    user_id,
                    conversation_id,
                    branch_id,
                    user_message_id,
                    assistant_message_id,
                    base_checkpoint_id,
                    revision,
                    _json(tool_call),
                    _json(resume),
                    _json(settings),
                    ts,
                    ts,
                ),
            )
            await self._conn().execute(
                """INSERT INTO retrieval_events
                   (id,conversation_id,checkpoint_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()),
                    conversation_id,
                    base_checkpoint_id or None,
                    "approval_required",
                    _json({"approvalId": aid, "revision": revision, "toolCall": tool_call}),
                    ts,
                ),
            )
            await self._conn().commit()
        return await self.pending_approval(user_id, aid) or {}

    async def pending_approval(self, user_id: str, approval_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            "SELECT * FROM pending_approvals WHERE id=? AND user_id=?",
            (approval_id, user_id),
        )).fetchone()
        if row is None:
            return None
        id_to_ref = await self._subquestion_ref_map(str(row["conversation_id"]))
        return {
            "id": str(row["id"]),
            "conversationId": str(row["conversation_id"]),
            "branchId": str(row["branch_id"]),
            "userMessageId": str(row["user_message_id"]),
            "assistantMessageId": str(row["assistant_message_id"]),
            "baseCheckpointId": str(row["base_checkpoint_id"] or ""),
            "status": str(row["status"]),
            "revision": int(row["revision"]),
            "toolCall": self._present_subquestion_value(
                _loads(row["tool_call_json"], {}), id_to_ref
            ),
            "resume": self._present_subquestion_value(
                _loads(row["resume_json"], {}), id_to_ref
            ),
            "settings": _loads(row["settings_json"], {}),
            "createdAt": int(row["created_at"]),
            "updatedAt": int(row["updated_at"]),
        }

    async def pending_for_branch(self, user_id: str, branch_id: str) -> dict[str, Any] | None:
        row = await (await self._conn().execute(
            """SELECT id FROM pending_approvals WHERE user_id=? AND branch_id=? AND status='pending'
               ORDER BY updated_at DESC LIMIT 1""",
            (user_id, branch_id),
        )).fetchone()
        return await self.pending_approval(user_id, str(row["id"])) if row is not None else None

    async def claim_pending_approval(
        self,
        user_id: str,
        approval_id: str,
        revision: int,
        action: str,
    ) -> dict[str, Any] | None:
        if action not in {"approve", "revise", "cancel"}:
            raise ValueError("Unsupported approval action")
        pending = await self.pending_approval(user_id, approval_id)
        if pending is None:
            return None
        await self.require_conversation_writable(
            user_id, str(pending["conversationId"])
        )
        next_status = {"approve": "approved", "revise": "revising", "cancel": "cancelled"}[action]
        async with self._write_lock:
            await self._conn().execute("BEGIN IMMEDIATE")
            try:
                current = await self.pending_approval(user_id, approval_id)
                if current is None:
                    await self._conn().rollback()
                    return None
                if current["status"] != "pending" or current["revision"] != revision:
                    raise RuntimeError("stale_approval")
                await self._conn().execute(
                    "UPDATE pending_approvals SET status=?,updated_at=? WHERE id=? AND user_id=?",
                    (next_status, now_ms(), approval_id, user_id),
                )
                await self._conn().execute(
                    """INSERT INTO retrieval_events
                       (id,conversation_id,checkpoint_id,event_type,payload_json,created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()),
                        current["conversationId"],
                        current["baseCheckpointId"] or None,
                        f"approval_{action}",
                        _json({"approvalId": approval_id, "revision": revision}),
                        now_ms(),
                    ),
                )
                await self._conn().commit()
                current["status"] = next_status
                return current
            except Exception:
                await self._conn().rollback()
                raise

    async def update_assistant_waiting(
        self,
        conversation_id: str,
        assistant_message_id: str,
        *,
        payload: dict[str, Any],
    ) -> None:
        async with self._write_lock:
            await self._conn().execute(
                """UPDATE messages SET status='waiting_approval',payload_json=?,updated_at=?
                   WHERE id=? AND conversation_id=?""",
                (_json(payload), now_ms(), assistant_message_id, conversation_id),
            )
            await self._conn().commit()

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
