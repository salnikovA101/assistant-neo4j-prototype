"""Safe SQLite maintenance commands."""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import time
from pathlib import Path

from server.core.app_store import AppStore
from server.utils.config import load_config


async def _backup(destination: str) -> None:
    config = load_config()
    store = AppStore(
        os.getenv("APP_DB_PATH", config.app_db_path),
        workspaces=config.workspaces,
    )
    await store.open()
    try:
        await store.backup(destination)
    finally:
        await store.close()
    print(destination)


def _restore(source: str, *, force: bool) -> None:
    if not force:
        raise SystemExit("restore requires --force; stop the app container first")
    source_path = Path(source).expanduser().resolve()
    target_path = Path(os.getenv("APP_DB_PATH", "data/assistant.db")).expanduser().resolve()
    if source_path == target_path:
        raise SystemExit("source and target database are the same file")
    if not source_path.is_file():
        raise SystemExit(f"backup not found: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source_db:
        integrity = source_db.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise SystemExit(f"backup integrity_check failed: {integrity}")
        if target_path.exists():
            safety = target_path.with_name(f"{target_path.name}.pre-restore-{int(time.time())}.db")
            with sqlite3.connect(target_path) as current, sqlite3.connect(safety) as safety_db:
                current.backup(safety_db)
            print(f"safety backup: {safety}")
        with sqlite3.connect(target_path) as target_db:
            source_db.backup(target_db)
    print(target_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обслуживание SQLite Neo4j Assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("destination")
    restore = sub.add_parser("restore")
    restore.add_argument("source")
    restore.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "backup":
        asyncio.run(_backup(args.destination))
    elif args.command == "restore":
        _restore(args.source, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
