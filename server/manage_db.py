"""Safe SQLite maintenance commands."""

from __future__ import annotations

import argparse
import asyncio
import os

from server.core.app_store import AppStore


async def _backup(destination: str) -> None:
    store = AppStore(os.getenv("APP_DB_PATH", "data/assistant.db"))
    await store.open()
    try:
        await store.backup(destination)
    finally:
        await store.close()
    print(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Обслуживание SQLite Neo4j Assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("destination")
    args = parser.parse_args()
    if args.command == "backup":
        asyncio.run(_backup(args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
