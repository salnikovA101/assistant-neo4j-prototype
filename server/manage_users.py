"""Account administration for ``docker compose exec app python -m server.manage_users``."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from server.core.app_store import AppStore


def _password(confirm: bool = True) -> str:
    first = getpass.getpass("Пароль: ")
    if confirm and first != getpass.getpass("Повторите пароль: "):
        raise ValueError("Пароли не совпадают")
    return first


async def _run(args: argparse.Namespace) -> int:
    store = AppStore(os.getenv("APP_DB_PATH", "data/assistant.db"))
    await store.open()
    try:
        if args.command == "create":
            user = await store.create_user(args.login, _password())
            print(f"Создан пользователь {user.username}")
        elif args.command == "list":
            rows = await store.list_users()
            if not rows:
                print("Пользователей нет")
            for row in rows:
                state = "active" if row["is_active"] else "disabled"
                print(f'{row["username"]}\t{state}')
        elif args.command == "reset-password":
            if not await store.reset_password(args.login, _password()):
                raise ValueError("Пользователь не найден")
            print("Пароль изменён, активные сессии отозваны")
        elif args.command in {"disable", "enable"}:
            active = args.command == "enable"
            if not await store.set_user_active(args.login, active):
                raise ValueError("Пользователь не найден")
            print("Пользователь включён" if active else "Пользователь заблокирован")
        elif args.command == "revoke-sessions":
            await store.revoke_user_sessions(args.login)
            print("Активные сессии отозваны")
        return 0
    finally:
        await store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Управление аккаунтами Neo4j Assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "reset-password", "disable", "enable", "revoke-sessions"):
        child = sub.add_parser(command)
        child.add_argument("login")
    sub.add_parser("list")
    try:
        return asyncio.run(_run(parser.parse_args()))
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
