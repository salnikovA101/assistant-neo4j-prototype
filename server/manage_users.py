"""Account administration for ``docker compose exec app python -m server.manage_users``."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from server.core.app_store import AppStore
from server.utils.config import load_config


def _password(confirm: bool = True) -> str:
    first = getpass.getpass("Пароль: ")
    if confirm and first != getpass.getpass("Повторите пароль: "):
        raise ValueError("Пароли не совпадают")
    return first


async def _run(args: argparse.Namespace) -> int:
    config = load_config()
    store = AppStore(
        os.getenv("APP_DB_PATH", config.app_db_path),
        default_run_id=config.run_id,
    )
    await store.open()
    try:
        if args.command == "create":
            user = await store.create_user(
                args.login, _password(), run_id=getattr(args, "run_id", None)
            )
            print(f"Создан пользователь {user.username}\trun_id={user.run_id}")
        elif args.command == "list":
            rows = await store.list_users()
            if not rows:
                print("Пользователей нет")
            for row in rows:
                state = "active" if row["is_active"] else "disabled"
                print(f'{row["username"]}\t{state}\trun_id={row["run_id"]}')
        elif args.command == "set-run-id":
            result = await store.set_user_run_id(args.login, args.run_id)
            if result is None:
                raise ValueError("Пользователь не найден")
            print(
                f'{result["username"]}: {result["previousRunId"]} -> {result["runId"]}; '
                f'чатов только для чтения: {result["readOnlyConversations"]}'
            )
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
        if command == "create":
            child.add_argument("--run-id", default=None)
    set_run_id = sub.add_parser("set-run-id")
    set_run_id.add_argument("login")
    set_run_id.add_argument("run_id")
    sub.add_parser("list")
    try:
        return asyncio.run(_run(parser.parse_args()))
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
