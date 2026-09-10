import sqlite3

import pytest

from server.manage_db import _restore


def make_db(path, value: str) -> None:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE marker(value TEXT)")
        db.execute("INSERT INTO marker(value) VALUES(?)", (value,))


def test_restore_checks_force_and_creates_safety_copy(tmp_path, monkeypatch):
    source = tmp_path / "backup.db"
    target = tmp_path / "live.db"
    make_db(source, "new")
    make_db(target, "old")
    monkeypatch.setenv("APP_DB_PATH", str(target))

    with pytest.raises(SystemExit):
        _restore(str(source), force=False)

    _restore(str(source), force=True)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT value FROM marker").fetchone()[0] == "new"
    safety = list(tmp_path.glob("live.db.pre-restore-*.db"))
    assert len(safety) == 1
    with sqlite3.connect(safety[0]) as db:
        assert db.execute("SELECT value FROM marker").fetchone()[0] == "old"
