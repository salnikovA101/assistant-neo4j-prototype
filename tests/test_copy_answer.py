"""Copy of an assistant answer: Markdown as on screen, with [n] and a source list."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
TSC = WEB / "node_modules" / ".bin" / "tsc"


def _run_copy(tmp_path: Path, text: str) -> str:
    if not TSC.exists():
        pytest.skip("web TypeScript compiler is not installed")
    out = tmp_path / "copyAnswer"
    compiled = subprocess.run(
        [
            str(TSC),
            "src/copyAnswer.ts",
            "--outDir",
            str(out),
            "--module",
            "es2022",
            "--target",
            "es2022",
            "--skipLibCheck",
        ],
        cwd=WEB,
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        raise AssertionError(compiled.stderr or compiled.stdout)
    (tmp_path / "package.json").write_text('{"type":"module"}\n', encoding="utf-8")
    runner = tmp_path / "run-copy.mjs"
    runner.write_text(
        "import { readFileSync } from 'node:fs';\n"
        "import { answerMarkdownForCopy } from './copyAnswer/copyAnswer.js';\n"
        "const input = JSON.parse(readFileSync(0, 'utf8'));\n"
        "process.stdout.write(answerMarkdownForCopy(input));\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(runner)],
        input=json.dumps(text),
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout


def test_copy_keeps_citation_numbers_and_lists_files(tmp_path):
    source = (
        "Кефиран подавляет Listeria [1]. Плёнка меняет цвет [1][2].\n\n"
        "### Источники\n"
        "[1] Hashim et al. Anthocyanins.pdf\n"
        "[2] Priyadarshi R. colorants.pdf\n"
    )
    copied = _run_copy(tmp_path, source)
    assert "https://" not in copied
    assert "http://" not in copied
    assert "Кефиран подавляет Listeria [1]. Плёнка меняет цвет [1][2]." in copied
    assert "(Hashim et al. Anthocyanins.pdf)" not in copied
    assert copied.endswith(
        "### Источники\n"
        "[1] - Hashim et al. Anthocyanins.pdf\n"
        "[2] - Priyadarshi R. colorants.pdf\n"
    )


def test_copy_renames_gaps_and_leaves_code_spans(tmp_path):
    source = (
        "Факт [1] и код `[2]`.\n\n"
        "```\nkeep [1] here\n```\n\n"
        "### GAPS\n"
        "пробелов нет\n\n"
        "### Источники\n"
        "[1] only.pdf\n"
    )
    copied = _run_copy(tmp_path, source)
    assert "Факт [1] и код `[2]`." in copied
    assert "keep [1] here" in copied
    assert "### Пробелы в данных" in copied
    assert "### GAPS" not in copied
    assert "[1] - only.pdf" in copied


def test_copy_without_bibliography_keeps_markdown(tmp_path):
    source = "Справка по интерфейсу.\n\n### GAPS\nпробелов нет\n"
    copied = _run_copy(tmp_path, source)
    assert copied == "Справка по интерфейсу.\n\n### Пробелы в данных\nпробелов нет\n"


def test_frontend_copy_does_not_invent_https_urls():
    copy_ts = (WEB / "src" / "copyAnswer.ts").read_text(encoding="utf-8")
    clipboard = (WEB / "src" / "clipboard.ts").read_text(encoding="utf-8")
    chat = (WEB / "src" / "components" / "ChatThread.tsx").read_text(encoding="utf-8")
    assert "https://" not in copy_ts
    assert "http://" not in copy_ts
    assert "isSecureContext" in clipboard
    assert "execCommand" in clipboard
    assert "clipboard timeout" in clipboard
    assert "answerMarkdownForCopy" in chat
    assert "IconCopy" in chat
    assert '"Копировать"' in chat
    assert 'className="copy-action"' in chat
    assert r"^###[ \t]*Источники(?:\s|$)" in copy_ts
    assert r"^\[(\d+)\]\s+(.+?)\s*$" in copy_ts
    assert "[${id}] - ${name}" in copy_ts
    assert "(${name})" not in copy_ts
