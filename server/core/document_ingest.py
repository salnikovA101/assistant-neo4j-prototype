"""User PDF ingest: validation, stub client, and a later HTTP adapter.

The UI talks to this assistant in batches of files. While
``document_ingest_url`` is empty, metadata is stored and bytes are discarded.
When a processing service exists, set the URL: the same POST forwards the PDFs
to ``{url}/batches`` and keeps the returned per-file statuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi import UploadFile


MAX_FILES = 10
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILENAME_LEN = 180
PDF_MAGIC = b"%PDF"
ALLOWED_CONTENT_TYPES = frozenset(
    {"", "application/pdf", "application/x-pdf", "application/octet-stream"}
)
DOCUMENT_STATUSES = frozenset(
    {"unavailable", "queued", "processing", "completed", "failed", "cancelled"}
)
STUB_MESSAGE = "Обработка документов будет подключена позже."


class DocumentIngestError(ValueError):
    """User-facing ingest failure."""


@dataclass(frozen=True)
class IngestFile:
    filename: str
    size: int
    content_type: str


@dataclass(frozen=True)
class IngestItemAck:
    filename: str
    status: str
    eta_seconds: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class IngestAck:
    status: str
    eta_seconds: int | None
    items: list[IngestItemAck]
    message: str | None = None


class DocumentIngestClient(Protocol):
    requires_content: bool

    async def ingest_batch(self, files: list[tuple[IngestFile, bytes | None]]) -> IngestAck:
        ...


def normalize_filename(name: str | None) -> str:
    raw = Path(str(name or "")).name.strip()
    if not raw or raw in {".", ".."}:
        raise DocumentIngestError("У файла нет имени")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise DocumentIngestError("Имя файла содержит недопустимые символы")
    if len(raw) > MAX_FILENAME_LEN:
        raise DocumentIngestError(f"Имя файла длиннее {MAX_FILENAME_LEN} символов")
    if not raw.lower().endswith(".pdf"):
        raise DocumentIngestError("Можно загрузить только PDF")
    return raw


def normalize_content_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _coerce_status(value: Any, fallback: str) -> str:
    status = str(value or "").strip().lower()
    return status if status in DOCUMENT_STATUSES else fallback


def _coerce_eta(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        eta = int(value)
    except (TypeError, ValueError):
        return None
    return eta if eta >= 0 else None


def validate_pdf(filename: str, header: bytes, content_type: str, size: int) -> None:
    normalize_filename(filename)
    if size <= 0:
        raise DocumentIngestError("Файл пустой")
    if size > MAX_FILE_BYTES:
        raise DocumentIngestError("Файл больше 50 МБ")
    ctype = normalize_content_type(content_type)
    if ctype not in ALLOWED_CONTENT_TYPES:
        raise DocumentIngestError("Можно загрузить только PDF")
    if not header.startswith(PDF_MAGIC):
        raise DocumentIngestError("Можно загрузить только PDF")


async def read_pdf_upload(
    upload: UploadFile, *, keep_content: bool
) -> tuple[IngestFile, bytes | None]:
    filename = normalize_filename(upload.filename)
    content_type = normalize_content_type(upload.content_type) or "application/pdf"
    if keep_content:
        data = await upload.read(MAX_FILE_BYTES + 1)
        validate_pdf(filename, data[:8], content_type, min(len(data), MAX_FILE_BYTES + 1))
        if len(data) > MAX_FILE_BYTES:
            raise DocumentIngestError("Файл больше 50 МБ")
        return IngestFile(filename, len(data), content_type), data

    header = await upload.read(8)
    size = len(header)
    if size == 0:
        raise DocumentIngestError("Файл пустой")
    validate_pdf(filename, header, content_type, size)
    while True:
        chunk = await upload.read(256 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_FILE_BYTES:
            raise DocumentIngestError("Файл больше 50 МБ")
    return IngestFile(filename, size, content_type), None


def stub_ack(files: list[IngestFile]) -> IngestAck:
    return IngestAck(
        status="unavailable",
        eta_seconds=None,
        message=STUB_MESSAGE,
        items=[
            IngestItemAck(filename=item.filename, status="unavailable")
            for item in files
        ],
    )


class StubDocumentIngestClient:
    requires_content = False

    async def ingest_batch(self, files: list[tuple[IngestFile, bytes | None]]) -> IngestAck:
        return stub_ack([item for item, _content in files])


class HttpDocumentIngestClient:
    """POST multipart ``files`` to ``{base_url}/batches`` when the service exists."""

    requires_content = True

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def ingest_batch(self, files: list[tuple[IngestFile, bytes | None]]) -> IngestAck:
        payload = []
        for meta, content in files:
            if content is None:
                raise DocumentIngestError("Не удалось прочитать файл для отправки")
            payload.append(("files", (meta.filename, content, meta.content_type)))
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/batches", files=payload)
        except httpx.HTTPError as exc:
            raise DocumentIngestError("Сервис обработки документов недоступен") from exc
        if response.status_code >= 400:
            raise DocumentIngestError("Сервис обработки документов не принял файлы")
        try:
            data = response.json()
        except ValueError as exc:
            raise DocumentIngestError("Сервис обработки документов вернул некорректный ответ") from exc
        if not isinstance(data, dict):
            raise DocumentIngestError("Сервис обработки документов вернул некорректный ответ")
        return parse_remote_ack(data, [meta for meta, _content in files])


def parse_remote_ack(data: dict[str, Any], originals: list[IngestFile]) -> IngestAck:
    remote_items = data.get("items")
    rows = remote_items if isinstance(remote_items, list) else []
    leftover = [row for row in rows if isinstance(row, dict)]
    mapped: list[IngestItemAck] = []
    for original in originals:
        match_index = next(
            (
                index
                for index, row in enumerate(leftover)
                if str(row.get("filename") or "") == original.filename
            ),
            None,
        )
        row = leftover.pop(match_index) if match_index is not None else {}
        error = row.get("error")
        mapped.append(
            IngestItemAck(
                filename=original.filename,
                status=_coerce_status(row.get("status"), "queued"),
                eta_seconds=_coerce_eta(row.get("etaSeconds") if "etaSeconds" in row else row.get("eta_seconds")),
                error=str(error).strip() if error not in (None, "") else None,
            )
        )
    fallback = "queued" if mapped else "unavailable"
    message = data.get("message")
    return IngestAck(
        status=_coerce_status(data.get("status"), fallback),
        eta_seconds=_coerce_eta(data.get("etaSeconds") if "etaSeconds" in data else data.get("eta_seconds")),
        message=str(message).strip() if message not in (None, "") else None,
        items=mapped,
    )


def build_document_ingest_client(url: str | None) -> StubDocumentIngestClient | HttpDocumentIngestClient:
    base = str(url or "").strip()
    if base:
        return HttpDocumentIngestClient(base)
    return StubDocumentIngestClient()
