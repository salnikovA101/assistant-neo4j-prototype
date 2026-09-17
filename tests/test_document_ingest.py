from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from starlette.datastructures import Headers, UploadFile

from server.core.app import document_batches_create, document_batches_list, document_batch_item_delete
from server.core.app_store import AppStore
from server.core.document_ingest import (
    DocumentIngestError,
    HttpDocumentIngestClient,
    IngestFile,
    StubDocumentIngestClient,
    build_document_ingest_client,
    normalize_filename,
    parse_remote_ack,
    read_pdf_upload,
    validate_pdf,
)


def _pdf(name: str = "paper.pdf", body: bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n") -> UploadFile:
    return UploadFile(
        BytesIO(body),
        filename=name,
        headers=Headers({"content-type": "application/pdf"}),
    )


def _request(store: AppStore, user, client=None) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(account_user=user),
        app=SimpleNamespace(
            state=SimpleNamespace(
                app_store=store,
                document_ingest=client or StubDocumentIngestClient(),
            )
        ),
    )


def test_normalize_filename_rejects_paths_and_non_pdf():
    assert normalize_filename("C:/tmp/review.PDF") == "review.PDF"
    with pytest.raises(DocumentIngestError, match="только PDF"):
        normalize_filename("notes.txt")
    with pytest.raises(DocumentIngestError, match="имени"):
        normalize_filename("..")


def test_validate_pdf_requires_magic_and_size():
    validate_pdf("ok.pdf", b"%PDF-1.7\n", "application/pdf", 12)
    with pytest.raises(DocumentIngestError, match="только PDF"):
        validate_pdf("ok.pdf", b"%PDF-1.7\n", "text/plain", 12)
    with pytest.raises(DocumentIngestError, match="только PDF"):
        validate_pdf("ok.pdf", b"not-a-pdf", "application/pdf", 9)
    with pytest.raises(DocumentIngestError, match="пустой"):
        validate_pdf("ok.pdf", b"%PDF", "application/pdf", 0)


@pytest.mark.asyncio
async def test_read_pdf_upload_stub_discards_bytes_and_counts_size():
    payload = b"%PDF-1.4\n" + (b"x" * 2048)
    meta, content = await read_pdf_upload(_pdf(body=payload), keep_content=False)
    assert meta.filename == "paper.pdf"
    assert meta.size == len(payload)
    assert content is None
    kept_meta, kept = await read_pdf_upload(_pdf(body=payload), keep_content=True)
    assert kept == payload
    assert kept_meta.size == len(payload)


@pytest.mark.asyncio
async def test_store_isolates_batches_and_deletes_last_item(tmp_path):
    store = AppStore(
        str(tmp_path / "docs.db"),
        workspaces={"packaging": "pack-run", "kefir": "kefir-run"},
    )
    await store.open()
    try:
        owner = await store.create_user("doc-owner", "a sufficiently long password")
        other = await store.create_user("doc-other", "another sufficiently long password")
        created = await store.create_document_batch(
            owner.id,
            "packaging",
            status="unavailable",
            eta_seconds=None,
            message="later",
            items=[
                {"filename": "a.pdf", "size": 10, "content_type": "application/pdf", "status": "unavailable"},
                {"filename": "b.pdf", "size": 20, "content_type": "application/pdf", "status": "unavailable"},
            ],
        )
        assert created["status"] == "unavailable"
        assert [item["filename"] for item in created["items"]] == ["a.pdf", "b.pdf"]
        assert await store.list_document_batches(other.id, "packaging") == []
        leftover = await store.delete_document_batch_item(
            owner.id, "packaging", created["id"], created["items"][0]["id"]
        )
        assert leftover is not None
        assert [item["filename"] for item in leftover["items"]] == ["b.pdf"]
        assert await store.delete_document_batch_item(
            owner.id, "packaging", created["id"], leftover["items"][0]["id"]
        ) is None
        assert await store.list_document_batches(owner.id, "packaging") == []
        with pytest.raises(KeyError):
            await store.get_document_batch(owner.id, "packaging", created["id"])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_create_endpoint_accepts_pdf_stub_and_rejects_other_types(tmp_path):
    store = AppStore(str(tmp_path / "docs-api.db"))
    await store.open()
    try:
        user = await store.create_user("doc-api", "a sufficiently long password")
        request = _request(store, user)
        created = await document_batches_create(request, files=[_pdf("one.pdf"), _pdf("two.pdf")])
        assert created.status_code == 202
        payload = json.loads(created.body)
        assert payload["status"] == "unavailable"
        assert payload["message"]
        assert [item["filename"] for item in payload["items"]] == ["one.pdf", "two.pdf"]
        listed = await document_batches_list(request)
        assert listed["items"][0]["id"] == payload["id"]

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as error:
            await document_batches_create(
                request,
                files=[
                    UploadFile(
                        BytesIO(b"not pdf"),
                        filename="notes.txt",
                        headers=Headers({"content-type": "text/plain"}),
                    )
                ],
            )
        assert error.value.status_code == 400
        remaining = await document_batch_item_delete(
            request, payload["id"], payload["items"][0]["id"]
        )
        assert remaining["batch"]["items"][0]["filename"] == "two.pdf"
    finally:
        await store.close()


def test_build_client_switches_on_url():
    assert isinstance(build_document_ingest_client(""), StubDocumentIngestClient)
    assert isinstance(build_document_ingest_client("https://ingest.example"), HttpDocumentIngestClient)


def test_parse_remote_ack_maps_by_filename():
    ack = parse_remote_ack(
        {
            "status": "processing",
            "etaSeconds": 120,
            "items": [
                {"filename": "b.pdf", "status": "queued", "etaSeconds": 90},
                {"filename": "a.pdf", "status": "processing", "etaSeconds": 30},
            ],
        },
        [IngestFile("a.pdf", 1, "application/pdf"), IngestFile("b.pdf", 2, "application/pdf")],
    )
    assert ack.status == "processing"
    assert ack.items[0].status == "processing"
    assert ack.items[1].eta_seconds == 90


@pytest.mark.asyncio
async def test_http_client_posts_multipart(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": "queued",
                "etaSeconds": 45,
                "items": [{"filename": "paper.pdf", "status": "queued", "etaSeconds": 45}],
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, files):
            captured["url"] = url
            captured["files"] = files
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    client = HttpDocumentIngestClient("https://ingest.example/v1")
    ack = await client.ingest_batch(
        [(IngestFile("paper.pdf", 4, "application/pdf"), b"%PDF")]
    )
    assert captured["url"] == "https://ingest.example/v1/batches"
    assert captured["files"][0][0] == "files"
    assert ack.status == "queued"
    assert ack.items[0].eta_seconds == 45
