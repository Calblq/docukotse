import pymupdf
from sqlalchemy import func, select, text

from app.db.models import Chunk, Document
from app.services.ingestion import ingest_document

PAGE_1 = "CERTIFICATE OF REGISTRATION\nPlate No: NBC-1234\nOwner: Juan Dela Cruz"
PAGE_2 = "OFFICIAL RECEIPT\nOR No: 7788123\nAmount Paid: PHP 1,820.00"


def make_pdf(pages: list[str]) -> bytes:
    pdf = pymupdf.open()
    for body in pages:
        page = pdf.new_page()
        if body:
            page.insert_text((72, 72), body, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


def upload(client, headers, data: bytes, vehicle_id=None, doc_type="or_cr"):
    form = {"doc_type": doc_type}
    if vehicle_id is not None:
        form["vehicle_id"] = str(vehicle_id)
    return client.post(
        "/documents",
        files={"file": ("orcr.pdf", data, "application/pdf")},
        data=form,
        headers=headers,
    )


def create_vehicle(client, headers) -> int:
    body = {"plate": "NBC 1234", "make": "Toyota", "model": "Vios", "year": 2019}
    return client.post("/vehicles", json=body, headers=headers).json()["id"]


def test_upload_ingests_pages_into_chunks(client, staff_headers, db, fake_storage):
    vid = create_vehicle(client, staff_headers)
    resp = upload(client, staff_headers, make_pdf([PAGE_1, PAGE_2]), vehicle_id=vid)
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "pending"
    doc_id = resp.json()["id"]

    # TestClient runs background tasks before returning, so ingestion is done.
    doc = client.get(f"/documents/{doc_id}", headers=staff_headers).json()
    assert doc["status"] == "ready", doc["error"]
    assert doc["page_count"] == 2
    assert len(fake_storage.objects) == 1

    chunks = db.scalars(select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.page)).all()
    assert [c.page for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].text.startswith("Vehicle: NBC 1234 Toyota Vios | Doc: or_cr | Page 1\n")
    assert "Juan Dela Cruz" in chunks[0].text
    assert "nbc1234" in chunks[0].text_norm
    assert len(chunks[0].embedding) == 1024


def test_full_text_and_normalized_columns_are_searchable(client, staff_headers, db):
    upload(client, staff_headers, make_pdf([PAGE_1, PAGE_2]))
    tsv_hit = db.scalar(
        text("SELECT page FROM chunks WHERE tsv @@ plainto_tsquery('simple', '7788123')")
    )
    assert tsv_hit == 2
    # 'NBC 1234', 'NBC-1234' and 'NBC1234' all match the normalized column
    norm_hit = db.scalar(text("SELECT page FROM chunks WHERE text_norm LIKE '%nbc1234%'"))
    assert norm_hit == 1


def test_reingest_is_idempotent(client, staff_headers, db, fake_storage, fake_embedder):
    doc_id = upload(client, staff_headers, make_pdf([PAGE_1, PAGE_2])).json()["id"]
    count = select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
    assert db.scalar(count) == 2

    status = ingest_document(doc_id, storage=fake_storage, embedder=fake_embedder)
    assert status == "ready"
    assert db.scalar(count) == 2


def test_pdf_without_text_fails_with_message(client, staff_headers):
    doc_id = upload(client, staff_headers, make_pdf(["", ""])).json()["id"]
    doc = client.get(f"/documents/{doc_id}", headers=staff_headers).json()
    assert doc["status"] == "failed"
    # Blank pages: OCR finds nothing (or isn't installed), and the message says so.
    assert "text" in doc["error"]


def test_embedder_failure_marks_document_failed(client, staff_headers, fake_storage, db):
    doc_id = upload(client, staff_headers, make_pdf([PAGE_1])).json()["id"]

    class BrokenEmbedder:
        def embed(self, texts):
            raise ConnectionError("ollama is down")

    assert ingest_document(doc_id, storage=fake_storage, embedder=BrokenEmbedder()) == "failed"
    db.expire_all()
    doc = db.get(Document, doc_id)
    assert "ollama is down" in doc.error
    # The failed re-run rolled back, so the earlier good chunks are untouched.
    assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id))


def test_rejects_non_pdf(client, staff_headers):
    resp = client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"doc_type": "other"},
        headers=staff_headers,
    )
    assert resp.status_code == 415


def test_rejects_unknown_vehicle(client, staff_headers):
    assert upload(client, staff_headers, make_pdf([PAGE_1]), vehicle_id=999).status_code == 404


def test_signed_url_and_admin_delete(client, staff_headers, admin_headers, fake_storage, db):
    doc_id = upload(client, staff_headers, make_pdf([PAGE_1])).json()["id"]

    url = client.get(f"/documents/{doc_id}/file", headers=staff_headers).json()["url"]
    assert url.startswith("http://storage.test/documents/")

    assert client.delete(f"/documents/{doc_id}", headers=staff_headers).status_code == 403
    assert client.delete(f"/documents/{doc_id}", headers=admin_headers).status_code == 204
    assert fake_storage.objects == {}
    assert db.scalar(select(func.count()).select_from(Chunk)) == 0


def test_reprocess_reads_the_document_again(client, staff_headers, db, fake_embedder):
    doc_id = upload(client, staff_headers, make_pdf([PAGE_1, PAGE_2])).json()["id"]
    calls = fake_embedder.calls
    resp = client.post(f"/documents/{doc_id}/reprocess", headers=staff_headers)
    assert resp.status_code == 202
    doc = client.get(f"/documents/{doc_id}", headers=staff_headers).json()
    assert doc["status"] == "ready"
    assert fake_embedder.calls > calls  # embedded again
    count = select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id)
    assert db.scalar(count) == 2  # chunks replaced, not duplicated


def test_reprocess_unknown_document_is_404(client, staff_headers):
    assert client.post("/documents/999/reprocess", headers=staff_headers).status_code == 404
