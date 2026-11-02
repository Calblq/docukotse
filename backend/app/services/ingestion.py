"""Ingestion pipeline: stored PDF → per-page text → chunks → embeddings → chunks table."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import EMBED_DIM, Chunk, Document
from app.db.session import SessionLocal
from app.services.chunking import chunk_text, context_header, normalize_for_match
from app.services.ocr import ocr_available, ocr_page
from app.services.providers import Embedder, get_embedder
from app.services.storage import Storage, get_storage

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """A document cannot be ingested; the message is shown to users."""


@dataclass
class PageText:
    text: str
    ocr_confidence: float | None = None  # set when the text came from OCR


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Text of each page (index 0 = page 1). Pages without a text layer are OCR'd."""
    settings = get_settings()
    use_ocr = settings.ocr_enabled and ocr_available()
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            pages = []
            for page in pdf:
                text = page.get_text("text")
                if use_ocr and len(text.strip()) < settings.ocr_min_text_chars:
                    result = ocr_page(page, settings.ocr_languages, settings.ocr_dpi)
                    if len(result.text.strip()) > len(text.strip()):
                        pages.append(PageText(result.text, result.confidence))
                        continue
                pages.append(PageText(text))
            return pages
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise IngestionError(f"Could not read PDF: {exc}") from exc


def no_text_message() -> str:
    if not get_settings().ocr_enabled or not ocr_available():
        return "No text layer found, and OCR is not available here (Tesseract isn't installed)."
    return (
        "Couldn't read any text from this file. If it's a photo, retake it flat, close up "
        "and in good light."
    )


def build_chunks(doc: Document, pages: list[PageText], max_words: int, overlap: int) -> list[Chunk]:
    """Chunk every page and prepend the context header. Embeddings are filled in later."""
    vehicle = doc.vehicle
    chunks: list[Chunk] = []
    for page_no, page in enumerate(pages, start=1):
        header = context_header(
            plate=vehicle.plate if vehicle else None,
            make=vehicle.make if vehicle else None,
            model=vehicle.model if vehicle else None,
            doc_type=doc.doc_type,
            page=page_no,
        )
        for body in chunk_text(page.text, max_words, overlap):
            text = f"{header}\n{body}"
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    page=page_no,
                    chunk_index=len(chunks),
                    text=text,
                    text_norm=normalize_for_match(text),
                    ocr_confidence=page.ocr_confidence,
                )
            )
    return chunks


def embed_chunks(chunks: list[Chunk], embedder: Embedder, batch_size: int) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed([c.text for c in batch])
        if len(vectors) != len(batch):
            raise IngestionError("Embedder returned the wrong number of vectors")
        for chunk, vector in zip(batch, vectors, strict=True):
            if len(vector) != EMBED_DIM:
                raise IngestionError(f"Expected {EMBED_DIM}-dim embeddings, got {len(vector)}")
            chunk.embedding = vector


def ingest_document(
    document_id: int,
    *,
    storage: Storage | None = None,
    embedder: Embedder | None = None,
    session_factory: Callable[[], Session] = SessionLocal,
) -> str:
    """Ingest one document and return its final status ('ready' or 'failed').

    Idempotent: existing chunks for the document are replaced. Errors never
    propagate; they are recorded on the document row instead.
    """
    settings = get_settings()
    storage = storage or get_storage()
    embedder = embedder or get_embedder()

    with session_factory() as db:
        doc = db.get(Document, document_id)
        if doc is None:
            logger.warning("ingest_document: document %s not found", document_id)
            return "failed"
        doc.status, doc.error = "processing", None
        db.commit()

        try:
            pages = extract_pages(storage.get_bytes(doc.file_key))
            chunks = build_chunks(
                doc, pages, settings.chunk_size_words, settings.chunk_overlap_words
            )
            if not chunks:
                raise IngestionError(no_text_message())
            embed_chunks(chunks, embedder, settings.embed_batch_size)

            db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            db.add_all(chunks)
            doc.page_count, doc.status = len(pages), "ready"
            ocr_confs = [p.ocr_confidence for p in pages if p.ocr_confidence is not None]
            doc.ocr_confidence = min(ocr_confs) if ocr_confs else None
            db.commit()
            logger.info(
                "Ingested document %s: %d pages, %d chunks", doc.id, len(pages), len(chunks)
            )
        except Exception as exc:
            logger.exception("Ingestion failed for document %s", document_id)
            db.rollback()
            doc = db.get(Document, document_id)
            doc.status = "failed"
            if isinstance(exc, IngestionError):
                doc.error = str(exc)[:1000]
            else:
                doc.error = f"{type(exc).__name__}: {exc}"[:1000]
            db.commit()
        return doc.status
