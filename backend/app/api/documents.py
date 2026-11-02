import re
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.config import get_settings
from app.db.models import Document, User, Vehicle
from app.db.session import get_db
from app.schemas import DocStatus, DocType, DocumentOut, SignedUrl
from app.services.files import UnsupportedFile, to_pdf
from app.services.ingestion import ingest_document
from app.services.providers import Embedder, get_embedder
from app.services.storage import Storage, get_storage

router = APIRouter(prefix="/documents", tags=["documents"])

_SIGNED_URL_TTL = 600


def _safe_filename(name: str | None) -> str:
    name = (name or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:200] or "document.pdf"


def _get_or_404(db: Session, document_id: int) -> Document:
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: DocType = Form(...),
    vehicle_id: int | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    embedder: Embedder = Depends(get_embedder),
) -> Document:
    """Store a PDF or a photo and queue it for ingestion. Poll GET /documents/{id} for status.

    Photos (JPG, PNG, WEBP, HEIC) are converted to a one-page PDF and read with OCR.
    """
    if vehicle_id is not None and db.get(Vehicle, vehicle_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Vehicle not found")

    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")
    try:
        data = to_pdf(data)
    except UnsupportedFile as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from None

    file_key = f"documents/{uuid.uuid4()}.pdf"
    storage.put_bytes(file_key, data, "application/pdf")

    doc = Document(
        vehicle_id=vehicle_id,
        doc_type=doc_type,
        filename=_safe_filename(file.filename),
        file_key=file_key,
        mime_type="application/pdf",
        status="pending",
        uploaded_by=user.id,
    )
    db.add(doc)
    db.commit()

    background.add_task(ingest_document, doc.id, storage=storage, embedder=embedder)
    return doc


@router.get("", response_model=list[DocumentOut], dependencies=[Depends(get_current_user)])
def list_documents(
    vehicle_id: int | None = None,
    doc_type: DocType | None = None,
    status_: DocStatus | None = Query(None, alias="status"),
    db: Session = Depends(get_db),
) -> list[Document]:
    query = select(Document).order_by(Document.id.desc())
    if vehicle_id is not None:
        query = query.where(Document.vehicle_id == vehicle_id)
    if doc_type is not None:
        query = query.where(Document.doc_type == doc_type)
    if status_ is not None:
        query = query.where(Document.status == status_)
    return list(db.scalars(query))


@router.get("/{document_id}", response_model=DocumentOut, dependencies=[Depends(get_current_user)])
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    return _get_or_404(db, document_id)


@router.get(
    "/{document_id}/file", response_model=SignedUrl, dependencies=[Depends(get_current_user)]
)
def get_document_file(
    document_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> SignedUrl:
    """Short-lived signed URL. Append `#page=N` to open the PDF at a cited page."""
    doc = _get_or_404(db, document_id)
    url = storage.presigned_url(doc.file_key, doc.filename, expires=_SIGNED_URL_TTL)
    return SignedUrl(url=url, expires_in=_SIGNED_URL_TTL)


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(get_current_user)],
)
def reprocess_document(
    document_id: int,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    embedder: Embedder = Depends(get_embedder),
) -> Document:
    """Read the stored file again (e.g. after OCR improvements) and replace its chunks."""
    doc = _get_or_404(db, document_id)
    if doc.status in ("pending", "processing"):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Document is already being read")
    doc.status, doc.error = "pending", None
    db.commit()
    background.add_task(ingest_document, doc.id, storage=storage, embedder=embedder)
    return doc


@router.delete(
    "/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)]
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> Response:
    """Admin only. Removes the stored file, the row, and its chunks (cascade)."""
    doc = _get_or_404(db, document_id)
    storage.delete(doc.file_key)
    db.delete(doc)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
