"""Test fixtures.

Unit tests run anywhere. Tests using the `client`/`db` fixtures need Postgres with
pgvector (`docker compose up db`) and are skipped if it is unreachable. They run against
a separate `cardocs_test` database, never the demo one.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://cardocs:cardocs@localhost:5432/cardocs_test"
)
os.environ["WARM_UP_MODELS"] = "false"  # tests use fake providers

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.models import EMBED_DIM, User  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.providers import get_embedder, get_llm, get_reranker  # noqa: E402
from app.services.storage import get_storage  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]
TABLES = "messages, conversations, chunks, documents, vehicles, users"


class FakeStorage:
    """In-memory stand-in for the object store."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def ensure_bucket(self) -> None:
        pass

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def presigned_url(self, key: str, filename: str, expires: int = 600) -> str:
        return f"http://storage.test/{key}?expires={expires}"


class FakeEmbedder:
    """Deterministic embeddings so tests never need Ollama."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []
        for t in texts:
            seed = hashlib.sha256(t.encode()).digest()
            vectors.append([seed[i % len(seed)] / 255.0 for i in range(EMBED_DIM)])
        return vectors


class FakeLLM:
    """Returns scripted replies in order (dicts are JSON-encoded) and records each call."""

    def __init__(self, replies: list | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list[dict]] = []

    def generate(self, messages, *, json_schema=None, temperature=0.0) -> str:
        self.calls.append(messages)
        reply = (
            self.replies.pop(0)
            if self.replies
            else {"answerable": False, "answer": "", "used_excerpts": [], "supporting_quote": ""}
        )
        return reply if isinstance(reply, str) else json.dumps(reply)


class FakeReranker:
    """Scores by overlap of content words with the query, so relevance is predictable."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        from app.services.retrieval import keyword_terms

        q = set(keyword_terms(query))
        return [len(q & set(keyword_terms(t))) / max(len(q), 1) for t in texts]


_db_unreachable = False


@pytest.fixture(scope="session")
def migrated_db():
    global _db_unreachable
    if not _db_unreachable:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            _db_unreachable = True
    if _db_unreachable:  # checked once; connection attempts are slow on Windows
        pytest.skip("Postgres not reachable; start it with `docker compose up -d db`")
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")
    return engine


@pytest.fixture
def db(migrated_db):
    with migrated_db.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    with SessionLocal() as session:
        yield session


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(db, fake_storage, fake_embedder, fake_llm):
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_user(db, email: str, password: str, role: str) -> User:
    user = User(email=email, password_hash=hash_password(password), role=role)
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def corpus(db, fake_embedder) -> dict:
    """Three vehicles with a few one-page chunks each, inserted directly (no PDF parsing)."""
    from app.db.models import Chunk, Document, Vehicle
    from app.services.chunking import context_header, normalize_for_match

    owner = _make_user(db, "seed@test.local", "x", "admin")
    vehicles = {
        "vios": Vehicle(plate="NBC 1234", make="Toyota", model="Vios", year=2019),
        "city": Vehicle(plate="ABC 5678", make="Honda", model="City", year=2020),
        "mirage": Vehicle(plate="DAE 9012", make="Mitsubishi", model="Mirage G4", year=2018),
    }
    db.add_all(vehicles.values())
    db.flush()
    pages = [
        ("vios", "deed_of_sale", "vios_deed.pdf", [
            "DEED OF ABSOLUTE SALE\nSeller: Juan Dela Cruz\nBuyer: Maria Santos\n"
            "Price: PHP 485,000.00\nDate of sale: September 18, 2024",
        ]),
        ("vios", "or_cr", "vios_orcr.pdf", [
            "CERTIFICATE OF REGISTRATION\nPlate No.: NBC 1234\nEngine No.: 2NR1234567\n"
            "Registration valid until: March 31, 2025",
            "OFFICIAL RECEIPT\nOR No.: 8820391457\nAmount paid: PHP 1,820.00",
        ]),
        ("city", "insurance", "city_insurance.pdf", [
            "MOTOR CAR POLICY\nPolicy No.: SMI-MC-2024-0057210\n"
            "Deductible: PHP 3,000.00 per claim\nActs of Nature: covered",
        ]),
        ("mirage", "or_cr", "mirage_orcr.pdf", [
            "CERTIFICATE OF REGISTRATION\nPlate No.: DAE 9012\nOwner: Ana Reyes",
        ]),
    ]  # fmt: skip
    docs = {}
    for key, doc_type, filename, bodies in pages:
        v = vehicles[key]
        doc = Document(
            vehicle_id=v.id,
            doc_type=doc_type,
            filename=filename,
            file_key=f"k/{filename}",
            mime_type="application/pdf",
            status="ready",
            page_count=len(bodies),
            uploaded_by=owner.id,
        )
        db.add(doc)
        db.flush()
        for page, body in enumerate(bodies, start=1):
            header = context_header(
                plate=v.plate, make=v.make, model=v.model, doc_type=doc_type, page=page
            )
            chunk_text = f"{header}\n{body}"
            db.add(
                Chunk(
                    document_id=doc.id,
                    page=page,
                    chunk_index=page - 1,
                    text=chunk_text,
                    text_norm=normalize_for_match(chunk_text),
                    embedding=fake_embedder.embed([chunk_text])[0],
                )
            )
        docs[filename] = doc
    db.commit()
    return {"vehicles": vehicles, "docs": docs}


def _token(client: TestClient, email: str, password: str) -> str:
    resp = client.post("/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture
def admin_headers(client, db) -> dict:
    _make_user(db, "admin@test.local", "admin-pass", "admin")
    return {"Authorization": f"Bearer {_token(client, 'admin@test.local', 'admin-pass')}"}


@pytest.fixture
def staff_headers(client, db) -> dict:
    _make_user(db, "staff@test.local", "staff-pass", "staff")
    return {"Authorization": f"Bearer {_token(client, 'staff@test.local', 'staff-pass')}"}
