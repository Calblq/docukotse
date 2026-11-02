"""Seed demo users, vehicles and sample documents. Safe to re-run.

Inside Docker (recommended):
    docker compose exec api python scripts/seed.py
Options:
    --no-docs   only create users and vehicles
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "backend", Path("/app")):  # local checkout, or the api container
    if (candidate / "app").is_dir():
        sys.path.insert(0, str(candidate))
        break

from sqlalchemy import select  # noqa: E402

from app.db.models import Document, User, Vehicle  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.files import to_pdf  # noqa: E402
from app.services.ingestion import ingest_document  # noqa: E402
from app.services.storage import get_storage  # noqa: E402

SAMPLES_DIR = next(
    (p for p in (HERE.parent / "samples", Path("/samples")) if (p / "manifest.json").exists()),
    None,
)


def seed_users(db) -> User:
    accounts = [
        (
            os.environ.get("SEED_ADMIN_EMAIL", "admin@cardocs.local"),
            os.environ.get("SEED_ADMIN_PASSWORD", "admin123"),
            "admin",
        ),
        (
            os.environ.get("SEED_STAFF_EMAIL", "staff@cardocs.local"),
            os.environ.get("SEED_STAFF_PASSWORD", "staff123"),
            "staff",
        ),
    ]
    for email, password, role in accounts:
        email = email.strip().lower()
        if db.scalar(select(User).where(User.email == email)) is None:
            db.add(User(email=email, password_hash=hash_password(password), role=role))
            print(f"  user  {email} ({role})")
    db.commit()
    return db.scalar(select(User).where(User.role == "admin"))


def seed_vehicles(db, vehicles: list[dict]) -> dict[str, Vehicle]:
    by_plate = {}
    for v in vehicles:
        vehicle = db.scalar(select(Vehicle).where(Vehicle.plate == v["plate"]))
        if vehicle is None:
            vehicle = Vehicle(**v)
            db.add(vehicle)
            print(f"  vehicle {v['plate']} {v['make']} {v['model']}")
        by_plate[v["plate"]] = vehicle
    db.commit()
    return by_plate


def seed_documents(db, docs: list[dict], vehicles: dict[str, Vehicle], uploader: User) -> None:
    storage = get_storage()
    storage.ensure_bucket()
    for entry in docs:
        existing = db.scalar(select(Document).where(Document.filename == entry["file"]))
        if existing is not None and existing.status == "ready":
            print(f"  skip  {entry['file']} (already ready)")
            continue
        if existing is None:
            key = f"documents/{uuid.uuid4()}.pdf"
            pdf = to_pdf((SAMPLES_DIR / entry["file"]).read_bytes())  # photos become PDFs
            storage.put_bytes(key, pdf, "application/pdf")
            existing = Document(
                vehicle_id=vehicles[entry["plate"]].id,
                doc_type=entry["doc_type"],
                filename=entry["file"],
                file_key=key,
                mime_type="application/pdf",
                uploaded_by=uploader.id,
            )
            db.add(existing)
            db.commit()
        status = ingest_document(existing.id, storage=storage)
        db.refresh(existing)
        note = f" - {existing.error}" if existing.error else ""
        print(f"  doc   {entry['file']}: {status}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-docs", action="store_true", help="skip document ingestion")
    args = parser.parse_args()

    if SAMPLES_DIR is None:
        sys.exit("samples/manifest.json not found. Run scripts/make_sample_docs.py first.")
    manifest = json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))

    with SessionLocal() as db:
        print("Users:")
        admin = seed_users(db)
        print("Vehicles:")
        vehicles = seed_vehicles(db, manifest["vehicles"])
        if not args.no_docs:
            print("Documents (ingesting; needs Ollama with bge-m3):")
            seed_documents(db, manifest["documents"], vehicles, admin)
    print("Done.")


if __name__ == "__main__":
    main()
