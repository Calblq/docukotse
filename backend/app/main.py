"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import auth, chat, documents, vehicles
from app.config import get_settings
from app.db.session import engine
from app.services.providers import warm_up_in_background

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if get_settings().warm_up_models:
        warm_up_in_background()
    yield


app = FastAPI(title="DocuKotse", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(vehicles.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness plus a database check."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
