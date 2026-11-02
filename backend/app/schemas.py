"""Request and response models."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["admin", "staff"]
VehicleStatus = Literal["acquired", "listed", "sold"]
DocType = Literal["or_cr", "deed_of_sale", "insurance", "inspection", "financing", "id", "other"]
DocStatus = Literal["pending", "processing", "ready", "failed"]


def canonical_plate(plate: str | None) -> str | None:
    """Store plates one way: uppercase, letters and digits separated by one space ('ABC 1234')."""
    if plate is None:
        return None
    compact = re.sub(r"[\s\-]+", "", plate).upper()
    if not compact:
        return None
    return re.sub(r"^([A-Z]+)(\d+)$", r"\1 \2", compact)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    role: Role


class VehicleBase(BaseModel):
    plate: str | None = Field(default=None, max_length=16)
    vin: str | None = Field(default=None, max_length=32)
    make: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=64)
    year: int | None = Field(default=None, ge=1950, le=2100)
    status: VehicleStatus = "acquired"

    @field_validator("plate")
    @classmethod
    def _canonical_plate(cls, value: str | None) -> str | None:
        return canonical_plate(value)


class VehicleCreate(VehicleBase):
    pass


class VehicleUpdate(BaseModel):
    plate: str | None = Field(default=None, max_length=16)
    vin: str | None = Field(default=None, max_length=32)
    make: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=64)
    year: int | None = Field(default=None, ge=1950, le=2100)
    status: VehicleStatus | None = None

    @field_validator("plate")
    @classmethod
    def _canonical_plate(cls, value: str | None) -> str | None:
        return canonical_plate(value)


class VehicleOut(VehicleBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    vehicle_id: int | None
    doc_type: DocType
    filename: str
    mime_type: str
    status: DocStatus
    error: str | None
    page_count: int | None
    ocr_confidence: float | None = None
    uploaded_by: int
    created_at: datetime


class SignedUrl(BaseModel):
    url: str
    expires_in: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    vehicle_id: int | None = None
    doc_type: DocType | None = None
    conversation_id: int | None = None


class CitationOut(BaseModel):
    n: int
    chunk_id: int
    document_id: int
    filename: str
    doc_type: str
    page: int
    snippet: str
    ocr_confidence: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    abstained: bool
    abstain_reason: Literal["low_confidence", "llm_not_answerable"] | None = None
    verified: bool
    supporting_quote: str
    conversation_id: int
    message_id: int
