"""Question-specific rules for model answers, on top of the checks in verification.py."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.services.chunking import normalize_for_match
from app.services.retrieval import Candidate
from app.services.verification import VerificationResult, quote_spans, verify_answer

_BUYER_RE = re.compile(r"\b(bought|buyer|buy|purchas\w*|bumili|bumibili|binili|nakabili)\b", re.I)
_REGISTRATION_STATUS_RE = re.compile(
    r"\b(still registered|still valid|registration\b.*\b(valid|expire\w*|current|active)|"
    r"expired?|expires|nakarehistro|rehistrado|registered pa|paso na)\b",
    re.I,
)
_EXPIRY_CLAIM_RE = re.compile(r"valid until|expir|until|hanggang", re.I)
_EXPIRY_LINE_RE = re.compile(r"valid until|expir", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ASKS_AMOUNT_RE = re.compile(r"\b(how much|magkano)\b", re.I)
# A money amount, not just any number: dates like 04/18/2019 don't count.
_AMOUNT_RE = re.compile(
    r"\b(?:PHP|P)\s?\d|₱\s?\d|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2}\b|\bpesos?\b", re.I
)


def buyer_answer_has_deed(question: str, data: dict, candidates: list[Candidate]) -> bool:
    """A question about who bought a car must be answered from a deed of sale.

    Only a deed names a buyer. Given an OR/CR, the model tends to name the registered owner
    instead, and a prompt rule alone didn't stop it. Other questions always pass.
    """
    if not _BUYER_RE.search(question):
        return True
    cited = [candidates[n - 1] for n in data["used_excerpts"] if 1 <= n <= len(candidates)]
    return any(c.doc_type == "deed_of_sale" for c in cited)


def only_restates_question(data: dict, question: str) -> bool:
    """True when the answer just repeats what the question already said.

    For "what car is XYZ7788?", quoting "PLATE NO.: XYZ7788" answers nothing. Each quoted
    line's value (the part after "LABEL:") and the answer are checked against the question.
    """
    q = normalize_for_match(question)
    parts = [p.split(":", 1)[-1] for p in quote_spans(data["supporting_quote"])]
    values = [normalize_for_match(p) for p in parts if normalize_for_match(p)]
    quote_restates = bool(values) and all(v in q for v in values)
    answer = normalize_for_match(data["answer"])
    return quote_restates and (not answer or answer in q or not data["answerable"])


def asks_registration_status(question: str) -> bool:
    """Does the question ask whether a registration is still valid, or when it expires?"""
    return bool(_REGISTRATION_STATUS_RE.search(question))


def registration_without_expiry(
    db: Session, question: str, candidates: list[Candidate], vehicle_id: int | None
) -> list[Candidate]:
    """For a registration-status question, the OR/CR excerpts about the vehicle asked about,
    if none of those documents has an expiry line; otherwise [] (the LLM answers).

    Decided in code because the model reads issue dates and form text ("is duly registered")
    as proof of validity, whatever the prompt says.
    """
    if not asks_registration_status(question):
        return []
    if vehicle_id is not None:
        about = [c for c in candidates if c.vehicle_id == vehicle_id and c.doc_type == "or_cr"]
        doc_ids = set(
            db.scalars(
                select(Document.id).where(
                    Document.vehicle_id == vehicle_id, Document.doc_type == "or_cr"
                )
            )
        )
    else:  # an unlinked document: the ones containing the plate/VIN from the question
        about = [c for c in candidates if c.identifier_matches > 0 and c.doc_type == "or_cr"]
        doc_ids = {c.document_id for c in about}
    if not about or not doc_ids:
        return []
    texts = db.scalars(select(Chunk.text).where(Chunk.document_id.in_(doc_ids)))
    if any(_EXPIRY_LINE_RE.search(t) for t in texts):
        return []
    return about


def check_answer(
    data: dict | None, candidates: list[Candidate], question: str = ""
) -> VerificationResult:
    """Verify a parsed answer against its cited excerpts and the rules for its question."""
    if data is None:
        return VerificationResult(False, ["The model did not return valid JSON."])
    cited = [n for n in data["used_excerpts"] if 1 <= n <= len(candidates)]
    if not cited:
        return VerificationResult(False, ["The answer cites no valid excerpt numbers."])
    problems = verify_answer(
        data["answer"],
        data["supporting_quote"],
        [candidates[n - 1].text for n in cited],
        question=question,
    ).problems
    # Date-only answers happen even at temperature 0; the retry then adds the amount.
    if _ASKS_AMOUNT_RE.search(question) and not _AMOUNT_RE.search(data["answer"]):
        problems.append("The question asks how much, but the answer has no amount.")
    if asks_registration_status(question):
        if not _YEAR_RE.search(data["answer"]):
            problems.append(
                "The question asks whether the registration is valid, but the answer gives no "
                "date. Give the expiry or registration dates from the excerpts, or say the "
                "documents don't show when it expires."
            )
        if (
            _EXPIRY_CLAIM_RE.search(data["answer"])
            and not _EXPIRY_LINE_RE.search(data["supporting_quote"])
            and "don't show" not in data["answer"].lower()
        ):
            problems.append(
                'The answer gives an expiry date, but the quote has no "valid until" or '
                "expiry line, so that date is not an expiry date."
            )
    return VerificationResult(not problems, problems)
