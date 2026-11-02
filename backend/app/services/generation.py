"""Question answering: retrieve → abstain check → grounded JSON answer → verify (retry once)."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.answer_checks import (
    asks_registration_status,
    buyer_answer_has_deed,
    check_answer,
    only_restates_question,
    registration_without_expiry,
)
from app.services.providers import Embedder, LLMProvider, Reranker
from app.services.retrieval import (
    Candidate,
    RetrievalConfig,
    RetrievalResult,
    resolve_vehicle,
    retrieve,
    vehicle_named,
)
from app.services.verification import VerificationResult, quote_in_excerpts

logger = logging.getLogger(__name__)

PARSE_FAILURE_ANSWER = "The model's answer could not be parsed."
NOT_FOUND_ANSWER = (
    "I couldn't find this in the uploaded documents. / Wala ito sa mga na-upload na dokumento."
)
NO_EXPIRY_ANSWER = (
    'The documents don\'t show when this registration expires: the OR/CR has no "valid until" '
    "date, so they can't tell whether it is still registered. Upload the latest OR (Official "
    "Receipt) of registration to check. / Hindi makikita sa mga dokumento kung hanggang "
    "kailan ang rehistro."
)

SYSTEM_PROMPT = """\
You answer questions from staff of a used-car dealership in the Philippines.
Use ONLY the numbered document excerpts in the user message.

Rules:
- Use only facts written in the excerpts. Never use outside knowledge, and never guess.
- If the excerpts do not contain the answer, set "answerable" to false and leave the other \
fields empty.
- Keep "answer" short: 1-2 sentences, but answer every part of the question (for example \
both the date and the amount).
- Copy numbers, amounts, dates, names and plate numbers exactly as they appear in the excerpts.
- "supporting_quote" is a short span (under 25 words) copied character-for-character from an \
excerpt, containing the key fact. Do not paraphrase or join text from different lines. If the \
answer uses several lines (for example a date and an amount), copy each line and separate \
them with " | ".
- "used_excerpts" lists the numbers of the excerpts the answer comes from.
- Excerpts may be about different vehicles. Check the "Vehicle:" line of each excerpt and use \
only the ones about the vehicle the question asks about. If it says "not linked", tell which \
vehicle the excerpt is about from the plate number or VIN written in its text.
- Document roles matter. The registered owner on an OR/CR or the assured on an insurance \
policy is not a buyer: only a deed of sale names a buyer and a sale price. The OR (Official \
Receipt) number and the CR (Certificate of Registration) number are different numbers: answer \
with the one the question asks for, or say it is not there.
- Answer in the language of the question (English, Filipino or Taglish)."""

# The model writes fields in this order. With "answerable" first it refused or mixed up
# fields it then quoted correctly (chassis vs engine number: 4/5 -> 5/5 with this order).
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "supporting_quote": {"type": "string"},
        "answer": {"type": "string"},
        "used_excerpts": {"type": "array", "items": {"type": "integer"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["supporting_quote", "answer", "used_excerpts", "answerable"],
}

ANSWER_ALL_PARTS = "Answer every part of the question."

RESTATED_HINT = (
    'Your quote "{quote}" only repeats what the question already says, so it does not answer '
    "it. Read the other lines of the excerpts about that vehicle and quote the ones that "
    "answer the question (for example, for which car it is: MAKE/BRAND, SERIES or MODEL and "
    "YEAR MODEL). If no line answers it, set answerable to false."
)

REGISTRATION_HINT = (
    "(Today is {today}. Whether a registration is still valid depends on a date written in "
    'the excerpts, such as "Registration valid until". Printed form text like "is duly '
    'registered" is on every certificate and proves nothing, and other dates (the date a '
    "certificate was issued, an OR date) are not expiry dates. If no excerpt has a "
    '"valid until" or expiry line, say the documents don\'t show when the registration '
    "expires, and give the registration dates they do show. Don't write today's date in the "
    "answer.)"
)

TRANSLATE_PROMPT = (
    "You translate questions from staff of a used-car dealership in the Philippines into "
    "natural English. Keep the exact meaning of dealership words (benta/bentahan/nabenta = "
    "sale/sold, bumili = bought, nagbenta = sold (seller), hulog = installment, may-ari = "
    "owner, rehistrado pa = still registered, paso na = expired). Keep names, plate numbers, "
    "numbers and dates exactly as written. Reply with the English question only.\n\n"
    "Question: {question}"
)

_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"(?:does|do|did) not (?:\w+ly )?(?:provide|mention|contain|include|state|specify|say)"
    r"|not (?:mentioned|provided|stated|specified|found|available)"
    r"|no (?:information|mention|record)"
    r"|(?:cannot|can't) (?:be )?(?:found|determined|answered)"
    r"|(?:walang|wala(?:ng)? (?:sa|ito)) (?:impormasyon|nakasaad|binanggit|dokumento)"
    r"|hindi (?:nakasaad|binanggit|nabanggit|makita)"
    r")\b",
    re.IGNORECASE,
)

# Common Filipino function words; any of these means the question is (at least partly)
# Filipino, so it is also searched in English.
_FILIPINO_WORDS = frozenset(
    "ang ng mga sa si ni kay nina ano sino kanino kailan saan magkano ilan ilang paano bakit "
    "ba po yung iyong nang ito iyan yan naman lang pa may mayroon meron wala hindi".split()
)
_WORD_RE = re.compile(r"[a-zA-Z]+")


@dataclass
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    abstain_rerank_threshold: float = 0.2
    abstain_vector_threshold: float = 0.5
    abstain_identifier_threshold: float = 0.05  # top chunk contains the plate/VIN asked about
    verify: bool = True
    auto_scope: bool = True  # restrict search to the one vehicle a question names
    translate_taglish: bool = True  # also search with an English translation of Taglish
    max_attempts: int = 2  # first try + one retry after failed verification


@dataclass
class Citation:
    n: int
    chunk_id: int
    document_id: int
    filename: str
    doc_type: str
    page: int
    snippet: str
    ocr_confidence: float | None = None  # set if the cited page was read from a photo/scan


@dataclass
class ChatAnswer:
    answer: str
    citations: list[Citation]
    abstained: bool
    verified: bool
    supporting_quote: str = ""
    abstain_reason: str | None = None  # "low_confidence" | "llm_not_answerable"
    scoped_vehicle_id: int | None = None  # vehicle the search was restricted to
    problems: list[str] = field(default_factory=list)
    attempts: int = 0
    retrieval: RetrievalResult | None = None
    timings: dict[str, float] = field(default_factory=dict)


def build_messages(
    question: str, candidates: list[Candidate], translation: str | None = None
) -> list[dict]:
    def source(c: Candidate) -> str:
        ocr = ", read by OCR from a photo/scan" if c.ocr_confidence is not None else ""
        return f"file: {c.filename}, page {c.page}{ocr}"

    excerpts = "\n\n".join(
        f"[{i}] ({source(c)})\n{c.text}" for i, c in enumerate(candidates, start=1)
    )
    user = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    if translation:  # helps the model read Taglish; it still answers in the user's language
        user += (
            f"\n(In English, only to help you understand it: {translation} Answer in the "
            "language of the original question, using the documents' own words; don't "
            "translate them.)"
        )
    if asks_registration_status(question):
        today = date.today()
        user += "\n" + REGISTRATION_HINT.format(today=f"{today:%B} {today.day}, {today.year}")
    # Last thing the model reads: the same rule in the system prompt alone gets ignored.
    user += f"\n{ANSWER_ALL_PARTS}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_llm_json(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return None
    data.setdefault("answerable", True)
    data.setdefault("supporting_quote", "")
    used = data.get("used_excerpts")
    data["used_excerpts"] = (
        [int(u) for u in used if isinstance(u, int)] if isinstance(used, list) else []
    )
    # Models sometimes set answerable=true and then write a refusal as the answer.
    if data["answerable"] and looks_like_refusal(data["answer"]):
        data["answerable"] = False
    return data


def looks_like_refusal(answer: str) -> bool:
    """True if the answer text says the excerpts don't contain the answer."""
    return not answer.strip() or bool(_REFUSAL_RE.search(answer))


def _snippet(c: Candidate, quote: str = "", limit: int = 500) -> str:
    """Excerpt of the chunk body for display, keeping line breaks and starting near the quote."""
    text = "\n".join(" ".join(line.split()) for line in c.body.splitlines() if line.strip())
    start = 0
    q = " ".join(quote.split()).lower()
    idx = text.lower().find(q) if q else -1
    if idx > limit // 2:  # quote is far down: start a few lines above it
        start = text.rfind("\n", 0, max(0, idx - 120)) + 1
    excerpt = text[start : start + limit]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + limit < len(text) else ""
    return prefix + excerpt.rstrip() + suffix


def _citations(candidates: list[Candidate], numbers: list[int], quote: str = "") -> list[Citation]:
    out = []
    for n in numbers:
        c = candidates[n - 1]
        out.append(
            Citation(
                n,
                c.chunk_id,
                c.document_id,
                c.filename,
                c.doc_type,
                c.page,
                _snippet(c, quote),
                c.ocr_confidence,
            )
        )
    return out


def with_conversation_vehicle(db: Session, question: str, earlier: list[str]) -> str:
    """Add the vehicle from earlier in the conversation to a follow-up like "who owns it?",
    which names none and so would find nothing.

    `earlier` holds the conversation's previous user questions, newest first.
    """
    if vehicle_named(db, question) is not None:
        return question
    for previous in earlier:
        named = vehicle_named(db, previous)
        if named is not None:
            return f"{question} (about {named})"
    return question


def looks_filipino(question: str) -> bool:
    return any(w.lower() in _FILIPINO_WORDS for w in _WORD_RE.findall(question))


def translate_to_english(llm: LLMProvider, question: str) -> str | None:
    """English version of a Filipino/Taglish question, used only to search.

    The reranker scores Taglish questions against English documents poorly, so retrieval
    uses both. Returns None if the model is unavailable or answers with something odd.
    """
    try:
        text = llm.generate(
            [{"role": "user", "content": TRANSLATE_PROMPT.format(question=question)}],
            temperature=0.0,
        )
    except Exception:
        logger.warning("Query translation failed; searching with the original question only")
        return None
    text = text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""
    return text if 3 <= len(text) <= 3 * len(question) + 40 else None


def answer_question(
    db: Session,
    question: str,
    *,
    embedder: Embedder,
    llm: LLMProvider,
    reranker: Reranker | None,
    config: PipelineConfig,
    vehicle_id: int | None = None,
    doc_type: str | None = None,
) -> ChatAnswer:
    """Answer from retrieved chunks only, abstaining or flagging 'unverified' when unsure."""
    start = time.perf_counter()
    if vehicle_id is None and config.auto_scope:
        vehicle_id = resolve_vehicle(db, question)
    translation = None
    if config.translate_taglish and looks_filipino(question):
        t0 = time.perf_counter()
        translation = translate_to_english(llm, question)
        translate_s = time.perf_counter() - t0
    result = retrieve(
        db,
        question,
        embedder,
        reranker,
        config.retrieval,
        vehicle_id=vehicle_id,
        doc_type=doc_type,
        alt_queries=[translation] if translation else None,
    )
    timings = dict(result.timings)
    if translation is not None:
        timings["translate"] = translate_s
    candidates = result.candidates

    threshold = (
        config.abstain_rerank_threshold if result.reranked else config.abstain_vector_threshold
    )
    if result.reranked and candidates and candidates[0].identifier_matches > 0:
        # The question names an exact identifier and the best chunk contains it: we have the
        # right document even if the reranker is unsure. The LLM and verification decide.
        threshold = min(threshold, config.abstain_identifier_threshold)
    if not candidates or result.confidence < threshold:
        timings["total"] = time.perf_counter() - start
        return ChatAnswer(
            NOT_FOUND_ANSWER,
            [],
            abstained=True,
            verified=True,
            abstain_reason="low_confidence",
            scoped_vehicle_id=vehicle_id,
            retrieval=result,
            timings=timings,
        )

    checked = registration_without_expiry(db, question, candidates, vehicle_id)
    if checked:
        timings["total"] = time.perf_counter() - start
        return ChatAnswer(
            NO_EXPIRY_ANSWER,
            _citations(checked, list(range(1, len(checked) + 1))),
            abstained=False,
            verified=True,
            scoped_vehicle_id=vehicle_id,
            retrieval=result,
            timings=timings,
        )

    messages = build_messages(question, candidates, translation)
    data: dict | None = None
    check = VerificationResult(False)
    attempts = 0
    t0 = time.perf_counter()
    while attempts < config.max_attempts:
        attempts += 1
        raw = llm.generate(messages, json_schema=ANSWER_SCHEMA, temperature=0.0)
        data = parse_llm_json(raw)
        if (
            data is not None
            and data["answerable"]
            and not buyer_answer_has_deed(question, data, candidates)
        ):
            data["answerable"] = False  # only a deed of sale says who bought a car
            break
        if (
            data is not None
            and attempts < config.max_attempts
            and only_restates_question(data, question)
        ):
            # Ask again from scratch: as a follow-up to its own answer, the model repeats it
            # word for word.
            hint = RESTATED_HINT.format(quote=data["supporting_quote"].strip())
            question_msg = {**messages[1], "content": messages[1]["content"] + "\n\n" + hint}
            messages = [messages[0], question_msg]
            continue
        if data is not None and not data["answerable"]:
            quote = data["supporting_quote"].strip()
            cited = [
                candidates[n - 1].text for n in data["used_excerpts"] if 1 <= n <= len(candidates)
            ]
            if attempts < config.max_attempts and quote and quote_in_excerpts(quote, cited):
                # Self-contradiction: "not answerable", yet it quoted real evidence.
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f'You set answerable to false but quoted "{quote}", which is '
                        "in the excerpts. If that text answers the question, set answerable to "
                        "true and answer from it. If it does not, keep answerable false.",
                    },
                ]
                continue
            break
        if not config.verify:
            check = VerificationResult(data is not None)
            break
        check = check_answer(data, candidates, question)
        if check.ok:
            break
        logger.info("Verification failed (attempt %d): %s", attempts, check.problems)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Your answer failed verification:\n- "
                + "\n- ".join(check.problems)
                + "\nAnswer again. Copy the supporting quote exactly from one excerpt, and use "
                "only numbers, dates and plates that appear in the excerpts you cite. If the "
                "excerpts do not answer the question, set answerable to false.",
            },
        ]
    timings["generate"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - start

    common = dict(
        attempts=attempts, retrieval=result, timings=timings, scoped_vehicle_id=vehicle_id
    )
    if data is not None and not data["answerable"]:
        return ChatAnswer(
            NOT_FOUND_ANSWER,
            [],
            abstained=True,
            verified=True,
            abstain_reason="llm_not_answerable",
            **common,
        )

    if data is not None and check.ok:
        cited = sorted({n for n in data["used_excerpts"] if 1 <= n <= len(candidates)})
        return ChatAnswer(
            data["answer"],
            _citations(
                candidates,
                cited or list(range(1, len(candidates) + 1)),
                data["supporting_quote"],
            ),
            abstained=False,
            verified=config.verify,
            supporting_quote=data["supporting_quote"],
            **common,
        )

    # Unverified: show the answer (if any) with every retrieved chunk so staff can check.
    answer = data["answer"] if data is not None else PARSE_FAILURE_ANSWER
    return ChatAnswer(
        answer,
        _citations(candidates, list(range(1, len(candidates) + 1))),
        abstained=False,
        verified=False,
        supporting_quote=(data or {}).get("supporting_quote", ""),
        problems=check.problems,
        **common,
    )


def config_from_settings() -> PipelineConfig:
    """The production pipeline configuration (hybrid + rerank + verification)."""
    s = get_settings()
    return PipelineConfig(
        retrieval=RetrievalConfig(
            mode="hybrid",
            top_k=s.retrieval_top_k,
            final_k=s.final_k,
            rerank=s.rerank_enabled,
            rerank_candidates=s.rerank_candidates,
        ),
        abstain_rerank_threshold=s.abstain_rerank_threshold,
        abstain_vector_threshold=s.abstain_vector_threshold,
        abstain_identifier_threshold=s.abstain_identifier_threshold,
    )
