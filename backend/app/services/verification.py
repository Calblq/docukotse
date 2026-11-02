"""Check an LLM answer against the excerpts it cites, in code (no second LLM call).

1. The supporting quote must appear verbatim (ignoring case, spacing and punctuation)
   in a cited excerpt.
2. Every number, plate and month name in the answer must appear in the cited excerpts.
"""

import re
from dataclasses import dataclass, field

from app.services.chunking import normalize_for_match

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Plates are written in capitals ("NBC 1234"); matching lowercase too would flag "at 245" in
# "resided at 245 Example Street" as a plate.
_PLATE_RE = re.compile(r"\b[A-Z]{2,3}[\s-]?\d{3,5}\b")
_CITATION_MARK_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")  # "[1]" or "[1, 2]" in the answer
_SPACE_RE = re.compile(r"\s+")
_CODE_TOKEN_RE = re.compile(r"[A-Za-z0-9]{5,}")
_PUNCT_RE = re.compile(r"[^\w\s]|_")

# English month -> Filipino spelling. Either spelling in the excerpt satisfies the answer.
MONTHS = {
    "january": "enero", "february": "pebrero", "march": "marso", "april": "abril",
    "may": "mayo", "june": "hunyo", "july": "hulyo", "august": "agosto",
    "september": "setyembre", "october": "oktubre", "november": "nobyembre",
    "december": "disyembre",
}  # fmt: skip
_MONTH_LOOKUP = {name: en for en, fil in MONTHS.items() for name in (en, fil)}
# "may" is also a common Filipino word ("there is"), so it is not checked as a month.
_MONTH_RE = re.compile(
    r"\b(" + "|".join(n for n in _MONTH_LOOKUP if n != "may") + r")\b", re.IGNORECASE
)


@dataclass
class VerificationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def _norm_text(s: str) -> str:
    """Words only: quotes must match verbatim up to case, spacing and punctuation.

    Punctuation is layout, not content ("Chassis No / VIN" vs "Chassis No. / VIN", dot
    leaders of any length). Numbers are still checked exactly by the number check.
    """
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", s)).strip().lower()


def canonical_number(token: str) -> str:
    """'485,000.00' -> '485000', '6.50' -> '6.5', '0031875' -> '0031875'."""
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token


def numbers_in(text: str) -> set[str]:
    return {canonical_number(m) for m in _NUMBER_RE.findall(text)}


def quote_spans(quote: str) -> list[str]:
    """The quote's separately quoted spans: " | "-separated, or one per line."""
    return [p for p in re.split(r" \| |\n", quote) if p.strip()]


def quote_in_excerpts(quote: str, excerpts: list[str]) -> bool:
    """Every span of the quote (separated by " | " or line breaks) must appear verbatim in a
    cited excerpt.

    Several spans let a multi-part answer ("paid on 04/18/2019, PHP 6,418.35") quote each
    line it uses. They are checked one by one because the lines may not be adjacent in the
    excerpt.
    """
    texts = [_norm_text(e) for e in excerpts]
    parts = [_norm_text(p) for p in quote_spans(quote)]
    parts = [p for p in parts if p]
    return bool(parts) and all(any(p in t for t in texts) for p in parts)


def _month_written_as_number(month: str, text: str) -> bool:
    """'April' matches a numeric date such as 04/18/2019 or 4-18-2019 in the text."""
    number = list(MONTHS).index(_MONTH_LOOKUP[month]) + 1
    return re.search(rf"\b0?{number}[/-]\d{{1,2}}[/-]\d{{2,4}}\b", text) is not None


def _codes(text: str) -> set[str]:
    """Identifier-like tokens: 5+ characters mixing letters and digits (engine, chassis,
    VIN, policy numbers), uppercased."""
    return {
        t.upper()
        for t in _CODE_TOKEN_RE.findall(text)
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    }


def verify_answer(
    answer: str, quote: str, excerpts: list[str], question: str = ""
) -> VerificationResult:
    """Verify `answer` and `quote` against the texts of the excerpts the model cited.

    `question` lets identifiers the user typed (e.g. the plate) be repeated in the answer
    without having to appear in the quote.
    """
    if not excerpts:
        return VerificationResult(False, ["The answer cites no excerpts."])

    problems = []
    if not quote_in_excerpts(quote, excerpts):
        problems.append(
            f'The supporting quote was not found verbatim in the cited excerpts: "{quote}"'
        )

    answer_clean = _CITATION_MARK_RE.sub(" ", answer)
    joined = "\n".join(excerpts)

    # An identifier given as the answer must be in the quote itself, as a whole token.
    # Otherwise an engine no. passes when quoted from a chassis no. that ends with it.
    unquoted = sorted(_codes(answer_clean) - _codes(quote) - _codes(question))
    if unquoted:
        problems.append(
            f"These identifiers are not in the supporting quote: {', '.join(unquoted)}. Quote "
            "the line that contains the value you answer with."
        )

    missing_numbers = sorted(numbers_in(answer_clean) - numbers_in(joined))
    if missing_numbers:
        problems.append(f"Numbers not in the cited excerpts: {', '.join(missing_numbers)}")

    joined_norm = normalize_for_match(joined)
    missing_plates = sorted(
        {p.upper() for p in _PLATE_RE.findall(answer_clean)
         if normalize_for_match(p) not in joined_norm}
    )  # fmt: skip
    if missing_plates:
        problems.append(f"Plates/IDs not in the cited excerpts: {', '.join(missing_plates)}")

    joined_lower = joined.lower()
    missing_months = sorted(
        {m.lower() for m in _MONTH_RE.findall(answer_clean)
         if not any(re.search(rf"\b{n}\b", joined_lower)
                    for n in (_MONTH_LOOKUP[m.lower()], MONTHS[_MONTH_LOOKUP[m.lower()]]))
         and not _month_written_as_number(m.lower(), joined_lower)}
    )  # fmt: skip
    if missing_months:
        problems.append(f"Months not in the cited excerpts: {', '.join(missing_months)}")

    return VerificationResult(not problems, problems)
