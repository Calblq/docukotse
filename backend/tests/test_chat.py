import json

import pytest
from sqlalchemy import func, select

from app.db.models import Conversation, Message

GOOD = {
    "answerable": True,
    "answer": "Maria Santos bought it for PHP 485,000.00.",
    "used_excerpts": [1],
    "supporting_quote": "Buyer: Maria Santos",
}
NOT_ANSWERABLE = {"answerable": False, "answer": "", "used_excerpts": [], "supporting_quote": ""}
DEED_SCOPE = {"doc_type": "deed_of_sale"}


def ask(client, headers, question, **extra):
    resp = client.post("/chat", json={"question": question, **extra}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_grounded_answer_is_verified_with_citation(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [GOOD]
    body = ask(client, staff_headers, "Who bought the Vios and for how much?", **DEED_SCOPE)
    assert body["verified"] is True
    assert body["abstained"] is False
    assert body["answer"] == GOOD["answer"]
    assert [(c["filename"], c["page"]) for c in body["citations"]] == [("vios_deed.pdf", 1)]
    assert "Maria Santos" in body["citations"][0]["snippet"]
    assert len(fake_llm.calls) == 1


def test_prompt_contains_numbered_excerpts_with_sources(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [GOOD]
    ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    system, user = fake_llm.calls[0]
    assert "ONLY" in system["content"]
    assert "[1] (file: vios_deed.pdf, page 1)" in user["content"]
    assert "Question: Who bought the Vios?\nAnswer every part of the question." in user["content"]


def test_hallucinated_number_triggers_retry(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [{**GOOD, "answer": "Maria Santos paid PHP 500,000."}, GOOD]
    body = ask(client, staff_headers, "Who bought the Vios and for how much?", **DEED_SCOPE)
    assert body["verified"] is True
    assert body["answer"] == GOOD["answer"]
    assert len(fake_llm.calls) == 2
    retry_prompt = fake_llm.calls[1][-1]["content"]
    assert "failed verification" in retry_prompt
    assert "500000" in retry_prompt


def test_failing_twice_returns_unverified_with_all_chunks(client, staff_headers, corpus, fake_llm):
    bad = {**GOOD, "supporting_quote": "The buyer paid in cash"}
    fake_llm.replies = [bad, bad]
    body = ask(client, staff_headers, "Who signed the deed for the NBC 1234 Vios?")
    assert body["verified"] is False
    assert body["abstained"] is False
    assert len(body["citations"]) >= 2  # every retrieved chunk, for manual checking
    assert len(fake_llm.calls) == 2


def test_invalid_json_is_unverified(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = ["not json", "still not json"]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["verified"] is False


def test_model_saying_not_answerable_abstains(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [NOT_ANSWERABLE]
    body = ask(client, staff_headers, "Who was the buyer of the Vios' lawyer?", **DEED_SCOPE)
    assert body["abstained"] is True
    assert body["abstain_reason"] == "llm_not_answerable"
    assert body["citations"] == []


def test_low_retrieval_confidence_abstains_without_llm_call(
    client, staff_headers, corpus, fake_llm
):
    body = ask(client, staff_headers, "xyzzy plugh frobnicate")
    assert body["abstained"] is True
    assert body["abstain_reason"] == "low_confidence"
    assert "couldn't find" in body["answer"]
    assert fake_llm.calls == []


def test_question_naming_a_vehicle_only_searches_its_documents(
    client, staff_headers, corpus, fake_llm
):
    # Only the City has an insurance policy. Scoped to the Mirage, the City's policy never
    # reaches the model, so it cannot be returned as the Mirage's.
    fake_llm.replies = [NOT_ANSWERABLE]
    body = ask(client, staff_headers, "What is the insurance policy number of the Mirage?")
    assert body["abstained"] is True
    prompts = " ".join(m["content"] for call in fake_llm.calls for m in call)
    assert "city_insurance.pdf" not in prompts
    assert "SMI-MC-2024-0057210" not in prompts


def test_messages_are_stored_in_a_conversation(client, staff_headers, corpus, fake_llm, db):
    fake_llm.replies = [GOOD, GOOD]
    first = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    conv_id = first["conversation_id"]
    second = ask(client, staff_headers, "Who bought it?", conversation_id=conv_id, **DEED_SCOPE)
    assert second["conversation_id"] == conv_id
    assert db.scalar(select(func.count()).select_from(Conversation)) == 1
    stored = db.scalars(select(Message).order_by(Message.id)).all()
    assert [m.role for m in stored] == ["user", "assistant", "user", "assistant"]
    assert stored[1].verified is True
    assert stored[1].citations[0]["filename"] == "vios_deed.pdf"


def test_other_users_conversation_is_not_found(
    client, staff_headers, admin_headers, corpus, fake_llm
):
    fake_llm.replies = [GOOD]
    conv = ask(client, admin_headers, "Who bought the Vios?", **DEED_SCOPE)["conversation_id"]
    resp = client.post(
        "/chat", json={"question": "hi", "conversation_id": conv}, headers=staff_headers
    )
    assert resp.status_code == 404


def test_unknown_vehicle_filter_is_404(client, staff_headers, corpus):
    resp = client.post("/chat", json={"question": "hi", "vehicle_id": 999}, headers=staff_headers)
    assert resp.status_code == 404


def test_chat_requires_login(client, db):
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_refusal_written_as_an_answer_counts_as_abstaining(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [
        {**GOOD, "answer": "The documents do not provide information about the buyer."}
    ]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["abstained"] is True
    assert body["citations"] == []


def test_llm_outage_returns_503(client, staff_headers, corpus, fake_llm):
    import httpx

    def down(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    fake_llm.generate = down
    resp = client.post(
        "/chat", json={"question": "Who bought the Vios?", **DEED_SCOPE}, headers=staff_headers
    )
    assert resp.status_code == 503
    assert "Ollama" in resp.json()["detail"]


def test_citations_from_photos_carry_ocr_confidence(client, staff_headers, corpus, fake_llm, db):
    from app.db.models import Chunk

    deed = corpus["docs"]["vios_deed.pdf"]
    db.query(Chunk).filter(Chunk.document_id == deed.id).update({"ocr_confidence": 81.5})
    db.commit()
    fake_llm.replies = [GOOD]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["citations"][0]["ocr_confidence"] == 81.5
    assert "read by OCR" in fake_llm.calls[0][1]["content"]


@pytest.mark.parametrize(
    ("question", "filipino"),
    [
        ("Magkano ang bentahan ng Almera?", True),
        ("Sino ang bumili ng Vios?", True),
        ("Ilang buwan na ang nabayaran?", True),
        ("Who bought the Vios?", False),
        ("What is the OR number for NBC 1234?", False),
    ],
)
def test_looks_filipino(question, filipino):
    from app.services.generation import looks_filipino

    assert looks_filipino(question) is filipino


def test_taglish_question_is_also_searched_in_english(client, staff_headers, corpus, fake_llm):
    # First LLM call translates the question; the second answers it.
    fake_llm.replies = ["Who bought the Vios and for how much?", GOOD]
    body = ask(client, staff_headers, "Sino ang bumili ng Vios at magkano?", **DEED_SCOPE)
    assert body["verified"] is True
    translate_call, answer_call = fake_llm.calls
    assert "into natural English" in translate_call[0]["content"]
    # The model still answers the original question, not the translation.
    prompt = answer_call[1]["content"]
    assert "Question: Sino ang bumili ng Vios at magkano?" in prompt
    # The translation is shown too, to help the model read Taglish.
    assert "Who bought the Vios and for how much?" in prompt  # the English translation


def test_translation_failure_falls_back_to_the_original_question(
    client, staff_headers, corpus, fake_llm
):
    import httpx

    replies = iter([httpx.ConnectError("down"), json.dumps(GOOD)])

    def flaky(messages, **kwargs):
        fake_llm.calls.append(messages)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    fake_llm.generate = flaky
    body = ask(client, staff_headers, "Sino ang bumili ng Vios?", **DEED_SCOPE)
    assert body["verified"] is True


class UnsureReranker:
    """Scores every chunk low, like the real reranker on terse OCR'd form text."""

    def score(self, query, texts):
        return [0.1] * len(texts)


def test_named_plate_found_in_top_chunk_lowers_the_abstain_bar(
    client, staff_headers, corpus, fake_llm
):
    from app.main import app
    from app.services.providers import get_reranker

    app.dependency_overrides[get_reranker] = lambda: UnsureReranker()
    fake_llm.replies = [
        {**GOOD, "answer": "Engine No.: 2NR1234567", "supporting_quote": "Engine No.: 2NR1234567"}
    ]
    # Plate in the question, and in the best chunk: answered despite the low score.
    body = ask(client, staff_headers, "What is the engine number of NBC-1234?")
    assert body["abstained"] is False
    assert len(fake_llm.calls) == 1
    # No identifier: the normal threshold applies and it abstains without calling the LLM.
    body = ask(client, staff_headers, "Who notarized the Vios deed?")
    assert body["abstained"] is True
    assert body["abstain_reason"] == "low_confidence"
    assert len(fake_llm.calls) == 1


def test_not_answerable_with_real_quoted_evidence_is_asked_again(
    client, staff_headers, corpus, fake_llm
):
    contradictory = {
        **NOT_ANSWERABLE,
        "used_excerpts": [1],
        "supporting_quote": "Buyer: Maria Santos",
    }
    fake_llm.replies = [contradictory, GOOD]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["abstained"] is False
    assert body["verified"] is True
    assert "You set answerable to false" in fake_llm.calls[1][-1]["content"]


def test_not_answerable_with_invented_quote_abstains_at_once(
    client, staff_headers, corpus, fake_llm
):
    invented = {**NOT_ANSWERABLE, "used_excerpts": [1], "supporting_quote": "Buyer: Pedro Cruz"}
    fake_llm.replies = [invented]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["abstained"] is True
    assert len(fake_llm.calls) == 1


def test_buyer_answers_must_come_from_a_deed_of_sale(client, staff_headers, corpus, fake_llm):
    # The only excerpt is the Vios OR/CR: its registered owner is not a buyer.
    owner_as_buyer = {**GOOD, "answer": "NBC 1234", "supporting_quote": "Plate No.: NBC 1234"}
    fake_llm.replies = [owner_as_buyer]
    body = ask(client, staff_headers, "Who bought the Vios?", doc_type="or_cr")
    assert body["abstained"] is True
    assert body["abstain_reason"] == "llm_not_answerable"
    # From the deed of sale the same question is answered.
    fake_llm.replies = [GOOD]
    body = ask(client, staff_headers, "Who bought the Vios?", **DEED_SCOPE)
    assert body["abstained"] is False


def test_quote_that_only_repeats_the_plate_is_asked_again_from_scratch(
    client, staff_headers, corpus, fake_llm
):
    restated = {
        **NOT_ANSWERABLE,
        "used_excerpts": [1],
        "supporting_quote": "Plate No.: NBC 1234",
    }
    engine = {
        "answerable": True,
        "answer": "2NR1234567",
        "used_excerpts": [1],
        "supporting_quote": "Engine No.: 2NR1234567",
    }
    fake_llm.replies = [restated, engine]
    body = ask(client, staff_headers, "What is the engine number of NBC 1234?", doc_type="or_cr")
    assert body["abstained"] is False
    assert body["verified"] is True
    retry = fake_llm.calls[1]
    assert len(retry) == 2  # system + question, not a follow-up to the first answer
    assert "only repeats what the question already says" in retry[-1]["content"]


@pytest.mark.parametrize(
    ("quote", "answer", "answerable", "restates"),
    [
        ("PLATE NO.: XYZ7788", "", False, True),
        ("PLATE NO.: XYZ7788", "XYZ7788", True, True),
        ("PLATE NO.: XYZ7788", "A Nissan Patrol.", True, False),
        ("MAKE/BRAND: NISSAN\nSERIES: PATROL 4X4", "A Nissan Patrol.", True, False),
        ("PLATE NO.: XYZ7788 | MAKE/BRAND: NISSAN", "", False, False),
    ],
)
def test_only_restates_question(quote, answer, answerable, restates):
    from app.services.answer_checks import only_restates_question

    data = {"supporting_quote": quote, "answer": answer, "answerable": answerable}
    assert only_restates_question(data, "what car is xyz 7788?") is restates


def test_how_much_answered_without_an_amount_is_retried(client, staff_headers, corpus, fake_llm):
    date_only = {
        "answerable": True,
        "answer": "It was sold on September 18, 2024.",
        "used_excerpts": [1],
        "supporting_quote": "Date of sale: September 18, 2024",
    }
    with_price = {
        "answerable": True,
        "answer": "On September 18, 2024, for PHP 485,000.00.",
        "used_excerpts": [1],
        "supporting_quote": "Price: PHP 485,000.00 | Date of sale: September 18, 2024",
    }
    fake_llm.replies = [date_only, with_price]
    body = ask(client, staff_headers, "When was the Vios sold and how much?", **DEED_SCOPE)
    assert body["verified"] is True
    assert "485,000.00" in body["answer"]
    assert "asks how much" in fake_llm.calls[1][-1]["content"]


def test_registration_status_without_expiry_date_is_answered_in_code(
    client, staff_headers, corpus, fake_llm
):
    # The Mirage's OR/CR has no "valid until" line: the app says so, without asking the LLM.
    body = ask(client, staff_headers, "Is the Mirage DAE 9012 still registered?")
    assert body["abstained"] is False
    assert "don't show when this registration expires" in body["answer"]
    assert [c["filename"] for c in body["citations"]] == ["mirage_orcr.pdf"]
    assert fake_llm.calls == []


def test_registration_status_with_expiry_date_goes_to_the_llm(
    client, staff_headers, corpus, fake_llm
):
    expired = {
        "answerable": True,
        "answer": "No, it expired on March 31, 2025.",
        "used_excerpts": [1],
        "supporting_quote": "Registration valid until: March 31, 2025",
    }
    fake_llm.replies = [expired]
    body = ask(client, staff_headers, "Is the Vios NBC 1234 still registered?", doc_type="or_cr")
    assert body["verified"] is True
    assert "March 31, 2025" in body["answer"]
    assert "Today is" in fake_llm.calls[0][-1]["content"]


def test_expiry_claim_without_an_expiry_line_fails_verification(
    client, staff_headers, corpus, fake_llm
):
    issue_date_as_expiry = {
        "answerable": True,
        "answer": "Yes, it is valid until 8820391457.",
        "used_excerpts": [1],
        "supporting_quote": "OR No.: 8820391457",
    }
    fake_llm.replies = [issue_date_as_expiry, issue_date_as_expiry]
    body = ask(client, staff_headers, "Is the Vios NBC 1234 still registered?", doc_type="or_cr")
    assert body["verified"] is False


def test_follow_up_without_a_vehicle_uses_the_one_asked_about_before(
    client, staff_headers, corpus, fake_llm, db
):
    engine = {
        "answerable": True,
        "answer": "2NR1234567",
        "used_excerpts": [1],
        "supporting_quote": "Engine No.: 2NR1234567",
    }
    fake_llm.replies = [GOOD, engine]
    first = ask(client, staff_headers, "Who bought the Vios NBC 1234?", **DEED_SCOPE)
    second = ask(
        client,
        staff_headers,
        "What is its engine number?",
        conversation_id=first["conversation_id"],
        doc_type="or_cr",
    )
    assert second["abstained"] is False
    assert "(about Toyota Vios NBC 1234)" in fake_llm.calls[1][-1]["content"]
    # The conversation keeps the question as typed.
    stored = db.scalars(select(Message.content).where(Message.role == "user")).all()
    assert stored[-1] == "What is its engine number?"


def test_follow_up_naming_another_vehicle_is_not_changed(client, staff_headers, corpus, fake_llm):
    fake_llm.replies = [GOOD, NOT_ANSWERABLE]
    first = ask(client, staff_headers, "Who bought the Vios NBC 1234?", **DEED_SCOPE)
    ask(
        client,
        staff_headers,
        "Who owns DAE 9012?",
        conversation_id=first["conversation_id"],
    )
    assert "(about" not in fake_llm.calls[-1][-1]["content"]
