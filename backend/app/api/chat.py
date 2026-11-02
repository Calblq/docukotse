from dataclasses import asdict

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Conversation, Message, User, Vehicle
from app.db.session import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services.generation import (
    answer_question,
    config_from_settings,
    with_conversation_vehicle,
)
from app.services.providers import (
    Embedder,
    LLMProvider,
    Reranker,
    get_embedder,
    get_llm,
    get_reranker,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMProvider = Depends(get_llm),
    reranker: Reranker | None = Depends(get_reranker),
) -> ChatResponse:
    """Answer a question from the uploaded documents, with citations.

    `abstained=true`: nothing relevant was found, so no answer is given.
    `verified=false`: the answer could not be checked against its sources; all retrieved
    excerpts are returned as citations so staff can check them by hand.
    """
    if body.vehicle_id is not None and db.get(Vehicle, body.vehicle_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Vehicle not found")

    question = body.question
    if body.conversation_id is not None:
        conversation = db.get(Conversation, body.conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if body.vehicle_id is None:
            earlier = db.scalars(
                select(Message.content)
                .where(Message.conversation_id == conversation.id, Message.role == "user")
                .order_by(Message.id.desc())
                .limit(10)
            )
            question = with_conversation_vehicle(db, question, list(earlier))
    else:
        conversation = Conversation(user_id=user.id, vehicle_id=body.vehicle_id)
        db.add(conversation)
        db.flush()

    try:
        result = answer_question(
            db,
            question,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            config=config_from_settings(),
            vehicle_id=body.vehicle_id,
            doc_type=body.doc_type,
        )
    except httpx.HTTPError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is not responding. Check that Ollama is running.",
        ) from None

    citations = [asdict(c) for c in result.citations]
    db.add(Message(conversation_id=conversation.id, role="user", content=body.question))
    reply = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=result.answer,
        citations=citations,
        verified=result.verified,
        abstained=result.abstained,
    )
    db.add(reply)
    db.commit()

    return ChatResponse(
        answer=result.answer,
        citations=citations,
        abstained=result.abstained,
        abstain_reason=result.abstain_reason,
        verified=result.verified,
        supporting_quote=result.supporting_quote,
        conversation_id=conversation.id,
        message_id=reply.id,
    )
