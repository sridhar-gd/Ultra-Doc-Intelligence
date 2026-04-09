# backend/agents/qa_agent.py

import asyncio
import logging

import anthropic
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from services.embedder import rerank_chunks as cross_encoder_rerank
from services.guardrails import (
    check_similarity_gate,
    evaluate_answer,
    GuardrailTriggered,
    is_refusal,
)
from prompts.qa_system_prompt import (
    QA_SYSTEM_PROMPT,
    format_chunks_for_prompt,
    build_qa_user_message,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChunkUsedOutput(BaseModel):
    """
    A single retrieved chunk returned in the API response.

    Fields:
        chunk_id:      Internal UUID of the chunk (for traceability/debugging).
        text:          Actual document text content — either the AI-contextualised
                       version (contextual_text from Contextualizer.py, preferred)
                       or the raw extracted chunk_text from the PDF (fallback).
                       This is ALWAYS real document content, never metadata.
        document_name: Name of the source document this chunk belongs to.
                       Always set for batch (multi-doc) queries.
                       Set for single-doc queries when document_name was provided.
    """
    chunk_id:      str | None    = None
    text:          str
    document_name: str | None    = None


class QAResponse(BaseModel):
    """
    Structured Q&A response returned to the /ask API endpoint.

    Confidence sub-scores (faithfulness, answer_relevancy, context_relevancy)
    are included so callers can see the breakdown and build their own UI indicators.

    Key field notes:
        confidence_score:
            Composite quality score (0–1). Weights are query-type aware.
            Single-doc:  0.55×faithfulness + 0.25×answer_relevancy + 0.20×context_relevancy
            Cross-doc:   0.60×faithfulness + 0.20×answer_relevancy + 0.20×context_relevancy
            Expected range for correct answers: 0.88–0.95

        faithfulness:
            Fraction of answer claims supported by retrieved context (0–1).
            1.0 = fully grounded. 0.0 = hallucination detected or refusal.

        answer_relevancy  (claim coverage):
            Whether the answer is responsive to the question and contains a
            concrete finding. Mean of two binary LLM judgments (responsive +
            concrete). Handles comparison queries correctly — not subject to
            embedding-space drift from comparison verbs.
            1.0 = responsive and concrete. 0.0 = vague or refusal.

        context_relevancy:
            Single-doc: sigmoid(rerank_score) from BAAI cross-encoder (0–1).
            Cross-doc:  source coverage — fraction of expected documents
            represented in the retrieved chunks (0–1).

        source_documents:
            Document names for grounded answers; None when guardrail_triggered.
    """
    answer:              str
    source_documents:    list[str] | None   = None
    confidence_score:    float              = Field(ge=0.0, le=1.0)
    faithfulness:        float              = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevancy:    float              = Field(default=0.0, ge=0.0, le=1.0)
    context_relevancy:   float              = Field(default=0.0, ge=0.0, le=1.0)
    guardrail_triggered: bool               = False
    chunks_used:         list[ChunkUsedOutput] = Field(default_factory=list)
    low_confidence_warning: bool            = False


qa_agent = Agent(
    model=AnthropicModel(
        model_name=settings.anthropic_model,
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    ),
    tools=[],
    system_prompt=(
        "You are Ultra Doc-Intelligence, a logistics document Q&A assistant. "
        "Answer questions strictly from provided context chunks. "
        "If no relevant chunks are found, respond: 'Not found in document.' "
        "Never use external knowledge. Always cite the source section."
    ),
    retries=2,
)


def _format_chunks_multi_doc(chunks: list[dict]) -> str:
    """Format cross-document chunks with source labels for prompting."""
    parts: list[str] = []
    for chunk in chunks:
        doc_label   = chunk.get("document_name") or "Document"
        chunk_text  = chunk.get("contextual_text") or chunk.get("chunk_text", "")
        parts.append(f"[SOURCE: {doc_label}]\n{chunk_text}")
    return "\n\n".join(parts)


def _call_claude_for_answer(
    question: str,
    chunks: list[dict],
    multi_doc: bool = False,
) -> str:
    """Generate a grounded answer from retrieved chunks using Claude."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if multi_doc:
        formatted_context = _format_chunks_multi_doc(chunks)
        system_prompt = (
            QA_SYSTEM_PROMPT.format(context_chunks=formatted_context)
            + "\n\n"
            + "IMPORTANT: The context above contains chunks from MULTIPLE documents. "
            "Each chunk is prefixed with [SOURCE: <document_name>]. "
            "When comparing values across documents, always cite the source document "
            "for each value so the answer is fully attributed. "
            "If a value differs between documents, state both values and their sources explicitly."
        )
    else:
        formatted_context = format_chunks_for_prompt(chunks)
        system_prompt     = QA_SYSTEM_PROMPT.format(context_chunks=formatted_context)

    user_message = build_qa_user_message(question)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()


async def run_qa(
    question: str,
    document_id: str,
    top_k: int = 3,
    document_name: str | None = None,
) -> QAResponse:
    """Run single-document RAG question answering."""
    logger.info(
        f"[QA] question='{question[:80]}' | "
        f"doc={document_id[:8]} | top_k={top_k}"
    )

    from services.retriever import hybrid_retrieve

    try:
        retrieved_chunks = await hybrid_retrieve(
            query=question,
            document_id=document_id,
            top_k_final=top_k,
            apply_reranking=True,
        )
    except Exception as exc:
        logger.error(f"[QA] Retrieval failed: {exc}")
        raise RuntimeError(f"Retrieval failed: {exc}") from exc

    if not retrieved_chunks:
        logger.warning(f"[QA] No chunks retrieved for doc={document_id[:8]}")
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
        )

    top_chunk      = retrieved_chunks[0]
    top_similarity = float(top_chunk.get("similarity", 0.0))

    try:
        check_similarity_gate(
            top_similarity=top_similarity,
            top_chunk=top_chunk,
        )
    except GuardrailTriggered as gt:
        logger.warning(f"[QA] Guardrail triggered: {gt.reason}")
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
        )

    try:
        answer = await asyncio.to_thread(
            _call_claude_for_answer,
            question,
            retrieved_chunks,
            False,
        )
    except Exception as exc:
        logger.error(f"[QA] Claude answer generation failed: {exc}")
        raise RuntimeError(f"Answer generation failed: {exc}") from exc

    try:
        scoring = evaluate_answer(
            answer=answer,
            chunks=retrieved_chunks,
            question=question,
        )
    except Exception as exc:
        logger.error(f"[QA] Confidence scoring failed: {exc}")
        scoring = {
            "confidence_score":       0.5,
            "faithfulness":           0.5,
            "answer_relevancy":       0.5,
            "context_relevancy":      0.5,
            "low_confidence_warning": False,
        }

    if is_refusal(answer):
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
            low_confidence_warning=True,
        )

    doc_label = document_name or top_chunk.get("section_heading") or "Document"

    chunks_used_out = [
        ChunkUsedOutput(
            chunk_id=c.get("chunk_id"),
            text=c.get("contextual_text") or c.get("chunk_text", ""),
            document_name=document_name,
        )
        for c in retrieved_chunks
    ]

    logger.info(
        f"[QA] ✅ Done | "
        f"confidence={scoring['confidence_score']} | "
        f"faithfulness={scoring['faithfulness']} | "
        f"claim_coverage={scoring['answer_relevancy']} | "
        f"context={scoring['context_relevancy']}"
    )

    return QAResponse(
        answer=answer,
        source_documents=[doc_label],
        confidence_score=scoring["confidence_score"],
        faithfulness=scoring["faithfulness"],
        answer_relevancy=scoring["answer_relevancy"],
        context_relevancy=scoring["context_relevancy"],
        guardrail_triggered=False,
        chunks_used=chunks_used_out,
        low_confidence_warning=scoring["low_confidence_warning"],
    )


_MAX_TOTAL_CHUNKS = 10


async def _retrieve_for_document(
    question: str,
    document_id: str,
    document_name: str,
    top_k: int,
) -> list[dict]:
    """Retrieve chunks for one document and tag them with document name."""
    from services.retriever import hybrid_retrieve

    try:
        chunks = await hybrid_retrieve(
            query=question,
            document_id=document_id,
            top_k_final=top_k,
            apply_reranking=True,
        )
        for chunk in chunks:
            chunk["document_name"] = document_name
        logger.debug(
            f"[QA-multi] Retrieved {len(chunks)} chunks from "
            f"doc={document_id[:8]} ({document_name})"
        )
        return chunks
    except Exception as exc:
        logger.warning(
            f"[QA-multi] Retrieval failed for doc={document_id[:8]} "
            f"({document_name}): {exc}"
        )
        return []


async def run_qa_multi(
    question: str,
    documents: list[tuple[str, str]],
    top_k: int = 3,
) -> QAResponse:
    """Run cross-document RAG question answering for a batch."""
    doc_names = [name for _, name in documents]
    logger.info(
        f"[QA-multi] question='{question[:80]}' | "
        f"docs={len(documents)} | top_k_per_doc={top_k} | "
        f"files={doc_names}"
    )

    retrieval_tasks = [
        _retrieve_for_document(
            question=question,
            document_id=doc_id,
            document_name=doc_name,
            top_k=top_k,
        )
        for doc_id, doc_name in documents
    ]
    per_doc_results: list[list[dict]] = await asyncio.gather(*retrieval_tasks)

    merged_chunks: list[dict] = []
    for chunks in per_doc_results:
        merged_chunks.extend(chunks)

    logger.info(
        f"[QA-multi] Merged {len(merged_chunks)} chunks from "
        f"{len(documents)} documents before global re-rank"
    )

    if not merged_chunks:
        logger.warning("[QA-multi] No chunks retrieved from any document in batch")
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
        )

    final_chunks = await cross_encoder_rerank(
        query=question,
        chunks=merged_chunks,
        top_k=min(_MAX_TOTAL_CHUNKS, len(merged_chunks)),
    )

    top_chunk      = final_chunks[0]
    top_similarity = float(top_chunk.get("similarity", 0.0))

    try:
        check_similarity_gate(
            top_similarity=top_similarity,
            top_chunk=top_chunk,
        )
    except GuardrailTriggered as gt:
        logger.warning(f"[QA-multi] Guardrail triggered: {gt.reason}")
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
        )

    try:
        answer = await asyncio.to_thread(
            _call_claude_for_answer,
            question,
            final_chunks,
            True,
        )
    except Exception as exc:
        logger.error(f"[QA-multi] Claude answer generation failed: {exc}")
        raise RuntimeError(f"Answer generation failed: {exc}") from exc

    try:
        scoring = evaluate_answer(
            answer=answer,
            chunks=final_chunks,
            question=question,
        )
    except Exception as exc:
        logger.error(f"[QA-multi] Confidence scoring failed: {exc}")
        scoring = {
            "confidence_score":       0.5,
            "faithfulness":           0.5,
            "answer_relevancy":       0.5,
            "context_relevancy":      0.5,
            "low_confidence_warning": False,
        }

    if is_refusal(answer):
        return QAResponse(
            answer="Not found in document.",
            source_documents=None,
            confidence_score=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_relevancy=0.0,
            guardrail_triggered=True,
            chunks_used=[],
            low_confidence_warning=True,
        )

    chunks_used_out = [
        ChunkUsedOutput(
            chunk_id=c.get("chunk_id"),
            text=c.get("contextual_text") or c.get("chunk_text", ""),
            document_name=c.get("document_name"),
        )
        for c in final_chunks
    ]

    logger.info(
        f"[QA-multi] ✅ Done | "
        f"confidence={scoring['confidence_score']} | "
        f"faithfulness={scoring['faithfulness']} | "
        f"claim_coverage={scoring['answer_relevancy']} | "
        f"source_coverage={scoring['context_relevancy']} | "
        f"chunks_used={len(chunks_used_out)}"
    )

    return QAResponse(
        answer=answer,
        source_documents=doc_names,
        confidence_score=scoring["confidence_score"],
        faithfulness=scoring["faithfulness"],
        answer_relevancy=scoring["answer_relevancy"],
        context_relevancy=scoring["context_relevancy"],
        guardrail_triggered=False,
        chunks_used=chunks_used_out,
        low_confidence_warning=scoring["low_confidence_warning"],
    )
