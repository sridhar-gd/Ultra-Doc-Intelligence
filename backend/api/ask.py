# backend/api/ask.py

import logging

from fastapi import APIRouter, HTTPException, status

from agents.qa_agent import run_qa, run_qa_multi, QAResponse
from schemas.request_schemas import AskRequest
from schemas.response_schemas import AskResponse, ChunkUsed, ErrorResponse
from services.ingestion_jobs import get_batch, get_job, JobStatus
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ask", tags=["Question Answering"])


def _resolve_single_document(job_id: str) -> tuple[str, str]:
    """Resolve a ready job to (document_id, filename)."""
    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No ingestion job found for job_id='{job_id}'. "
                "Check that the job_id was returned by POST /upload."
            ),
        )

    if job.status != JobStatus.READY or not job.document_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Job '{job_id}' is not ready for querying "
                f"(current status: '{job.status.value}'). "
                "Poll GET /ingestion-status/{job_id} and retry once status='ready'."
            ),
        )

    return job.document_id, job.filename


async def _resolve_batch_documents(batch_id: str) -> list[tuple[str, str]]:
    """Resolve a batch to all ready (document_id, filename) pairs."""
    batch = get_batch(batch_id)

    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No ingestion batch found for batch_id='{batch_id}'. "
                "Check that the batch_id was returned by POST /upload."
            ),
        )

    all_jobs = [get_job(jid) for jid in batch.job_ids]
    all_jobs = [j for j in all_jobs if j is not None]

    ready_docs: list[tuple[str, str]] = [
        (job.document_id, job.filename)
        for job in all_jobs
        if job.status == JobStatus.READY and job.document_id
    ]

    if not ready_docs:
        total      = len(all_jobs)
        processing = sum(1 for j in all_jobs if j.status in (JobStatus.PENDING, JobStatus.PROCESSING))
        failed     = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Batch '{batch_id}' has {total} document(s) but none are ready for querying. "
                f"({processing} still processing, {failed} failed.) "
                "Poll GET /ingestion-status/batch/{batch_id} and retry once status='ready'."
            ),
        )

    not_ready = len(all_jobs) - len(ready_docs)
    if not_ready > 0:
        logger.warning(
            f"[/ask] batch={batch_id[:8]} — {not_ready} of {len(all_jobs)} documents "
            f"not yet ready; querying the {len(ready_docs)} that are ready."
        )

    logger.info(
        f"[/ask] batch={batch_id[:8]} resolved to "
        f"{len(ready_docs)} ready documents: "
        f"{[name for _, name in ready_docs]}"
    )

    return ready_docs


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question for a single job or a whole batch."""
    if not request.is_batch_query:
        document_id, document_name = _resolve_single_document(request.job_id)  # type: ignore[arg-type]

        logger.info(
            f"[/ask] single-doc | job_id={request.job_id[:8]} | "  # type: ignore[index]
            f"document_id={document_id[:8]} | "
            f"question='{request.question[:60]}' | top_k={request.top_k}"
        )

        try:
            qa_result: QAResponse = await run_qa(
                question=request.question,
                document_id=document_id,
                top_k=request.top_k,
                document_name=document_name,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )
        except RuntimeError as exc:
            logger.error(f"[/ask] Pipeline error (single-doc): {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Q&A pipeline failed: {str(exc)}",
            )
        except Exception as exc:
            logger.error(f"[/ask] Unexpected error (single-doc): {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during question answering.",
            )

    else:
        logger.info(
            f"[/ask] batch | batch_id={request.batch_id[:8]} | "  # type: ignore[index]
            f"question='{request.question[:60]}' | top_k_per_doc={request.top_k}"
        )

        documents = await _resolve_batch_documents(request.batch_id)  # type: ignore[arg-type]

        try:
            qa_result = await run_qa_multi(
                question=request.question,
                documents=documents,
                top_k=request.top_k,
            )
        except RuntimeError as exc:
            logger.error(f"[/ask] Pipeline error (batch): {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Q&A pipeline failed: {str(exc)}",
            )
        except Exception as exc:
            logger.error(f"[/ask] Unexpected error (batch): {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during question answering.",
            )

    chunks_used_out = [
        ChunkUsed(
            chunk_id=c.chunk_id or "",
            text=c.text,
            document_name=c.document_name,
        )
        for c in qa_result.chunks_used
    ]

    return AskResponse(
        answer=qa_result.answer,
        source_documents=qa_result.source_documents,
        confidence_score=qa_result.confidence_score,
        faithfulness=qa_result.faithfulness,
        answer_relevancy=qa_result.answer_relevancy,
        context_relevancy=qa_result.context_relevancy,
        guardrail_triggered=qa_result.guardrail_triggered,
        chunks_used=chunks_used_out,
        low_confidence_warning=qa_result.low_confidence_warning,
    )