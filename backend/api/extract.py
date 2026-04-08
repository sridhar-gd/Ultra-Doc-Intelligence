# backend/api/extract.py

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from agents.extraction_agent import run_extraction
from schemas.request_schemas import ExtractRequest
from schemas.response_schemas import (
    BatchExtractionResponse,
    BatchExtractionResult,
    ErrorResponse,
)
from schemas.shipment_schema import ShipmentData
from services.ingestion_jobs import JobStatus, get_batch, get_job
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/extract", tags=["Structured Extraction"])


def _resolve_job_for_extraction(job_id: str) -> tuple[str, str]:
    """
    Resolve a job ID to its document ID and filename.
    Only jobs in `ready` state are accepted for extraction.
    """
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
                f"Job '{job_id}' is not ready for extraction "
                f"(current status: '{job.status.value}'). "
                "Poll GET /ingestion-status/{job_id} and retry once status='ready'."
            ),
        )

    return job.document_id, job.filename


async def _extract_one_safe(
    job_id: str,
    document_id: str,
    filename: str,
) -> BatchExtractionResult:
    """
    Run extraction for a single document and return a result object.
    Exceptions are captured and returned as failed statuses.
    """
    try:
        shipment: ShipmentData = await run_extraction(
            document_id=document_id,
            document_name=filename,
        )
        return BatchExtractionResult(
            job_id=job_id,
            document_id=document_id,
            filename=filename,
            status="success",
            shipment_data=shipment.model_dump(mode="json"),
        )
    except Exception as exc:
        logger.error(
            f"[/extract/batch] Extraction failed | job={job_id} | "
            f"doc={document_id[:8]} | file='{filename}' | error={exc}",
            exc_info=True,
        )
        return BatchExtractionResult(
            job_id=job_id,
            document_id=document_id,
            filename=filename,
            status="failed",
            error=str(exc),
        )


@router.post(
    "",
    response_model=ShipmentData,
    status_code=status.HTTP_200_OK,
    responses={
        200: {},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def extract_shipment_data(request: ExtractRequest) -> ShipmentData:
    """
    Extract structured shipment fields for one ready ingestion job.
    The endpoint resolves document identity from `job_id`.
    """
    document_id, document_name = _resolve_job_for_extraction(request.job_id)

    logger.info(
        f"[/extract] job_id={request.job_id[:8]} | "
        f"document_id={document_id[:8]} | name={document_name!r}"
    )

    try:
        return await run_extraction(
            document_id=document_id,
            document_name=document_name,
        )

    except ValueError as exc:

        detail = str(exc)
        logger.error(f"[/extract] Validation error: {detail}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )

    except RuntimeError as exc:
        logger.error(f"[/extract] Pipeline error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction pipeline failed: {str(exc)}",
        )

    except Exception as exc:
        logger.error(f"[/extract] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during field extraction.",
        )


@router.post(
    "/batch",
    response_model=BatchExtractionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def extract_batch(batch_id: str) -> BatchExtractionResponse:
    """
    Extract shipment fields for every ready document in a batch.
    Not-ready jobs are returned as skipped.
    """
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No upload batch found for batch_id='{batch_id}'.",
        )

    logger.info(
        f"[/extract/batch] batch={batch_id} | total_jobs={batch.total} | "
        f"ready={batch.ready} | pending={batch.pending} | failed={batch.failed}"
    )

    to_extract: list[tuple[str, str, str]] = [] 
    skipped_results: list[BatchExtractionResult] = []

    for job_id in batch.job_ids:
        job = get_job(job_id)
        if not job:
            continue

        if job.status == JobStatus.READY and job.document_id:
            to_extract.append((job.job_id, job.document_id, job.filename))
        else:
            reason = (
                f"Ingestion job status is '{job.status.value}' — "
                "only 'ready' documents can be extracted."
            )
            skipped_results.append(
                BatchExtractionResult(
                    job_id=job.job_id,
                    document_id=job.document_id,
                    filename=job.filename,
                    status="skipped",
                    error=reason,
                )
            )

    semaphore = asyncio.Semaphore(settings.ingestion_concurrency)

    async def _guarded(job_id: str, document_id: str, filename: str) -> BatchExtractionResult:
        async with semaphore:
            return await _extract_one_safe(job_id, document_id, filename)

    extracted_results: list[BatchExtractionResult] = []
    if to_extract:
        extracted_results = list(
            await asyncio.gather(*[_guarded(jid, did, fn) for jid, did, fn in to_extract])
        )

    all_results = extracted_results + skipped_results

    job_order = {jid: idx for idx, jid in enumerate(batch.job_ids)}
    all_results.sort(key=lambda r: job_order.get(r.job_id, 999))

    succeeded = sum(1 for r in all_results if r.status == "success")
    failed    = sum(1 for r in all_results if r.status == "failed")
    skipped   = sum(1 for r in all_results if r.status == "skipped")

    logger.info(
        f"[/extract/batch] batch={batch_id} complete | "
        f"succeeded={succeeded} failed={failed} skipped={skipped}"
    )

    return BatchExtractionResponse(
        batch_id=batch_id,
        total=len(all_results),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        results=all_results,
    )