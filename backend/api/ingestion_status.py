# backend/api/ingestion_status.py

import logging

from fastapi import APIRouter, HTTPException, status

from schemas.response_schemas import BatchStatusResponse, ErrorResponse, JobStatusResponse
from services.ingestion_jobs import get_batch, get_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion-status", tags=["Ingestion Status"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get status of a single ingestion job",
    description=(
        "Returns the current status of one document ingestion job. "
        "Poll until `status` is `ready` (success) or `failed` (error). "
        "When `status=ready` the `document_id` field is populated and can "
        "be used in POST /ask and POST /extract."
    ),
    responses={
        200: {"description": "Job found — check `status` field"},
        404: {"model": ErrorResponse, "description": "No job found for the given job_id"},
    },
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingestion job found for job_id='{job_id}'.",
        )
    return JobStatusResponse(**job.model_dump())


@router.get(
    "/batch/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Get status of all jobs in an upload batch",
    description=(
        "Returns aggregated progress for every file submitted in a single "
        "POST /upload batch. The `ready` + `failed` counts can be compared "
        "against `total` to determine when the batch has finished processing."
    ),
    responses={
        200: {"description": "Batch found — check per-job statuses"},
        404: {"model": ErrorResponse, "description": "No batch found for the given batch_id"},
    },
)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ingestion batch found for batch_id='{batch_id}'.",
        )

    jobs = []
    for jid in batch.job_ids:
        job = get_job(jid)
        if job:
            jobs.append(JobStatusResponse(**job.model_dump()))

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        total=batch.total,
        pending=batch.pending,
        processing=batch.processing,
        ready=batch.ready,
        failed=batch.failed,
        jobs=jobs,
        created_at=batch.created_at,
    )