# backend/api/upload.py

import asyncio
import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from agents.ingestion_agent import run_ingestion
from schemas.response_schemas import BatchUploadResponse, ErrorResponse, JobSummary
from services.ingestion_jobs import (
    JobStatus,
    create_batch,
    create_job,
    update_job,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/upload", tags=["Document Ingestion"])


def _validate_file(file: UploadFile) -> None:
    """Validate filename and extension for one uploaded file."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more uploaded files has no filename.",
        )

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '.{ext}' for '{file.filename}'. "
                f"Allowed: {', '.join(settings.allowed_extensions)}"
            ),
        )


def _compute_sha256(content: bytes) -> str:
    """Return the SHA-256 hex digest for file bytes."""
    return hashlib.sha256(content).hexdigest()


async def _read_and_save_temp(file: UploadFile) -> tuple[str, str]:
    """Read bytes, compute hash, and save a temp file."""
    content = await file.read()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"'{file.filename}' ({len(content) // (1024 * 1024)} MB) exceeds "
                f"the {settings.max_upload_size_mb} MB limit."
            ),
        )

    file_hash = _compute_sha256(content)

    ext = Path(file.filename).suffix.lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")

    with open(temp_path, "wb") as f:
        f.write(content)

    logger.debug(
        f"Saved '{file.filename}' → {temp_path} "
        f"({len(content)} bytes, sha256={file_hash[:12]}…)"
    )
    return temp_path, file_hash


# Background worker

async def _ingest_one(
    job_id: str,
    temp_path: str,
    filename: str,
    file_hash: str,
) -> None:
    """Ingest one file and update its job status."""
    update_job(job_id, status=JobStatus.PROCESSING)
    try:
        result = await run_ingestion(
            file_path=temp_path,
            document_name=filename,
            file_hash=file_hash,
        )

        update_job(
            job_id,
            status=JobStatus.READY,
            document_id=result.document_id,
            chunks_created=result.chunks_created,
        )

        if result.cached:
            logger.info(
                f"[Ingestion] ♻️  job={job_id} | file='{filename}' | "
                f"CACHED → doc_id={result.document_id}"
            )
        else:
            logger.info(
                f"[Ingestion] ✅ job={job_id} | file='{filename}' | "
                f"doc_id={result.document_id} | chunks={result.chunks_created}"
            )
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error=str(exc))
        logger.error(
            f"[Ingestion] ❌ job={job_id} | file='{filename}' | error={exc}",
            exc_info=True,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except OSError:
                pass


async def _ingest_batch(jobs: list[tuple[str, str, str, str]]) -> None:
    """Ingest a batch with bounded parallelism."""
    semaphore = asyncio.Semaphore(settings.ingestion_concurrency)

    async def _guarded(job_id: str, temp_path: str, filename: str, file_hash: str) -> None:
        async with semaphore:
            await _ingest_one(job_id, temp_path, filename, file_hash)

    await asyncio.gather(*[_guarded(jid, tp, fn, fh) for jid, tp, fn, fh in jobs])


@router.post(
    "/",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> BatchUploadResponse:
    """Queue a batch upload and start background ingestion."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided.",
        )

    if len(files) > settings.max_files_per_batch:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Too many files: {len(files)} submitted, "
                f"maximum per batch is {settings.max_files_per_batch}."
            ),
        )

    for file in files:
        _validate_file(file)

    saved: list[tuple[str, str, str]] = []
    for file in files:
        temp_path, file_hash = await _read_and_save_temp(file)
        saved.append((temp_path, file.filename, file_hash))

    # Create job registry entries
    job_objects = [create_job(batch_id="__pending__", filename=fn) for _, fn, _ in saved]

    # Create the batch (rewire batch_id into each job)
    batch = create_batch([j.job_id for j in job_objects])
    for job in job_objects:
        job.batch_id = batch.batch_id

    # Build the background task payload: (job_id, temp_path, filename, file_hash)
    task_args: list[tuple[str, str, str, str]] = [
        (job.job_id, temp_path, filename, file_hash)
        for job, (temp_path, filename, file_hash) in zip(job_objects, saved)
    ]

    # Schedule concurrent ingestion as a single background task
    background_tasks.add_task(_ingest_batch, task_args)

    logger.info(
        f"Batch {batch.batch_id} accepted: {len(files)} file(s) → "
        f"jobs {[j.job_id for j in job_objects]}"
    )

    return BatchUploadResponse(
        batch_id=batch.batch_id,
        total_files=len(job_objects),
        jobs=[
            JobSummary(job_id=j.job_id, filename=j.filename, status=j.status)
            for j in job_objects
        ],
    )