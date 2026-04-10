# backend/services/ingestion_jobs.py

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import get_settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class IngestionJob(BaseModel):
    job_id: str = Field(..., description="UUID identifying this ingestion job")
    batch_id: str = Field(..., description="UUID of the batch this job belongs to")
    filename: str = Field(..., description="Original uploaded filename")
    status: JobStatus = JobStatus.PENDING
    document_id: Optional[str] = Field(None, description="Supabase document UUID (set when ready)")
    chunks_created: int = 0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionBatch(BaseModel):
    batch_id: str = Field(..., description="UUID identifying this upload batch")
    job_ids: list[str] = Field(default_factory=list)
    total: int = 0
    pending: int = 0
    processing: int = 0
    ready: int = 0
    failed: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_JOBS: dict[str, IngestionJob] = {}
_BATCHES: dict[str, IngestionBatch] = {}
_DB_READY = False
_DB_DISABLED_REASON: str | None = None


def _use_db() -> bool:
    settings = get_settings()
    return bool((settings.database_url or "").strip())


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    settings = get_settings()
    db_url = settings.database_url.strip()
    if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, future=True, pool_pre_ping=True)


def _ensure_tables() -> None:
    global _DB_READY, _DB_DISABLED_REASON
    if _DB_READY or _DB_DISABLED_REASON:
        return
    if not _use_db():
        _DB_DISABLED_REASON = "DATABASE_URL not set; using in-memory job registry."
        logger.warning(_DB_DISABLED_REASON)
        return
    try:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingestion_batches (
                    batch_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    document_id TEXT NULL,
                    chunks_created INTEGER NOT NULL DEFAULT 0,
                    error TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """))
            conn.execute(text(
                "ALTER TABLE ingestion_jobs DROP CONSTRAINT IF EXISTS ingestion_jobs_batch_id_fkey;"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_batch_id ON ingestion_jobs (batch_id);"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs (status);"
            ))
        _DB_READY = True
        logger.info("Persistent ingestion job tables ready.")
    except Exception as exc:
        _DB_DISABLED_REASON = f"Failed to init persistent job tables: {exc}"
        logger.warning("%s Falling back to in-memory registry.", _DB_DISABLED_REASON)


def _row_to_job(row) -> IngestionJob:
    return IngestionJob(
        job_id=row.job_id,
        batch_id=row.batch_id,
        filename=row.filename,
        status=JobStatus(row.status),
        document_id=row.document_id,
        chunks_created=row.chunks_created or 0,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def create_job(batch_id: str, filename: str) -> IngestionJob:
    _ensure_tables()
    job = IngestionJob(job_id=str(uuid.uuid4()), batch_id=batch_id, filename=filename)
    if _DB_READY:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ingestion_jobs
                    (job_id, batch_id, filename, status, document_id, chunks_created, error, created_at, updated_at)
                    VALUES (:job_id, :batch_id, :filename, :status, NULL, 0, NULL, NOW(), NOW());
                """),
                {
                    "job_id": job.job_id,
                    "batch_id": batch_id,
                    "filename": filename,
                    "status": job.status.value,
                },
            )
        return job

    _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[IngestionJob]:
    _ensure_tables()
    if _DB_READY:
        engine = _get_engine()
        with engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT job_id, batch_id, filename, status, document_id, chunks_created, error, created_at, updated_at
                    FROM ingestion_jobs
                    WHERE job_id = :job_id
                    LIMIT 1;
                """),
                {"job_id": job_id},
            ).fetchone()
        return _row_to_job(row) if row else None
    return _JOBS.get(job_id)


def update_job(
    job_id: str,
    status: JobStatus,
    document_id: Optional[str] = None,
    chunks_created: int = 0,
    error: Optional[str] = None,
) -> None:
    _ensure_tables()
    if _DB_READY:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE ingestion_jobs
                    SET
                        status = :status,
                        document_id = COALESCE(:document_id, document_id),
                        chunks_created = CASE WHEN :chunks_created > 0 THEN :chunks_created ELSE chunks_created END,
                        error = CASE WHEN :error_is_set THEN :error ELSE error END,
                        updated_at = NOW()
                    WHERE job_id = :job_id;
                """),
                {
                    "job_id": job_id,
                    "status": status.value,
                    "document_id": document_id,
                    "chunks_created": chunks_created,
                    "error": error,
                    "error_is_set": error is not None,
                },
            )
        return

    job = _JOBS.get(job_id)
    if not job:
        return
    job.status = status
    job.updated_at = datetime.now(timezone.utc)
    if document_id is not None:
        job.document_id = document_id
    if chunks_created:
        job.chunks_created = chunks_created
    if error is not None:
        job.error = error


def create_batch(job_ids: list[str]) -> IngestionBatch:
    _ensure_tables()
    batch_id = str(uuid.uuid4())

    if _DB_READY:
        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO ingestion_batches (batch_id, created_at) VALUES (:batch_id, NOW());"),
                {"batch_id": batch_id},
            )
            for jid in job_ids:
                conn.execute(
                    text("""
                        UPDATE ingestion_jobs
                        SET batch_id = :batch_id, updated_at = NOW()
                        WHERE job_id = :job_id;
                    """),
                    {"batch_id": batch_id, "job_id": jid},
                )
            created_at = conn.execute(
                text("SELECT created_at FROM ingestion_batches WHERE batch_id = :batch_id;"),
                {"batch_id": batch_id},
            ).scalar_one()
        return IngestionBatch(
            batch_id=batch_id,
            job_ids=job_ids,
            total=len(job_ids),
            pending=len(job_ids),
            created_at=created_at,
        )

    batch = IngestionBatch(
        batch_id=batch_id,
        job_ids=job_ids,
        total=len(job_ids),
        pending=len(job_ids),
    )
    _BATCHES[batch.batch_id] = batch
    for jid in job_ids:
        if jid in _JOBS:
            _JOBS[jid].batch_id = batch.batch_id
    return batch


def get_batch(batch_id: str) -> Optional[IngestionBatch]:
    _ensure_tables()
    if _DB_READY:
        engine = _get_engine()
        with engine.begin() as conn:
            batch_row = conn.execute(
                text("SELECT batch_id, created_at FROM ingestion_batches WHERE batch_id = :batch_id LIMIT 1;"),
                {"batch_id": batch_id},
            ).fetchone()
            if not batch_row:
                return None
            job_rows = conn.execute(
                text("""
                    SELECT job_id, status
                    FROM ingestion_jobs
                    WHERE batch_id = :batch_id
                    ORDER BY created_at ASC;
                """),
                {"batch_id": batch_id},
            ).fetchall()

        counters = dict.fromkeys(JobStatus, 0)
        job_ids: list[str] = []
        for row in job_rows:
            job_ids.append(row.job_id)
            counters[JobStatus(row.status)] += 1

        return IngestionBatch(
            batch_id=batch_row.batch_id,
            job_ids=job_ids,
            total=len(job_ids),
            pending=counters[JobStatus.PENDING],
            processing=counters[JobStatus.PROCESSING],
            ready=counters[JobStatus.READY],
            failed=counters[JobStatus.FAILED],
            created_at=batch_row.created_at,
        )

    batch = _BATCHES.get(batch_id)
    if not batch:
        return None
    counters = dict.fromkeys(JobStatus, 0)
    for jid in batch.job_ids:
        job = _JOBS.get(jid)
        if job:
            counters[job.status] += 1
    batch.pending = counters[JobStatus.PENDING]
    batch.processing = counters[JobStatus.PROCESSING]
    batch.ready = counters[JobStatus.READY]
    batch.failed = counters[JobStatus.FAILED]
    return batch
