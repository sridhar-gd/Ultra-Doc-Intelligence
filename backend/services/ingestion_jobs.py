# backend/services/ingestion_jobs.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# Status enum

class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


# Per-file job model

class IngestionJob(BaseModel):
    job_id:        str = Field(..., description="UUID identifying this ingestion job")
    batch_id:      str = Field(..., description="UUID of the batch this job belongs to")
    filename:      str = Field(..., description="Original uploaded filename")
    status:        JobStatus = JobStatus.PENDING
    document_id:   Optional[str] = Field(None, description="Supabase document UUID (set when ready)")
    chunks_created: int = 0
    error:         Optional[str] = None
    created_at:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Batch model

class IngestionBatch(BaseModel):
    batch_id:   str = Field(..., description="UUID identifying this upload batch")
    job_ids:    list[str] = Field(default_factory=list)
    total:      int = 0
    pending:    int = 0
    processing: int = 0
    ready:      int = 0
    failed:     int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# In-process registries

_JOBS:    dict[str, IngestionJob]    = {}
_BATCHES: dict[str, IngestionBatch] = {}


# Job CRUD

def create_job(batch_id: str, filename: str) -> IngestionJob:
    """Create and register a new PENDING ingestion job."""
    job = IngestionJob(
        job_id=str(uuid.uuid4()),
        batch_id=batch_id,
        filename=filename,
    )
    _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[IngestionJob]:
    return _JOBS.get(job_id)


def update_job(
    job_id: str,
    status: JobStatus,
    document_id: Optional[str] = None,
    chunks_created: int = 0,
    error: Optional[str] = None,
) -> None:
    """Update a job's status and optional result fields."""
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


# Batch CRUD

def create_batch(job_ids: list[str]) -> IngestionBatch:
    """Create and register a new batch for the given job IDs."""
    batch = IngestionBatch(
        batch_id=str(uuid.uuid4()),
        job_ids=job_ids,
        total=len(job_ids),
        pending=len(job_ids),
    )
    _BATCHES[batch.batch_id] = batch
    return batch


def get_batch(batch_id: str) -> Optional[IngestionBatch]:
    """Return a batch with up-to-date counters derived from its jobs."""
    batch = _BATCHES.get(batch_id)
    if not batch:
        return None

    # Recompute counters from live job state
    counters = {s: 0 for s in JobStatus}
    for jid in batch.job_ids:
        job = _JOBS.get(jid)
        if job:
            counters[job.status] += 1

    batch.pending    = counters[JobStatus.PENDING]
    batch.processing = counters[JobStatus.PROCESSING]
    batch.ready      = counters[JobStatus.READY]
    batch.failed     = counters[JobStatus.FAILED]
    return batch