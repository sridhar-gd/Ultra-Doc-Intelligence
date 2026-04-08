# backend/schemas/request_schemas.py

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class AskRequest(BaseModel):
    """
    POST /ask — ask a natural-language question about one or more uploaded documents.

    Scope selection (exactly one must be provided):
        job_id:    Query a single document by its ingestion job UUID.
                   The system resolves the document_id internally from the job
                   registry — no need to call GET /ingestion-status first.
                   The job must have status=ready before querying.
        batch_id:  Query ALL ready documents in a previously uploaded batch.
                   The system resolves every document_id that belongs to this
                   batch (status=ready), runs hybrid retrieval against each one
                   in parallel, merges and re-ranks the candidate chunks across
                   documents, and sends the labelled cross-document context to
                   Claude in a single prompt — enabling true cross-document
                   comparisons (e.g. "Compare ceramics weight between carrier
                   RC and shipper RC").

    Other fields:
        question:  Natural-language question (3–1000 chars, non-blank).
        top_k:     Number of final chunks per document to retrieve (1–10, default 3).
                   For batch queries this is applied per-document before the
                   global merge+re-rank, so the LLM context may contain up to
                   top_k × N chunks (capped internally at 10 total).

    Note:
        document_id and document_name have been intentionally removed from this
        schema. POST /upload already returns job_id — using job_id here keeps
        the API consistent and eliminates the need for an extra
        GET /ingestion-status call just to retrieve a document_id.
    """

    job_id: Optional[str] = Field(
        default=None,
        description=(
            "Ingestion job UUID for a single document (returned by POST /upload). "
            "The system resolves the internal document_id automatically. "
            "Mutually exclusive with batch_id."
        ),
        examples=["e7d1a2b3-4c5d-6e7f-8a9b-0c1d2e3f4a5b"],
    )
    batch_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID of an upload batch (returned by POST /upload). "
            "All ready documents in this batch will be queried together. "
            "Mutually exclusive with job_id."
        ),
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about the document content",
        examples=["What is the carrier rate and who is the driver?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of final chunks per document to include in the LLM context "
            "window (1–10). For batch queries, chunks from all documents are "
            "merged and globally re-ranked; the final context is capped at 10 chunks total."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank or whitespace only.")
        return v.strip()

    @model_validator(mode="after")
    def exactly_one_scope(self) -> "AskRequest":
        """
        Enforce that exactly one of job_id or batch_id is provided.
        Both missing or both present are both invalid.
        """
        has_job   = bool(self.job_id   and self.job_id.strip())
        has_batch = bool(self.batch_id and self.batch_id.strip())

        if has_job and has_batch:
            raise ValueError(
                "Provide either 'job_id' or 'batch_id', not both. "
                "Use job_id for a single-document query, "
                "or batch_id to query all documents in a batch."
            )
        if not has_job and not has_batch:
            raise ValueError(
                "One of 'job_id' or 'batch_id' is required. "
                "Use job_id for a single-document query, "
                "or batch_id to query all documents in a batch."
            )
        return self

    @property
    def is_batch_query(self) -> bool:
        """True when the request targets a batch of documents."""
        return self.batch_id is not None


class ExtractRequest(BaseModel):
    """
    POST /extract — trigger structured field extraction for a single document.

    Accepts the job_id returned by POST /upload. The system resolves the
    internal document_id and filename automatically from the job registry,
    so callers do not need to call GET /ingestion-status first.

    The job must have status=ready before extraction can run.
    """

    job_id: str = Field(
        ...,
        description=(
            "Ingestion job UUID for the document to extract (returned by POST /upload). "
            "The system resolves document_id and filename internally. "
            "The job must have status=ready."
        ),
        examples=["e7d1a2b3-4c5d-6e7f-8a9b-0c1d2e3f4a5b"],
    )