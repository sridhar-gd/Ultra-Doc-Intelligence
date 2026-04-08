# backend/schemas/response_schemas.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from services.ingestion_jobs import JobStatus


# /upload response (multi-file batch)

class JobSummary(BaseModel):
    """Per-file summary returned immediately after a batch upload."""

    job_id:   str = Field(..., description="Poll GET /ingestion-status/{job_id} for this file's progress")
    filename: str = Field(..., description="Original uploaded filename")
    status:   JobStatus = Field(default=JobStatus.PENDING)


class BatchUploadResponse(BaseModel):
    """
    Returned immediately after POST /upload with multiple files.

    The response is non-blocking — ingestion runs in the background.
    Poll GET /ingestion-status/batch/{batch_id} for overall progress,
    or GET /ingestion-status/{job_id} for a specific file.
    """

    batch_id:   str = Field(..., description="UUID for this upload batch — use in /ingestion-status/batch/{batch_id}")
    total_files: int = Field(..., description="Number of files accepted for ingestion")
    jobs:        list[JobSummary] = Field(..., description="One entry per uploaded file")
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


# /ingestion-status responses

class JobStatusResponse(BaseModel):
    """Full status of a single ingestion job."""

    job_id:         str
    batch_id:       str
    filename:       str
    status:         JobStatus
    document_id:    Optional[str] = Field(None, description="Available once status=ready")
    chunks_created: int = 0
    error:          Optional[str] = Field(None, description="Error message when status=failed")
    created_at:     datetime
    updated_at:     datetime


class BatchStatusResponse(BaseModel):
    """Aggregated status of all jobs in a batch."""

    batch_id:   str
    total:      int
    pending:    int
    processing: int
    ready:      int
    failed:     int
    jobs:       list[JobStatusResponse]
    created_at: datetime


# /ask response

class ChunkUsed(BaseModel):
    """
    A single retrieved chunk returned in the /ask response.

    Fields:
        chunk_id:      Internal UUID of the chunk (traceability/debugging).
        text:          Actual document text content used as LLM context.
                       This is `contextual_text` (AI-enriched by Contextualizer.py)
                       if contextualisation ran, otherwise the raw `chunk_text`
                       extracted from the PDF. Always real document content.
        document_name: Name of the source document this chunk belongs to.
                       Always populated for batch queries; present for single-doc
                       queries when document_name was provided in the request.
    """
    chunk_id:      str            = Field(..., description="UUID of the chunk in the DB")
    text:          str            = Field(..., description="Document text used as context (contextualised or raw)")
    document_name: Optional[str]  = Field(None, description="Source document name for this chunk")


class AskResponse(BaseModel):
    """
    Response schema for POST /ask.

    Supports both single-document and cross-document (batch_id) queries.

    CROSS-DOCUMENT QUERIES (batch_id):
    ────────────────────────────────────
    When a batch_id is provided, the system:
      1. Resolves all ready document_ids in the batch.
      2. Runs hybrid retrieval in parallel against each document.
      3. Labels every retrieved chunk with its source document name.
      4. Merges and globally re-ranks chunks across all documents.
      5. Sends the labelled cross-document context to Claude in one prompt,
         enabling attributed comparisons ("In carrier_rc.pdf, X is Y.
         In shipper_rc.pdf, X is Z.").

    When `guardrail_triggered` is true, `source_documents` is null

    CONFIDENCE SCORING (v3 — query-aware three-signal approach):
    ─────────────────────────────────────────────────────────────
    Weights adapt to query type, detected automatically from retrieved chunks.

    Single-document:
        confidence_score = 0.55 * faithfulness
                         + 0.25 * answer_relevancy   (claim coverage)
                         + 0.20 * context_relevancy

    Cross-document (batch_id query, chunks span > 1 document):
        confidence_score = 0.60 * faithfulness
                         + 0.20 * answer_relevancy   (claim coverage)
                         + 0.20 * context_relevancy  (source coverage)

    faithfulness:
        Fraction of answer claims supported by retrieved context.
        Computed by Claude Sonnet (LLM-as-judge). Understands that "#10000"
        and "10,000 units" are the same fact — no false penalties.
        1.0 = fully grounded | 0.0 = hallucinated or refusal

    answer_relevancy  (claim coverage):
        Whether the answer is directly responsive to the question and
        contains a concrete finding (specific value, date, comparison, etc.).
        Scored by Claude Sonnet as two binary dimensions (responsive + concrete),
        averaged to a float. Immune to comparison-verb embedding drift —
        works correctly on questions like "Compare X between doc A and doc B."
        1.0 = responsive and concrete | 0.0 = refusal or vague non-answer

    context_relevancy:
        Single-doc: sigmoid(rerank_score) from BAAI cross-encoder — measures
        retrieval quality for the top chunk. Zero extra latency.
        Cross-doc:  source coverage — fraction of expected source documents
        represented in the retrieved chunks. 1.0 = all documents contributed
        at least one chunk to the answer context.

    EXPECTED RANGES:
        Correct, fully grounded answer:    confidence_score 0.88-0.95
        Correct but incomplete:            0.65-0.85
        Partially grounded / vague:        0.45-0.65
        Refusal ("Not found"):             0.00-0.10
    """
    answer: str = Field(
        ...,
        description="LLM-generated answer grounded in retrieved context, or 'Not found in document.'",
    )
    source_documents: Optional[list[str]] = Field(
        None,
        description=(
            "Documents that contributed to a grounded answer. "
            "Null when guardrail_triggered is true. "
            "Single-doc: one entry (filename or section label). Batch: one per file used."
        ),
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Composite quality score (0-1). "
            "Single-doc: 0.55*faithfulness + 0.25*answer_relevancy + 0.20*context_relevancy. "
            "Cross-doc:  0.60*faithfulness + 0.20*answer_relevancy + 0.20*context_relevancy."
        ),
    )
    faithfulness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of answer claims supported by retrieved context (Claude Sonnet LLM-as-judge)",
    )
    answer_relevancy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Claim coverage score: whether the answer is responsive to the question "
            "and contains a concrete finding. Mean of two binary judgments (responsive, concrete) "
            "scored by Claude Sonnet. Handles comparison queries correctly."
        ),
    )
    context_relevancy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Single-doc: retrieval quality — sigmoid(rerank_score) from BAAI cross-encoder. "
            "Cross-doc: source coverage — fraction of expected documents represented in retrieved chunks."
        ),
    )
    guardrail_triggered: bool = Field(
        default=False,
        description=(
            "True when the system refused to answer (retrieval gate, similarity gate, or document refusal). "
            "When true, source_documents is null and chunks_used is empty."
        ),
    )
    chunks_used: list[ChunkUsed] = Field(
        default_factory=list,
        description=(
            "The top-K chunks used as context for this answer. "
            "For batch queries, chunks from multiple documents are included; "
            "each chunk's document_name field identifies its source document."
        ),
    )
    low_confidence_warning: bool = Field(
        default=False,
        description="True if confidence_score is below the low-confidence threshold in config",
    )


# /extract/batch response

class BatchExtractionResult(BaseModel):
    """
    Extraction outcome for a single document within a batch extraction run.

    When `status` is `"success"`, `shipment_data` is populated.
    When `status` is `"failed"`,  `error` explains the failure and
    `shipment_data` is None.  `status` is `"skipped"` when the ingestion
    job for this file has not yet reached the `ready` state.
    """
    job_id:       str                     = Field(..., description="Ingestion job UUID for this file")
    document_id:  Optional[str]           = Field(None, description="Supabase document UUID")
    filename:     str                     = Field(..., description="Original uploaded filename")
    status:       str                     = Field(..., description="success | failed | skipped")
    shipment_data: Optional[dict]         = Field(None, description="Extracted ShipmentData fields (null on failure/skip)")
    error:        Optional[str]           = Field(None, description="Error message when status=failed")


class BatchExtractionResponse(BaseModel):
    """
    Response for POST /extract/batch.

    Contains one `BatchExtractionResult` entry per document in the batch.
    Documents whose ingestion job is not yet `ready` are returned with
    `status="skipped"` so the caller can retry them individually later.
    """
    batch_id:     str                          = Field(..., description="The batch_id passed in the request")
    total:        int                          = Field(..., description="Total documents in the batch")
    succeeded:    int                          = Field(..., description="Number of successful extractions")
    failed:       int                          = Field(..., description="Number of failed extractions")
    skipped:      int                          = Field(..., description="Documents not yet ready for extraction")
    results:      list[BatchExtractionResult]  = Field(..., description="Per-document extraction results")


# Error

class ErrorResponse(BaseModel):
    """Generic error response body."""
    error:       str            = Field(..., description="Short error code or message")
    detail:      Optional[str]  = Field(None, description="Detailed explanation for debugging")
    document_id: Optional[str]  = Field(None, description="Related document UUID if applicable")