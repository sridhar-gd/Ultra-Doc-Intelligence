export type JobStatus = "pending" | "processing" | "ready" | "failed";

export type JobSummary = {
  job_id: string;
  filename: string;
  status: JobStatus;
};

export type BatchUploadResponse = {
  batch_id: string;
  total_files: number;
  jobs: JobSummary[];
  submitted_at: string;
};

export type JobStatusResponse = {
  job_id: string;
  batch_id: string;
  filename: string;
  status: JobStatus;
  document_id: string | null;
  chunks_created: number;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type BatchStatusResponse = {
  batch_id: string;
  total: number;
  pending: number;
  processing: number;
  ready: number;
  failed: number;
  jobs: JobStatusResponse[];
  created_at: string;
};

export type ChunkUsed = {
  chunk_id: string;
  text: string;
  document_name?: string | null;
};

export type AskResponse = {
  answer: string;
  /** Present when the answer is grounded; null when guardrail_triggered */
  source_documents: string[] | null;
  confidence_score: number;
  faithfulness: number;
  answer_relevancy: number;
  context_relevancy: number;
  guardrail_triggered: boolean;
  chunks_used: ChunkUsed[];
  low_confidence_warning: boolean;
};

export type BatchExtractionResult = {
  job_id: string;
  document_id: string | null;
  filename: string;
  status: "success" | "failed" | "skipped";
  shipment_data: Record<string, unknown> | null;
  error: string | null;
};

export type BatchExtractionResponse = {
  batch_id: string;
  total: number;
  succeeded: number;
  failed: number;
  skipped: number;
  results: BatchExtractionResult[];
};

export type HealthResponse = {
  status: string;
  checks: Record<string, string>;
};
