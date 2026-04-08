import type {
  AskResponse,
  BatchExtractionResponse,
  BatchStatusResponse,
  BatchUploadResponse,
  HealthResponse,
  JobStatusResponse,
} from "../types/api";

const defaultBase = "http://127.0.0.1:8000";

export function getApiBase(): string {
  const raw = import.meta.env.VITE_API_URL;
  if (typeof raw === "string" && raw.trim()) {
    return raw.replace(/\/$/, "");
  }
  return defaultBase;
}

async function parseJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!text) {
    throw new Error(`Empty response (${res.status})`);
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(text.slice(0, 200));
  }
}

function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const d = (body as { detail?: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d) && d[0] && typeof d[0] === "object" && "msg" in d[0]) {
      return String((d[0] as { msg: string }).msg);
    }
  }
  return `Request failed (${status})`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBase()}/health`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(errorMessage(res.status, body));
  }
  return parseJson<HealthResponse>(res);
}

export async function uploadDocuments(files: File[]): Promise<BatchUploadResponse> {
  const fd = new FormData();
  for (const file of files) fd.append("files", file);
  const res = await fetch(`${getApiBase()}/upload`, {
    method: "POST",
    body: fd,
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(errorMessage(res.status, body));
  }
  return body as BatchUploadResponse;
}

export type AskPayload = {
  job_id?: string;
  batch_id?: string;
  question: string;
  top_k?: number;
};

export async function askQuestion(payload: AskPayload): Promise<AskResponse> {
  const res = await fetch(`${getApiBase()}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_id: payload.job_id,
      batch_id: payload.batch_id,
      question: payload.question,
      top_k: payload.top_k ?? 3,
    }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(errorMessage(res.status, body));
  }
  return body as AskResponse;
}

export type ExtractPayload = {
  job_id: string;
};

export async function extractShipment(payload: ExtractPayload): Promise<unknown> {
  const res = await fetch(`${getApiBase()}/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(errorMessage(res.status, body));
  }
  return body;
}

export async function extractBatch(batchId: string): Promise<BatchExtractionResponse> {
  const res = await fetch(`${getApiBase()}/extract/batch?batch_id=${encodeURIComponent(batchId)}`, {
    method: "POST",
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(res.status, body));
  return body as BatchExtractionResponse;
}

export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${getApiBase()}/ingestion-status/${encodeURIComponent(jobId)}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(res.status, body));
  return body as JobStatusResponse;
}

export async function fetchBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const res = await fetch(`${getApiBase()}/ingestion-status/batch/${encodeURIComponent(batchId)}`);
  const body = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errorMessage(res.status, body));
  return body as BatchStatusResponse;
}
