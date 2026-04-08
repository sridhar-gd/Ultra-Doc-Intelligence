import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import {
  askQuestion,
  extractBatch,
  extractShipment,
  fetchBatchStatus,
  fetchHealth,
  fetchJobStatus,
  uploadDocuments,
} from "./api/client";
import type {
  AskResponse,
  BatchExtractionResponse,
  BatchStatusResponse,
  BatchUploadResponse,
  HealthResponse,
  JobStatus,
  JobStatusResponse,
} from "./types/api";

type Tab = "upload" | "ask" | "extract" | "status";
type AskScope = "job" | "batch";
type ExtractScope = "job" | "batch";

export default function App() {
  const [tab, setTab] = useState<Tab>("upload");
  const [lastUpload, setLastUpload] = useState<BatchUploadResponse | null>(null);
  const [activeBatchId, setActiveBatchId] = useState("");
  const [activeJobId, setActiveJobId] = useState("");
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        setHealth(await fetchHealth());
      } catch (e) {
        setHealthError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  useEffect(() => {
    if (!activeBatchId) return;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const status = await fetchBatchStatus(activeBatchId);
        setBatchStatus(status);
        if (status.ready + status.failed < status.total) {
          timer = window.setTimeout(() => void poll(), 2000);
        }
      } catch {
        timer = window.setTimeout(() => void poll(), 3000);
      }
    };
    void poll();
    return () => {
      if (timer) window.clearTimeout(timer);
    };
  }, [activeBatchId]);

  const readyJobs = useMemo(
    () => (batchStatus?.jobs ?? []).filter((j) => j.status === "ready"),
    [batchStatus],
  );

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Ultraship AI Workspace</p>
          <h1>Document intelligence for logistics documents</h1>
          <p className="subtitle">
            Upload multiple files, monitor ingestion live, ask across a batch, and run extraction in one flow.
          </p>
        </div>
      </header>

      <nav className="tabs">
        {(["upload", "ask", "extract", "status"] as const).map((id) => (
          <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>
            {id[0].toUpperCase() + id.slice(1)}
          </button>
        ))}
      </nav>

      {tab === "upload" && (
        <UploadPanel
          onUploaded={(res) => {
            setLastUpload(res);
            setActiveBatchId(res.batch_id);
            setActiveJobId(res.jobs[0]?.job_id ?? "");
            setTab("status");
          }}
        />
      )}
      {tab === "status" && (
        <StatusPanel
          health={health}
          healthError={healthError}
          batchId={activeBatchId}
          batchStatus={batchStatus}
          onPickJob={setActiveJobId}
          onRefreshHealth={async () => {
            try {
              setHealthError(null);
              setHealth(await fetchHealth());
            } catch (e) {
              setHealthError(e instanceof Error ? e.message : String(e));
            }
          }}
        />
      )}
      {tab === "ask" && (
        <AskPanel
          defaultBatchId={activeBatchId}
          defaultJobId={activeJobId}
          readyJobs={readyJobs}
          onGotoStatus={() => setTab("status")}
        />
      )}
      {tab === "extract" && (
        <ExtractPanel defaultBatchId={activeBatchId} defaultJobId={activeJobId} readyJobs={readyJobs} />
      )}

      {lastUpload && (
        <footer className="session-strip">
          <span>Latest batch</span>
          <code>{lastUpload.batch_id}</code>
          <span>{lastUpload.total_files} file(s)</span>
          <span>{batchStatus ? `${batchStatus.ready}/${batchStatus.total} ready` : "Polling..."}</span>
        </footer>
      )}
    </div>
  );
}

function UploadPanel({ onUploaded }: { onUploaded: (r: BatchUploadResponse) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BatchUploadResponse | null>(null);

  const submit = async () => {
    if (files.length === 0) {
      setError("Choose one or more files.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await uploadDocuments(files);
      setResult(r);
      onUploaded(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel elevated">
      <h2>Upload & ingest</h2>
      <div className="field">
        <label htmlFor="file">Documents</label>
        <input
          id="file"
          type="file"
          multiple
          accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
          onChange={(ev) => {
            const selected = Array.from(ev.target.files ?? []);
            setFiles((prev) => {
              const merged = [...prev, ...selected];
              const seen = new Set<string>();
              return merged.filter((f) => {
                const key = `${f.name}-${f.size}-${f.lastModified}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
              });
            });
            setResult(null);
            setError(null);
            ev.currentTarget.value = "";
          }}
        />
        <p className="hint">You can select multiple files at once and keep adding more before upload.</p>
      </div>
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((f) => (
            <li key={`${f.name}-${f.size}`}>{f.name}</li>
          ))}
        </ul>
      )}
      <div className="actions">
        <button type="button" className="btn primary" disabled={loading || files.length === 0} onClick={() => void submit()}>
          {loading ? "Ingesting…" : "Upload"}
        </button>
        <button type="button" className="btn secondary" disabled={loading || files.length === 0} onClick={() => setFiles([])}>
          Clear files
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && <UploadResultCard result={result} />}
    </section>
  );
}

function UploadResultCard({ result }: { result: BatchUploadResponse }) {
  return (
    <div className="card success">
      <h3>Batch queued</h3>
      <p>
        <strong>{result.total_files}</strong> files accepted. Batch ID: <code>{result.batch_id}</code>
      </p>
      <ul className="file-list">
        {result.jobs.map((j) => (
          <li key={j.job_id}>
            <span>
              {j.filename}
              <br />
              <small>
                job_id: <code>{j.job_id}</code>
              </small>
            </span>
            <span className="chip">{j.status}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function AskPanel({
  defaultBatchId,
  defaultJobId,
  readyJobs,
  onGotoStatus,
}: {
  defaultBatchId: string;
  defaultJobId: string;
  readyJobs: JobStatusResponse[];
  onGotoStatus: () => void;
}) {
  const [scope, setScope] = useState<AskScope>("batch");
  const [batchId, setBatchId] = useState(defaultBatchId);
  const [jobId, setJobId] = useState(defaultJobId);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AskResponse | null>(null);

  const submit = async () => {
    const scopeValue = scope === "batch" ? batchId.trim() : jobId.trim();
    if (!scopeValue) {
      setError(`${scope}_id is required.`);
      return;
    }
    if (question.trim().length < 3) {
      setError("Question must be at least 3 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const r = await askQuestion({
        job_id: scope === "job" ? scopeValue : undefined,
        batch_id: scope === "batch" ? scopeValue : undefined,
        question: question.trim(),
        top_k: topK,
      });
      setAnswer(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <h2>Ask a question</h2>
      <div className="grid two">
        <div className="field">
          <label>Scope</label>
          <div className="segmented">
            <button className={scope === "batch" ? "active" : ""} onClick={() => setScope("batch")}>
              batch_id
            </button>
            <button className={scope === "job" ? "active" : ""} onClick={() => setScope("job")}>
              job_id
            </button>
          </div>
        </div>
        <div className="field">
          <label>{scope}_id</label>
          <input
            value={scope === "batch" ? batchId : jobId}
            onChange={(e) => (scope === "batch" ? setBatchId(e.target.value) : setJobId(e.target.value))}
            placeholder={`Enter ${scope}_id`}
            autoComplete="off"
          />
        </div>
      </div>
      {readyJobs.length > 0 && (
        <div className="hint">
          Ready jobs: {readyJobs.length} &middot; quick select{" "}
          {readyJobs.slice(0, 3).map((j) => (
            <button key={j.job_id} className="inline-link" onClick={() => { setScope("job"); setJobId(j.job_id); }}>
              {j.filename}
            </button>
          ))}
        </div>
      )}
      <div className="field">
        <label htmlFor="q">Question</label>
        <textarea
          id="q"
          rows={4}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is the carrier rate and who is the driver?"
        />
      </div>
      <div className="field narrow">
        <label htmlFor="topk">top_k (1–10)</label>
        <input
          id="topk"
          type="number"
          min={1}
          max={10}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value) || 3)}
        />
      </div>
      <div className="actions">
        <button type="button" className="btn primary" disabled={loading} onClick={() => void submit()}>
          {loading ? "Asking…" : "Ask"}
        </button>
        <button type="button" className="btn secondary" onClick={onGotoStatus}>
          Open live status
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {answer && <AskResultCard response={answer} />}
    </section>
  );
}

function AskResultCard({ response }: { response: AskResponse }) {
  const pct = (n: number) => `${Math.round(n * 100)}%`;
  return (
    <div className="card answer">
      <div className="answer-head">
        <h3>Answer</h3>
      </div>
      <div className="answer-text">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{response.answer}</ReactMarkdown>
      </div>
      <div className="metrics">
        <Metric label="confidence" value={pct(response.confidence_score)} warn={response.low_confidence_warning} />
        <Metric label="faithfulness" value={pct(response.faithfulness)} />
        <Metric label="answer relevancy" value={pct(response.answer_relevancy)} />
        <Metric label="context relevancy" value={pct(response.context_relevancy)} />
        <span className={`badge ${response.guardrail_triggered ? "bad" : "ok"}`}>
          guardrail {response.guardrail_triggered ? "triggered" : "clear"}
        </span>
        {response.low_confidence_warning && !response.guardrail_triggered && (
          <span className="badge warn">low confidence warning</span>
        )}
      </div>
      {response.source_documents != null && response.source_documents.length > 0 && (
        <div className="source-list">
          {response.source_documents.map((name) => (
            <span key={name} className="chip">
              {name}
            </span>
          ))}
        </div>
      )}
      {response.chunks_used.length > 0 && (
        <details className="chunks">
          <summary>Chunks used ({response.chunks_used.length})</summary>
          <ul className="chunk-list">
            {response.chunks_used.map((c) => (
              <li key={c.chunk_id}>
                <div className="chunk-meta">
                  <code>chunk_id: {c.chunk_id}</code>
                  <span>{c.document_name ? `document_name: ${c.document_name}` : "document_name: Document"}</span>
                </div>
                <pre className="chunk-text">{c.text}</pre>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function Metric({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <span className={`metric ${warn ? "warn" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </span>
  );
}

function ExtractPanel({
  defaultBatchId,
  defaultJobId,
  readyJobs,
}: {
  defaultBatchId: string;
  defaultJobId: string;
  readyJobs: JobStatusResponse[];
}) {
  const [scope, setScope] = useState<ExtractScope>("job");
  const [batchId, setBatchId] = useState(defaultBatchId);
  const [jobId, setJobId] = useState(defaultJobId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<unknown>(null);

  const submit = async () => {
    const scopeValue = scope === "batch" ? batchId.trim() : jobId.trim();
    if (!scopeValue) {
      setError(`${scope}_id is required.`);
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const r =
        scope === "job" ? await extractShipment({ job_id: scopeValue }) : await extractBatch(scopeValue);
      setData(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel">
      <h2>Structured extraction</h2>
      <div className="grid two">
        <div className="field">
          <label>Scope</label>
          <div className="segmented">
            <button className={scope === "job" ? "active" : ""} onClick={() => setScope("job")}>
              job_id
            </button>
            <button className={scope === "batch" ? "active" : ""} onClick={() => setScope("batch")}>
              batch_id
            </button>
          </div>
        </div>
        <div className="field">
          <label>{scope}_id</label>
          <input
            value={scope === "batch" ? batchId : jobId}
            onChange={(e) => (scope === "batch" ? setBatchId(e.target.value) : setJobId(e.target.value))}
            placeholder={`Enter ${scope}_id`}
            autoComplete="off"
          />
        </div>
      </div>
      {readyJobs.length > 0 && scope === "job" && (
        <div className="hint">
          Ready jobs:{" "}
          {readyJobs.slice(0, 4).map((j) => (
            <button key={j.job_id} className="inline-link" onClick={() => setJobId(j.job_id)}>
              {j.filename}
            </button>
          ))}
        </div>
      )}
      <div className="actions">
        <button type="button" className="btn primary" disabled={loading} onClick={() => void submit()}>
          {loading ? "Extracting…" : scope === "job" ? "Extract job" : "Extract batch"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {data !== null && <ExtractOutput data={data} />}
    </section>
  );
}

function ExtractOutput({ data }: { data: unknown }) {
  const maybeBatch = data as BatchExtractionResponse;
  if (maybeBatch && typeof maybeBatch === "object" && Array.isArray(maybeBatch.results)) {
    return (
      <div className="card">
        <h3>Batch extraction summary</h3>
        <p>
          {maybeBatch.succeeded} succeeded / {maybeBatch.failed} failed / {maybeBatch.skipped} skipped
        </p>
        <ul className="file-list">
          {maybeBatch.results.map((r) => (
            <li key={r.job_id}>
              <div className="file-list-main">
                <strong>{r.filename}</strong>
                <br />
                <small>
                  job_id: <code>{r.job_id}</code>
                </small>
                {r.document_id && (
                  <>
                    <br />
                    <small>
                      document_id: <code>{r.document_id}</code>
                    </small>
                  </>
                )}
                {r.error && (
                  <>
                    <br />
                    <small className="error">error: {r.error}</small>
                  </>
                )}
                {r.shipment_data && (
                  <pre className="json-out">{JSON.stringify(r.shipment_data, null, 2)}</pre>
                )}
              </div>
              <span className={`chip ${r.status}`}>{r.status}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  return <pre className="json-out">{JSON.stringify(data, null, 2)}</pre>;
}

function StatusPanel({
  health,
  healthError,
  batchId,
  batchStatus,
  onRefreshHealth,
  onPickJob,
}: {
  health: HealthResponse | null;
  healthError: string | null;
  batchId: string;
  batchStatus: BatchStatusResponse | null;
  onRefreshHealth: () => void;
  onPickJob: (jobId: string) => void;
}) {
  const [jobLookup, setJobLookup] = useState("");
  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  return (
    <section className="panel">
      <h2>Live status</h2>
      <div className="actions">
        <button type="button" className="btn secondary" onClick={onRefreshHealth}>
          Refresh health
        </button>
      </div>
      {healthError && <p className="error">{healthError}</p>}
      {health && (
        <div className="card">
          <h3>GET /health</h3>
          <p className={`status-line ${health.status === "healthy" ? "ok" : "degraded"}`}>
            status: <strong>{health.status}</strong>
          </p>
          <ul className="check-list">
            {Object.entries(health.checks).map(([k, v]) => (
              <li key={k}>
                <span className="check-key">{k}</span>
                <span className={v === "ok" ? "check-ok" : "check-bad"}>{v}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="card">
        <h3>Batch tracker</h3>
        {batchId ? (
          <>
            <p>
              batch_id: <code>{batchId}</code>
            </p>
            {batchStatus ? (
              <>
                <div className="metrics">
                  <Metric label="total" value={String(batchStatus.total)} />
                  <Metric label="ready" value={String(batchStatus.ready)} />
                  <Metric label="processing" value={String(batchStatus.processing)} />
                  <Metric label="failed" value={String(batchStatus.failed)} />
                </div>
                <ul className="file-list">
                  {batchStatus.jobs.map((j) => (
                    <li key={j.job_id}>
                      <span>
                        <button className="inline-link" onClick={() => onPickJob(j.job_id)}>
                          {j.filename}
                        </button>
                        <br />
                        <small>
                          job_id: <code>{j.job_id}</code>
                        </small>
                      </span>
                      <StatusChip status={j.status} />
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>Waiting for first poll result...</p>
            )}
          </>
        ) : (
          <p>No active batch yet. Upload files first.</p>
        )}
      </div>
      <div className="card">
        <h3>Single job lookup</h3>
        <div className="actions">
          <input
            className="inline-input"
            value={jobLookup}
            onChange={(e) => setJobLookup(e.target.value)}
            placeholder="Enter job_id"
          />
          <button
            className="btn secondary"
            onClick={() =>
              void (async () => {
                try {
                  setJobError(null);
                  setJobStatus(await fetchJobStatus(jobLookup.trim()));
                } catch (e) {
                  setJobStatus(null);
                  setJobError(e instanceof Error ? e.message : String(e));
                }
              })()
            }
          >
            Check job
          </button>
        </div>
        {jobError && <p className="error">{jobError}</p>}
        {jobStatus && (
          <p>
            {jobStatus.filename} <StatusChip status={jobStatus.status} />{" "}
            {jobStatus.document_id ? <code>{jobStatus.document_id}</code> : "no document_id yet"}
          </p>
        )}
      </div>
    </section>
  );
}

function StatusChip({ status }: { status: JobStatus }) {
  return <span className={`chip ${status}`}>{status}</span>;
}
