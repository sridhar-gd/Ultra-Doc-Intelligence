# Ultra Doc-Intelligence Architecture

## Objective

Ultra Doc-Intelligence is an agentic document intelligence stack for logistics operations. It ingests carrier and customer shipping documents (PDF, DOCX, TXT), builds searchable vectorized knowledge, answers natural-language questions with grounded context, and extracts structured shipment JSON.

The product emphasizes asynchronous multi-file ingestion, hybrid retrieval, guardrails and confidence scoring, and both single-document and cross-document (batch) workflows.

---

## Product scope

### Core capabilities

- Batch upload and async ingestion (`POST /upload` + background workers)
- Pollable ingestion status (`GET /ingestion-status/{job_id}`, `GET /ingestion-status/batch/{batch_id}`)
- Grounded Q&A (`POST /ask`) over one ready job or a whole ready batch
- Structured extraction (`POST /extract`, `POST /extract/batch`)
- Health (`GET /health`, `GET /`)

### Frontend (high level)

The UI under `frontend/` is a React + TypeScript + Vite single-page app. It drives the full demo flow: multi-file upload with live ingestion status, batch or single-job **Ask** (markdown answers, metrics, guardrail badge, optional chunk disclosure), and **Extract** (JSON including `shipment_data`). It calls the FastAPI backend via a small typed client; styling is plain CSS (dark theme). It is intentionally thinner than the backend: no duplicate business logic—only orchestration, forms, and presentation.

---

## How to run

**Backend** (from `ultra-doc-intelligence/backend`, with `.env` configured):

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend** (from `ultra-doc-intelligence/frontend`):

```bash
cd frontend
npm install
npm run dev
```

Open the app in the browser at:

**[http://localhost:5173](http://localhost:5173)**

(API docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**)

On Windows PowerShell, use the same commands after `cd` to the respective folders.

---

## High-level architecture

```mermaid
flowchart LR
    subgraph Client
        FE[React SPA]
    end
    subgraph API[FastAPI]
        RU[Upload]
        RA[Ask]
        RE[Extract]
        RS[Status]
    end
    subgraph Workers[Background]
        ING[Ingestion pipeline]
    end
    subgraph Data
        JOB[Job registry]
        SB[(Supabase / pgvector)]
    end
    FE --> RU & RA & RE & RS
    RU --> JOB
    RU --> ING
    ING --> JOB
    ING --> SB
    RA --> JOB
    RA --> QA[QA: retrieve + rerank + Claude]
    QA --> SB
    RE --> JOB
    RE --> EXT[Extraction: Claude + schema]
    EXT --> SB
    RS --> JOB
```



---

## Detailed technical components (pipeline order)

1. **Configuration** (`config.py`) — Pydantic Settings: Anthropic/OpenAI keys, exact model IDs, embedding dimensions, retrieval k-values, guardrail thresholds, upload limits, CORS.
2. **Document parsing** (`services/parser.py`) — Docling conversion to markdown/text; load-id / document-type heuristics for metadata.
3. **Chunking** (`services/chunker.py`) — Primary: Docling **HybridChunker**. Fallback: **LangChain** `RecursiveCharacterTextSplitter` when the primary path is unavailable; preserves headings, order, and page hints where present.
4. **Contextualization** (`services/contextualizer.py`, `prompts/contextual_retrieval_prompt.py`) — Per-chunk context prefixes via Anthropic. The prompt marks the **full-document block with Anthropic prompt cache** (`cache_control: ephemeral`) so repeated chunk calls reuse cached prefix tokens; usage metrics log cache read vs fresh input.
5. **Embeddings** (`services/embedder.py`) — **OpenAI** embeddings via LangChain `OpenAIEmbeddings(model=settings.embedding_model_name)`. Default model: `text-embedding-3-small` with `embedding_dimensions=1536`.
6. **Cross-encoder reranking** (`services/embedder.py`) — HuggingFace cross-encoder via LangChain community `HuggingFaceCrossEncoder(model_name=settings.reranker_model_name)`. Default reranker model: `BAAI/bge-reranker-base`.
7. **Vector storage & dedup** (`db/vector_store.py`, `db/client.py`) — Supabase tables for documents, chunks, embeddings; **SHA-256 file hash** lookup skips re-ingestion for identical content.
8. **Hybrid retrieval** (`services/retriever.py`) — Per-document **pgvector** cosine match + **pg_trgm** keyword RPCs; **Reciprocal Rank Fusion (RRF)** merges lists; cross-encoder **rerank** produces final ordering.
9. **Batch QA** (`agents/qa_agent.py`) — Parallel retrieve per document, **merge + global cross-encoder rerank** so scores are comparable across files; answers via **Anthropic Messages API** (`model=settings.anthropic_model`, default `claude-sonnet-4-20250514`) with `prompts/qa_system_prompt.py`. **Refusal detection** aligns `guardrail_triggered`, empty `chunks_used`, and `source_documents: null` with “not in document” outcomes.
10. **Pydantic AI agents & tools** (`agents/ingestion_agent.py`, `agents/qa_agent.py`, `tools/*.py`) — **Pydantic AI** `Agent` wiring with Anthropic models; tools such as `retrieve_chunks` / `rerank_chunks` (LangChain tool adapters) support tool-calling style orchestration alongside the main direct-call QA path.
11. **Guardrails & confidence scoring** (`services/guardrails.py`) — Similarity gate on top retrieved chunk; **LLM-as-judge** (Claude Sonnet via `settings.anthropic_model`) for faithfulness and claim coverage; query-aware composite score; `is_refusal()` for API alignment.
12. **Structured extraction** (`agents/extraction_agent.py`, `schemas/shipment_schema.py`, `prompts/extraction_system_prompt.py`) — Full-chunk context from DB; Claude output validated as **ShipmentData**.
13. **Job registry** (`services/ingestion_jobs.py`) — In-memory batches and jobs (volatile across restarts).

---

## Runtime model stack and scoring defaults

- **Primary LLM (QA + extraction + contextualizer + judge):** `settings.anthropic_model` (default `claude-sonnet-4-20250514`)
- **Claude max tokens (QA/extract):** `anthropic_max_tokens=1024`
- **Contextualizer max tokens:** `contextualizer_max_tokens=200`
- **Embedding model:** `settings.embedding_model_name` (default `text-embedding-3-small`)
- **Embedding dimension:** `embedding_dimensions=1536`
- **Cross-encoder reranker:** `settings.reranker_model_name` (default `BAAI/bge-reranker-base`)

All values are configurable via environment variables (`.env`) through `config.py`.

---

## RAGAS-style confidence implementation

Confidence is computed in `services/guardrails.py::evaluate_answer()` using three signals:

1. **Faithfulness** (`_compute_faithfulness`)
  - LLM-judge output: `total_claims`, `supported_claims`
  - Formula: `supported_claims / total_claims` (**RAGAS faithfulness-style formula**)
2. **Answer relevancy** (`_compute_claim_coverage`)
  - Binary checks for `responsive` and `concrete`
  - Score: mean of those two binaries
3. **Context relevancy**
  - Single-doc: `sigmoid(rerank_score)` from cross-encoder  
  - Cross-doc: source coverage across expected documents

Composite weighting:

- **Single-doc:** `0.55 * faithfulness + 0.25 * answer_relevancy + 0.20 * context_relevancy`
- **Cross-doc:** `0.60 * faithfulness + 0.20 * answer_relevancy + 0.20 * context_relevancy`

Thresholds:

- **Similarity gate:** `guardrail_similarity_threshold=0.25`
- **Low confidence warning:** `guardrail_low_confidence_threshold=0.20`

---

## Architecture

This repository follows a **layered FastAPI backend** with optional React frontend:


| Layer         | Responsibility                                     | Key locations                                                                                     |
| ------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **API**       | HTTP routes, validation, job/batch resolution      | `backend/api/*.py`, `schemas/request_schemas.py`, `schemas/response_schemas.py`                   |
| **Agents**    | QA, ingestion orchestration, extraction            | `backend/agents/*.py`, `backend/tools/*.py`                                                       |
| **Services**  | Parsing, chunking, embeddings, retrieval, scoring  | `backend/services/*.py`                                                                           |
| **Data**      | Supabase client, pgvector RPCs, chunk persistence  | `backend/db/client.py`, `backend/db/vector_store.py`, SQL in `backend/db/init_supabase_schema.py` |
| **Config**    | Environment-driven settings                        | `backend/config.py`                                                                               |
| **App entry** | Lifespan, CORS, router registration, logging setup | `backend/main.py`                                                                                 |


**Request flow (simplified):** client → FastAPI router → resolve `job_id` / `batch_id` via in-memory registry (`services/ingestion_jobs.py`) → QA or extraction agent → hybrid retrieval against Supabase (`services/retriever.py`) → Anthropic for generation or judging (`agents/qa_agent.py`, `services/guardrails.py`) → `AskResponse` / `ShipmentData` per `schemas/`.

The mermaid diagram in [High-level architecture](#high-level-architecture) shows upload → background ingestion → vector DB → ask/extract.

---

## Chunking strategy

Chunking is implemented in `backend/services/chunker.py`.

1. **Primary path — Docling `HybridChunker`**
  - Uses tokenizer `BAAI/bge-large-en-v1.5` and `max_tokens=settings.chunk_size` (default 512).  
  - `merge_peers=True` merges small adjacent chunks in the same section.  
  - Metadata per chunk: `section_heading`, `page_number` (from Docling provenance when available), `doc_type`, `load_id` from the parsed document.
2. **Quality gate**
  - If Docling returns too few chunks (`_MIN_ACCEPTABLE_CHUNKS`, default 3) or any chunk exceeds `_MAX_ACCEPTABLE_TOKENS` (default 1200 estimated tokens), the pipeline **falls back** to LangChain.
3. **Fallback — `RecursiveCharacterTextSplitter`**
  - Splits markdown with separators `["\n\n", "\n", ". ", " ", ""]`, using `chunk_size` and `chunk_overlap` from config (scaled in code for character length).  
  - `section_heading` is inferred via nearest markdown heading before the chunk (`_extract_nearest_heading`).  
  - `page_number` is not set in fallback mode.
4. **Downstream**
  - Chunks are contextualized (`services/contextualizer.py`), embedded (`services/embedder.py`), and stored (`db/vector_store.py`). Search and prompts prefer `contextual_text` when present.

---

## Retrieval method

Hybrid retrieval lives in `backend/services/retriever.py` and uses Supabase RPCs defined in `init_supabase_schema.py`.

1. **Query embedding** — The user question is embedded with OpenAI via `embed_texts()` (`services/embedder.py`).
2. **Two parallel legs (per document)**
  - **Semantic:** `match_chunks` RPC — pgvector cosine similarity on stored embeddings.  
  - **Keyword:** `keyword_search_chunks` RPC — `pg_trgm` similarity on `contextual_text`.
3. **Fusion** — Results are merged with **Reciprocal Rank Fusion** (`_RRF_K = 60`): each list contributes `1 / (k + rank)` per chunk; combined score sorts a deduplicated candidate set.
4. **Reranking** — Top candidates (up to `retrieval_top_k_rerank_input`, default 15) are re-scored with the **cross-encoder** (`rerank_chunks` in `embedder.py`). Final context uses `retrieval_top_k_final` (default 3) unless overridden by `top_k` in the Ask API.
5. **Batch / cross-document** — `agents/qa_agent.py` runs retrieval per ready document, merges candidates, and applies a **global** rerank so scores are comparable across files; chunks carry `document_name` for attribution in `AskResponse`.

---

## Guardrails approach

Guardrails are centered in `backend/services/guardrails.py` and wired through `agents/qa_agent.py` and `api/ask.py`.

1. **Pre-generation relevance gate** — `check_similarity_gate()` runs **before** the main LLM answer. It uses the best available score: **sigmoid(cross-encoder logit)** when `rerank_score` exists, else cosine similarity from pgvector. If the score is below `guardrail_similarity_threshold` (default `0.25`), it raises `GuardrailTriggered` — the API surfaces this as a refusal path (`guardrail_triggered`, empty `chunks_used`, `source_documents: null`).
2. **Refusal alignment** — `is_refusal()` detects standardized “not in document” style answers so scores and flags stay consistent with user-facing behavior.
3. **No extra gate on faithfulness** — Faithfulness and claim coverage are evaluated **after** an answer is produced (for scoring and `confidence_score`), unless the similarity gate already stopped the pipeline.

Threshold tuning notes are documented on `Settings.guardrail_similarity_threshold` in `config.py` (lower for form-like logistics PDFs, higher for dense prose).

---

## Confidence scoring method

Confidence is the **composite** from `evaluate_answer()` in `services/guardrails.py` (see also [RAGAS-style confidence implementation](#ragas-style-confidence-implementation-actual-code-path) above).


| Signal                              | Meaning in this codebase                                                | How it is computed                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Faithfulness**                    | Grounding of the answer in retrieved chunks                             | LLM judge returns `total_claims` / `supported_claims`; ratio = **RAGAS-style faithfulness**                              |
| `**answer_relevancy`** (API field)  | Responsiveness + concreteness                                           | LLM judge: binary `responsive` and `concrete`; score = mean                                                              |
| `**context_relevancy`** (API field) | Single-doc: top-chunk retrieval quality; cross-doc: **source coverage** | Single-doc: `sigmoid(rerank_score)` or cosine; cross-doc: fraction of distinct `document_name` in chunks vs expected set |


**Weighting:** single-document vs cross-document blends differ (`0.55/0.25/0.20` vs `0.60/0.20/0.20`).  
**Warning:** `low_confidence_warning` is true when `confidence_score < guardrail_low_confidence_threshold` (default `0.20`).  
**Edge cases:** empty chunks → zeros; refusals → zero composite with warning; judge JSON parse errors fall back to **0.5** on affected signals (logged).

---

## Failure cases


| Symptom                                         | Typical cause                                                            | Where it shows up                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| **404** on ask/extract                          | Unknown `job_id` or `batch_id`                                           | `api/ask.py`, `api/extract.py`                                             |
| **422** on ask/extract                          | Job not `ready`, or batch has no ready docs                              | Same                                                                       |
| **Guardrail / empty context**                   | Retrieval relevance below threshold                                      | `GuardrailTriggered`, `guardrail_triggered=true`                           |
| **“Not found in document.”**                    | Model refusal or off-topic question                                      | Prompt rules + `is_refusal()`                                              |
| **Ingestion failed**                            | Parse error, Docling failure, DB error                                   | Job `status=failed`, `error` on status API                                 |
| **Poor chunking**                               | Docling unusable → fallback splitter; very large chunks trigger fallback | `chunker.py`                                                               |
| **Confidence looks “stuck” at 0.5** on a signal | LLM judge returned non-JSON or malformed output                          | `guardrails.py` fallback                                                   |
| **Lost upload state**                           | Server restart                                                           | `ingestion_jobs.py` is **in-memory** only                                  |
| **Schema / DB mismatch**                        | Old DB without migrations                                                | Run `db/init_supabase_schema.py` (includes legacy column/index migrations) |


---

## Improvement ideas

### 1. LangSmith (observability)

Langsmith can trace **each LLM call** (QA, contextualizer, extraction, judge prompts in `guardrails.py`), record **token usage and latency**, and compare runs over time. Integration would wrap Anthropic/OpenAI calls or use LangChain callbacks where applicable, without changing core RAG logic — primarily instrumentation in `agents/*.py`, `services/contextualizer.py`, and `services/guardrails.py`.

### 2. Anthropic structured outputs

Anthropic’s **structured outputs** let the API return data constrained to a supplied schema instead of free text that must be cleaned (e.g. stripping ````json`fences and hoping`json.loads` succeeds).

**In this codebase:** `_compute_faithfulness`, `_compute_claim_coverage`, and extraction flows currently rely on **plain-text JSON** in the message body with manual cleanup. Structured outputs could:

- Return **typed** judge results (`total_claims`, `supported_claims`, `responsive`, `concrete`) with fewer parse failures and **no fence-stripping**.
- Align extraction even more tightly with `ShipmentData` / Pydantic models, reducing retry loops in PydanticAI.

### 3. Mem0 (long-term memory)

Mem0 is a **memory layer** for LLM apps: it stores and retrieves user- or session-level facts and preferences across conversations.

**Here,** each `/ask` is largely stateless aside from uploaded documents. Mem0 could remember **clarifications** (“when I say ‘the load’, I mean LD53657”), **frequently used filters**, or **broker-specific terminology** — injected as a short memory block into `qa_system_prompt.py` or the user message builder in `qa_agent.py`. Implementation would add a small adapter (API key, user/session IDs from your auth layer) and a **retrieve-then-merge** step before retrieval, being careful **not** to let memory override **grounded** answers from documents (memory as adjunct, not a source of logistics facts unless explicitly allowed).

### 4. Context relevancy (single-document): why it stalls near ~0.89–0.90 — and the best fix

For single-document queries, `context_relevancy` comes from `_compute_context_relevancy()` in `services/guardrails.py`, which uses `_best_relevance_score()` → **`sigmoid(rerank_score)`** when a cross-encoder `rerank_score` (logit) is present. A standard logistic maps moderate logits to the high‑but‑not‑saturated band (the in-file reference curve: logit **+2 → ~0.88**, **+4 → ~0.98**). So scores in the **~0.89–0.91** range frequently mean “strong but not extreme logit,” not necessarily a broken retriever. The same doc block also notes that logistics query/chunk pairs rarely produce extremely large logits.

**Single best improvement:** Replace the **standalone** `sigmoid(top_logit)` with a **per-query, candidate-relative score** computed over the **same reranked list** you already have when building context (the chunks passed into `evaluate_answer`, each ideally still carrying `rerank_score`). Concretely: take the rerank logits for those final candidates, apply **softmax across that set**, and set **`context_relevancy` to the softmax weight of the top-ranked chunk**. Optionally multiply by a **bounded** function of the top logit so that when *every* candidate’s logit is poor, the score does not pretend the winner is excellent. This measures **how dominant the winner is versus the other retrieved chunks for this question**, removes the artificial “everything tops out around 0.9 unless the top logit is very large” effect, and stays aligned with the cross-encoder ordering you already use in `services/embedder.py` and `services/retriever.py`.

---

## End-to-end workflow

### 1) Upload and ingestion

1. `POST /upload` with multipart `files`.
2. Validate extension, size, and batch count; compute SHA-256; write temp files.
3. Create `batch_id` and per-file `job_id`; return **202** immediately.
4. Background pipeline (bounded concurrency): parse → chunk → contextualize → embed → store; hash hit short-circuits to existing document.
5. Status: `pending → processing → ready | failed`.

### 2) Ask

1. `POST /ask` with `question`, `top_k`, and exactly one of `job_id` or `batch_id`.
2. Hybrid retrieve → RRF → cross-encoder rerank (global rerank for batch).
3. Claude generates answer from context only.
4. Guardrails and scoring; refusals set `guardrail_triggered: true`, `chunks_used: []`, `source_documents: null`.

### 3) Extraction

1. `POST /extract` with `job_id`, or `POST /extract/batch?batch_id=...`.
2. Resolve ready jobs; fetch chunk text; LLM → `ShipmentData` (or per-file batch rows with `success | failed | skipped`).

### 4) Status

- `GET /ingestion-status/{job_id}` — one job.
- `GET /ingestion-status/batch/{batch_id}` — aggregate counters and job list.

---

## API contracts (operational)


| Method | Path                                 | Notes                                                   |
| ------ | ------------------------------------ | ------------------------------------------------------- |
| POST   | `/upload`                            | Multipart `files`; **202** + `batch_id`, `jobs[]`       |
| GET    | `/ingestion-status/{job_id}`         | Job state, `document_id` when ready                     |
| GET    | `/ingestion-status/batch/{batch_id}` | Aggregate + `jobs[]`                                    |
| POST   | `/ask`                               | JSON: `question`, `top_k`, and `job_id` *or* `batch_id` |
| POST   | `/extract`                           | JSON: `job_id` → `ShipmentData`                         |
| POST   | `/extract/batch?batch_id=`           | Batch extraction summary + `results[]`                  |


OpenAPI entries for `/upload`, `/ask`, and `/extract` intentionally use minimal response documentation (success + **400** + **404** model hooks); the service may still return other HTTP codes (e.g. 422) for validation or readiness.

---

## Response shape notes you should rely on

### `/ask` response fields (`AskResponse`)

- `answer`
- `source_documents` (`null` when `guardrail_triggered=true`)
- `confidence_score`
- `faithfulness`
- `answer_relevancy`
- `context_relevancy`
- `guardrail_triggered`
- `chunks_used` (array of `{chunk_id, text, document_name}`)
- `low_confidence_warning`

### `/extract` response fields

`POST /extract` returns full `ShipmentData` (including `document_id` and `extraction_confidence`).

`POST /extract/batch` returns:

- `batch_id`, `total`, `succeeded`, `failed`, `skipped`
- `results[]` with `job_id`, `document_id`, `filename`, `status`, `shipment_data`, `error`

---

## Flowchart (sequence)

```mermaid
sequenceDiagram
    participant U as User / Frontend
    participant API as FastAPI
    participant JOB as Job registry
    participant ING as Ingestion worker
    participant DB as Supabase pgvector
    participant QA as QA pipeline
    participant EXT as Extraction

    U->>API: POST /upload
    API->>JOB: create batch + jobs
    API-->>U: 202 batch_id, job_ids
    API->>ING: background ingest
    ING->>DB: parse, chunk, contextualize, embed
    ING->>JOB: ready | failed

    U->>API: GET /ingestion-status/batch/{batch_id}
    API->>JOB: read state
    API-->>U: counters + jobs

    U->>API: POST /ask
    API->>JOB: resolve ready docs
    API->>QA: retrieve, rerank, Claude
    QA->>DB: hybrid + RPC
    QA-->>API: answer, chunks, scores
    API-->>U: AskResponse

    U->>API: POST /extract / extract/batch
    API->>JOB: resolve jobs
    API->>EXT: structured extract
    EXT->>DB: load chunks
    EXT-->>API: ShipmentData or batch results
    API-->>U: response
```



---

## Example scenarios (detailed)

### Example A — Batch Q&A: equipment from the rate confirmation only

**Setup:** One batch upload contains `LD53657-Carrier-RC.pdf` and `BOL53657_billoflading.pdf`. Both jobs reach `ready`.

**Request:**

```json
{
  "batch_id": "<your-batch-uuid>",
  "question": "What specific equipment type and size is required for this shipment according to the RC?",
  "top_k": 5
}
```

**Representative response shape:**

```json
{
  "answer": "The rate confirmation requires a 48 ft flatbed. Source: Equipment Requirements, Page 1",
  "source_documents": ["LD53657-Carrier-RC.pdf"],
  "confidence_score": 0.914,
  "faithfulness": 1.0,
  "answer_relevancy": 1.0,
  "context_relevancy": 0.82,
  "guardrail_triggered": false,
  "chunks_used": [
    {
      "chunk_id": "4a7568f2-3fd4-4f11-b6cf-2b1e3f6f8a10",
      "text": "Equipment required: Flatbed 48 ft ...",
      "document_name": "LD53657-Carrier-RC.pdf"
    }
  ],
  "low_confidence_warning": false
}
```

Retrieved context is ranked with a **global cross-encoder** pass so chunks from the BOL are less likely to appear in `chunks_used` when they are weak for this question.

### Example B — Off-topic question: guardrails and empty attribution

**Setup:** Same batch (logistics PDFs ingested and ready).

**Request:**

```json
{
  "batch_id": "<your-batch-uuid>",
  "question": "What is the most spoken language in the world?",
  "top_k": 5
}
```

**Representative response shape:**

```json
{
  "answer": "Not found in document.",
  "source_documents": null,
  "confidence_score": 0.0,
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_relevancy": 0.0,
  "guardrail_triggered": true,
  "chunks_used": [],
  "low_confidence_warning": true
}
```

This avoids implying that uploaded files supported an answer about general knowledge.

### Example C — Batch extraction with a not-ready file

**Request:** `POST /extract/batch?batch_id=<batch_uuid>` while one file is still `processing`.

**Expected behavior:** Ready jobs return `status: "success"` with `shipment_data`; not-ready jobs return `status: "skipped"` with an explanatory `error` string. The batch does not fail entirely because one file is still ingesting.

---

## Operational notes

- Job/batch state is **in-memory** and is lost on server restart.
- Upload is asynchronous; clients should poll ingestion status before Ask/Extract.
- Batch Ask/Extract only include **ready** documents; partial batches are supported.
- Concurrency is capped by `ingestion_concurrency` in config.
- Tuning: `config.py` and environment variables (`.env`).

---

