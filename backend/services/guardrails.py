# backend/services/guardrails.py

import json
import logging
import math
from functools import lru_cache

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Exception

class GuardrailTriggered(Exception):
    """
    Raised by check_similarity_gate() when the best retrieved chunk does not
    meet the minimum relevance threshold.

    Attributes:
        reason: Human-readable explanation (logged server-side only, NOT in API).
        score:  The relevance score that triggered the gate (for debug logging).
    """
    def __init__(self, reason: str, score: float = 0.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.score  = score


# Math helpers

def _sigmoid(x: float) -> float:
    """
    Normalise a BAAI cross-encoder logit (unbounded float) to [0, 1].
        -4 → 0.018   -2 → 0.119   0 → 0.500   +2 → 0.880   +4 → 0.982
    """
    return 1.0 / (1.0 + math.exp(-x))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two float vectors, clipped to [0, 1].
    Returns 0.0 for empty or mismatched vectors.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def _best_relevance_score(chunk: dict) -> tuple[float, str]:
    """
    Return the best available retrieval relevance score for a chunk.

    Priority:
        1. sigmoid(rerank_score) — cross-encoder logit, most accurate
        2. similarity            — raw cosine from pgvector, fallback

    Returns:
        (score: float in [0,1], label: str for logging)
    """
    raw_rerank = chunk.get("rerank_score")
    if raw_rerank is not None:
        normalised = _sigmoid(float(raw_rerank))
        return normalised, f"rerank(sigmoid({raw_rerank:.3f})={normalised:.3f})"

    cosine = float(chunk.get("similarity", 0.0))
    return cosine, f"cosine({cosine:.3f})"


def _is_refusal(answer: str) -> bool:
    """Return True if the LLM response is a 'not found' refusal."""
    lower = answer.lower().strip()
    return any(p in lower for p in [
        "not found in document",
        "cannot be found",
        "no information",
        "not mentioned",
        "not specified",
        "not available",
        "not provided",
    ])


def is_refusal(answer: str) -> bool:
    """Public alias — used by QA pipeline to align API guardrail flags with refusals."""
    return _is_refusal(answer)


# Lightweight Claude sonnet caller (for scoring only)

@lru_cache(maxsize=1)
def _get_anthropic_client():
    """Cached Anthropic SDK client — created once per process."""
    import anthropic
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def _sonnet_call(prompt: str, max_tokens: int = 200) -> str:
    """
    Lightweight Claude sonnet API call for confidence scoring tasks.

    Uses claude-sonnet-4-5-20251001 — the fastest, cheapest Claude model.
    Scoring prompts are structured (JSON output), so model intelligence
    requirements are minimal. temperature=0 for deterministic results.

    Args:
        prompt:     User message text (structured scoring prompt).
        max_tokens: Token budget. 200 covers claim lists, 80 covers one question.

    Returns:
        Raw response string from Claude sonnet.
    """
    client = _get_anthropic_client()
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# Signal 1 — Faithfulness

_FAITHFULNESS_PROMPT = """\
You are a strict factual grounding checker for a logistics document QA system.

RETRIEVED CONTEXT (from the document):
{context}

ANSWER TO CHECK:
{answer}

YOUR TASK:
Step 1: List every distinct factual claim made in the ANSWER.
        A claim is any specific assertion: a number, name, date, address, weight,
        description, or any other concrete fact stated in the answer.
Step 2: For each claim, check whether the RETRIEVED CONTEXT directly supports it.
        IMPORTANT: Numeric formatting differences are NOT mismatches.
          "#10000" and "10,000 units" refer to the same fact → SUPPORTED
          "56000 lbs" and "56,000 lbs" are identical → SUPPORTED
          Reasonable paraphrasing of the same fact → SUPPORTED
Step 3: Count total_claims and supported_claims.

Respond with ONLY a valid JSON object, no explanation, no markdown code fences:
{{"total_claims": <integer>, "supported_claims": <integer>}}

Special cases:
- If the answer is empty or says "not found in document" → {{"total_claims": 0, "supported_claims": 0}}
- If the answer makes no verifiable claims → {{"total_claims": 1, "supported_claims": 0}}
"""

def _compute_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Measure the fraction of answer claims supported by retrieved context.

    Uses Claude sonnet as LLM judge. sonnet understands semantic equivalence
    between "#10000" and "10,000 units", so formatting differences in structured
    documents do NOT penalise correct answers.

    Score = supported_claims / total_claims  (RAGAS faithfulness formula)

    Returns:
        Float in [0, 1].
        1.0 = all claims are grounded in context (perfect faithfulness)
        0.0 = refusal response, or no claims are supported (hallucination)
        0.5 = fallback on API/parse error (neutral, non-penalising)
    """
    if _is_refusal(answer) or not answer.strip():
        return 0.0

    context_str = "\n\n---\n\n".join(
        c.get("contextual_text") or c.get("chunk_text", "")
        for c in chunks
    ).strip()

    if not context_str:
        return 0.0

    prompt = _FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)

    try:
        raw   = _sonnet_call(prompt, max_tokens=150)
        clean = raw.replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)

        total     = int(data.get("total_claims", 0))
        supported = int(data.get("supported_claims", 0))

        if total == 0:
            return 0.0

        score = max(0.0, min(1.0, supported / total))
        logger.debug(f"[Faithfulness] {supported}/{total} claims → {score:.3f}")
        return score

    except Exception as exc:
        logger.warning(f"[Faithfulness] Scoring error ({exc}) → fallback 0.5")
        return 0.5   # neutral fallback


# Signal 2 — Claim Coverage

_CLAIM_COVERAGE_PROMPT = """\
You are evaluating whether a logistics QA answer is responsive and concrete.

QUESTION:
{question}

ANSWER:
{answer}

Score the answer on two dimensions:
1) responsive (0 or 1): Does the answer directly address what was asked?
   - For comparison questions: does it compare the requested entities?
   - For factual questions: does it provide the requested fact?
2) concrete (0 or 1): Does it include specific values, dates, amounts, names,
   or a clear finding (not vague hedging)?

Respond with ONLY valid JSON, no markdown:
{{"responsive": 0_or_1, "concrete": 0_or_1}}
"""

def _compute_claim_coverage(question: str, answer: str) -> float:
    """
    Score whether an answer is directly responsive and concrete.

    This replaces reverse-question embedding similarity for better stability on
    comparison-style questions. The score is the mean of two binary judgments:
    responsive and concrete.
    """
    if _is_refusal(answer) or not answer.strip():
        return 0.0

    try:
        raw = _sonnet_call(_CLAIM_COVERAGE_PROMPT.format(question=question, answer=answer), max_tokens=120).strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        responsive = 1 if int(data.get("responsive", 0)) == 1 else 0
        concrete = 1 if int(data.get("concrete", 0)) == 1 else 0
        score = (responsive + concrete) / 2.0
        logger.debug(
            f"[ClaimCoverage] responsive={responsive} concrete={concrete} → {score:.3f}"
        )
        return score
    except Exception as exc:
        logger.warning(f"[ClaimCoverage] Scoring failed ({exc}) → fallback 0.5")
        return 0.5

# Signal 3 — Context Relevancy

def _compute_context_relevancy(top_chunk: dict) -> float:
    """
    Measure how relevant the retrieved context is to the question.

    Uses sigmoid(rerank_score) from the BAAI cross-encoder — already computed
    during retrieval, zero extra latency. Falls back to cosine similarity.

    NOTE: Scores of 0.85-0.95 are expected and correct. The sigmoid function
    asymptotes toward 1.0 — reaching it would require a rerank logit of ~∞.
    A logit of +4.0 maps to sigmoid(4.0) = 0.982. Real logistics documents
    against a BAAI cross-encoder rarely produce logits that high.

    Returns:
        Float in [0, 1].
    """
    score, label = _best_relevance_score(top_chunk)
    logger.debug(f"[ContextRelevancy] {label}")
    return score


def _compute_source_coverage(chunks: list[dict], expected_docs: set[str]) -> float:
    """
    Measure how many expected documents are represented in retrieved chunks.

    Used for cross-document queries where a single top-chunk similarity is not
    enough to judge retrieval quality across multiple sources.
    """
    if not expected_docs:
        return 0.5

    retrieved_docs = {
        str(c.get("document_name", "")).strip()
        for c in chunks
        if str(c.get("document_name", "")).strip()
    }
    covered = len(retrieved_docs.intersection(expected_docs))
    score = covered / len(expected_docs)
    logger.debug(
        f"[SourceCoverage] covered={covered}/{len(expected_docs)} docs → {score:.3f}"
    )
    return max(0.0, min(1.0, score))


# Hard gate

def check_similarity_gate(
    top_similarity: float,
    top_chunk: dict | None = None,
) -> None:
    """
    Hard relevance gate. Raises GuardrailTriggered if the best retrieved
    chunk does not meet the minimum relevance threshold.

    Runs BEFORE LLM generation and confidence scoring — fast, no API cost.
    Uses sigmoid(rerank_score) when available (most accurate), falls back to
    raw cosine similarity.

    Threshold calibration (config.py):
        0.25 — structured form docs (BOL, RC, Invoice): low similarity is EXPECTED
        0.55+ — dense prose docs (manuals, contracts)

    Args:
        top_similarity: Raw cosine similarity (used as fallback).
        top_chunk:      Top chunk dict — provides rerank_score when available.

    Raises:
        GuardrailTriggered: If best score < threshold. Reason logged only.
    """
    threshold = settings.guardrail_similarity_threshold

    if top_chunk is not None:
        score, score_label = _best_relevance_score(top_chunk)
    else:
        score       = float(top_similarity)
        score_label = f"cosine({score:.3f})"

    if score < threshold:
        reason = (
            f"Retrieval relevance below threshold "
            f"({score_label} < {threshold}). "
            "Document does not appear to contain a relevant answer."
        )
        logger.warning(f"Guardrail triggered: {score_label} < threshold={threshold}")
        raise GuardrailTriggered(reason=reason, score=score)

    logger.debug(f"Guardrail passed: {score_label} >= threshold={threshold}")


# Composite confidence score

def evaluate_answer(
    answer: str,
    chunks: list[dict],
    question: str = "",
) -> dict:
    """
    Compute confidence with query-aware scoring.

    Single-document weighting:
        0.55 * faithfulness + 0.25 * claim_coverage + 0.20 * context_relevancy
    Cross-document weighting:
        0.60 * faithfulness + 0.20 * claim_coverage + 0.20 * source_coverage

    For API compatibility, claim_coverage is returned in `answer_relevancy`, and
    source_coverage (cross-doc) or context_relevancy (single-doc) is returned in
    `context_relevancy`.
    """
    if not chunks:
        return {
            "confidence_score":       0.0,
            "faithfulness":           0.0,
            "answer_relevancy":       0.0,
            "context_relevancy":      0.0,
            "low_confidence_warning": True,
        }

    # Fast path: refusals always score zero
    if _is_refusal(answer):
        return {
            "confidence_score":       0.0,
            "faithfulness":           0.0,
            "answer_relevancy":       0.0,
            "context_relevancy":      round(_compute_context_relevancy(chunks[0]), 3),
            "low_confidence_warning": True,
        }

    top_chunk = chunks[0]

    # Signal 1: Faithfulness
    faithfulness = _compute_faithfulness(answer, chunks)

    # Signal 2: Claim coverage (stored in response field `answer_relevancy`)
    if question.strip():
        answer_relevancy = _compute_claim_coverage(question, answer)
    else:
        answer_relevancy = 0.5
        logger.debug("[ClaimCoverage] No question provided → neutral 0.5")

    expected_docs = {
        str(c.get("document_name", "")).strip()
        for c in chunks
        if str(c.get("document_name", "")).strip()
    }
    is_cross_doc = len(expected_docs) > 1

    # Signal 3: Retrieval quality (query-type aware)
    if is_cross_doc:
        context_relevancy = _compute_source_coverage(chunks, expected_docs)
    else:
        context_relevancy = _compute_context_relevancy(top_chunk)

    # Composite
    if is_cross_doc:
        confidence_score = round(
            0.60 * faithfulness
            + 0.20 * answer_relevancy
            + 0.20 * context_relevancy,
            3,
        )
    else:
        confidence_score = round(
            0.55 * faithfulness
            + 0.25 * answer_relevancy
            + 0.20 * context_relevancy,
            3,
        )
    confidence_score = max(0.0, min(1.0, confidence_score))

    low_confidence_warning = confidence_score < settings.guardrail_low_confidence_threshold

    logger.info(
        f"[Confidence] faithfulness={faithfulness:.3f} | "
        f"claim_coverage={answer_relevancy:.3f} | "
        f"context_relevancy={context_relevancy:.3f} | "
        f"is_cross_doc={is_cross_doc} | "
        f"composite={confidence_score:.3f} | "
        f"low_warning={low_confidence_warning}"
    )

    return {
        "confidence_score":       confidence_score,
        "faithfulness":           round(faithfulness, 3),
        "answer_relevancy":       round(answer_relevancy, 3),
        "context_relevancy":      round(context_relevancy, 3),
        "low_confidence_warning": low_confidence_warning,
    }