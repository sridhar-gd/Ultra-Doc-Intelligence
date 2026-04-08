# backend/utils/confidence.py

from services.guardrails import (
    evaluate_answer,
    check_similarity_gate,
    GuardrailTriggered,
)

__all__ = [
    "evaluate_answer",
    "check_similarity_gate",
    "GuardrailTriggered",
    "build_confidence_report",
]


def build_confidence_report(
    answer: str,
    chunks: list[dict],
    question: str = "",
) -> dict:
    """
    Build a confidence report from answer, chunks, and question.

    Delegates entirely to evaluate_answer() — the single source of truth
    for all scoring logic. Weights are query-type aware:

    Single-document:
        0.55 * faithfulness + 0.25 * claim_coverage + 0.20 * context_relevancy

    Cross-document (batch_id query, len(unique document_names in chunks) > 1):
        0.60 * faithfulness + 0.20 * claim_coverage + 0.20 * source_coverage

    The `weights_used` field in the returned dict reflects the branch taken.
    `answer_relevancy` in the result dict holds the claim_coverage score
    for API compatibility with the AskResponse schema.
    """
    if not chunks:
        return {
            "confidence_score":       0.0,
            "faithfulness":           0.0,
            "answer_relevancy":       0.0,
            "context_relevancy":      0.0,
            "low_confidence_warning": True,
            "weights_used": {
                "faithfulness":      0.55,
                "answer_relevancy":  0.25,
                "context_relevancy": 0.20,
            },
        }

    # Delegate to evaluate_answer — the single source of truth for scoring.
    # evaluate_answer detects cross-doc vs single-doc from the chunks themselves
    result = evaluate_answer(answer=answer, chunks=chunks, question=question)

    # Determine which weight branch was taken so callers can audit/log it.
    unique_docs = {
        str(c.get("document_name", "")).strip()
        for c in chunks
        if str(c.get("document_name", "")).strip()
    }
    is_cross_doc = len(unique_docs) > 1

    if is_cross_doc:
        result["weights_used"] = {
            "faithfulness":      0.60,
            "answer_relevancy":  0.20,   # claim_coverage stored here
            "context_relevancy": 0.20,   # source_coverage stored here
        }
    else:
        result["weights_used"] = {
            "faithfulness":      0.55,
            "answer_relevancy":  0.25,   # claim_coverage stored here
            "context_relevancy": 0.20,
        }

    return result