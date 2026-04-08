# backend/services/embedder.py

import asyncio
import logging
from functools import lru_cache

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_openai_embedder():
    """
    Load and cache the OpenAI embeddings client.
    Reused across requests to avoid repeated initialization cost.
    """
    from langchain_openai import OpenAIEmbeddings

    model_name = settings.embedding_model_name  # e.g. "text-embedding-3-small"
    logger.info(f"Initialising OpenAIEmbeddings: {model_name}...")
    embedder = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.openai_api_key,
    )
    logger.info(f"OpenAIEmbeddings ready: {model_name}")
    return embedder


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """
    Load and cache the cross-encoder reranker model.
    This model is reused for all rerank operations.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model_name = settings.reranker_model_name    # "BAAI/bge-reranker-base"
    logger.info(f"Loading BAAI cross-encoder: {model_name}...")
    model = HuggingFaceCrossEncoder(model_name=model_name)
    logger.info(f"BAAI cross-encoder ready: {model_name}")
    return model


# Embedding — OpenAI via LangChain

def _embed_sync(texts: list[str]) -> list[list[float]]:
    """
    Synchronously embed a list of texts.
    Intended to run in a worker thread via asyncio.to_thread.
    """
    if not texts:
        return []

    embedder = _get_openai_embedder()

    vectors: list[list[float]] = embedder.embed_documents(texts)

    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Asynchronously embed texts with the cached embedding model.
    Returns an empty list when no input texts are provided.
    """
    if not texts:
        return []
    return await asyncio.to_thread(_embed_sync, texts)

# Re-ranking — cross-encoder

def _rerank_sync(query: str, chunk_texts: list[str]) -> list[float]:
    """
    Synchronously score chunk relevance with the cross-encoder.
    Higher scores indicate stronger query-chunk relevance.
    """
    if not chunk_texts:
        return []

    model = _get_cross_encoder()
    pairs = [(query, chunk) for chunk in chunk_texts]
    
    scores: list[float] = model.score(pairs)

    return scores


async def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Re-rank chunks by cross-encoder score.
    Optionally truncate results to the top-K entries.
    """
    if not chunks:
        return []

    chunk_texts = [
        c.get("contextual_text") or c.get("chunk_text", "")
        for c in chunks
    ]

    scores: list[float] = await asyncio.to_thread(_rerank_sync, query, chunk_texts)

    scored_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]

    scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

    if top_k is not None:
        scored_chunks = scored_chunks[:top_k]

    if scored_chunks:
        logger.debug(
            f"Re-ranked {len(chunks)} chunks → returning top {len(scored_chunks)}. "
            f"Top score: {scored_chunks[0]['rerank_score']:.4f}"
        )

    return scored_chunks