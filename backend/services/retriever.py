# backend/services/retriever.py

import asyncio
import logging

from db.client import get_supabase_client
from services.embedder import embed_texts, rerank_chunks
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_RRF_K = 60


def _reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
) -> list[dict]:
    """Fuse semantic and keyword results using reciprocal-rank fusion."""
    chunks_by_id: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        cid = chunk["chunk_id"]
        if cid not in chunks_by_id:
            chunks_by_id[cid] = {**chunk, "rrf_score": 0.0}
        chunks_by_id[cid]["rrf_score"] += 1.0 / (_RRF_K + rank)
        chunks_by_id[cid]["similarity"] = chunk.get("similarity", 0.0)

    for rank, chunk in enumerate(keyword_results, start=1):
        cid = chunk["chunk_id"]
        if cid not in chunks_by_id:
            chunks_by_id[cid] = {**chunk, "rrf_score": 0.0, "similarity": 0.0}
        chunks_by_id[cid]["rrf_score"] += 1.0 / (_RRF_K + rank)
        chunks_by_id[cid]["keyword_score"] = chunk.get("keyword_score", 0.0)

    fused = sorted(chunks_by_id.values(), key=lambda c: c["rrf_score"], reverse=True)

    logger.debug(
        f"RRF fusion: {len(semantic_results)} semantic + {len(keyword_results)} keyword "
        f"→ {len(fused)} unique candidates"
    )
    return fused


async def _semantic_search(
    query_vector: list[float],
    document_id: str,
    top_k: int,
) -> list[dict]:
    """Run semantic chunk search for one document."""
    client = get_supabase_client()

    response = (
        client
        .rpc(
            "match_chunks",
            {
                "query_embedding": query_vector,
                "document_id_filter": document_id,
                "match_count": top_k,
                "similarity_threshold": 0.0,
            },
        )
        .execute()
    )

    results: list[dict] = response.data or []
    logger.debug(f"Semantic search returned {len(results)} chunks.")
    return results


async def _keyword_search(
    query_text: str,
    document_id: str,
    top_k: int,
) -> list[dict]:
    """Run keyword chunk search for one document."""
    client = get_supabase_client()

    response = (
        client
        .rpc(
            "keyword_search_chunks",
            {
                "query_text": query_text,
                "document_id_filter": document_id,
                "match_count": top_k,
            },
        )
        .execute()
    )

    results: list[dict] = response.data or []
    logger.debug(f"Keyword search returned {len(results)} chunks.")
    return results


async def hybrid_retrieve(
    query: str,
    document_id: str,
    top_k_final: int | None = None,
    apply_reranking: bool = True,
) -> list[dict]:
    """Run hybrid retrieval and return final ranked chunks."""
    top_k_final = top_k_final or settings.retrieval_top_k_final

    # Step 1: Embed the query.
    query_vectors = await embed_texts([query])
    query_vector = query_vectors[0]

    # Step 2: Parallel semantic + keyword search
    semantic_results, keyword_results = await asyncio.gather(
        _semantic_search(
            query_vector=query_vector,
            document_id=document_id,
            top_k=settings.retrieval_top_k_semantic,
        ),
        _keyword_search(
            query_text=query,
            document_id=document_id,
            top_k=settings.retrieval_top_k_keyword,
        ),
    )

    # Step 3: RRF fusion
    fused_candidates = _reciprocal_rank_fusion(semantic_results, keyword_results)
    top_candidates   = fused_candidates[:settings.retrieval_top_k_rerank_input]

    if not top_candidates:
        logger.warning(f"No chunks retrieved for document {document_id}.")
        return []

    # Step 4: Cross-encoder re-ranking
    enable_reranking = apply_reranking and settings.retrieval_enable_reranking
    if enable_reranking and len(top_candidates) > 1:
        final_chunks = await rerank_chunks(
            query=query,
            chunks=top_candidates,
            top_k=top_k_final,
        )
    else:
        if apply_reranking and not settings.retrieval_enable_reranking:
            logger.info("Cross-encoder reranking disabled by config; using RRF-only ranking.")
        final_chunks = top_candidates[:top_k_final]

    if final_chunks:
        logger.info(
            f"Hybrid retrieval complete: {len(final_chunks)} final chunks | "
            f"top similarity={final_chunks[0].get('similarity', 0.0):.3f}"
        )

    return final_chunks
