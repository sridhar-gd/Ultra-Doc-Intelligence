# backend/agents/ingestion_agent.py

import logging
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from services.parser import parse_document
from services.chunker import chunk_document
from services.contextualizer import contextualise_chunks
from services.embedder import embed_texts
from db.vector_store import (
    insert_document,
    insert_chunks_and_embeddings,
    update_document_status,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionResult(BaseModel):
    """Structured result of the full document ingestion pipeline."""

    document_id: str
    document_name: str
    load_id: str | None = None
    chunks_created: int
    page_count: int = 0
    table_count: int = 0
    status: str = "ready"
    cached: bool = False


ingestion_agent = Agent(
    model=AnthropicModel(
        model_name=settings.anthropic_model,
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    ),
    output_type=IngestionResult,
    system_prompt=(
        "You are a document ingestion coordinator. "
        "When given an ingestion result JSON, validate it and return it as a "
        "structured IngestionResult. Do not modify any values."
    ),
    retries=2,
)


async def run_ingestion(
    file_path: str,
    document_name: str,
    file_hash: str | None = None,
) -> IngestionResult:
    """Run the ingestion pipeline with optional hash-based dedup."""
    import os
    from db.vector_store import lookup_document_by_hash

    file_size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    document_id: str | None = None

    if file_hash:
        cached_doc = await lookup_document_by_hash(file_hash)
        if cached_doc:
            logger.info(
                f"[Ingestion] ♻️ Cache HIT '{document_name}' → doc_id={cached_doc['id']}"
            )
            return IngestionResult(
                document_id=cached_doc["id"],
                document_name=document_name,
                load_id=cached_doc.get("load_id"),
                chunks_created=cached_doc.get("chunks_count", 0),
                page_count=0,
                table_count=0,
                status="ready",
                cached=True,
            )

    try:
        logger.info(f"[Ingestion] Step 1/6: Parsing '{document_name}'...")
        parsed = await parse_document(file_path)

        logger.info(
            f"[Ingestion] Parsed: {parsed.page_count} pages | "
            f"{parsed.table_count} tables | type={parsed.detected_doc_type}"
        )

        logger.info("[Ingestion] Step 2/6: Creating document record...")
        document_id = await insert_document(
            document_name=document_name,
            file_size_bytes=file_size_bytes,
            load_id=parsed.detected_load_id,
            raw_markdown=parsed.markdown,
            file_hash=file_hash,
        )

        logger.info(f"[Ingestion] Step 3/6: Chunking (doc_id={document_id[:8]})...")
        chunks = chunk_document(parsed_doc=parsed, document_id=document_id)

        logger.info(f"[Ingestion] Produced {len(chunks)} chunks.")

        logger.info(f"[Ingestion] Step 4/6: Contextualising chunks...")
        ctx_chunks = await contextualise_chunks(
            chunks=chunks,
            doc_markdown=parsed.markdown,
            max_concurrency=5,
        )

        logger.info(f"[Ingestion] Contextualised {len(ctx_chunks)} chunks.")

        logger.info(f"[Ingestion] Step 5/6: Embedding chunks...")
        texts_to_embed = [c.contextual_text for c in ctx_chunks]
        vectors = await embed_texts(texts_to_embed)

        logger.info(f"[Ingestion] Generated {len(vectors)} embeddings.")

        logger.info("[Ingestion] Step 6/6: Storing in Supabase...")
        chunks_stored = await insert_chunks_and_embeddings(
            contextualised_chunks=ctx_chunks,
            embedding_vectors=vectors,
            model_name=settings.embedding_model_name,
        )

        await update_document_status(
            document_id=document_id,
            status="ready",
            chunks_count=chunks_stored,
        )

        result = IngestionResult(
            document_id=document_id,
            document_name=document_name,
            load_id=parsed.detected_load_id,
            chunks_created=chunks_stored,
            page_count=parsed.page_count,
            table_count=parsed.table_count,
            status="ready",
            cached=False,
        )

        logger.info(
            f"[Ingestion] ✅ Complete: doc_id={document_id} | chunks={chunks_stored}"
        )

        return result

    except Exception as exc:
        logger.error(
            f"[Ingestion] ❌ Failed for '{document_name}': {exc}",
            exc_info=True
        )

        if document_id:
            try:
                await update_document_status(document_id, status="failed")
            except Exception:
                pass

        raise RuntimeError(f"Ingestion failed for '{document_name}': {exc}") from exc