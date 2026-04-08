# backend/db/vector_store.py

import logging
import uuid
from datetime import datetime, timezone

from db.client import get_supabase_client
from services.contextualizer import ContextualisedChunk

logger = logging.getLogger(__name__)


async def lookup_document_by_hash(file_hash: str) -> dict | None:
    """
    Check whether a 'ready' document with the given SHA-256 file hash exists.
    """
    import asyncio

    def _lookup():
        client = get_supabase_client()
        return (
            client.table("documents")
            .select("id, filename, load_id, chunks_count")
            .eq("file_hash", file_hash)
            .eq("status", "ready")
            .limit(1)
            .execute()
        )

    response = await asyncio.to_thread(_lookup)
    rows = response.data or []

    if rows:
        logger.info(
            f"Cache HIT for file_hash={file_hash[:12]}… → document_id={rows[0]['id']}"
        )
        return rows[0]

    logger.debug(f"Cache MISS for file_hash={file_hash[:12]}…")
    return None


async def insert_document(
    document_name: str,
    file_size_bytes: int,
    load_id: str | None,
    raw_markdown: str,
    file_hash: str | None = None,
) -> str:
    """
    Insert a new document record.
    """
    import asyncio

    document_id = str(uuid.uuid4())

    payload = {
        "id":              document_id,
        "filename":        document_name,
        "file_hash":       file_hash,
        "file_size_bytes": file_size_bytes,
        "load_id":         load_id,
        "raw_markdown":    raw_markdown,
        "status":          "processing",
        "chunks_count":    0,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }

    def _insert():
        client = get_supabase_client()
        return client.table("documents").insert(payload).execute()

    response = await asyncio.to_thread(_insert)

    if not response.data:
        raise RuntimeError(f"Failed to insert document record for '{document_name}'.")

    logger.info(f"Document record created: {document_id} ({document_name})")
    return document_id


async def fetch_document_filename(document_id: str) -> str | None:
    """
    Return original filename.
    """
    import asyncio

    def _fetch():
        client = get_supabase_client()
        return (
            client.table("documents")
            .select("filename")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )

    response = await asyncio.to_thread(_fetch)
    rows = response.data or []
    if not rows:
        return None

    return rows[0].get("filename")


async def update_document_status(
    document_id: str,
    status: str,
    chunks_count: int = 0,
) -> None:
    """
    Update document status after ingestion.
    """
    import asyncio

    payload = {
        "status":       status,
        "chunks_count": chunks_count,
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }

    def _update():
        client = get_supabase_client()
        return (
            client.table("documents")
            .update(payload)
            .eq("id", document_id)
            .execute()
        )

    await asyncio.to_thread(_update)
    logger.info(f"Document {document_id} status → {status} (chunks: {chunks_count})")


async def insert_chunks_and_embeddings(
    contextualised_chunks: list[ContextualisedChunk],
    embedding_vectors: list[list[float]],
    model_name: str,
) -> int:
    import asyncio

    if len(contextualised_chunks) != len(embedding_vectors):
        raise ValueError("Mismatch between chunks and embeddings.")

    if not contextualised_chunks:
        return 0

    chunk_payloads = []
    chunk_ids = []

    for ctx_chunk in contextualised_chunks:
        chunk_id = str(uuid.uuid4())
        chunk_ids.append(chunk_id)
        orig = ctx_chunk.original_chunk

        chunk_payloads.append({
            "id":               chunk_id,
            "document_id":      orig.document_id,
            "chunk_text":       orig.chunk_text,
            "contextual_text":  ctx_chunk.contextual_text,
            "chunk_index":      orig.chunk_index,
            "page_number":      orig.page_number,
            "section_heading":  orig.section_heading,
            "doc_type":         orig.doc_type,
            "load_id":          orig.load_id,
            "token_count":      orig.token_count,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        })

    embedding_payloads = []

    for chunk_id, ctx_chunk, vector in zip(
        chunk_ids, contextualised_chunks, embedding_vectors
    ):
        embedding_payloads.append({
            "id":          str(uuid.uuid4()),
            "chunk_id":    chunk_id,
            "document_id": ctx_chunk.original_chunk.document_id,
            "embedding":   vector,
            "model_name":  model_name,
            "created_at":  datetime.now(timezone.utc).isoformat(),
        })

    _BATCH_SIZE = 50

    def _insert_batch(table: str, payloads: list[dict]) -> None:
        client = get_supabase_client()
        for i in range(0, len(payloads), _BATCH_SIZE):
            batch = payloads[i: i + _BATCH_SIZE]
            response = client.table(table).insert(batch).execute()
            if not response.data:
                raise RuntimeError(f"Insert failed for table '{table}' batch {i}")

    await asyncio.to_thread(_insert_batch, "chunks", chunk_payloads)
    logger.info(f"Inserted {len(chunk_payloads)} chunks.")

    await asyncio.to_thread(_insert_batch, "embeddings", embedding_payloads)
    logger.info(f"Inserted {len(embedding_payloads)} embeddings.")

    return len(chunk_payloads)


async def fetch_all_chunks(document_id: str) -> list[dict]:
    import asyncio

    def _fetch():
        client = get_supabase_client()
        return (
            client.table("chunks")
            .select(
                "id, document_id, chunk_text, contextual_text, chunk_index, "
                "page_number, section_heading, doc_type, load_id, token_count"
            )
            .eq("document_id", document_id)
            .order("chunk_index", desc=False)
            .execute()
        )

    response = await asyncio.to_thread(_fetch)
    chunks = response.data or []

    logger.info(f"Fetched {len(chunks)} chunks for document {document_id}.")
    return chunks


async def delete_document(document_id: str) -> None:
    import asyncio

    def _delete():
        client = get_supabase_client()
        return (
            client.table("documents")
            .delete()
            .eq("id", document_id)
            .execute()
        )

    await asyncio.to_thread(_delete)
    logger.info(f"Deleted document {document_id} and related data.")