# backend/tools/store_document_tool.py

import asyncio
import json
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pydantic_ai.ext.langchain import tool_from_langchain

from db.vector_store import (
    insert_document,
    insert_chunks_and_embeddings,
    update_document_status,
)
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Input schema

class StoreDocumentInput(BaseModel):
    """Input schema for the store_document tool."""

    filename: str = Field(..., description="Original filename of the uploaded document.")
    doc_type: str = Field(..., description="Detected doc type: carrier_rc | shipper_rc | bol | invoice | unknown")
    file_size_bytes: int = Field(..., description="File size in bytes.")
    load_id: Optional[str] = Field(None, description="Detected load ID (e.g. LD53657) or null.")
    raw_markdown: str = Field(..., description="Full Docling markdown output of the document.")
    contextualised_chunks_json: str = Field(
        ...,
        description=(
            "JSON string representing the list of contextualised chunk objects. "
            "Each object must have: chunk_text, contextual_text, chunk_index, "
            "page_number, section_heading, doc_type, load_id, token_count."
        ),
    )
    embedding_vectors_json: str = Field(
        ...,
        description=(
            "JSON string representing list of embedding vectors (list of list of float). "
            "Must be same length as contextualised_chunks_json."
        ),
    )


# LangChain BaseTool implementation

class StoreDocumentTool(BaseTool):
    """
    Persist a processed document to Supabase pgvector.

    Steps performed:
      1. Insert document row (status=processing)
      2. Bulk insert chunk rows
      3. Bulk insert embedding vectors
      4. Update document status → ready

    Returns a JSON object with the document_id and insertion stats.
    """

    name: str = "store_document"
    description: str = (
        "Store a fully processed document (chunks + embeddings) into the Supabase "
        "vector database. Call this tool AFTER parsing, chunking, contextualising, "
        "and embedding the document — as the final ingestion step. "
        "Input: document metadata, contextualised chunks JSON, and embedding vectors JSON. "
        "Output: document_id and chunk count."
    )
    args_schema: Type[BaseModel] = StoreDocumentInput
    return_direct: bool = False

    def _run(
        self,
        filename: str,
        doc_type: str,
        file_size_bytes: int,
        raw_markdown: str,
        contextualised_chunks_json: str,
        embedding_vectors_json: str,
        load_id: Optional[str] = None,
        run_manager: Optional[object] = None,
    ) -> str:
        try:
            result = asyncio.run(
                self._store_async(
                    filename=filename,
                    doc_type=doc_type,
                    file_size_bytes=file_size_bytes,
                    load_id=load_id,
                    raw_markdown=raw_markdown,
                    contextualised_chunks_json=contextualised_chunks_json,
                    embedding_vectors_json=embedding_vectors_json,
                )
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self._store_async(
                    filename=filename,
                    doc_type=doc_type,
                    file_size_bytes=file_size_bytes,
                    load_id=load_id,
                    raw_markdown=raw_markdown,
                    contextualised_chunks_json=contextualised_chunks_json,
                    embedding_vectors_json=embedding_vectors_json,
                )
            )
        return result

    async def _store_async(
        self,
        filename: str,
        doc_type: str,
        file_size_bytes: int,
        load_id: Optional[str],
        raw_markdown: str,
        contextualised_chunks_json: str,
        embedding_vectors_json: str,
    ) -> str:
        """
        Store a processed document and related vectors in Supabase.
        Handles insert, embedding persistence, and final status update.
        """
        # Parse JSON inputs
        try:
            ctx_chunks_data: list[dict] = json.loads(contextualised_chunks_json)
            embedding_vectors: list[list[float]] = json.loads(embedding_vectors_json)
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "message": f"JSON parse error: {e}"})

        if len(ctx_chunks_data) != len(embedding_vectors):
            return json.dumps({
                "status":  "error",
                "message": (
                    f"Chunk/vector count mismatch: "
                    f"{len(ctx_chunks_data)} chunks vs {len(embedding_vectors)} vectors."
                ),
            })

        document_id: str | None = None
        try:
            # Step 1: Insert document record
            document_id = await insert_document(
                document_name=filename,
                doc_type=doc_type,
                file_size_bytes=file_size_bytes,
                load_id=load_id,
                raw_markdown=raw_markdown,
            )

            # Step 2 & 3: Reconstruct ContextualisedChunk objects from JSON
            from dataclasses import dataclass

            @dataclass
            class _FlatChunk:
                chunk_text: str
                chunk_index: int
                document_id: str
                doc_type: str
                load_id: str | None
                page_number: int | None
                section_heading: str | None
                token_count: int

            from services.contextualizer import ContextualisedChunk as CC

            ctx_chunks_rebuilt = []
            for item in ctx_chunks_data:
                flat = _FlatChunk(
                    chunk_text=item.get("chunk_text", ""),
                    chunk_index=item.get("chunk_index", 0),
                    document_id=document_id,
                    doc_type=item.get("doc_type", doc_type),
                    load_id=item.get("load_id", load_id),
                    page_number=item.get("page_number"),
                    section_heading=item.get("section_heading"),
                    token_count=item.get("token_count", 0),
                )
                ctx_chunks_rebuilt.append(
                    CC(
                        original_chunk=flat,  
                        context_prefix=item.get("context_prefix", ""),
                        contextual_text=item.get("contextual_text", item.get("chunk_text", "")),
                    )
                )

            chunks_inserted = await insert_chunks_and_embeddings(
                contextualised_chunks=ctx_chunks_rebuilt,
                embedding_vectors=embedding_vectors,
                model_name=settings.embedding_model_name,
            )

            # Step 4: Mark document as ready
            await update_document_status(
                document_id=document_id,
                status="ready",
                chunks_count=chunks_inserted,
            )

            logger.info(
                f"store_document: '{filename}' stored | "
                f"doc_id={document_id} | chunks={chunks_inserted}"
            )

            return json.dumps({
                "status":         "success",
                "document_id":    document_id,
                "chunks_stored":  chunks_inserted,
                "doc_type":       doc_type,
                "load_id":        load_id,
            })

        except Exception as exc:
            logger.error(f"store_document failed for '{filename}': {exc}")
            if document_id:
                await update_document_status(document_id, status="failed")
            return json.dumps({
                "status":  "error",
                "message": str(exc),
                "document_id": document_id,
            })


# Wrap for PydanticAI

# Instantiate the LangChain tool

_store_document_lc_tool = StoreDocumentTool()
store_document_tool = tool_from_langchain(_store_document_lc_tool)