# backend/tools/check_document_tool.py

import asyncio
import json
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
try:
    from pydantic_ai.ext.langchain import tool_from_langchain
except ImportError:
    # pydantic_ai ext module is optional in some runtime builds.
    # Fallback keeps app startup working by passing through the original tool.
    def tool_from_langchain(tool):
        return tool

from db.vector_store import fetch_all_chunks

logger = logging.getLogger(__name__)


# Input schema

class CheckDocumentInput(BaseModel):
    """Input schema for the check_document tool."""

    document_id: str = Field(
        ...,
        description=(
            "UUID of the document to retrieve all chunks from. "
            "This is the document_id returned by POST /upload."
        ),
    )

# LangChain BaseTool implementation

class CheckDocumentTool(BaseTool):
    """
    Retrieve all chunks for a document in sequential order.

    Unlike the retrieve_chunks tool (which returns similarity-filtered top-K),
    this tool returns EVERY chunk ordered by chunk_index — providing full
    document context for structured field extraction.

    The Extraction Agent uses this to give Claude the entire document before
    filling the ShipmentData schema.
    """

    name: str = "check_document"
    description: str = (
        "Retrieve ALL chunks from a document in sequential order (by chunk_index). "
        "Use this tool to get the complete document content for structured data extraction. "
        "Unlike retrieve_chunks (which does similarity search), this returns every chunk "
        "in reading order — essential for extraction tasks where any field could be "
        "anywhere in the document. "
        "Input: document_id. "
        "Output: JSON object with all chunks and document metadata."
    )
    args_schema: Type[BaseModel] = CheckDocumentInput
    return_direct: bool = False

    def _run(
        self,
        document_id: str,
        run_manager: Optional[object] = None,
    ) -> str:
        try:
            result = asyncio.run(self._fetch_async(document_id))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._fetch_async(document_id))
        return result

    async def _fetch_async(self, document_id: str) -> str:
        """
        Fetch all chunks for a document and build a JSON response.
        Includes summary fields such as total chunks and token estimate.
        """
        try:
            chunks = await fetch_all_chunks(document_id=document_id)
        except Exception as exc:
            logger.error(f"check_document failed for {document_id}: {exc}")
            return json.dumps({
                "status":  "error",
                "message": str(exc),
                "document_id": document_id,
                "chunks":  [],
            })

        if not chunks:
            return json.dumps({
                "status":       "not_found",
                "message":      f"No chunks found for document_id={document_id}. "
                                "The document may not have been ingested yet.",
                "document_id":  document_id,
                "chunks":       [],
                "total_chunks": 0,
            })

        # Summarise chunk stats for the agent
        total_tokens = sum(c.get("token_count", 0) for c in chunks)
        sections     = list(dict.fromkeys(
            c.get("section_heading") for c in chunks
            if c.get("section_heading")
        ))

        logger.info(
            f"check_document: doc={document_id[:8]} | "
            f"chunks={len(chunks)} | total_tokens≈{total_tokens}"
        )

        return json.dumps({
            "status":        "success",
            "document_id":   document_id,
            "total_chunks":  len(chunks),
            "total_tokens":  total_tokens,
            "sections":      sections,
            "chunks":        chunks,
        }, ensure_ascii=False, default=str)


# Wrap for PydanticAI

# Instantiate the LangChain tool

_check_document_lc_tool = CheckDocumentTool()
check_document_tool = tool_from_langchain(_check_document_lc_tool)
