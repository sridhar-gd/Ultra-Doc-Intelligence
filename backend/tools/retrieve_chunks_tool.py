# backend/tools/retrieve_chunks_tool.py

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

from services.retriever import hybrid_retrieve
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Input schema

class RetrieveChunksInput(BaseModel):
    """Input schema for the retrieve_chunks tool."""

    query: str = Field(
        ...,
        description=(
            "The natural-language question or search query to retrieve relevant "
            "document chunks for. Should be the user's exact question."
        ),
    )
    document_id: str = Field(
        ...,
        description="UUID of the document to search within (from POST /upload response).",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Number of final chunks to retrieve after re-ranking (1-10). "
            "Default is 3. Use higher values for complex multi-part questions."
        ),
    )


# LangChain BaseTool implementation

class RetrieveChunksTool(BaseTool):
    """
    Hybrid semantic + keyword chunk retrieval tool.

    Performs a two-leg search:
      1. Dense leg: pgvector cosine similarity over configured embedding vectors
      2. Sparse leg: pg_trgm trigram similarity (BM25-style keyword match)
      3. RRF fusion: Reciprocal Rank Fusion to merge both result lists
      4. Cross-encoder re-ranking for final ordering

    Returns the top-K most relevant chunks as a JSON string.
    """

    name: str = "retrieve_chunks"
    description: str = (
        "Retrieve the most relevant document chunks for a query using hybrid "
        "semantic + keyword search and cross-encoder re-ranking. "
        "Use this tool FIRST whenever answering a question about a document. "
        "Input: the user's question and the document_id. "
        "Output: JSON array of the top relevant chunks with their text, section, "
        "page number, and similarity scores."
    )
    args_schema: Type[BaseModel] = RetrieveChunksInput
    return_direct: bool = False   # Result is passed back to the agent, not user

    def _run(
        self,
        query: str,
        document_id: str,
        top_k: int = 3,
        run_manager: Optional[object] = None,
    ) -> str:
        """
        Execute retrieval in a synchronous tool context.
        Wraps the async retriever and returns a JSON string payload.
        """
        try:
            chunks = asyncio.run(
                hybrid_retrieve(
                    query=query,
                    document_id=document_id,
                    top_k_final=top_k,
                    apply_reranking=True,
                )
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            chunks = loop.run_until_complete(
                hybrid_retrieve(
                    query=query,
                    document_id=document_id,
                    top_k_final=top_k,
                    apply_reranking=True,
                )
            )

        if not chunks:
            return json.dumps({
                "status": "no_results",
                "message": "No relevant chunks found in the document for this query.",
                "chunks": [],
                "top_similarity": 0.0,
            })

        # Serialise chunks to JSON for the agent to consume
        serialised = []
        for chunk in chunks:
            serialised.append({
                "chunk_id":        chunk.get("chunk_id"),
                "chunk_text":      chunk.get("chunk_text", ""),
                "contextual_text": chunk.get("contextual_text", ""),
                "section_heading": chunk.get("section_heading"),
                "page_number":     chunk.get("page_number"),
                "similarity":      round(chunk.get("similarity", 0.0), 4),
                "rerank_score":    round(chunk.get("rerank_score", 0.0), 4),
                "rrf_score":       round(chunk.get("rrf_score", 0.0), 6),
            })

        top_similarity = serialised[0]["similarity"] if serialised else 0.0

        result = {
            "status":         "success",
            "chunks_found":   len(serialised),
            "top_similarity": top_similarity,
            "chunks":         serialised,
        }

        logger.info(
            f"retrieve_chunks: query='{query[:50]}...' | "
            f"doc={document_id[:8]} | found={len(serialised)} | "
            f"top_sim={top_similarity:.3f}"
        )
        return json.dumps(result, ensure_ascii=False)

# Wrap for PydanticAI 

# Instantiate the LangChain tool
_retrieve_chunks_lc_tool = RetrieveChunksTool()
retrieve_chunks_tool = tool_from_langchain(_retrieve_chunks_lc_tool)
