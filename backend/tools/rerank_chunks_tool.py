# backend/tools/rerank_chunks_tool.py

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

from services.embedder import rerank_chunks

logger = logging.getLogger(__name__)


# Input schema

class RerankChunksInput(BaseModel):
    """Input schema for the rerank_chunks tool."""

    query: str = Field(
        ...,
        description="The original user question used to score chunk relevance.",
    )
    chunks_json: str = Field(
        ...,
        description=(
            "JSON string containing the list of chunk dicts to re-rank. "
            "Each chunk must have 'contextual_text' or 'chunk_text' key. "
            "This should be the 'chunks' array from the retrieve_chunks tool output."
        ),
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top chunks to return after re-ranking.",
    )


# LangChain BaseTool implementation

class RerankChunksTool(BaseTool):
    """
    Cross-encoder re-ranking tool.

    Uses the configured cross-encoder model to score
    each chunk against the query as a (query, chunk) pair.  This joint scoring
    is significantly more accurate than embedding cosine similarity alone.

    Input:  query string + JSON array of chunk dicts
    Output: JSON array of top-K chunks sorted by cross-encoder score (descending)
    """

    name: str = "rerank_chunks"
    description: str = (
        "Re-rank retrieved document chunks using a cross-encoder model for higher "
        "accuracy relevance scoring. Use this tool AFTER retrieve_chunks when you "
        "need to select the most relevant chunks for answering a question. "
        "Input: the original query and a JSON string of chunks from retrieve_chunks. "
        "Output: top-K chunks sorted by cross-encoder relevance score."
    )
    args_schema: Type[BaseModel] = RerankChunksInput
    return_direct: bool = False

    def _run(
        self,
        query: str,
        chunks_json: str,
        top_k: int = 3,
        run_manager: Optional[object] = None,
    ) -> str:
        """Synchronous re-ranking."""
        try:
            chunks: list[dict] = json.loads(chunks_json)
        except json.JSONDecodeError as e:
            return json.dumps({
                "status": "error",
                "message": f"Invalid chunks_json — could not parse JSON: {e}",
                "chunks": [],
            })

        if not chunks:
            return json.dumps({
                "status":  "no_input",
                "message": "No chunks provided to re-rank.",
                "chunks":  [],
            })

        try:
            reranked = asyncio.run(
                rerank_chunks(query=query, chunks=chunks, top_k=top_k)
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            reranked = loop.run_until_complete(
                rerank_chunks(query=query, chunks=chunks, top_k=top_k)
            )

        result = {
            "status":         "success",
            "chunks_returned": len(reranked),
            "chunks":         reranked,
        }

        logger.info(
            f"rerank_chunks: query='{query[:50]}' | "
            f"input={len(chunks)} → output={len(reranked)} chunks | "
            f"top_rerank_score={reranked[0].get('rerank_score', 0.0):.4f}"
            if reranked else ""
        )
        return json.dumps(result, ensure_ascii=False, default=str)

# Wrap for PydanticAI

# Instantiate the LangChain tool
_rerank_chunks_lc_tool = RerankChunksTool()
rerank_chunks_tool = tool_from_langchain(_rerank_chunks_lc_tool)
