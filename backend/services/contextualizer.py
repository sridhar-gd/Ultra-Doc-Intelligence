# backend/services/contextualizer.py

import asyncio
import logging
from dataclasses import dataclass

import anthropic

from config import get_settings
from prompts.contextual_retrieval_prompt import build_contextualizer_messages
from services.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


# Output dataclass

@dataclass
class ContextualisedChunk:
    """A chunk augmented with its Claude-generated context prefix."""

    original_chunk: DocumentChunk
    context_prefix: str        
    contextual_text: str       
    cached_tokens: int = 0     
    fresh_tokens: int = 0      
    output_tokens: int = 0     


# Anthropic async client singleton

def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return a configured async Anthropic client."""
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


# Batch contextualisation with concurrency control

async def contextualise_chunks(
    chunks: list[DocumentChunk],
    doc_markdown: str,
    max_concurrency: int = 5,
) -> list[ContextualisedChunk]:
    """
    Contextualise all chunks for a document using Anthropic's Contextual
    Retrieval technique.

    Processes chunks concurrently (up to max_concurrency at once) to balance
    speed vs API rate limits. The full document is passed with prompt caching
    so only the chunk-specific tokens are billed fresh after the first call.

    Args:
        chunks:          List of DocumentChunk objects from chunker.py
        doc_markdown:    Full document markdown (from parser.py ParsedDocument.markdown)
        max_concurrency: Max parallel Claude API calls (default 5)

    Returns:
        List of ContextualisedChunk objects in same order as input chunks.

    Raises:
        RuntimeError: If Claude API calls fail for any chunk.
    """
    if not chunks:
        return []

    client = _get_anthropic_client()
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[ContextualisedChunk | None] = [None] * len(chunks)

    async def _contextualise_one(idx: int, chunk: DocumentChunk) -> None:
        async with semaphore:
            try:
                messages = build_contextualizer_messages(
                    doc_content=doc_markdown,
                    chunk_content=chunk.chunk_text,
                )

                response = await client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=settings.contextualizer_max_tokens,
                    temperature=0.0,
                    messages=messages,
                )

                context_prefix: str = response.content[0].text.strip()

                usage = response.usage
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
                fresh_tokens  = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0

                contextual_text = f"{context_prefix}\n\n{chunk.chunk_text}"

                results[idx] = ContextualisedChunk(
                    original_chunk=chunk,
                    context_prefix=context_prefix,
                    contextual_text=contextual_text,
                    cached_tokens=cached_tokens,
                    fresh_tokens=fresh_tokens,
                    output_tokens=output_tokens,
                )

                logger.debug(
                    f"Chunk {idx}: context={context_prefix[:60]}... | "
                    f"cached={cached_tokens} fresh={fresh_tokens}"
                )

            except anthropic.RateLimitError:
                logger.warning(f"Rate limited on chunk {idx} — retrying after 10s...")
                await asyncio.sleep(10)

                messages = build_contextualizer_messages(
                    doc_content=doc_markdown,
                    chunk_content=chunk.chunk_text,
                )

                response = await client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=settings.contextualizer_max_tokens,
                    temperature=0.0,
                    messages=messages,
                )

                context_prefix = response.content[0].text.strip()
                usage = response.usage

                results[idx] = ContextualisedChunk(
                    original_chunk=chunk,
                    context_prefix=context_prefix,
                    contextual_text=f"{context_prefix}\n\n{chunk.chunk_text}",
                    cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                    fresh_tokens=getattr(usage, "input_tokens", 0) or 0,
                    output_tokens=getattr(usage, "output_tokens", 0) or 0,
                )

            except Exception as exc:
                logger.error(f"Failed to contextualise chunk {idx}: {exc}")
                results[idx] = ContextualisedChunk(
                    original_chunk=chunk,
                    context_prefix="",
                    contextual_text=chunk.chunk_text,
                )

    tasks = [
        asyncio.create_task(_contextualise_one(i, chunk))
        for i, chunk in enumerate(chunks)
    ]
    await asyncio.gather(*tasks)

    # Log cost summary
    total_cached = sum(r.cached_tokens for r in results if r)
    total_fresh  = sum(r.fresh_tokens  for r in results if r)
    total_output = sum(r.output_tokens for r in results if r)
    cache_pct    = (total_cached / max(total_fresh + total_cached, 1)) * 100

    logger.info(
        f"Contextualisation complete: {len(chunks)} chunks | "
        f"cached={total_cached} fresh={total_fresh} output={total_output} | "
        f"cache hit rate={cache_pct:.1f}%"
    )

    return [r for r in results if r is not None]