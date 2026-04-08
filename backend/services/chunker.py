# backend/services/chunker.py

import logging
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.parser import ParsedDocument
from config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# Output dataclass

@dataclass
class DocumentChunk:
    """A single chunk ready for contextualisation and embedding."""

    chunk_text: str          # Raw chunk text (before contextualisation)
    chunk_index: int         
    document_id: str         
    doc_type: str
    load_id: str | None
    page_number: int | None
    section_heading: str | None
    token_count: int         

# Token count helper

def _estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (whitespace-split word count).
    Good enough for chunking decisions — avoids a full tokenizer dependency.
    """
    return len(text.split())


# Section heading extractor from Markdown

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _extract_nearest_heading(text: str, full_markdown: str) -> str | None:
    """
    Find the nearest Markdown heading that appears BEFORE the chunk text
    in the full document.  Used to annotate fallback-chunked pieces.
    """
    chunk_pos = full_markdown.find(text[:80].strip())  # search by first 80 chars
    if chunk_pos == -1:
        return None

    # Find all headings before this position
    headings_before = [
        m.group(2).strip()
        for m in _HEADING_RE.finditer(full_markdown[:chunk_pos])
    ]
    return headings_before[-1] if headings_before else None


# Primary chunker — Docling HierarchicalChunker

def _chunk_with_docling(
    parsed_doc: ParsedDocument,
    document_id: str,
) -> list[DocumentChunk]:
    """
    Use Docling's native HierarchicalChunker to split by document structure
    (sections, headings, table blocks).

    Returns a list of DocumentChunk objects, or empty list if chunking fails
    or produces unusable output (triggers fallback).
    """
    try:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker import DocMeta
    except ImportError:
        logger.warning("Docling HybridChunker not available — using fallback chunker.")
        return []

    if parsed_doc.raw_docling_result is None:
        return []

    try:
        chunker = HybridChunker(
            tokenizer="BAAI/bge-large-en-v1.5",
            max_tokens=settings.chunk_size,
            merge_peers=True,   # merge small adjacent chunks of same section
        )
        docling_chunks = list(chunker.chunk(parsed_doc.raw_docling_result.document))
    except Exception as exc:
        logger.warning(f"Docling chunker failed: {exc} — falling back to LangChain splitter.")
        return []

    if not docling_chunks:
        return []

    results: list[DocumentChunk] = []
    for idx, dc in enumerate(docling_chunks):
        text = dc.text.strip()
        if not text or _estimate_tokens(text) < 5:
            continue  # skip empty / near-empty chunks

        # Extract metadata from Docling chunk
        meta: DocMeta | None = getattr(dc, "meta", None)
        headings: list[str] = getattr(meta, "headings", []) if meta else []
        section_heading = headings[-1] if headings else None

        # Page number — from first doc_item provenance
        page_number: int | None = None
        doc_items = getattr(meta, "doc_items", []) if meta else []
        if doc_items:
            prov_list = getattr(doc_items[0], "prov", [])
            if prov_list:
                page_number = getattr(prov_list[0], "page_no", None)

        results.append(
            DocumentChunk(
                chunk_text=text,
                chunk_index=idx,
                document_id=document_id,
                doc_type=parsed_doc.detected_doc_type,
                load_id=parsed_doc.detected_load_id,
                page_number=page_number,
                section_heading=section_heading,
                token_count=_estimate_tokens(text),
            )
        )

    logger.info(f"Docling chunker produced {len(results)} chunks.")
    return results

# Fallback chunker — LangChain RecursiveCharacterTextSplitter

def _chunk_with_langchain(
    parsed_doc: ParsedDocument,
    document_id: str,
) -> list[DocumentChunk]:
    """
    LangChain RecursiveCharacterTextSplitter on the raw Markdown export.
    Used when Docling's native chunker fails or produces too few chunks.

    Splits on: paragraph breaks → newlines → sentences → words (in that order)
    so chunks respect natural text boundaries as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=settings.chunk_size * 5,  
        chunk_overlap=settings.chunk_overlap * 5,
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks: list[str] = splitter.split_text(parsed_doc.markdown)

    results: list[DocumentChunk] = []
    for idx, text in enumerate(raw_chunks):
        text = text.strip()
        if not text or _estimate_tokens(text) < 5:
            continue

        section_heading = _extract_nearest_heading(text, parsed_doc.markdown)

        results.append(
            DocumentChunk(
                chunk_text=text,
                chunk_index=idx,
                document_id=document_id,
                doc_type=parsed_doc.detected_doc_type,
                load_id=parsed_doc.detected_load_id,
                page_number=None,  
                section_heading=section_heading,
                token_count=_estimate_tokens(text),
            )
        )

    logger.info(f"LangChain fallback chunker produced {len(results)} chunks.")
    return results


# Public interface

_MIN_ACCEPTABLE_CHUNKS = 3     # If Docling produces fewer, use fallback
_MAX_ACCEPTABLE_TOKENS = 1200  # If any chunk exceeds this, re-split with fallback


def chunk_document(
    parsed_doc: ParsedDocument,
    document_id: str,
) -> list[DocumentChunk]:
    """
    Chunk a parsed document into semantically meaningful pieces.

    Strategy:
      1. Try Docling HybridChunker (section-aware, structure-preserving)
      2. If insufficient results → fall back to LangChain RecursiveCharacterTextSplitter
      3. Re-index chunk_index to ensure sequential ordering

    Args:
        parsed_doc:   ParsedDocument from services/parser.py
        document_id:  UUID string for the parent document (FK metadata)

    Returns:
        List of DocumentChunk objects, ordered by chunk_index.

    Raises:
        ValueError: If no chunks could be produced from the document.
    """
    # Primary: Docling
    chunks = _chunk_with_docling(parsed_doc, document_id)

    # Quality gate: too few chunks or oversized chunks → use fallback
    oversized = any(c.token_count > _MAX_ACCEPTABLE_TOKENS for c in chunks)
    if len(chunks) < _MIN_ACCEPTABLE_CHUNKS or oversized:
        reason = "too few chunks" if len(chunks) < _MIN_ACCEPTABLE_CHUNKS else "oversized chunks"
        logger.info(
            f"Docling chunker result inadequate ({reason}). "
            f"Switching to LangChain RecursiveCharacterTextSplitter."
        )
        chunks = _chunk_with_langchain(parsed_doc, document_id)

    if not chunks:
        raise ValueError(
            f"No chunks produced for document '{parsed_doc.filename}'. "
            "Document may be empty or unparseable."
        )

    # Re-index to guarantee sequential 0-based chunk_index
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    logger.info(
        f"Final chunk count for '{parsed_doc.filename}': {len(chunks)}"
    )
    return chunks