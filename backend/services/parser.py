# backend/services/parser.py

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

# Dataclass for parsed output

@dataclass
class ParsedDocument:
    """Result of parsing a single document through Docling."""

    file_path: str
    filename: str
    markdown: str                        
    detected_doc_type: str = "unknown"  
    detected_load_id: str | None = None 
    page_count: int = 0
    table_count: int = 0
    raw_docling_result: object = field(default=None, repr=False)  

# Document type detection heuristics

_DOC_TYPE_PATTERNS: dict[str, list[str]] = {
    "carrier_rc": [
        r"carrier\s+rate\s+confirmation",
        r"carrier\s+confirmation",
        r"carrier\s+rate",
    ],
    "shipper_rc": [
        r"shipper\s+rate\s+confirmation",
        r"customer\s+rate\s+confirmation",
        r"shipper\s+confirmation",
    ],
    "bol": [
        r"bill\s+of\s+lading",
        r"\bbol\b",
        r"b/l\b",
    ],
    "invoice": [
        r"\binvoice\b",
        r"freight\s+invoice",
    ],
}

_LOAD_ID_PATTERN = re.compile(
    r"\b(LD\d{4,8}|LOAD[\s\-]?\d{4,8})\b",
    re.IGNORECASE,
)


def _detect_doc_type(text: str) -> str:
    """
    Heuristic detection of logistics document type.
    Checks lowercased text against known patterns.
    """
    lower = text.lower()
    for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                return doc_type
    return "unknown"


def _detect_load_id(text: str) -> str | None:
    """
    Extract the first load ID found in the document text.
    Pattern covers formats like: LD53657, LOAD53657, LOAD-53657
    """
    match = _LOAD_ID_PATTERN.search(text)
    if match:
        return match.group(0).upper().replace(" ", "").replace("-", "")
    return None


# Docling converter (initialised once)

def _build_converter() -> DocumentConverter:
    """
    Build and return a configured Docling DocumentConverter.

    PdfPipelineOptions controls:
      - do_ocr: Enable EasyOCR for scanned PDFs
      - do_table_structure: Enable TableFormer for table parsing
      - generate_page_images: Disabled (not needed for RAG text extraction)
    """
    pdf_options = PdfPipelineOptions(
        do_ocr=True,               # Handle scanned PDFs (EasyOCR)
        do_table_structure=True,   # TableFormer — ~94% table accuracy
        generate_page_images=False,
        generate_picture_images=False,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        }
    )


# Module-level singleton — loaded once when the service module is imported
_converter: DocumentConverter | None = None


def get_converter() -> DocumentConverter:
    """Return the module-level DocumentConverter singleton."""
    global _converter
    if _converter is None:
        logger.info("Initialising Docling DocumentConverter (one-time model load)...")
        _converter = _build_converter()
        logger.info("Docling DocumentConverter ready.")
    return _converter


# Core parse function

def _parse_sync(file_path: str) -> ParsedDocument:
    """
    Synchronous parse — runs Docling in the calling thread.
    Called via asyncio.to_thread() from parse_document() to avoid
    blocking the FastAPI event loop.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower().lstrip(".")
    if ext not in ("pdf", "docx", "txt"):
        raise ValueError(f"Unsupported file extension: .{ext}")

    logger.info(f"Parsing document: {path.name}")

    converter = get_converter()

    try:
        result = converter.convert(str(path))
    except Exception as exc:
        logger.error(f"Docling conversion failed for {path.name}: {exc}")
        raise RuntimeError(f"Document parsing failed: {exc}") from exc

    # Export to Markdown — primary text representation for chunking + embedding
    markdown_text: str = result.document.export_to_markdown()

    if not markdown_text.strip():
        raise ValueError(
            f"Docling produced empty output for {path.name}. "
            "File may be corrupted, password-protected, or image-only without OCR."
        )

    # Count tables extracted by Docling
    table_count = len(result.document.tables) if hasattr(result.document, "tables") else 0

    # Page count
    page_count = 0
    if hasattr(result.document, "pages"):
        page_count = len(result.document.pages)

    # Heuristic metadata extraction
    doc_type = _detect_doc_type(markdown_text)
    load_id  = _detect_load_id(markdown_text)

    logger.info(
        f"Parsed {path.name}: {page_count} pages, {table_count} tables, "
        f"doc_type={doc_type}, load_id={load_id}"
    )

    return ParsedDocument(
        file_path=str(path),
        filename=path.name,
        markdown=markdown_text,
        detected_doc_type=doc_type,
        detected_load_id=load_id,
        page_count=page_count,
        table_count=table_count,
        raw_docling_result=result,
    )


async def parse_document(file_path: str) -> ParsedDocument:
    """
    Async entry point for document parsing.

    Runs Docling in a thread pool (asyncio.to_thread) so it doesn't block
    the FastAPI event loop during the heavy model inference step.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        ParsedDocument dataclass with markdown text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file type is unsupported or output is empty.
        RuntimeError:      If Docling conversion fails.
    """
    return await asyncio.to_thread(_parse_sync, file_path)