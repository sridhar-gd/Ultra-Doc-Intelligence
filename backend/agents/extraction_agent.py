# backend/agents/extraction_agent.py

import asyncio
import json
import logging

import anthropic
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from schemas.shipment_schema import ShipmentData
from prompts.extraction_system_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    format_all_chunks_for_extraction,
)
from db.vector_store import fetch_all_chunks, fetch_document_filename
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Extraction Agent


extraction_agent = Agent(
    model=AnthropicModel(
        model_name=settings.anthropic_model,
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    ),
    output_type=ShipmentData,
    tools=[],
    system_prompt=(
        "You are a logistics document data extraction engine. "
        "Your task is to extract structured shipment fields from a document. "
        "Extract every field you can find and return a valid JSON object "
        "matching the ShipmentData schema. "
        "Set missing fields to null — never fabricate values. "
        "Copy values verbatim from the document. "
        "Convert all dates to ISO 8601 format. "
        "Extract rates as numbers only (no currency symbols)."
    ),
    retries=3,
)


# Extraction via Anthropic SDK

def _call_claude_for_extraction(document_chunks: list[dict]) -> str:
    """Call Claude and return raw extraction JSON text."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    formatted_chunks = format_all_chunks_for_extraction(document_chunks)
    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(document_chunks=formatted_chunks)

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        temperature=0.0,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": (
                "Please extract all shipment fields from the document chunks in the "
                "system prompt and return a valid JSON object matching the ShipmentData "
                "schema. Set any missing fields to null."
            ),
        }],
    )

    return response.content[0].text.strip()

# Main extraction pipeline

async def run_extraction(document_id: str, document_name: str) -> ShipmentData:
    """Extract structured shipment data for one stored document."""
    logger.info(
        f"[Extraction] Starting extraction for doc={document_id[:8]} "
        f"name={document_name!r}"
    )

    stored_name = await fetch_document_filename(document_id)
    if not stored_name:
        raise ValueError(
            f"No document found for document_id={document_id}."
        )
    if stored_name.strip() != document_name.strip():
        raise ValueError(
            "document_name does not match the uploaded file for this document_id "
            f"(expected {stored_name!r}, got {document_name.strip()!r})."
        )

    # Step 1: Fetch all document chunks (no similarity filter)
    chunks = await fetch_all_chunks(document_id=document_id)

    if not chunks:
        raise ValueError(
            f"No chunks found for document_id={document_id}. "
            "Ensure the document has been successfully ingested first."
        )

    logger.info(f"[Extraction] Retrieved {len(chunks)} chunks for extraction.")

    # Step 2: Call Claude for extraction 
    try:
        raw_json = await asyncio.to_thread(
            _call_claude_for_extraction,
            chunks,
        )
    except Exception as exc:
        logger.error(f"[Extraction] Claude call failed: {exc}")
        raise RuntimeError(f"Extraction LLM call failed: {exc}") from exc

    # Step 3: Parse and validate with Pydantic 
    # Strip markdown fences when claude returns the json
    cleaned_json = raw_json.strip()
    if cleaned_json.startswith("```"):
        lines = cleaned_json.split("\n")
        cleaned_json = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        raw_data = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        logger.error(f"[Extraction] JSON parse failed: {exc}\nRaw: {cleaned_json[:500]}")
        raise RuntimeError(f"Claude returned invalid JSON for extraction: {exc}") from exc

    # Inject document_id 
    raw_data["document_id"] = document_id
    raw_data["extraction_confidence"] = None  

    try:
        shipment_data = ShipmentData.model_validate(raw_data)
    except Exception as exc:
        logger.error(f"[Extraction] Pydantic validation failed: {exc}")
        raise RuntimeError(f"Extraction output failed schema validation: {exc}") from exc

    logger.info(
        f"[Extraction] ✅ Complete for doc={document_id[:8]} | "
        f"shipment_id={shipment_data.shipment_id} | "
        f"carrier={shipment_data.carrier_name}"
    )

    return shipment_data
