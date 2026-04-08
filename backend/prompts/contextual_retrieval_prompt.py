# backend/prompts/contextual_retrieval_prompt.py

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>

You are a logistics document analysis assistant. The document above is a
transportation or logistics document — it may be a Rate Confirmation (RC),
Bill of Lading (BOL), Shipment Instruction, or Invoice.
"""

CHUNK_CONTEXT_PROMPT = """
Here is a specific chunk from the document above:

<chunk>
{chunk_content}
</chunk>

Please give a short, succinct context (1-2 sentences) to situate this chunk
within the overall document. Focus on:
- Which load/shipment ID this refers to (if present)
- What section or topic this chunk covers (e.g. rate breakdown, pickup details,
carrier info, driver details, commodity, standing instructions)
- Any critical values mentioned (rates, dates, locations, IDs)

This context will be prepended to the chunk to improve search retrieval.
Answer ONLY with the situating context — no preamble, no explanation,
no bullet points. Plain sentences only.
"""

# Assembled prompt builder

def build_contextualizer_messages(doc_content: str, chunk_content: str) -> list[dict]:
    """
    Build Anthropic message content for contextual chunk augmentation.
    The full document block is marked cacheable while chunk text remains dynamic.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_content),
                    "cache_control": {"type": "ephemeral"},  # This will be cached across chunk calls.
                },
                {
                    "type": "text",
                    "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content), # This will not be cached. It is a fresh prompt for each chonk.
                },
            ],
        }
    ]