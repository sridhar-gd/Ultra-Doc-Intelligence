# backend/config.py

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.
    Pydantic BaseSettings auto-reads values from:
      1. Environment variables (takes precedence)
      2. A .env file in the project root

    Usage anywhere in the codebase:
        from config import get_settings
        settings = get_settings()
        print(settings.anthropic_api_key)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = Field(..., description="Anthropic API key")

    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic Claude model identifier",
    )

    anthropic_max_tokens: int = Field(
        default=1024,
        description="Max tokens for Claude completions",
    )

    contextualizer_max_tokens: int = Field(
        default=200,
        description="Max tokens when generating contextual chunk summaries",
    )

    # Supabase
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_anon_key: str = Field(..., description="Supabase anon/public key")
    supabase_service_role_key: str = Field(
        ..., description="Supabase service-role key (bypasses RLS — server-side only)"
    )

    database_url: str = Field(
        default="",
        description="Optional direct psycopg2 DSN for bulk pgvector / schema operations",
    )

    # OpenAI — used for embeddings only
    openai_api_key: str = Field(..., description="OpenAI API key (used for embeddings)")

    embedding_model_name: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name (used via langchain_openai.OpenAIEmbeddings)",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Vector dimensions produced by the embedding model",
    )

    # Re-ranking model
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="BAAI BGE cross-encoder reranker model (via HuggingFaceCrossEncoder)",
    )

    # Chunking
    chunk_size: int = Field(
        default=512,
        description="Max tokens per chunk (RecursiveCharacterTextSplitter fallback)",
    )

    chunk_overlap: int = Field(
        default=50,
        description="Token overlap between consecutive chunks",
    )

    # Retrieval
    retrieval_top_k_semantic: int = Field(
        default=10,
        description="Top-K chunks from pgvector cosine similarity search",
    )

    retrieval_top_k_keyword: int = Field(
        default=10,
        description="Top-K chunks from pg_trgm keyword search",
    )

    retrieval_top_k_rerank_input: int = Field(
        default=15,
        description="Candidates sent to cross-encoder after RRF fusion",
    )

    retrieval_top_k_final: int = Field(
        default=3,
        description="Final top-K chunks used in the LLM prompt",
    )
    retrieval_enable_reranking: bool = Field(
        default=False,
        description=(
            "Enable cross-encoder reranking. Disable on low-memory hosts to avoid "
            "loading the heavyweight BAAI reranker model at request time."
        ),
    )

    # Guardrails
    guardrail_similarity_threshold: float = Field(
        default=0.25,
        description=(
            "Min cosine similarity (or rerank score if available) to proceed with answering. "
            "Set low (0.20-0.35) for structured form docs (BOL, RC, invoice). "
            "Set higher (0.55-0.65) for dense prose docs."
        ),
    )

    guardrail_low_confidence_threshold: float = Field(
        default=0.20,
        description=(
            "Confidence score below which a low-confidence warning is attached to the answer. "
            "Set lower than guardrail_similarity_threshold."
        ),
    )

    # File Upload
    max_upload_size_mb: int = Field(
        default=50,
        description="Maximum allowed file upload size in megabytes",
    )

    max_files_per_batch: int = Field(
        default=10,
        description="Maximum number of files allowed in a single batch upload request",
    )

    allowed_extensions: list[str] = Field(
        default=["pdf", "docx", "txt"],
        description="Allowed document file extensions",
    )

    # Async ingestion concurrency
    ingestion_concurrency: int = Field(
        default=4,
        description=(
            "Maximum number of documents ingested concurrently within a single batch. "
            "Limits parallel calls to Anthropic and OpenAI to avoid rate-limit errors. "
            "Tune based on your API tier limits."
        ),
    )

    # FastAPI / Server
    app_title: str = Field(
        default="Ultra Doc-Intelligence API",
        description="FastAPI app title shown in /docs",
    )

    app_version: str = Field(
        default="0.1.0",
        description="API version string",
    )

    debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode (never True in production)",
    )
    warmup_models_on_startup: bool = Field(
        default=False,
        description=(
            "If True, eagerly load embedding/reranker models during app startup. "
            "Keep False on low-memory hosts (e.g., Railway) to avoid restart loops."
        ),
    )
    parser_pdf_do_ocr: bool = Field(
        default=False,
        description=(
            "Enable OCR path in Docling PDF parser. Keep False on constrained/cloud "
            "Linux runtimes where OCR backend native GUI/XCB libs may be unavailable."
        ),
    )

    # CORS
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        description="Allowed CORS origins for the React frontend",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Call get_settings() anywhere:

        from config import get_settings
        settings = get_settings()
    """
    return Settings()
