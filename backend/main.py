# backend/main.py

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings
from utils.logging import setup_logging

settings = get_settings()

setup_logging(debug=settings.debug)
logger = logging.getLogger(__name__)

from api.upload import router as upload_router
from api.ask import router as ask_router
from api.extract import router as extract_router
from api.ingestion_status import router as ingestion_status_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Runs startup logic before the first request and cleanup on shutdown.
    """
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_title} v{settings.app_version}")
    logger.info(f"Debug mode              : {settings.debug}")
    logger.info(f"Embedding model         : {settings.embedding_model_name}")
    logger.info(f"LLM model               : {settings.anthropic_model}")
    logger.info(f"Ingestion concurrency   : {settings.ingestion_concurrency}")
    logger.info(f"Max files per batch     : {settings.max_files_per_batch}")
    logger.info("=" * 60)

    if settings.warmup_models_on_startup:
        try:
            logger.info("Warming up embedding and reranker models (one-time load)...")
            from services.embedder import _get_openai_embedder, _get_cross_encoder
            _get_openai_embedder()
            _get_cross_encoder()
            logger.info("Models warmed up and ready.")
        except Exception as exc:
            logger.warning(f"Model warm-up failed (will load on first request): {exc}")
    else:
        logger.info(
            "Skipping startup model warm-up "
            "(set WARMUP_MODELS_ON_STARTUP=true to enable)."
        )

    try:
        from db.client import get_supabase_client
        client = get_supabase_client()
        client.table("documents").select("id").limit(1).execute()
        logger.info("Supabase connectivity verified.")
    except Exception as exc:
        logger.error(
            f"Supabase connectivity check failed: {exc}. "
            "Ensure the schema has been initialised and credentials are correct."
        )

    yield

    logger.info("Shutting down Ultra Doc-Intelligence API.")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Ultra Doc-Intelligence — Agentic RAG system for logistics documents. "
        "Upload Rate Confirmations, Bills of Lading, and Invoices; ask questions; "
        "and extract structured shipment data using Claude + pgvector. "
        "Multi-file uploads are processed asynchronously in the background."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    debug=settings.debug,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(extract_router)
app.include_router(ingestion_status_router)


@app.get("/", tags=["Health"], summary="API root")
async def root() -> dict:
    """Root endpoint — confirms the API is running."""
    return {
        "service": settings.app_title,
        "version": settings.app_version,
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    """
    Health check endpoint for load balancers and uptime monitors.
    Verifies Supabase connectivity and model availability.
    """
    checks: dict[str, str] = {
        "api":      "ok",
        "supabase": "unknown",
        "embedder": "unknown",
    }

    try:
        from db.client import get_supabase_client
        get_supabase_client().table("documents").select("id").limit(1).execute()
        checks["supabase"] = "ok"
    except Exception as exc:
        checks["supabase"] = f"error: {str(exc)[:80]}"

    try:
        from services.embedder import _get_openai_embedder
        _get_openai_embedder()
        checks["embedder"] = "ok"
    except Exception as exc:
        checks["embedder"] = f"error: {str(exc)[:80]}"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }
