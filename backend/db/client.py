# backend/db/client.py

import logging
from functools import lru_cache

from supabase import create_client, Client

from config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return a cached Supabase client."""
    settings = get_settings()

    logger.info(f"Initialising Supabase client for: {settings.supabase_url[:40]}...")

    client: Client = create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_service_role_key,
    )

    logger.info("Supabase client ready.")
    return client