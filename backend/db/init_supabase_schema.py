# backend/db/init_supabase_schema.py

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    BigInteger,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ORM Table Definitions

class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="processing")
    file_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    load_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    load_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    model_name: Mapped[str] = mapped_column(
        Text, nullable=False, default="text-embedding-3-small"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )


# Helpers

def _resolve_db_url() -> str:
    load_dotenv()
    db_url = (os.getenv("DATABASE_URL") or "").strip()

    if not db_url:
        print()
        print("❌  DATABASE_URL is not set in your .env file.")
        print()
        print("    Follow these steps to get it:")
        print("    1. Go to https://supabase.com/dashboard")
        print("    2. Open your project → Project Settings → Database")
        print("    3. Under 'Connection string', click the 'Session pooler' tab")
        print("    4. Copy the URI and replace [YOUR-PASSWORD] with your DB password")
        print("    5. Add it to your .env file:")
        print()
        print("       DATABASE_URL=postgresql://postgres.qdrkoyviknjebugruoec:YOUR_PASSWORD@aws-0-<region>.pooler.supabase.com:5432/postgres")
        print()
        print("    ⚠️  Use Session Pooler (port 5432), NOT Transaction Pooler (port 6543).")
        print("    ⚠️  Do NOT use db.qdrkoyviknjebugruoec.supabase.co — it does not")
        print("        resolve on new free-tier Supabase projects.")
        print()
        sys.exit(1)

    # Reject the deprecated direct-connection format
    if "db.qdrkoyviknjebugruoec.supabase.co" in db_url:
        print()
        print("❌  Your DATABASE_URL uses the deprecated direct-connection hostname:")
        print(f"    {db_url}")
        print()
        print("    This hostname (db.<ref>.supabase.co) does NOT resolve on new")
        print("    Supabase free-tier projects created after 2024.")
        print()
        print("    Fix: Supabase Dashboard → Project Settings → Database")
        print("         Copy the 'Session pooler' connection string (port 5432).")
        print()
        sys.exit(1)

    # Reject the transaction pooler (port 6543) 
    if ":6543/" in db_url:
        print()
        print("❌  Your DATABASE_URL is pointing to the Transaction Pooler (port 6543).")
        print("    Transaction Pooler does NOT support DDL statements like")
        print("    CREATE TABLE, CREATE INDEX, CREATE TRIGGER, etc.")
        print()
        print("    Fix: Supabase Dashboard → Project Settings → Database")
        print("         Use the 'Session pooler' connection string (port 5432).")
        print()
        sys.exit(1)

    # Normalise driver prefix
    if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        print("ℹ️  DSN prefix rewritten to postgresql+psycopg2://")

    return db_url


def _resolve_embedding_dim() -> int:
    raw = (
        os.getenv("EMBEDDING_DIMENSIONS")
        or os.getenv("embedding_dimensions")
        or "1536"
    ).strip()
    return int(raw)


def _resolve_embedding_model_name() -> str:
    return (os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-3-small").strip()


def _mask_url(url: str) -> str:
    """Mask password text in a DSN for logs."""
    try:
        if "@" in url:
            prefix, rest = url.split("@", 1)
            if ":" in prefix.split("//", 1)[-1]:
                user_part, _ = prefix.rsplit(":", 1)
                return f"{user_part}:***@{rest}"
    except Exception:
        pass
    return url


# Schema construction helpers

def _ensure_extensions(conn) -> None:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
    conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
    print("  ✅  Extensions: vector, pg_trgm, uuid-ossp")


def _create_indexes(conn) -> None:
    # documents
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_documents_load_id ON documents (load_id);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_documents_file_hash "
        "ON documents (file_hash) WHERE file_hash IS NOT NULL;"
    ))
    conn.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash_ready "
        "ON documents (file_hash) "
        "WHERE status = 'ready' AND file_hash IS NOT NULL;"
    ))
    # chunks
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_chunks_load_id ON chunks (load_id);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks (doc_type);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_chunks_contextual_trgm "
        "ON chunks USING GIN (contextual_text gin_trgm_ops);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_trgm "
        "ON chunks USING GIN (chunk_text gin_trgm_ops);"
    ))
    # embeddings
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings (document_id);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings (chunk_id);"
    ))
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw "
        "ON embeddings USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64);"
    ))
    print("  ✅  Indexes: B-tree (load_id, status, file_hash), GIN trigram, HNSW vector")


def _apply_legacy_schema_migrations(conn) -> None:
    """
    Backward-compatible migrations for older schemas.
    Safe to run repeatedly on both fresh and existing databases.
    """
    conn.execute(text("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'documents'
                  AND column_name = 'doc_type'
            ) THEN
                ALTER TABLE documents DROP COLUMN doc_type;
            END IF;
        END
        $$;
    """))
    conn.execute(text("DROP INDEX IF EXISTS idx_documents_doc_type;"))

    conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'documents'
                  AND column_name = 'file_hash'
            ) THEN
                ALTER TABLE documents ADD COLUMN file_hash TEXT;
            END IF;
        END
        $$;
    """))
    print("  ✅  Legacy migrations: drop documents.doc_type, ensure documents.file_hash")


def _create_view(conn) -> None:
    conn.execute(text("""
        CREATE OR REPLACE VIEW chunks_with_embeddings AS
        SELECT
            c.id                AS chunk_id,
            c.document_id,
            c.chunk_text,
            c.contextual_text,
            c.chunk_index,
            c.page_number,
            c.section_heading,
            c.doc_type,
            c.load_id,
            c.token_count,
            e.id                AS embedding_id,
            e.embedding,
            e.model_name
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id;
    """))
    print("  ✅  View: chunks_with_embeddings")


def _create_functions(conn, embedding_dim: int) -> None:
    # 1. Semantic similarity search
    conn.execute(text(f"""
        CREATE OR REPLACE FUNCTION match_chunks(
            query_embedding      vector({embedding_dim}),
            document_id_filter   UUID,
            match_count          INT     DEFAULT 10,
            similarity_threshold FLOAT   DEFAULT 0.0
        )
        RETURNS TABLE (
            chunk_id         UUID,
            document_id      UUID,
            chunk_text       TEXT,
            contextual_text  TEXT,
            chunk_index      INT,
            page_number      INT,
            section_heading  TEXT,
            doc_type         TEXT,
            load_id          TEXT,
            similarity       FLOAT
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                c.id             AS chunk_id,
                c.document_id,
                c.chunk_text,
                c.contextual_text,
                c.chunk_index,
                c.page_number,
                c.section_heading,
                c.doc_type,
                c.load_id,
                1 - (e.embedding <=> query_embedding) AS similarity
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE
                c.document_id = document_id_filter
                AND 1 - (e.embedding <=> query_embedding) >= similarity_threshold
            ORDER BY e.embedding <=> query_embedding
            LIMIT match_count;
        $$;
    """))

    # 2. Trigram keyword search
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION keyword_search_chunks(
            query_text           TEXT,
            document_id_filter   UUID,
            match_count          INT DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id         UUID,
            document_id      UUID,
            chunk_text       TEXT,
            contextual_text  TEXT,
            chunk_index      INT,
            page_number      INT,
            section_heading  TEXT,
            doc_type         TEXT,
            load_id          TEXT,
            keyword_score    FLOAT
        )
        LANGUAGE sql STABLE
        AS $$
            SELECT
                c.id             AS chunk_id,
                c.document_id,
                c.chunk_text,
                c.contextual_text,
                c.chunk_index,
                c.page_number,
                c.section_heading,
                c.doc_type,
                c.load_id,
                similarity(c.contextual_text, query_text) AS keyword_score
            FROM chunks c
            WHERE
                c.document_id = document_id_filter
                AND c.contextual_text % query_text
            ORDER BY keyword_score DESC
            LIMIT match_count;
        $$;
    """))

    # 3. updated_at trigger function + trigger
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$;
    """))
    conn.execute(text("DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;"))
    conn.execute(text("""
        CREATE TRIGGER trg_documents_updated_at
            BEFORE UPDATE ON documents
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """))
    print(f"  ✅  Functions: match_chunks(dim={embedding_dim}), keyword_search_chunks")
    print("  ✅  Trigger:   trg_documents_updated_at → update_updated_at_column()")


def _align_embedding_column(conn, embedding_dim: int) -> None:
    conn.execute(text("ALTER TABLE embeddings DROP COLUMN IF EXISTS embedding;"))
    conn.execute(text(
        f"ALTER TABLE embeddings ADD COLUMN embedding vector({embedding_dim}) NOT NULL;"
    ))
    print(f"  ✅  embeddings.embedding → vector({embedding_dim})")


def _align_model_name_default(conn, model_name: str) -> None:
    escaped = model_name.replace("'", "''")
    conn.execute(text(
        f"ALTER TABLE embeddings ALTER COLUMN model_name SET DEFAULT '{escaped}';"
    ))
    print(f"  ✅  embeddings.model_name default → '{model_name}'")


# Entry point

def main() -> None:
    db_url = _resolve_db_url()
    embedding_dim = _resolve_embedding_dim()
    embedding_model_name = _resolve_embedding_model_name()

    print()
    print("🚀  Ultra Doc-Intelligence — Supabase Schema Init")
    print("─" * 52)
    print(f"  DSN             : {_mask_url(db_url)}")
    print(f"  Embedding dim   : {embedding_dim}")
    print(f"  Embedding model : {embedding_model_name}")
    print("─" * 52)

    engine = create_engine(
        db_url,
        future=True,
        connect_args={"connect_timeout": 30},
    )

    try:
        with engine.begin() as conn:
            print("\n[1/7] Ensuring extensions ...")
            _ensure_extensions(conn)

            print("[2/7] Creating tables ...")
            Base.metadata.create_all(conn)
            print("  ✅  Tables: documents, chunks, embeddings")

            print("[3/7] Aligning embedding column ...")
            _align_embedding_column(conn, embedding_dim)
            _align_model_name_default(conn, embedding_model_name)

            print("[4/7] Applying legacy compatibility migrations ...")
            _apply_legacy_schema_migrations(conn)

            print("[5/7] Creating indexes ...")
            _create_indexes(conn)

            print("[6/7] Creating views ...")
            _create_view(conn)

            print("[7/7] Creating functions & triggers ...")
            _create_functions(conn, embedding_dim)

    except Exception as exc:
        print()
        print(f"❌  Schema init failed: {exc}")
        print()
        print("    Common causes:")
        print("    • Wrong port  — use port 5432 (Session Pooler), not 6543")
        print("    • Wrong host  — use pooler host from dashboard, not db.<ref>.supabase.co")
        print("    • Wrong password — double-check SUPABASE_PASSWORD in your .env")
        raise

    print()
    print("─" * 52)
    print("🎉  Schema initialisation complete!")
    print(f"  Tables  : documents, chunks, embeddings")
    print(f"  Indexes : B-tree, GIN trigram, HNSW vector")
    print(f"  View    : chunks_with_embeddings")
    print(f"  Funcs   : match_chunks, keyword_search_chunks")
    print(f"  Trigger : trg_documents_updated_at")
    print()


if __name__ == "__main__":
    main()