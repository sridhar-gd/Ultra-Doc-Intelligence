from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")

    engine = create_engine(db_url)

    with engine.begin() as conn:
        print("Running migration: add file_hash")

        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'file_hash'
                ) THEN
                    ALTER TABLE documents ADD COLUMN file_hash TEXT;
                END IF;
            END
            $$;
        """))

        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_hash_ready
            ON documents (file_hash)
            WHERE status = 'ready' AND file_hash IS NOT NULL;
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_file_hash
            ON documents (file_hash)
            WHERE file_hash IS NOT NULL;
        """))

        print("Migration complete")

if __name__ == "__main__":
    main()