from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")

    engine = create_engine(db_url)

    with engine.begin() as conn:
        print("Running migration: remove documents.doc_type")

        conn.execute(text("""
            DROP INDEX IF EXISTS idx_documents_doc_type;
        """))

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

        print("Migration complete")


if __name__ == "__main__":
    main()
