"""Persistence helpers for PostgreSQL and ChromaDB."""

from __future__ import annotations

import uuid
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from email_agent.models import ALLOWED_CATEGORIES
from email_agent.settings import CHROMA_COLLECTION_NAME, CHROMA_PATH, dict_row, get_database_url, psycopg


def get_postgres_connection() -> Any:
    """Return a PostgreSQL connection configured for dict-like rows."""

    if psycopg is None:
        raise RuntimeError(
            "psycopg não está instalado. Instale `psycopg[binary]` para usar o PostgreSQL."
        )
    return psycopg.connect(get_database_url(), row_factory=dict_row)


def init_postgres_db() -> None:
    """Create the `processed_emails` table required by the MVP."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_emails (
                    message_id VARCHAR(255) PRIMARY KEY,
                    sender TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_status
                ON processed_emails (status)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_category
                ON processed_emails (category)
                """
            )
        connection.commit()


def get_chroma_collection() -> Collection:
    """Return the local Chroma collection used as semantic memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def bootstrap_services() -> None:
    """Initialize local persistence layers."""

    init_postgres_db()
    _ = get_chroma_collection()


def save_processed_email(message_id: str, sender: str, subject: str, status: str, category: str) -> None:
    """Persist the triage result into PostgreSQL."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO processed_emails (message_id, sender, subject, status, category)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(message_id) DO UPDATE SET
                    sender = EXCLUDED.sender,
                    subject = EXCLUDED.subject,
                    status = EXCLUDED.status,
                    category = EXCLUDED.category,
                    updated_at = NOW()
                """,
                (message_id, sender, subject, status, category),
            )
        connection.commit()


def is_already_processed(message_id: str) -> bool:
    """Check whether the email was already handled previously."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM processed_emails WHERE message_id = %s", (message_id,))
            row = cursor.fetchone()
    return row is not None


def upsert_semantic_memory(
    message_id: str,
    email_data: dict[str, Any],
    learned_text: str,
    category: str,
) -> None:
    """Persist new semantic memory into ChromaDB."""

    collection = get_chroma_collection()
    document = (
        f"Categoria: {category}\n"
        f"Remetente: {email_data.get('sender', '')}\n"
        f"Assunto: {email_data.get('subject', '')}\n"
        f"Aprendizado: {learned_text}"
    )
    collection.upsert(
        ids=[f"{message_id}-{uuid.uuid4().hex[:8]}"],
        documents=[document],
        metadatas=[
            {
                "message_id": message_id,
                "sender": str(email_data.get("sender", "")),
                "subject": str(email_data.get("subject", "")),
                "category": category,
            }
        ],
    )


def get_dashboard_metrics() -> dict[str, Any]:
    """Compute the dashboard metrics used by frontend clients."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails")
            total_processed = int(cursor.fetchone()["total"])
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails WHERE status = 'EM_DUVIDA'")
            pending_review = int(cursor.fetchone()["total"])
            cursor.execute(
                """
                SELECT COUNT(*) AS total FROM processed_emails
                WHERE category IN ('CARREIRA_PRIORIDADE', 'PROJETOS_TECH')
                """
            )
            priority_count = int(cursor.fetchone()["total"])

    automated = max(total_processed - pending_review, 0)
    automation_rate = (automated / total_processed * 100.0) if total_processed else 0.0

    return {
        "emails_processados_hoje": total_processed,
        "prioritarios_detectados": priority_count,
        "fila_revisao": pending_review,
        "taxa_automacao": round(automation_rate, 1),
    }


def get_accumulated_category_counts() -> list[dict[str, Any]]:
    """Return the accumulated triage volume per category from PostgreSQL."""

    bootstrap_services()
    counts = {category: 0 for category in ALLOWED_CATEGORIES}

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT category, COUNT(*) AS total
                FROM processed_emails
                GROUP BY category
                """
            )
            rows = cursor.fetchall()

    for row in rows:
        row_dict = cast(dict[str, Any], row)
        category = str(row_dict["category"]).strip().upper()
        if category in counts:
            counts[category] = int(row_dict["total"])

    return [{"categoria": category, "quantidade": counts[category]} for category in ALLOWED_CATEGORIES]


def get_pending_review_rows() -> list[dict[str, Any]]:
    """Return pending manual-review rows from PostgreSQL."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT message_id, sender, subject, status, category
                FROM processed_emails
                WHERE status = 'EM_DUVIDA'
                ORDER BY sender, subject
                """
            )
            rows = cursor.fetchall()

    return [dict(cast(dict[str, Any], row)) for row in rows]

