"""Persistence helpers for PostgreSQL and ChromaDB."""

from __future__ import annotations

import uuid
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from email_agent.models import FINAL_LABELS, FewShotExample, PRIORITY_LEVELS, THEME_CATEGORIES
from email_agent.settings import CHROMA_COLLECTION_NAME, CHROMA_PATH, dict_row, get_database_url, psycopg


LEGACY_CATEGORY_MAP: dict[str, tuple[str, str, str, bool, bool, str]] = {
    "CARREIRA_PRIORIDADE": ("CARREIRA", "ALTA", "ALTA_CARREIRA", True, True, "ALTA"),
    "PROJETOS_TECH": ("PROJETOS_TECH", "ALTA", "ALTA_PROJETOS_TECH", True, True, "MEDIA"),
    "CRIATIVO_E_MAKER": ("CRIATIVO_MAKER", "BAIXA", "BAIXA_CRIATIVO_MAKER", False, False, "BAIXA"),
    "CURSOS_E_APRENDIZADO": ("CURSOS_APRENDIZADO", "BAIXA", "BAIXA_CURSOS_APRENDIZADO", False, False, "BAIXA"),
    "PESSOAL_FINANCEIRO": ("FINANCEIRO", "MEDIA", "MEDIA_FINANCEIRO", True, True, "MEDIA"),
    "NEWSLETTER_INFORMATIVO": ("NEWSLETTER", "BAIXA", "BAIXA_NEWSLETTER", False, False, "BAIXA"),
    "SPAM_LIXO": ("SPAM", "BAIXA", "BAIXA_SPAM", False, False, "BAIXA"),
    "OUTROS": ("OUTROS", "BAIXA", "BAIXA_OUTROS", False, False, "BAIXA"),
    "EM_DUVIDA": ("EM_DUVIDA", "BAIXA", "EM_DUVIDA", False, False, "BAIXA"),
}


def normalize_legacy_category(category: str) -> tuple[str, str, str, bool, bool, str]:
    """Map legacy one-shot categories into the new theme/priority structure."""

    normalized_category = str(category).strip().upper()
    return LEGACY_CATEGORY_MAP.get(
        normalized_category,
        ("EM_DUVIDA", "BAIXA", "EM_DUVIDA", False, False, "BAIXA"),
    )


def get_postgres_connection() -> Any:
    """Return a PostgreSQL connection configured for dict-like rows."""

    if psycopg is None:
        raise RuntimeError(
            "psycopg não está instalado. Instale `psycopg[binary]` para usar o PostgreSQL."
        )
    return psycopg.connect(get_database_url(), row_factory=dict_row, connect_timeout=3)


def init_postgres_db() -> None:
    """Create or migrate the `processed_emails` table required by the MVP."""

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
                    theme_category VARCHAR(100) NOT NULL DEFAULT 'EM_DUVIDA',
                    priority_level VARCHAR(20) NOT NULL DEFAULT 'BAIXA',
                    final_label VARCHAR(150) NOT NULL DEFAULT 'EM_DUVIDA',
                    needs_action BOOLEAN NOT NULL DEFAULT FALSE,
                    is_important BOOLEAN NOT NULL DEFAULT FALSE,
                    time_sensitivity VARCHAR(20) NOT NULL DEFAULT 'BAIXA',
                    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                    gmail_flagged BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS theme_category VARCHAR(100)")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS priority_level VARCHAR(20)")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS final_label VARCHAR(150)")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS needs_action BOOLEAN")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS is_important BOOLEAN")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS time_sensitivity VARCHAR(20)")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION")
            cursor.execute("ALTER TABLE processed_emails ADD COLUMN IF NOT EXISTS gmail_flagged BOOLEAN")
            cursor.execute(
                """
                UPDATE processed_emails
                SET
                    theme_category = COALESCE(theme_category, 'EM_DUVIDA'),
                    priority_level = COALESCE(priority_level, 'BAIXA'),
                    final_label = COALESCE(final_label, category, 'EM_DUVIDA'),
                    needs_action = COALESCE(needs_action, FALSE),
                    is_important = COALESCE(is_important, FALSE),
                    time_sensitivity = COALESCE(time_sensitivity, 'BAIXA'),
                    confidence = COALESCE(confidence, 0),
                    gmail_flagged = COALESCE(gmail_flagged, FALSE)
                """
            )
            for legacy_category, (
                theme_category,
                priority_level,
                final_label,
                needs_action,
                is_important,
                time_sensitivity,
            ) in LEGACY_CATEGORY_MAP.items():
                cursor.execute(
                    """
                    UPDATE processed_emails
                    SET
                        theme_category = %s,
                        priority_level = %s,
                        final_label = %s,
                        needs_action = %s,
                        is_important = %s,
                        time_sensitivity = %s,
                        gmail_flagged = CASE WHEN %s THEN TRUE ELSE gmail_flagged END
                    WHERE category = %s
                      AND (
                        final_label IS NULL
                        OR final_label = category
                        OR final_label NOT LIKE '%%_%%'
                        OR final_label = 'EM_DUVIDA'
                      )
                    """,
                    (
                        theme_category,
                        priority_level,
                        final_label,
                        needs_action,
                        is_important,
                        time_sensitivity,
                        final_label.startswith("ALTA_"),
                        legacy_category,
                    ),
                )
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN theme_category SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN priority_level SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN final_label SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN needs_action SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN is_important SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN time_sensitivity SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN confidence SET NOT NULL")
            cursor.execute("ALTER TABLE processed_emails ALTER COLUMN gmail_flagged SET NOT NULL")
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
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_theme_category
                ON processed_emails (theme_category)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_final_label
                ON processed_emails (final_label)
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


def save_processed_email(
    message_id: str,
    sender: str,
    subject: str,
    status: str,
    category: str,
    theme_category: str,
    priority_level: str,
    final_label: str,
    needs_action: bool,
    is_important: bool,
    time_sensitivity: str,
    confidence: float,
    gmail_flagged: bool,
) -> None:
    """Persist the triage result into PostgreSQL."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO processed_emails (
                    message_id, sender, subject, status, category, theme_category,
                    priority_level, final_label, needs_action, is_important,
                    time_sensitivity, confidence, gmail_flagged
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(message_id) DO UPDATE SET
                    sender = EXCLUDED.sender,
                    subject = EXCLUDED.subject,
                    status = EXCLUDED.status,
                    category = EXCLUDED.category,
                    theme_category = EXCLUDED.theme_category,
                    priority_level = EXCLUDED.priority_level,
                    final_label = EXCLUDED.final_label,
                    needs_action = EXCLUDED.needs_action,
                    is_important = EXCLUDED.is_important,
                    time_sensitivity = EXCLUDED.time_sensitivity,
                    confidence = EXCLUDED.confidence,
                    gmail_flagged = EXCLUDED.gmail_flagged,
                    updated_at = NOW()
                """,
                (
                    message_id,
                    sender,
                    subject,
                    status,
                    category,
                    theme_category,
                    priority_level,
                    final_label,
                    needs_action,
                    is_important,
                    time_sensitivity,
                    confidence,
                    gmail_flagged,
                ),
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
    theme_category: str,
    priority_level: str,
    final_label: str,
) -> None:
    """Persist new semantic memory into ChromaDB."""

    collection = get_chroma_collection()
    document = (
        f"Tema: {theme_category}\n"
        f"Prioridade: {priority_level}\n"
        f"Label final: {final_label}\n"
        f"Remetente: {email_data.get('sender', '')}\n"
        f"Assunto: {email_data.get('subject', '')}\n"
        f"Resumo factual: {learned_text}"
    )
    collection.upsert(
        ids=[f"{message_id}-{uuid.uuid4().hex[:8]}"],
        documents=[document],
        metadatas=[
            {
                "message_id": message_id,
                "sender": str(email_data.get("sender", "")),
                "subject": str(email_data.get("subject", "")),
                "theme_category": theme_category,
                "priority_level": priority_level,
                "final_label": final_label,
            }
        ],
    )


def query_semantic_examples(query_text: str, n_results: int = 8) -> list[FewShotExample]:
    """Retrieve semantically similar memory examples for few-shot construction."""

    collection = get_chroma_collection()
    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = cast(list[list[str]], result.get("documents", [[]]))
    metadatas = cast(list[list[dict[str, Any]]], result.get("metadatas", [[]]))
    distances = cast(list[list[float]], result.get("distances", [[]]))

    rows: list[FewShotExample] = []
    for index, document in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else 0.0
        theme_category = str(metadata.get("theme_category", metadata.get("category", "EM_DUVIDA"))).strip().upper()
        priority_level = str(metadata.get("priority_level", "BAIXA")).strip().upper()
        final_label = str(metadata.get("final_label", metadata.get("category", "EM_DUVIDA"))).strip().upper()
        if theme_category not in THEME_CATEGORIES or final_label not in FINAL_LABELS:
            normalized_theme, normalized_priority, normalized_final_label, _, _, _ = normalize_legacy_category(
                str(metadata.get("category", final_label))
            )
            theme_category = normalized_theme
            priority_level = normalized_priority
            final_label = normalized_final_label
        if theme_category not in THEME_CATEGORIES:
            theme_category = "EM_DUVIDA"
        if priority_level not in PRIORITY_LEVELS:
            priority_level = "BAIXA"
        if final_label not in FINAL_LABELS:
            final_label = "EM_DUVIDA"
        rows.append(
            {
                "document": document,
                "theme_category": theme_category,
                "priority_level": priority_level,
                "final_label": final_label,
                "sample_role": "CANDIDATE",
                "distance": float(distance),
            }
        )
    return rows


def get_dashboard_metrics() -> dict[str, Any]:
    """Compute the dashboard metrics used by frontend clients."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails")
            total_processed = int(cursor.fetchone()["total"])
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails WHERE status = 'EM_DUVIDA'")
            pending_review = int(cursor.fetchone()["total"])
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails WHERE priority_level = 'ALTA'")
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
    """Return the accumulated triage volume per final label from PostgreSQL."""

    bootstrap_services()
    counts = {label: 0 for label in FINAL_LABELS}

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT final_label, COUNT(*) AS total
                FROM processed_emails
                GROUP BY final_label
                """
            )
            rows = cursor.fetchall()

    for row in rows:
        row_dict = cast(dict[str, Any], row)
        final_label = str(row_dict["final_label"]).strip().upper()
        if final_label in counts:
            counts[final_label] = int(row_dict["total"])

    return [{"categoria": label, "quantidade": counts[label]} for label in FINAL_LABELS]


def get_pending_review_rows() -> list[dict[str, Any]]:
    """Return pending manual-review rows from PostgreSQL."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    message_id,
                    sender,
                    subject,
                    status,
                    category,
                    theme_category,
                    priority_level,
                    final_label,
                    needs_action,
                    is_important,
                    time_sensitivity,
                    confidence,
                    gmail_flagged
                FROM processed_emails
                WHERE status = 'EM_DUVIDA'
                ORDER BY sender, subject
                """
            )
            rows = cursor.fetchall()

    return [dict(cast(dict[str, Any], row)) for row in rows]


def get_recent_processed_rows(limit: int = 50) -> list[dict[str, Any]]:
    """Return recently processed emails for audit and manual correction."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    message_id,
                    sender,
                    subject,
                    status,
                    category,
                    theme_category,
                    priority_level,
                    final_label,
                    needs_action,
                    is_important,
                    time_sensitivity,
                    confidence,
                    gmail_flagged,
                    created_at,
                    updated_at
                FROM processed_emails
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

    return [dict(cast(dict[str, Any], row)) for row in rows]
