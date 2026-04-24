"""Persistence helpers for PostgreSQL and ChromaDB."""

from __future__ import annotations

import json
import hashlib
import uuid
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection

from email_agent.models import FINAL_LABELS, FewShotExample, PRIORITY_LEVELS, THEME_CATEGORIES
from email_agent.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    WRITING_FEEDBACK_COLLECTION_NAME,
    WRITING_RUNS_COLLECTION_NAME,
    WRITING_SAMPLES_COLLECTION_NAME,
    dict_row,
    get_database_url,
    psycopg,
)
from writing_agent.models import (
    DEFAULT_GENRE_CARDS,
    DEFAULT_INSPIRATION_PROFILES,
    DEFAULT_STYLE_PROFILES,
    GENRE_CARD_IDS,
    STYLE_REASON_TAGS,
)


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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS supervisor_executions (
                    execution_id VARCHAR(64) PRIMARY KEY,
                    supervisor_name VARCHAR(100) NOT NULL,
                    user_request TEXT NOT NULL,
                    requested_agent VARCHAR(100),
                    selected_agents_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    execution_plan_json TEXT NOT NULL,
                    writing_task_type VARCHAR(50),
                    tone TEXT,
                    audience TEXT,
                    context_notes TEXT,
                    email_limit INTEGER,
                    output_preview TEXT,
                    agent_outputs_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_style_profiles (
                    profile_id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    owner_scope VARCHAR(100) NOT NULL,
                    voice_traits_json TEXT NOT NULL,
                    structure_traits_json TEXT NOT NULL,
                    rhetorical_traits_json TEXT NOT NULL,
                    lexical_traits_json TEXT NOT NULL,
                    dos_json TEXT NOT NULL,
                    donts_json TEXT NOT NULL,
                    sample_text_ids_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_genre_cards (
                    genre_id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    primary_goal VARCHAR(50) NOT NULL,
                    expected_structure_json TEXT NOT NULL,
                    tone_defaults_json TEXT NOT NULL,
                    length_defaults_json TEXT NOT NULL,
                    quality_checklist_json TEXT NOT NULL,
                    typical_openings_json TEXT NOT NULL,
                    typical_closings_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_inspiration_profiles (
                    inspiration_profile_id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    traits_to_borrow_json TEXT NOT NULL,
                    forbidden_behaviors_json TEXT NOT NULL,
                    transformation_strength VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_samples (
                    sample_id VARCHAR(64) PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_scope VARCHAR(100) NOT NULL,
                    text_content TEXT NOT NULL,
                    editorial_summary TEXT NOT NULL,
                    style_profile_id VARCHAR(100),
                    genre_id VARCHAR(100),
                    text_fingerprint VARCHAR(64),
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_executions (
                    execution_id VARCHAR(64) PRIMARY KEY,
                    user_request TEXT NOT NULL,
                    task_type VARCHAR(50) NOT NULL,
                    goal VARCHAR(50) NOT NULL,
                    genre_id VARCHAR(100) NOT NULL,
                    style_profile_id VARCHAR(100),
                    inspiration_profile_id VARCHAR(100),
                    audience TEXT NOT NULL,
                    tone_override TEXT NOT NULL,
                    source_notes TEXT NOT NULL,
                    must_include_json TEXT NOT NULL,
                    must_avoid_json TEXT NOT NULL,
                    reference_text TEXT NOT NULL,
                    normalized_brief_json TEXT NOT NULL,
                    applied_style_profile_json TEXT NOT NULL,
                    applied_genre_card_json TEXT NOT NULL,
                    retrieved_examples_preview_json TEXT NOT NULL,
                    outline_text TEXT NOT NULL,
                    draft_text TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    quality_report_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_versions (
                    version_id VARCHAR(64) PRIMARY KEY,
                    execution_id VARCHAR(64) NOT NULL REFERENCES antese_executions(execution_id) ON DELETE CASCADE,
                    stage VARCHAR(50) NOT NULL,
                    action_label VARCHAR(100) NOT NULL,
                    output_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS antese_feedback (
                    feedback_id VARCHAR(64) PRIMARY KEY,
                    execution_id VARCHAR(64) REFERENCES antese_executions(execution_id) ON DELETE CASCADE,
                    version_id VARCHAR(64) REFERENCES antese_versions(version_id) ON DELETE CASCADE,
                    chosen_version_id VARCHAR(64),
                    rejected_version_id VARCHAR(64),
                    liked BOOLEAN,
                    style_match_score DOUBLE PRECISION,
                    usefulness_score DOUBLE PRECISION,
                    creativity_score DOUBLE PRECISION,
                    faithfulness_score DOUBLE PRECISION,
                    notes TEXT NOT NULL DEFAULT '',
                    reason_tags_json TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS requested_agent VARCHAR(100)")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS selected_agents_json TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS reason TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS execution_plan_json TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS writing_task_type VARCHAR(50)")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS tone TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS audience TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS context_notes TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS email_limit INTEGER")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS output_preview TEXT")
            cursor.execute("ALTER TABLE supervisor_executions ADD COLUMN IF NOT EXISTS agent_outputs_json TEXT")
            cursor.execute("ALTER TABLE antese_samples ADD COLUMN IF NOT EXISTS text_fingerprint VARCHAR(64)")
            cursor.execute(
                """
                UPDATE antese_samples
                SET text_fingerprint = md5(lower(regexp_replace(text_content, '\\s+', ' ', 'g')))
                WHERE text_fingerprint IS NULL
                """
            )
            cursor.execute(
                """
                UPDATE supervisor_executions
                SET
                    selected_agents_json = COALESCE(selected_agents_json, '[]'),
                    reason = COALESCE(reason, ''),
                    execution_plan_json = COALESCE(execution_plan_json, '[]'),
                    agent_outputs_json = COALESCE(agent_outputs_json, '[]')
                """
            )
            cursor.execute("ALTER TABLE supervisor_executions ALTER COLUMN selected_agents_json SET NOT NULL")
            cursor.execute("ALTER TABLE supervisor_executions ALTER COLUMN reason SET NOT NULL")
            cursor.execute("ALTER TABLE supervisor_executions ALTER COLUMN execution_plan_json SET NOT NULL")
            cursor.execute("ALTER TABLE supervisor_executions ALTER COLUMN agent_outputs_json SET NOT NULL")
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_antese_samples_text_fingerprint
                ON antese_samples (text_fingerprint)
                WHERE text_fingerprint IS NOT NULL
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_samples_updated_at
                ON antese_samples (updated_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_samples_genre_id
                ON antese_samples (genre_id)
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
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_supervisor_executions_created_at
                ON supervisor_executions (created_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_executions_created_at
                ON antese_executions (created_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_executions_genre_id
                ON antese_executions (genre_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_versions_execution_id
                ON antese_versions (execution_id, created_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_antese_feedback_execution_id
                ON antese_feedback (execution_id, created_at DESC)
                """
            )
        connection.commit()


def get_chroma_collection() -> Collection:
    """Return the local Chroma collection used as semantic memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def get_writing_samples_collection() -> Collection:
    """Return the Chroma collection used for writing sample memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=WRITING_SAMPLES_COLLECTION_NAME)


def get_writing_feedback_collection() -> Collection:
    """Return the Chroma collection used for writing feedback memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=WRITING_FEEDBACK_COLLECTION_NAME)


def get_writing_runs_collection() -> Collection:
    """Return the Chroma collection used for Antese run memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=WRITING_RUNS_COLLECTION_NAME)


def _seed_style_profiles() -> None:
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            for item in DEFAULT_STYLE_PROFILES:
                cursor.execute(
                    """
                    INSERT INTO antese_style_profiles (
                        profile_id, name, owner_scope, voice_traits_json, structure_traits_json,
                        rhetorical_traits_json, lexical_traits_json, dos_json, donts_json, sample_text_ids_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (profile_id) DO NOTHING
                    """,
                    (
                        item["profile_id"],
                        item["name"],
                        item["owner_scope"],
                        json.dumps(item["voice_traits"], ensure_ascii=False),
                        json.dumps(item["structure_traits"], ensure_ascii=False),
                        json.dumps(item["rhetorical_traits"], ensure_ascii=False),
                        json.dumps(item["lexical_traits"], ensure_ascii=False),
                        json.dumps(item["dos"], ensure_ascii=False),
                        json.dumps(item["donts"], ensure_ascii=False),
                        json.dumps(item["sample_text_ids"], ensure_ascii=False),
                    ),
                )
        connection.commit()


def _seed_genre_cards() -> None:
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            for item in DEFAULT_GENRE_CARDS:
                cursor.execute(
                    """
                    INSERT INTO antese_genre_cards (
                        genre_id, name, primary_goal, expected_structure_json, tone_defaults_json,
                        length_defaults_json, quality_checklist_json, typical_openings_json, typical_closings_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (genre_id) DO NOTHING
                    """,
                    (
                        item["genre_id"],
                        item["name"],
                        item["primary_goal"],
                        json.dumps(item["expected_structure"], ensure_ascii=False),
                        json.dumps(item["tone_defaults"], ensure_ascii=False),
                        json.dumps(item["length_defaults"], ensure_ascii=False),
                        json.dumps(item["quality_checklist"], ensure_ascii=False),
                        json.dumps(item["typical_openings"], ensure_ascii=False),
                        json.dumps(item["typical_closings"], ensure_ascii=False),
                    ),
                )
        connection.commit()


def _seed_inspiration_profiles() -> None:
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            for item in DEFAULT_INSPIRATION_PROFILES:
                cursor.execute(
                    """
                    INSERT INTO antese_inspiration_profiles (
                        inspiration_profile_id, name, traits_to_borrow_json,
                        forbidden_behaviors_json, transformation_strength
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (inspiration_profile_id) DO NOTHING
                    """,
                    (
                        item["inspiration_profile_id"],
                        item["name"],
                        json.dumps(item["traits_to_borrow"], ensure_ascii=False),
                        json.dumps(item["forbidden_behaviors"], ensure_ascii=False),
                        item["transformation_strength"],
                    ),
                )
        connection.commit()


def bootstrap_services() -> None:
    """Initialize local persistence layers."""

    init_postgres_db()
    _ = get_chroma_collection()
    _ = get_writing_samples_collection()
    _ = get_writing_feedback_collection()
    _ = get_writing_runs_collection()
    _seed_style_profiles()
    _seed_genre_cards()
    _seed_inspiration_profiles()


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


def save_supervisor_execution(
    supervisor_name: str,
    user_request: str,
    requested_agent: str | None,
    selected_agents: list[str],
    reason: str,
    execution_plan: list[str],
    writing_task_type: str,
    tone: str,
    audience: str,
    context_notes: str,
    email_limit: int | None,
    agent_outputs: list[dict[str, Any]],
    output_preview: str,
) -> str:
    """Persist one Micelio supervisor execution in PostgreSQL."""

    bootstrap_services()
    execution_id = uuid.uuid4().hex
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO supervisor_executions (
                    execution_id,
                    supervisor_name,
                    user_request,
                    requested_agent,
                    selected_agents_json,
                    reason,
                    execution_plan_json,
                    writing_task_type,
                    tone,
                    audience,
                    context_notes,
                    email_limit,
                    output_preview,
                    agent_outputs_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    execution_id,
                    supervisor_name,
                    user_request,
                    requested_agent,
                    json.dumps(selected_agents, ensure_ascii=False),
                    reason,
                    json.dumps(execution_plan, ensure_ascii=False),
                    writing_task_type,
                    tone,
                    audience,
                    context_notes,
                    email_limit,
                    output_preview,
                    json.dumps(agent_outputs, ensure_ascii=False),
                ),
            )
        connection.commit()
    return execution_id


def get_recent_supervisor_executions(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent Micelio executions with parsed JSON fields."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    execution_id,
                    supervisor_name,
                    user_request,
                    requested_agent,
                    selected_agents_json,
                    reason,
                    execution_plan_json,
                    writing_task_type,
                    tone,
                    audience,
                    context_notes,
                    email_limit,
                    output_preview,
                    agent_outputs_json,
                    created_at,
                    updated_at
                FROM supervisor_executions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["selected_agents"] = json.loads(str(row_dict.pop("selected_agents_json", "[]")))
        row_dict["execution_plan"] = json.loads(str(row_dict.pop("execution_plan_json", "[]")))
        row_dict["agent_outputs"] = json.loads(str(row_dict.pop("agent_outputs_json", "[]")))
        parsed_rows.append(row_dict)
    return parsed_rows


def _parse_json_column(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _normalize_text_for_fingerprint(text_content: str) -> str:
    return " ".join(str(text_content).split()).strip().lower()


def _build_text_fingerprint(text_content: str) -> str:
    normalized = _normalize_text_for_fingerprint(text_content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_antese_style_profiles() -> list[dict[str, Any]]:
    """Return style profiles available to Antese."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    profile_id,
                    name,
                    owner_scope,
                    voice_traits_json,
                    structure_traits_json,
                    rhetorical_traits_json,
                    lexical_traits_json,
                    dos_json,
                    donts_json,
                    sample_text_ids_json,
                    created_at,
                    updated_at
                FROM antese_style_profiles
                ORDER BY
                    CASE WHEN profile_id = 'personal_default' THEN 0 ELSE 1 END,
                    name ASC
                """
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["voice_traits"] = _parse_json_column(row_dict.pop("voice_traits_json", "{}"), {})
        row_dict["structure_traits"] = _parse_json_column(row_dict.pop("structure_traits_json", "{}"), {})
        row_dict["rhetorical_traits"] = _parse_json_column(row_dict.pop("rhetorical_traits_json", "{}"), {})
        row_dict["lexical_traits"] = _parse_json_column(row_dict.pop("lexical_traits_json", "{}"), {})
        row_dict["dos"] = _parse_json_column(row_dict.pop("dos_json", "[]"), [])
        row_dict["donts"] = _parse_json_column(row_dict.pop("donts_json", "[]"), [])
        row_dict["sample_text_ids"] = _parse_json_column(row_dict.pop("sample_text_ids_json", "[]"), [])
        parsed_rows.append(row_dict)
    return parsed_rows


def save_antese_style_profile(
    profile_id: str,
    name: str,
    owner_scope: str,
    voice_traits: dict[str, Any],
    structure_traits: dict[str, Any],
    rhetorical_traits: dict[str, Any],
    lexical_traits: dict[str, Any],
    dos: list[str],
    donts: list[str],
    sample_text_ids: list[str],
) -> dict[str, Any]:
    """Create or update one Antese style profile."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO antese_style_profiles (
                    profile_id, name, owner_scope, voice_traits_json, structure_traits_json,
                    rhetorical_traits_json, lexical_traits_json, dos_json, donts_json, sample_text_ids_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    owner_scope = EXCLUDED.owner_scope,
                    voice_traits_json = EXCLUDED.voice_traits_json,
                    structure_traits_json = EXCLUDED.structure_traits_json,
                    rhetorical_traits_json = EXCLUDED.rhetorical_traits_json,
                    lexical_traits_json = EXCLUDED.lexical_traits_json,
                    dos_json = EXCLUDED.dos_json,
                    donts_json = EXCLUDED.donts_json,
                    sample_text_ids_json = EXCLUDED.sample_text_ids_json,
                    updated_at = NOW()
                """,
                (
                    profile_id,
                    name,
                    owner_scope,
                    json.dumps(voice_traits, ensure_ascii=False),
                    json.dumps(structure_traits, ensure_ascii=False),
                    json.dumps(rhetorical_traits, ensure_ascii=False),
                    json.dumps(lexical_traits, ensure_ascii=False),
                    json.dumps(dos, ensure_ascii=False),
                    json.dumps(donts, ensure_ascii=False),
                    json.dumps(sample_text_ids, ensure_ascii=False),
                ),
            )
        connection.commit()
    return next(item for item in get_antese_style_profiles() if item["profile_id"] == profile_id)


def get_antese_genre_cards() -> list[dict[str, Any]]:
    """Return genre cards available to Antese."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    genre_id,
                    name,
                    primary_goal,
                    expected_structure_json,
                    tone_defaults_json,
                    length_defaults_json,
                    quality_checklist_json,
                    typical_openings_json,
                    typical_closings_json,
                    created_at,
                    updated_at
                FROM antese_genre_cards
                ORDER BY name ASC
                """
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["expected_structure"] = _parse_json_column(row_dict.pop("expected_structure_json", "[]"), [])
        row_dict["tone_defaults"] = _parse_json_column(row_dict.pop("tone_defaults_json", "[]"), [])
        row_dict["length_defaults"] = _parse_json_column(row_dict.pop("length_defaults_json", "{}"), {})
        row_dict["quality_checklist"] = _parse_json_column(row_dict.pop("quality_checklist_json", "[]"), [])
        row_dict["typical_openings"] = _parse_json_column(row_dict.pop("typical_openings_json", "[]"), [])
        row_dict["typical_closings"] = _parse_json_column(row_dict.pop("typical_closings_json", "[]"), [])
        parsed_rows.append(row_dict)
    return parsed_rows


def save_antese_genre_card(
    genre_id: str,
    name: str,
    primary_goal: str,
    expected_structure: list[str],
    tone_defaults: list[str],
    length_defaults: dict[str, Any],
    quality_checklist: list[str],
    typical_openings: list[str],
    typical_closings: list[str],
) -> dict[str, Any]:
    """Create or update one Antese genre card."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO antese_genre_cards (
                    genre_id, name, primary_goal, expected_structure_json, tone_defaults_json,
                    length_defaults_json, quality_checklist_json, typical_openings_json, typical_closings_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (genre_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    primary_goal = EXCLUDED.primary_goal,
                    expected_structure_json = EXCLUDED.expected_structure_json,
                    tone_defaults_json = EXCLUDED.tone_defaults_json,
                    length_defaults_json = EXCLUDED.length_defaults_json,
                    quality_checklist_json = EXCLUDED.quality_checklist_json,
                    typical_openings_json = EXCLUDED.typical_openings_json,
                    typical_closings_json = EXCLUDED.typical_closings_json,
                    updated_at = NOW()
                """,
                (
                    genre_id,
                    name,
                    primary_goal,
                    json.dumps(expected_structure, ensure_ascii=False),
                    json.dumps(tone_defaults, ensure_ascii=False),
                    json.dumps(length_defaults, ensure_ascii=False),
                    json.dumps(quality_checklist, ensure_ascii=False),
                    json.dumps(typical_openings, ensure_ascii=False),
                    json.dumps(typical_closings, ensure_ascii=False),
                ),
            )
        connection.commit()
    return next(item for item in get_antese_genre_cards() if item["genre_id"] == genre_id)


def get_antese_inspiration_profiles() -> list[dict[str, Any]]:
    """Return inspiration profiles used as style modifiers."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    inspiration_profile_id,
                    name,
                    traits_to_borrow_json,
                    forbidden_behaviors_json,
                    transformation_strength,
                    created_at,
                    updated_at
                FROM antese_inspiration_profiles
                ORDER BY name ASC
                """
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["traits_to_borrow"] = _parse_json_column(row_dict.pop("traits_to_borrow_json", "[]"), [])
        row_dict["forbidden_behaviors"] = _parse_json_column(row_dict.pop("forbidden_behaviors_json", "[]"), [])
        parsed_rows.append(row_dict)
    return parsed_rows


def save_antese_sample(
    title: str,
    source_scope: str,
    text_content: str,
    editorial_summary: str,
    style_profile_id: str | None,
    genre_id: str | None,
    metadata: dict[str, Any],
) -> str:
    """Persist one writing sample and index it in Chroma."""

    bootstrap_services()
    normalized_fingerprint = _build_text_fingerprint(text_content)
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT sample_id, style_profile_id, genre_id, metadata_json
                FROM antese_samples
                WHERE text_fingerprint = %s
                LIMIT 1
                """,
                (normalized_fingerprint,),
            )
            existing_row = cursor.fetchone()
            if existing_row:
                existing = dict(cast(dict[str, Any], existing_row))
                sample_id = str(existing["sample_id"])
                existing_metadata = _parse_json_column(existing.get("metadata_json", "{}"), {})
                merged_metadata = {**existing_metadata, **metadata}
                cursor.execute(
                    """
                    UPDATE antese_samples
                    SET
                        title = %s,
                        source_scope = %s,
                        editorial_summary = %s,
                        style_profile_id = %s,
                        genre_id = %s,
                        metadata_json = %s,
                        updated_at = NOW()
                    WHERE sample_id = %s
                    """,
                    (
                        title,
                        source_scope,
                        editorial_summary,
                        style_profile_id or existing.get("style_profile_id"),
                        genre_id or existing.get("genre_id"),
                        json.dumps(merged_metadata, ensure_ascii=False),
                        sample_id,
                    ),
                )
                connection.commit()
                get_writing_samples_collection().upsert(
                    ids=[sample_id],
                    documents=[f"{title}\n\n{text_content}".strip()],
                    metadatas=[
                        {
                            "sample_id": sample_id,
                            "title": title,
                            "source_scope": source_scope,
                            "style_profile_id": style_profile_id or str(existing.get("style_profile_id", "")),
                            "genre_id": genre_id or str(existing.get("genre_id", "")),
                            "editorial_summary": editorial_summary,
                            "text_fingerprint": normalized_fingerprint,
                        }
                    ],
                )
                return sample_id

            sample_id = uuid.uuid4().hex
            cursor.execute(
                """
                INSERT INTO antese_samples (
                    sample_id, title, source_scope, text_content, editorial_summary,
                    style_profile_id, genre_id, text_fingerprint, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sample_id,
                    title,
                    source_scope,
                    text_content,
                    editorial_summary,
                    style_profile_id,
                    genre_id,
                    normalized_fingerprint,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        connection.commit()

    get_writing_samples_collection().upsert(
        ids=[sample_id],
        documents=[f"{title}\n\n{text_content}".strip()],
        metadatas=[
            {
                "sample_id": sample_id,
                "title": title,
                "source_scope": source_scope,
                "style_profile_id": style_profile_id or "",
                "genre_id": genre_id or "",
                "editorial_summary": editorial_summary,
                "text_fingerprint": normalized_fingerprint,
            }
        ],
    )
    return sample_id


def get_antese_samples(limit: int = 200) -> list[dict[str, Any]]:
    """Return persisted writing samples for audit and manual genre correction."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    sample_id,
                    title,
                    source_scope,
                    text_content,
                    editorial_summary,
                    style_profile_id,
                    genre_id,
                    text_fingerprint,
                    metadata_json,
                    created_at,
                    updated_at
                FROM antese_samples
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["metadata"] = _parse_json_column(row_dict.pop("metadata_json", "{}"), {})
        parsed_rows.append(row_dict)
    return parsed_rows


def update_antese_sample_genre(sample_id: str, genre_id: str | None) -> dict[str, Any]:
    """Update the manual genre classification of one writing sample and sync Chroma metadata."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE antese_samples
                SET genre_id = %s, updated_at = NOW()
                WHERE sample_id = %s
                RETURNING
                    sample_id,
                    title,
                    source_scope,
                    text_content,
                    editorial_summary,
                    style_profile_id,
                    genre_id,
                    text_fingerprint,
                    metadata_json,
                    created_at,
                    updated_at
                """,
                (genre_id, sample_id),
            )
            row = cursor.fetchone()
        connection.commit()

    if not row:
        raise ValueError("Sample não encontrado para reclassificação.")

    row_dict = dict(cast(dict[str, Any], row))
    metadata = _parse_json_column(row_dict.pop("metadata_json", "{}"), {})
    get_writing_samples_collection().upsert(
        ids=[sample_id],
        documents=[f"{row_dict['title']}\n\n{row_dict['text_content']}".strip()],
        metadatas=[
            {
                "sample_id": sample_id,
                "title": row_dict["title"],
                "source_scope": row_dict["source_scope"],
                "style_profile_id": row_dict.get("style_profile_id") or "",
                "genre_id": row_dict.get("genre_id") or "",
                "editorial_summary": row_dict["editorial_summary"],
                "text_fingerprint": row_dict.get("text_fingerprint") or "",
                **({"sample_type": metadata.get("sample_type", "")} if metadata else {}),
            }
        ],
    )
    row_dict["metadata"] = metadata
    return row_dict


def query_writing_samples(
    query_text: str,
    style_profile_id: str | None = None,
    genre_id: str | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant writing samples from the dedicated Antese memory."""

    bootstrap_services()
    collection = get_writing_samples_collection()
    result = collection.query(
        query_texts=[query_text],
        n_results=max(1, n_results),
        include=["documents", "metadatas", "distances"],
    )
    documents = cast(list[list[str]], result.get("documents", [[]]))
    metadatas = cast(list[list[dict[str, Any]]], result.get("metadatas", [[]]))
    distances = cast(list[list[float]], result.get("distances", [[]]))

    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        row = {
            "document": document,
            "distance": float(distances[0][index]) if distances and distances[0] and index < len(distances[0]) else 0.0,
            "sample_id": str(metadata.get("sample_id", "")),
            "title": str(metadata.get("title", "")),
            "source_scope": str(metadata.get("source_scope", "")),
            "style_profile_id": str(metadata.get("style_profile_id", "")),
            "genre_id": str(metadata.get("genre_id", "")),
            "editorial_summary": str(metadata.get("editorial_summary", "")),
        }
        if style_profile_id and row["style_profile_id"] and row["style_profile_id"] != style_profile_id:
            continue
        if genre_id and row["genre_id"] and row["genre_id"] != genre_id:
            continue
        rows.append(row)
    return rows


def query_writing_feedback_memory(query_text: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Retrieve relevant editorial feedback from dedicated memory."""

    bootstrap_services()
    collection = get_writing_feedback_collection()
    try:
        result = collection.query(
            query_texts=[query_text],
            n_results=max(1, n_results),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents = cast(list[list[str]], result.get("documents", [[]]))
    metadatas = cast(list[list[dict[str, Any]]], result.get("metadatas", [[]]))
    distances = cast(list[list[float]], result.get("distances", [[]]))

    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        rows.append(
            {
                "document": document,
                "distance": float(distances[0][index]) if distances and distances[0] and index < len(distances[0]) else 0.0,
                "execution_id": str(metadata.get("execution_id", "")),
                "version_id": str(metadata.get("version_id", "")),
                "reason_tags": _parse_json_column(metadata.get("reason_tags", "[]"), []),
            }
        )
    return rows


def query_writing_runs_memory(query_text: str, n_results: int = 3) -> list[dict[str, Any]]:
    """Retrieve prior Antese runs from dedicated run memory."""

    bootstrap_services()
    collection = get_writing_runs_collection()
    try:
        result = collection.query(
            query_texts=[query_text],
            n_results=max(1, n_results),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    documents = cast(list[list[str]], result.get("documents", [[]]))
    metadatas = cast(list[list[dict[str, Any]]], result.get("metadatas", [[]]))
    distances = cast(list[list[float]], result.get("distances", [[]]))

    rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents[0] if documents else []):
        metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        rows.append(
            {
                "document": document,
                "distance": float(distances[0][index]) if distances and distances[0] and index < len(distances[0]) else 0.0,
                "execution_id": str(metadata.get("execution_id", "")),
                "task_type": str(metadata.get("task_type", "")),
                "genre_id": str(metadata.get("genre_id", "")),
            }
        )
    return rows


def save_antese_execution(
    user_request: str,
    task_type: str,
    goal: str,
    genre_id: str,
    style_profile_id: str | None,
    inspiration_profile_id: str | None,
    audience: str,
    tone_override: str,
    source_notes: str,
    must_include: list[str],
    must_avoid: list[str],
    reference_text: str,
    normalized_brief: dict[str, Any],
    applied_style_profile: dict[str, Any],
    applied_genre_card: dict[str, Any],
    retrieved_examples_preview: list[dict[str, Any]],
    outline_text: str,
    draft_text: str,
    final_text: str,
    quality_report: dict[str, Any],
) -> str:
    """Persist one Antese execution and index it in memory."""

    bootstrap_services()
    execution_id = uuid.uuid4().hex
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO antese_executions (
                    execution_id, user_request, task_type, goal, genre_id, style_profile_id,
                    inspiration_profile_id, audience, tone_override, source_notes, must_include_json,
                    must_avoid_json, reference_text, normalized_brief_json, applied_style_profile_json,
                    applied_genre_card_json, retrieved_examples_preview_json, outline_text, draft_text,
                    final_text, quality_report_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    execution_id,
                    user_request,
                    task_type,
                    goal,
                    genre_id,
                    style_profile_id,
                    inspiration_profile_id,
                    audience,
                    tone_override,
                    source_notes,
                    json.dumps(must_include, ensure_ascii=False),
                    json.dumps(must_avoid, ensure_ascii=False),
                    reference_text,
                    json.dumps(normalized_brief, ensure_ascii=False),
                    json.dumps(applied_style_profile, ensure_ascii=False),
                    json.dumps(applied_genre_card, ensure_ascii=False),
                    json.dumps(retrieved_examples_preview, ensure_ascii=False),
                    outline_text,
                    draft_text,
                    final_text,
                    json.dumps(quality_report, ensure_ascii=False),
                ),
            )
        connection.commit()

    get_writing_runs_collection().upsert(
        ids=[execution_id],
        documents=[f"{user_request}\n\n{final_text}".strip()],
        metadatas=[
            {
                "execution_id": execution_id,
                "task_type": task_type,
                "genre_id": genre_id,
                "style_profile_id": style_profile_id or "",
            }
        ],
    )
    return execution_id


def save_antese_version(
    execution_id: str,
    stage: str,
    action_label: str,
    output_text: str,
    metadata: dict[str, Any],
) -> str:
    """Persist one Antese version snapshot."""

    bootstrap_services()
    version_id = uuid.uuid4().hex
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO antese_versions (
                    version_id, execution_id, stage, action_label, output_text, metadata_json
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    version_id,
                    execution_id,
                    stage,
                    action_label,
                    output_text,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
        connection.commit()
    return version_id


def get_antese_executions(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent Antese executions with parsed JSON fields."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    execution_id,
                    user_request,
                    task_type,
                    goal,
                    genre_id,
                    style_profile_id,
                    inspiration_profile_id,
                    audience,
                    tone_override,
                    source_notes,
                    must_include_json,
                    must_avoid_json,
                    reference_text,
                    normalized_brief_json,
                    applied_style_profile_json,
                    applied_genre_card_json,
                    retrieved_examples_preview_json,
                    outline_text,
                    draft_text,
                    final_text,
                    quality_report_json,
                    created_at,
                    updated_at
                FROM antese_executions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["must_include"] = _parse_json_column(row_dict.pop("must_include_json", "[]"), [])
        row_dict["must_avoid"] = _parse_json_column(row_dict.pop("must_avoid_json", "[]"), [])
        row_dict["normalized_brief"] = _parse_json_column(row_dict.pop("normalized_brief_json", "{}"), {})
        row_dict["applied_style_profile"] = _parse_json_column(row_dict.pop("applied_style_profile_json", "{}"), {})
        row_dict["applied_genre_card"] = _parse_json_column(row_dict.pop("applied_genre_card_json", "{}"), {})
        row_dict["retrieved_examples_preview"] = _parse_json_column(
            row_dict.pop("retrieved_examples_preview_json", "[]"),
            [],
        )
        row_dict["quality_report"] = _parse_json_column(row_dict.pop("quality_report_json", "{}"), {})
        parsed_rows.append(row_dict)
    return parsed_rows


def get_antese_execution_versions(execution_id: str) -> list[dict[str, Any]]:
    """Return all stored versions for one Antese execution."""

    bootstrap_services()
    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT version_id, execution_id, stage, action_label, output_text, metadata_json, created_at
                FROM antese_versions
                WHERE execution_id = %s
                ORDER BY created_at ASC
                """,
                (execution_id,),
            )
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["metadata"] = _parse_json_column(row_dict.pop("metadata_json", "{}"), {})
        parsed_rows.append(row_dict)
    return parsed_rows


def save_antese_feedback(
    execution_id: str | None,
    version_id: str | None,
    chosen_version_id: str | None,
    rejected_version_id: str | None,
    liked: bool | None,
    style_match_score: float | None,
    usefulness_score: float | None,
    creativity_score: float | None,
    faithfulness_score: float | None,
    notes: str,
    reason_tags: list[str],
) -> str:
    """Persist structured feedback and index it in dedicated memory."""

    bootstrap_services()
    feedback_id = uuid.uuid4().hex
    normalized_reason_tags = [tag for tag in reason_tags if tag in STYLE_REASON_TAGS]

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO antese_feedback (
                    feedback_id, execution_id, version_id, chosen_version_id, rejected_version_id,
                    liked, style_match_score, usefulness_score, creativity_score, faithfulness_score,
                    notes, reason_tags_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    feedback_id,
                    execution_id,
                    version_id,
                    chosen_version_id,
                    rejected_version_id,
                    liked,
                    style_match_score,
                    usefulness_score,
                    creativity_score,
                    faithfulness_score,
                    notes,
                    json.dumps(normalized_reason_tags, ensure_ascii=False),
                ),
            )
        connection.commit()

    feedback_document = (
        f"Feedback editorial. Notas: {notes or 'Sem notas.'} "
        f"Tags: {', '.join(normalized_reason_tags) or 'sem tags'}."
    )
    get_writing_feedback_collection().upsert(
        ids=[feedback_id],
        documents=[feedback_document],
        metadatas=[
            {
                "feedback_id": feedback_id,
                "execution_id": execution_id or "",
                "version_id": version_id or "",
                "reason_tags": json.dumps(normalized_reason_tags, ensure_ascii=False),
            }
        ],
    )
    return feedback_id


def get_antese_feedback_history(execution_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return persisted feedback rows for Antese executions."""

    bootstrap_services()
    query = """
        SELECT
            feedback_id,
            execution_id,
            version_id,
            chosen_version_id,
            rejected_version_id,
            liked,
            style_match_score,
            usefulness_score,
            creativity_score,
            faithfulness_score,
            notes,
            reason_tags_json,
            created_at
        FROM antese_feedback
    """
    params: tuple[Any, ...]
    if execution_id:
        query += " WHERE execution_id = %s"
        query += " ORDER BY created_at DESC LIMIT %s"
        params = (execution_id, max(1, limit))
    else:
        query += " ORDER BY created_at DESC LIMIT %s"
        params = (max(1, limit),)

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        row_dict["reason_tags"] = _parse_json_column(row_dict.pop("reason_tags_json", "[]"), [])
        parsed_rows.append(row_dict)
    return parsed_rows
