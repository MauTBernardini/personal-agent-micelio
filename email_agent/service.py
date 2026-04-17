"""Service layer composing storage, providers and workflow for app clients."""

from __future__ import annotations

import os
from typing import Any

from email_agent.email_provider import get_email_message, move_email_message
from email_agent.llm import LLMProviderError, build_factual_memory_summary
from email_agent.models import MANUAL_REVIEW_CATEGORIES, MANUAL_REVIEW_PRIORITY_LEVELS
from email_agent.settings import (
    CHROMA_PATH,
    GoogleCredentials,
    build,
    get_anthropic_model_name,
    get_email_provider,
    get_gemini_fallback_model_name,
    get_gemini_model_name,
    get_gmail_client_id,
    get_gmail_client_secret,
    get_gmail_credentials_file,
    get_llm_provider,
    get_openai_model_name,
    get_triage_batch_size,
)
from email_agent.storage import (
    bootstrap_services,
    get_accumulated_category_counts,
    get_dashboard_metrics,
    get_pending_review_rows,
    get_recent_processed_rows,
    save_processed_email,
    upsert_semantic_memory,
)
from email_agent.workflow import run_triage_workflow, summarize_category_counts


def get_review_queue() -> list[dict[str, Any]]:
    """Return pending manual-review emails, enriched with provider payload."""

    queue: list[dict[str, Any]] = []
    for row_dict in get_pending_review_rows():
        email_payload = get_email_message(str(row_dict["message_id"])) or {}
        queue.append(
            {
                **row_dict,
                "category": row_dict.get("final_label", row_dict.get("category", "EM_DUVIDA")),
                "body": email_payload.get("body", ""),
            }
        )
    return queue


def get_processed_emails(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent processed emails enriched with provider payload when available."""

    rows = get_recent_processed_rows(limit=limit)
    emails: list[dict[str, Any]] = []
    for row_dict in rows:
        email_payload = get_email_message(str(row_dict["message_id"])) or {}
        emails.append(
            {
                **row_dict,
                "category": row_dict.get("final_label", row_dict.get("category", "EM_DUVIDA")),
                "body": email_payload.get("body", ""),
                "labels": email_payload.get("labels", []),
            }
        )
    return emails


def manual_reclassify_email(
    message_id: str,
    forced_category: str,
    forced_priority_level: str = "MEDIA",
) -> dict[str, Any]:
    """Update PostgreSQL and inject a manual-learning note into ChromaDB."""

    if forced_category not in MANUAL_REVIEW_CATEGORIES:
        raise ValueError(f"Categoria manual inválida: {forced_category}")
    if forced_priority_level not in MANUAL_REVIEW_PRIORITY_LEVELS:
        raise ValueError(f"Prioridade manual inválida: {forced_priority_level}")

    bootstrap_services()
    email_data = get_email_message(message_id)
    if email_data is None:
        raise ValueError(f"E-mail não encontrado no provider ativo: {message_id}")

    if forced_category == "EM_DUVIDA":
        final_label = "EM_DUVIDA"
        status = "EM_DUVIDA"
        gmail_flagged = False
        needs_action = False
        is_important = False
        time_sensitivity = "BAIXA"
        priority_level = "BAIXA"
    else:
        priority_level = forced_priority_level
        final_label = f"{priority_level}_{forced_category}"
        status = "PROCESSADO"
        gmail_flagged = priority_level == "ALTA"
        needs_action = priority_level != "BAIXA"
        is_important = priority_level != "BAIXA" or forced_category in {"CARREIRA", "PROJETOS_TECH", "FINANCEIRO"}
        time_sensitivity = priority_level

    save_processed_email(
        message_id=message_id,
        sender=str(email_data["sender"]),
        subject=str(email_data["subject"]),
        status=status,
        category=final_label,
        theme_category=forced_category,
        priority_level=priority_level,
        final_label=final_label,
        needs_action=needs_action,
        is_important=is_important,
        time_sensitivity=time_sensitivity,
        confidence=0.99,
        gmail_flagged=gmail_flagged,
    )

    move_result: dict[str, Any] | None = None
    if final_label != "EM_DUVIDA":
        move_result = move_email_message(message_id, final_label, should_flag=gmail_flagged)

    manual_summary = build_factual_memory_summary(
        email_data,
        {
            "theme_category": forced_category,
            "priority_level": priority_level,
            "final_label": final_label,
            "needs_action": needs_action,
        },
    )
    learned_text = (
        "Aprendizado manual do revisor: este e-mail foi reclassificado para "
        f"{final_label} após análise humana. {manual_summary}"
    )
    upsert_semantic_memory(
        message_id=message_id,
        email_data=email_data,
        learned_text=learned_text,
        theme_category=forced_category,
        priority_level=priority_level,
        final_label=final_label,
    )

    return {
        "message_id": message_id,
        "forced_category": forced_category,
        "forced_priority_level": priority_level,
        "forced_final_label": final_label,
        "updated": True,
        "move_result": move_result,
    }


def get_connection_status() -> dict[str, str]:
    """Expose simple connection health for frontend sidebars."""

    postgres_status = "OFFLINE"
    try:
        bootstrap_services()
        postgres_status = "OK"
    except Exception:
        postgres_status = "OFFLINE"

    chroma_status = "OK" if CHROMA_PATH.exists() else "OFFLINE"

    gmail_status = "MOCK"
    if get_email_provider() == "gmail":
        credentials_ready = bool(get_gmail_client_id() and get_gmail_client_secret()) or get_gmail_credentials_file().exists()
        deps_ready = build is not None and GoogleCredentials is not None
        if credentials_ready and deps_ready:
            gmail_status = "CONFIGURADO"
        elif not credentials_ready:
            gmail_status = "SEM_CREDENTIALS_JSON"
        else:
            gmail_status = "DEPENDENCIAS_PENDENTES"

    llm_provider = get_llm_provider()
    if llm_provider == "gemini":
        llm_status = "CONFIGURADO" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "NAO_CONFIGURADO"
        llm_model = f"{get_gemini_model_name()} -> {get_gemini_fallback_model_name()}"
    elif llm_provider == "openai":
        llm_status = "CONFIGURADO" if os.getenv("OPENAI_API_KEY") else "NAO_CONFIGURADO"
        llm_model = get_openai_model_name()
    elif llm_provider == "anthropic":
        llm_status = "CONFIGURADO" if os.getenv("ANTHROPIC_API_KEY") else "NAO_CONFIGURADO"
        llm_model = get_anthropic_model_name()
    else:
        llm_status = "HEURISTICO"
        llm_model = "local"

    return {
        "gmail": gmail_status,
        "postgres": postgres_status,
        "chroma": chroma_status,
        "llm_provider": llm_provider.upper(),
        "llm_status": llm_status,
        "llm_model": llm_model,
    }


def get_manual_review_categories() -> tuple[str, ...]:
    return MANUAL_REVIEW_CATEGORIES


def get_manual_review_priorities() -> tuple[str, ...]:
    return MANUAL_REVIEW_PRIORITY_LEVELS


def run_triage_and_collect(limit: int | None = None) -> dict[str, Any]:
    """Run the workflow and return a frontend-oriented payload."""

    effective_limit = limit if limit is not None else get_triage_batch_size()
    results = run_triage_workflow(limit=effective_limit)
    return {
        "batch_limit": effective_limit,
        "processed": len(results),
        "results": results,
        "category_counts": summarize_category_counts(results),
    }


def build_dashboard_payload() -> dict[str, Any]:
    """Return the full dashboard payload consumed by frontend clients."""

    return {
        "metrics": get_dashboard_metrics(),
        "category_counts": get_accumulated_category_counts(),
    }


def bootstrap_on_import() -> None:
    try:
        bootstrap_services()
    except Exception:
        pass


bootstrap_on_import()
