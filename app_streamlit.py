"""Streamlit frontend for the personal email triage agent.

This UI talks only to the local FastAPI backend and keeps business rules out of
the presentation layer.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime
from typing import Any
from urllib import error, request

import pandas as pd
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_TIMEOUT_SECONDS = 15.0
DEFAULT_TRIAGE_TIMEOUT_SECONDS = 180.0
PRIORITY_THEMES = {"CARREIRA", "PROJETOS_TECH", "FINANCEIRO"}
REVIEW_CATEGORY = "EM_DUVIDA"


st.set_page_config(
    page_title="Agente Pessoal de Triagem de E-mails",
    page_icon="📬",
    layout="wide",
)


def get_api_base_url() -> str:
    return os.getenv("AGENT_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _get_timeout_from_env(env_name: str, default: float) -> float:
    raw_value = os.getenv(env_name, str(default)).strip()
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return default


def get_default_api_timeout_seconds() -> float:
    return _get_timeout_from_env("AGENT_API_TIMEOUT_SECONDS", DEFAULT_API_TIMEOUT_SECONDS)


def get_triage_timeout_seconds() -> float:
    return _get_timeout_from_env("AGENT_TRIAGE_TIMEOUT_SECONDS", DEFAULT_TRIAGE_TIMEOUT_SECONDS)


def api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> Any:
    """Call the FastAPI backend and return the parsed JSON response."""

    url = f"{get_api_base_url()}{path}"
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method.upper())
    timeout = timeout_seconds if timeout_seconds is not None else get_default_api_timeout_seconds()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw_body).get("detail", raw_body)
        except json.JSONDecodeError:
            detail = raw_body or str(exc)
        raise RuntimeError(str(detail)) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "A API demorou mais do que o tempo limite configurado para responder. "
            "Aumente `AGENT_TRIAGE_TIMEOUT_SECONDS` no `.env`, reduza o batch da triagem "
            "ou verifique se o provider LLM/Gmail está lento."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            "API FastAPI indisponível. Suba o backend com `uvicorn api_server:app --reload`."
        ) from exc

    if not raw_body:
        return None
    return json.loads(raw_body)


def inject_custom_css() -> None:
    """Apply a more intentional visual language to the Streamlit app."""

    st.markdown(
        """
        <style>
        :root {
            --micelio-ink: #132a13;
            --micelio-green: #2d6a4f;
            --micelio-green-soft: #d8f3dc;
            --micelio-sand: #f8f3e8;
            --micelio-amber: #c17c00;
            --micelio-red: #9b2226;
            --micelio-line: rgba(19, 42, 19, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(216, 243, 220, 0.9), transparent 30%),
                linear-gradient(180deg, #fcfbf7 0%, #f4efe5 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(216, 243, 220, 0.9), rgba(248, 243, 232, 0.96));
            border: 1px solid var(--micelio-line);
            border-radius: 24px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 16px 40px rgba(19, 42, 19, 0.08);
            margin-bottom: 1rem;
        }

        .hero-eyebrow {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--micelio-green);
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1.05;
            color: var(--micelio-ink);
            font-weight: 800;
            margin: 0.35rem 0 0.5rem;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: rgba(19, 42, 19, 0.78);
            max-width: 920px;
            margin-bottom: 0.9rem;
        }

        .status-strip {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.7rem;
        }

        .status-pill {
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid var(--micelio-line);
            background: rgba(255, 255, 255, 0.72);
            color: var(--micelio-ink);
        }

        .section-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--micelio-line);
            border-radius: 20px;
            padding: 1rem 1rem 0.6rem;
            box-shadow: 0 12px 28px rgba(19, 42, 19, 0.05);
        }

        .category-chip-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.45rem 0 1rem;
        }

        .category-chip {
            border-radius: 14px;
            padding: 0.55rem 0.75rem;
            border: 1px solid var(--micelio-line);
            min-width: 180px;
            background: #ffffffcc;
        }

        .category-chip strong {
            display: block;
            font-size: 0.82rem;
            color: var(--micelio-ink);
            margin-bottom: 0.1rem;
        }

        .category-chip span {
            color: rgba(19, 42, 19, 0.68);
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Prepare frontend-only session state buckets."""

    st.session_state.setdefault("last_triage_category_counts", [])
    st.session_state.setdefault("last_triage_processed", 0)
    st.session_state.setdefault("last_triage_results", [])
    st.session_state.setdefault("triage_history", [])


def category_style_label(category: str) -> str:
    """Return a short semantic label for category emphasis."""

    if category.startswith("ALTA_"):
        return "Alta prioridade"
    if category.startswith("MEDIA_"):
        return "Média prioridade"
    if category in PRIORITY_THEMES:
        return "Prioridade"
    if category == REVIEW_CATEGORY:
        return "Revisão"
    return "Fluxo padrão"


def normalize_rows_for_chart(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert category count rows into a display-friendly dataframe."""

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = pd.DataFrame({"categoria": [], "quantidade": []})
    dataframe["grupo"] = dataframe["categoria"].apply(category_style_label)
    return dataframe


def record_triage_history(processed: int, category_counts: list[dict[str, Any]]) -> None:
    """Store a lightweight execution history for the current frontend session."""

    summary_parts = [
        f"{row['categoria']}: {row['quantidade']}"
        for row in category_counts
        if int(row["quantidade"]) > 0
    ]
    st.session_state["triage_history"].insert(
        0,
        {
            "executado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processados": processed,
            "resumo": ", ".join(summary_parts) if summary_parts else "Sem novos e-mails",
        },
    )
    st.session_state["triage_history"] = st.session_state["triage_history"][:10]


def render_sidebar() -> dict[str, Any] | None:
    """Render the Streamlit sidebar controls and return the latest backend status."""

    st.sidebar.title("Controle do Agente")
    st.sidebar.caption(f"Backend API: `{get_api_base_url()}`")

    try:
        status = api_request("GET", "/status")
    except RuntimeError as exc:
        status = None
        st.sidebar.error(str(exc))

    if st.sidebar.button("Rodar Triagem", use_container_width=True, type="primary"):
        try:
            with st.spinner("Executando a triagem via FastAPI..."):
                response = api_request("POST", "/triage", timeout_seconds=get_triage_timeout_seconds())
                category_counts = response.get("category_counts", [])
                processed = int(response.get("processed", 0))
                st.session_state["last_triage_results"] = response.get("results", [])
                st.session_state["last_triage_category_counts"] = category_counts
                st.session_state["last_triage_processed"] = processed
                record_triage_history(processed, category_counts)

            if processed:
                st.sidebar.success(f"Triagem concluída: {processed} novo(s) e-mail(s) processado(s).")
            else:
                st.sidebar.info("Nenhum novo e-mail para processar no provider ativo no momento.")
        except RuntimeError as exc:
            st.sidebar.error(f"Falha na triagem: {exc}")
            st.error(f"Falha ao chamar a API de triagem: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Status das conexões")

    if status is not None:
        st.sidebar.caption(f"Gmail: {status['gmail']}")
        st.sidebar.caption(f"PostgreSQL: {status['postgres']}")
        st.sidebar.caption(f"Chroma: {status['chroma']}")
        st.sidebar.caption(f"LLM Provider: {status['llm_provider']}")
        st.sidebar.caption(f"LLM Status: {status['llm_status']}")
        st.sidebar.caption(f"LLM Model: {status['llm_model']}")

    history = st.session_state.get("triage_history", [])
    if history:
        st.sidebar.divider()
        st.sidebar.subheader("Últimas Execuções")
        for item in history[:5]:
            st.sidebar.caption(f"{item['executado_em']} • {item['processados']} e-mails")

    return status


def render_hero(status: dict[str, Any] | None, dashboard_payload: dict[str, Any]) -> None:
    """Render the main header block with operational context."""

    metrics = dashboard_payload["metrics"]
    provider = status["llm_provider"] if status else "N/A"
    provider_status = status["llm_status"] if status else "OFFLINE"
    gmail_status = status["gmail"] if status else "OFFLINE"
    review_count = metrics["fila_revisao"]
    processed = metrics["emails_processados_hoje"]

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">Painel operacional</div>
            <div class="hero-title">Triagem de E-mails com backend FastAPI</div>
            <div class="hero-subtitle">
                O frontend agora está desacoplado do motor do agente e conversa com a API local
                para executar triagem, consolidar métricas, revisar dúvidas e acompanhar o
                comportamento por categoria.
            </div>
            <div class="status-strip">
                <div class="status-pill">Processados: {processed}</div>
                <div class="status-pill">Em revisão: {review_count}</div>
                <div class="status-pill">Gmail: {gmail_status}</div>
                <div class="status-pill">LLM: {provider} / {provider_status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(dashboard_payload: dict[str, Any]) -> None:
    """Render the dashboard metric cards."""

    metrics = dashboard_payload["metrics"]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Processados Hoje", metrics["emails_processados_hoje"])
    col2.metric("Prioritários", metrics["prioritarios_detectados"])
    col3.metric("Em Revisão", metrics["fila_revisao"])
    col4.metric("Taxa de Automação", f"{metrics['taxa_automacao']}%")


def render_category_summary_chips(rows: list[dict[str, Any]], title: str) -> None:
    """Render compact highlight chips for categories with non-zero values."""

    visible_rows = [row for row in rows if int(row["quantidade"]) > 0]
    st.markdown(f"**{title}**")
    if not visible_rows:
        st.caption("Nenhuma categoria com volume registrado neste recorte.")
        return

    chips_markup = "".join(
        (
            f"<div class='category-chip'>"
            f"<strong>{row['categoria']}</strong>"
            f"<span>{row['quantidade']} e-mail(s) • {category_style_label(str(row['categoria']))}</span>"
            f"</div>"
        )
        for row in visible_rows
    )
    st.markdown(f"<div class='category-chip-row'>{chips_markup}</div>", unsafe_allow_html=True)


def render_category_breakdowns(dashboard_payload: dict[str, Any]) -> None:
    """Render accumulated and last-run category counters with charts."""

    accumulated_rows = dashboard_payload["category_counts"]
    last_run_rows = st.session_state.get(
        "last_triage_category_counts",
        [{"categoria": row["categoria"], "quantidade": 0} for row in accumulated_rows],
    )

    accumulated_total = sum(int(row["quantidade"]) for row in accumulated_rows)
    last_run_total = sum(int(row["quantidade"]) for row in last_run_rows)

    st.subheader("Categorização por Categoria")
    st.caption(
        "A visão acumulada vem do PostgreSQL. A visão da última execução reflete apenas a rodada "
        "mais recente disparada neste frontend via FastAPI."
    )

    render_category_summary_chips(last_run_rows, "Destaques da Última Execução")

    accumulated_df = normalize_rows_for_chart(accumulated_rows)
    last_run_df = normalize_rows_for_chart(last_run_rows)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**Acumulado**")
        st.metric("Total Categorizado", accumulated_total)
        st.bar_chart(accumulated_df, x="categoria", y="quantidade", color="#2d6a4f")
        st.dataframe(accumulated_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("**Última Execução**")
        st.metric("Total Categorizado", last_run_total)
        if last_run_total == 0:
            st.info("Nenhuma execução registrada nesta sessão ainda.")
        st.bar_chart(last_run_df, x="categoria", y="quantidade", color="#c17c00")
        st.dataframe(last_run_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_execution_history() -> None:
    """Render a local session history of triage runs."""

    history = st.session_state.get("triage_history", [])
    st.subheader("Histórico Local da Sessão")
    st.caption(
        "Este histórico é apenas da sessão atual do frontend. Ele ajuda a comparar rapidamente "
        "as execuções feitas desde que a página foi aberta."
    )

    if not history:
        st.info("Nenhuma execução registrada nesta sessão ainda.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)


def render_reclassification_controls(
    item: dict[str, Any],
    categories: list[str],
    priorities: list[str],
    endpoint_prefix: str,
) -> None:
    """Render manual recategorization controls for a processed email."""

    message_id = str(item["message_id"])
    current_theme = str(item.get("theme_category", "EM_DUVIDA"))
    current_priority = str(item.get("priority_level", "MEDIA"))

    select_col1, select_col2, select_col3 = st.columns([2, 1, 1])
    theme_index = categories.index(current_theme) if current_theme in categories else 0
    priority_index = priorities.index(current_priority) if current_priority in priorities else 1

    selected_category = select_col1.selectbox(
        "Tema manual",
        options=categories,
        index=theme_index,
        key=f"{endpoint_prefix}_theme_{message_id}",
    )
    selected_priority = select_col2.selectbox(
        "Prioridade manual",
        options=priorities,
        index=priority_index,
        key=f"{endpoint_prefix}_priority_{message_id}",
        disabled=selected_category == "EM_DUVIDA",
    )

    if select_col3.button("Salvar", key=f"{endpoint_prefix}_save_{message_id}", use_container_width=True, type="primary"):
        try:
            result = api_request(
                "POST",
                f"/{endpoint_prefix}/{message_id}/reclassify",
                {"category": selected_category, "priority_level": selected_priority},
            )
        except RuntimeError as exc:
            st.error(f"Falha ao reclassificar e-mail: {exc}")
        else:
            st.success(
                f"E-mail {result['message_id']} atualizado para "
                f"{result['forced_final_label']} e retroalimentado na memória semântica."
            )
            st.rerun()


def render_review_queue() -> None:
    """Render emails currently waiting for manual review."""

    st.subheader("Fila de Revisão Manual")
    st.caption(
        "Os e-mails classificados como `EM_DUVIDA` aparecem aqui para decisão humana. "
        "Cada ação atualiza o PostgreSQL e injeta aprendizado no ChromaDB."
    )

    try:
        review_queue = api_request("GET", "/review-queue")
        categories_payload = api_request("GET", "/manual-review-categories")
    except RuntimeError as exc:
        st.error(f"Não foi possível carregar a fila de revisão: {exc}")
        return

    categories = categories_payload["categories"]
    priorities = categories_payload["priorities"]
    if not review_queue:
        st.success("Nenhum e-mail aguardando revisão manual.")
        return

    st.caption(f"{len(review_queue)} e-mail(s) aguardando intervenção humana.")

    for item in review_queue:
        header = f"{item['subject']} • {item['sender']}"
        with st.expander(header, expanded=False):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.write(f"**Message ID:** {item['message_id']}")
            meta_col2.write(f"**Status atual:** {item['status']}")
            meta_col3.write(f"**Categoria atual:** {item['category']}")
            st.caption(
                f"Tema atual: {item.get('theme_category', 'EM_DUVIDA')} • "
                f"Prioridade atual: {item.get('priority_level', 'BAIXA')} • "
                f"Confiança: {item.get('confidence', 0)}"
            )

            st.write("**Corpo do e-mail:**")
            st.write(item.get("body", "Sem corpo disponível."))

            render_reclassification_controls(item, categories, priorities, endpoint_prefix="review-queue")


def render_last_execution_results() -> None:
    """Show the emails and classifications produced in the latest frontend-triggered run."""

    results = st.session_state.get("last_triage_results", [])
    st.subheader("E-mails vs Classificações da Última Execução")
    st.caption(
        "Esta seção mostra exatamente os e-mails processados na rodada mais recente disparada por este frontend, "
        "junto com a classificação final produzida pelo agente."
    )

    if not results:
        st.info("Ainda não há resultados de triagem nesta sessão.")
        return

    for item in results:
        email_data = item.get("email_data", {})
        classification = item.get("classification_result", {})
        header = (
            f"{email_data.get('subject', 'Sem assunto')} • "
            f"{classification.get('final_label', 'EM_DUVIDA')} • "
            f"{email_data.get('sender', 'Sem remetente')}"
        )
        with st.expander(header, expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**Tema:** {classification.get('theme_category', 'EM_DUVIDA')}")
            col2.write(f"**Prioridade:** {classification.get('priority_level', 'BAIXA')}")
            col3.write(f"**Label final:** {classification.get('final_label', 'EM_DUVIDA')}")
            col4.write(f"**Confiança:** {classification.get('confidence', 0)}")
            st.write("**Resumo da decisão**")
            st.write(
                f"Tema: {classification.get('motivo_tema', 'N/D')}  \n"
                f"Prioridade: {classification.get('motivo_prioridade', 'N/D')}"
            )
            st.write("**Corpo do e-mail:**")
            st.write(email_data.get("body", "Sem corpo disponível."))


def render_processed_email_audit() -> None:
    """Render a searchable audit view with manual reclassification for processed emails."""

    st.subheader("Auditoria e Reclassificação Manual")
    st.caption(
        "Aqui você pode revisar os e-mails já processados, comparar o conteúdo com a classificação aplicada "
        "e corrigir tema/prioridade manualmente. Cada correção atualiza o PostgreSQL e adiciona aprendizado ao Chroma."
    )

    try:
        processed_emails = api_request("GET", "/processed-emails?limit=40")
        categories_payload = api_request("GET", "/manual-review-categories")
    except RuntimeError as exc:
        st.error(f"Não foi possível carregar os e-mails processados: {exc}")
        return

    categories = categories_payload["categories"]
    priorities = categories_payload["priorities"]
    if not processed_emails:
        st.info("Nenhum e-mail processado disponível para auditoria ainda.")
        return

    for item in processed_emails:
        header = (
            f"{item['subject']} • {item.get('final_label', item.get('category', 'EM_DUVIDA'))} • "
            f"{item['sender']}"
        )
        with st.expander(header, expanded=False):
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            meta_col1.write(f"**Status:** {item['status']}")
            meta_col2.write(f"**Tema:** {item.get('theme_category', 'EM_DUVIDA')}")
            meta_col3.write(f"**Prioridade:** {item.get('priority_level', 'BAIXA')}")
            meta_col4.write(f"**Atualizado em:** {item.get('updated_at', '-')}")
            st.caption(
                f"Label final: {item.get('final_label', 'EM_DUVIDA')} • "
                f"Ação: {item.get('needs_action', False)} • "
                f"Importância: {item.get('is_important', False)} • "
                f"Flag Gmail: {item.get('gmail_flagged', False)}"
            )
            st.write("**Corpo do e-mail:**")
            st.write(item.get("body", "Sem corpo disponível."))
            render_reclassification_controls(item, categories, priorities, endpoint_prefix="processed-emails")


def main() -> None:
    """Main entry point for the Streamlit app."""

    initialize_session_state()
    inject_custom_css()
    status = render_sidebar()

    try:
        dashboard_payload = api_request("GET", "/dashboard")
    except RuntimeError as exc:
        st.error(f"Não foi possível carregar o dashboard: {exc}")
        st.info("Suba o backend com `uvicorn api_server:app --reload` e recarregue a página.")
        return

    render_hero(status, dashboard_payload)
    render_metrics(dashboard_payload)

    overview_tab, execution_tab, review_tab = st.tabs(["Visão Geral", "Execuções & Correções", "Revisão Manual"])

    with overview_tab:
        st.divider()
        render_category_breakdowns(dashboard_payload)
        st.divider()
        render_execution_history()

    with execution_tab:
        st.divider()
        render_last_execution_results()
        st.divider()
        render_processed_email_audit()

    with review_tab:
        st.divider()
        render_review_queue()


if __name__ == "__main__":
    main()
