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
DEFAULT_ANTESE_STYLE_PROFILES = [
    {
        "profile_id": "personal_default",
        "name": "Personal Default",
        "owner_scope": "personal",
        "voice_traits": {"formalidade": "equilibrada", "assertividade": "alta", "concretude": "alta"},
        "structure_traits": {"abertura": "direta com contexto", "transicoes": "claras"},
        "rhetorical_traits": {"exemplos": "frequentes", "contraste": "forte"},
        "lexical_traits": {"simplicidade_vocabular": "alta", "preferencia_verbal": "verbos fortes"},
        "dos": ["Explique conceitos com clareza.", "Use exemplos concretos."],
        "donts": ["Nao soar genérico.", "Nao copiar referências literalmente."],
    },
    {
        "profile_id": "workspace_specific",
        "name": "Workspace Specific",
        "owner_scope": "workspace",
        "voice_traits": {"formalidade": "profissional", "assertividade": "alta", "concretude": "alta"},
        "structure_traits": {"abertura": "situacao + contexto", "transicoes": "fortes"},
        "rhetorical_traits": {"exemplos": "frequentes", "contraste": "moderado"},
        "lexical_traits": {"simplicidade_vocabular": "media", "termos_tecnicos": "altos quando pertinentes"},
        "dos": ["Conecte a escrita ao contexto do projeto."],
        "donts": ["Nao deixe o texto genérico."],
    },
    {
        "profile_id": "temporary_session_profile",
        "name": "Temporary Session Profile",
        "owner_scope": "session",
        "voice_traits": {"formalidade": "adaptativa", "assertividade": "media", "concretude": "media"},
        "structure_traits": {"abertura": "adaptativa", "transicoes": "claras"},
        "rhetorical_traits": {"exemplos": "moderados", "contraste": "moderado"},
        "lexical_traits": {"simplicidade_vocabular": "media", "termos_tecnicos": "adaptativos"},
        "dos": ["Adaptar a voz ao contexto corrente."],
        "donts": ["Nao cristalizar uma voz indevida."],
    },
]
DEFAULT_ANTESE_GENRE_CARDS = [
    {
        "genre_id": "brainstorm_notes",
        "name": "Brainstorm Notes",
        "primary_goal": "EXPLORAR",
        "expected_structure": ["framing", "angles", "examples", "next steps"],
        "tone_defaults": ["exploratorio", "claro"],
        "quality_checklist": ["Variedade de angulos", "Foco no objetivo"],
    },
    {
        "genre_id": "outline",
        "name": "Outline",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["title", "hook", "sections", "closing"],
        "tone_defaults": ["estruturado", "direto"],
        "quality_checklist": ["Sequencia logica", "Cobertura do objetivo"],
    },
    {
        "genre_id": "linkedin_post",
        "name": "LinkedIn Post",
        "primary_goal": "PERSUADIR",
        "expected_structure": ["hook", "story/insight", "takeaways", "closing"],
        "tone_defaults": ["profissional", "humano"],
        "quality_checklist": ["Abertura forte", "Valor pratico"],
    },
    {
        "genre_id": "essay_article",
        "name": "Essay Article",
        "primary_goal": "REFLETIR",
        "expected_structure": ["hook", "context", "argument", "examples", "conclusion"],
        "tone_defaults": ["ensaistico", "reflexivo"],
        "quality_checklist": ["Tese clara", "Progressao argumentativa"],
    },
    {
        "genre_id": "newsletter",
        "name": "Newsletter",
        "primary_goal": "INFORMAR",
        "expected_structure": ["opening", "highlights", "commentary", "closing"],
        "tone_defaults": ["curatorial", "caloroso"],
        "quality_checklist": ["Clareza editorial", "Bom ritmo"],
    },
    {
        "genre_id": "rewrite_clarity",
        "name": "Rewrite Clarity",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["preserve core", "improve flow", "tighten wording"],
        "tone_defaults": ["claro", "preciso"],
        "quality_checklist": ["Preservar sentido", "Melhorar clareza"],
    },
    {
        "genre_id": "consolidated_memo",
        "name": "Consolidated Memo",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["context", "points", "decisions", "next steps"],
        "tone_defaults": ["objetivo", "profissional"],
        "quality_checklist": ["Consolidacao fiel", "Proximos passos claros"],
    },
    {
        "genre_id": "poem",
        "name": "Poema",
        "primary_goal": "REFLETIR",
        "expected_structure": ["imagem inicial", "movimento", "virada", "fecho ressonante"],
        "tone_defaults": ["lirico", "sensorial", "condensado"],
        "quality_checklist": ["Imagens fortes", "Economia verbal", "Ritmo perceptivel"],
    },
    {
        "genre_id": "chronicle",
        "name": "Crônica",
        "primary_goal": "REFLETIR",
        "expected_structure": ["cena cotidiana", "observacao", "deslocamento", "fecho reflexivo"],
        "tone_defaults": ["intimo", "observacional", "fluido"],
        "quality_checklist": ["Cena concreta", "Voz autoral", "Virada reflexiva"],
    },
    {
        "genre_id": "short_story",
        "name": "Conto",
        "primary_goal": "EXPLORAR",
        "expected_structure": ["situacao inicial", "tensao", "virada", "desfecho"],
        "tone_defaults": ["narrativo", "concentrado", "imagetico"],
        "quality_checklist": ["Conflito claro", "Atmosfera consistente", "Final com impacto"],
    },
    {
        "genre_id": "novel_excerpt",
        "name": "Romance (trecho)",
        "primary_goal": "EXPLORAR",
        "expected_structure": ["imersao", "desenvolvimento de cena", "subtexto", "gancho"],
        "tone_defaults": ["narrativo", "expandido", "atmosferico"],
        "quality_checklist": ["Cena sustentada", "Subtexto", "Gancho para continuidade"],
    },
]
DEFAULT_ANTESE_INSPIRATION_PROFILES = [
    {
        "inspiration_profile_id": "essayistic",
        "name": "Mais ensaistico",
        "traits_to_borrow": ["cadencia mais contemplativa", "transicoes mais suaves"],
        "forbidden_behaviors": ["imitar trechos famosos"],
        "transformation_strength": "media",
    },
    {
        "inspiration_profile_id": "aphoristic",
        "name": "Mais aforistico",
        "traits_to_borrow": ["frases mais condensadas", "fechamentos mais cortantes"],
        "forbidden_behaviors": ["fragmentar demais o texto"],
        "transformation_strength": "baixa",
    },
    {
        "inspiration_profile_id": "journalistic",
        "name": "Mais jornalistico",
        "traits_to_borrow": ["abertura mais informativa", "clareza e economia verbal"],
        "forbidden_behaviors": ["neutralidade artificial"],
        "transformation_strength": "media",
    },
    {
        "inspiration_profile_id": "intimate_reflective",
        "name": "Mais intimo/reflexivo",
        "traits_to_borrow": ["proximidade com o leitor", "fechamento com introspeccao"],
        "forbidden_behaviors": ["sentimentalismo excessivo"],
        "transformation_strength": "media",
    },
]


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
    st.session_state.setdefault("last_micelio_response", None)
    st.session_state.setdefault("last_antese_response", None)
    st.session_state.setdefault("last_antese_payload", None)
    st.session_state.setdefault("last_antese_segmentation_preview", None)
    st.session_state.setdefault("selected_antese_sample_id", None)
    st.session_state.setdefault("triage_history", [])


def parse_multiline_items(raw_text: str) -> list[str]:
    """Split multiline or comma-separated text into a compact list."""

    items: list[str] = []
    for raw_line in raw_text.replace(",", "\n").splitlines():
        normalized = raw_line.strip().lstrip("-").strip()
        if normalized:
            items.append(normalized)
    return items


def parse_json_text(raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON textbox conservatively for Streamlit forms."""

    raw_value = raw_text.strip()
    if not raw_value:
        return fallback
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON inválido: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("O campo JSON deve conter um objeto.")
    return parsed


def load_antese_catalog_with_fallback() -> tuple[dict[str, Any], bool]:
    """Load Antese catalog endpoints, falling back to local defaults on 404."""

    fallback_catalog = {
        "style_profiles": DEFAULT_ANTESE_STYLE_PROFILES,
        "genre_cards": DEFAULT_ANTESE_GENRE_CARDS,
        "inspiration_profiles": DEFAULT_ANTESE_INSPIRATION_PROFILES,
        "executions": [],
    }
    try:
        style_profiles_payload = api_request("GET", "/antese/style-profiles")
        genre_cards_payload = api_request("GET", "/antese/genre-cards")
        inspiration_payload = api_request("GET", "/antese/inspiration-profiles")
        try:
            history_payload = api_request("GET", "/antese/executions?limit=12")
        except RuntimeError:
            history_payload = {"executions": []}
        return (
            {
                "style_profiles": style_profiles_payload.get("style_profiles", []),
                "genre_cards": genre_cards_payload.get("genre_cards", []),
                "inspiration_profiles": inspiration_payload.get("inspiration_profiles", []),
                "executions": history_payload.get("executions", []),
            },
            False,
        )
    except RuntimeError as exc:
        if "Not Found" in str(exc):
            return fallback_catalog, True
        raise


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


def render_micelio_workspace() -> None:
    """Render an open-ended workspace for the Micelio supervisor."""

    st.subheader("Micélio Supervisor")
    st.caption(
        "Faça um pedido aberto ao supervisor. O Micélio decide se chama o Enzima, o Antese "
        "ou ambos, e registra a execução no histórico persistido."
    )

    try:
        agents_payload = api_request("GET", "/agents")
        execution_history_payload = api_request("GET", "/micelio/executions?limit=12")
    except RuntimeError as exc:
        st.error(f"Não foi possível carregar os agentes disponíveis: {exc}")
        return

    available_agents = agents_payload.get("agents", [])
    persisted_executions = execution_history_payload.get("executions", [])
    with st.form("micelio_form"):
        user_request = st.text_area(
            "Pedido ao Micélio",
            placeholder="Ex.: Leia meus e-mails mais importantes e me ajude a transformar isso em um rascunho de update semanal.",
            height=140,
        )
        requested_agent = st.selectbox(
            "Forçar agente (opcional)",
            options=["AUTO", "ENZIMA", "ANTESE"],
            index=0,
        )
        col1, col2 = st.columns(2)
        writing_task_type = col1.selectbox("Modo preferencial do Antese", options=["BRAINSTORM", "OUTLINE", "DRAFT", "CONSOLIDATE", "REWRITE"], index=0)
        email_limit = col2.number_input("Limite de e-mails para Enzima", min_value=1, max_value=50, value=3)
        tone = st.text_input("Tom", value="claro e profissional")
        audience = st.text_input("Público-alvo", value="leitores gerais")
        context_notes = st.text_area(
            "Notas de apoio",
            placeholder="Cole rascunhos, tópicos ou contexto adicional para o Antese.",
            height=120,
        )
        submitted = st.form_submit_button("Executar Micélio", type="primary", use_container_width=True)

    if submitted:
        if not user_request.strip():
            st.warning("Descreva um pedido para o Micélio antes de executar.")
        else:
            payload = {
                "user_request": user_request,
                "requested_agent": None if requested_agent == "AUTO" else requested_agent,
                "writing_task_type": writing_task_type,
                "context_notes": context_notes,
                "tone": tone,
                "audience": audience,
                "email_limit": int(email_limit),
            }
            try:
                with st.spinner("Micélio está planejando e acionando os agentes..."):
                    st.session_state["last_micelio_response"] = api_request(
                        "POST",
                        "/micelio/orchestrate",
                        payload,
                        timeout_seconds=get_triage_timeout_seconds(),
                    )
            except RuntimeError as exc:
                st.error(f"Falha ao executar o Micélio: {exc}")

    st.markdown("**Agentes registrados**")
    for agent in available_agents:
        st.caption(
            f"{agent['agent_name']}: {agent['description']} "
            f"({', '.join(agent.get('task_types', []))})"
        )

    response = st.session_state.get("last_micelio_response")
    if not response:
        st.info("Nenhuma execução do Micélio nesta sessão ainda.")
        return

    st.divider()
    st.markdown("**Plano do Micélio**")
    st.write(f"Agentes escolhidos: {', '.join(response.get('selected_agents', []))}")
    st.write(f"Motivo: {response.get('reason', 'N/D')}")
    for step in response.get("execution_plan", []):
        st.write(f"- {step}")

    st.divider()
    st.markdown("**Saídas dos Agentes**")
    for item in response.get("agent_outputs", []):
        agent_name = item.get("agent_name", "AGENTE")
        result = item.get("result", {})
        with st.expander(f"{agent_name}", expanded=True):
            if agent_name == "ANTESE":
                st.caption(
                    f"Modo: {result.get('task_type', 'N/D')} • "
                    f"Tom: {result.get('tone', 'N/D')} • "
                    f"Público: {result.get('audience', 'N/D')}"
                )
                st.write(result.get("output_text", "Sem saída."))
            else:
                st.write(
                    f"Processados: {result.get('processed', 0)} • "
                    f"Lote: {result.get('batch_limit', 'N/D')}"
                )
                category_counts = result.get("category_counts", [])
                if category_counts:
                    visible = [row for row in category_counts if int(row.get("quantidade", 0)) > 0]
                    if visible:
                        st.dataframe(pd.DataFrame(visible), use_container_width=True, hide_index=True)
                for triage_item in result.get("results", [])[:5]:
                    email_data = triage_item.get("email_data", {})
                    classification = triage_item.get("classification_result", {})
                    st.write(
                        f"- {email_data.get('subject', 'Sem assunto')} -> "
                        f"{classification.get('final_label', 'EM_DUVIDA')}"
                    )

    st.divider()
    st.markdown("**Histórico Persistido do Micélio**")
    st.caption(
        "Estas execuções ficam salvas no PostgreSQL, independentemente da sessão atual do frontend."
    )
    if not persisted_executions:
        st.info("Nenhuma execução persistida do Micélio ainda.")
        return

    history_rows = [
        {
            "executado_em": item.get("created_at", "-"),
            "execution_id": item.get("execution_id", "-"),
            "agentes": ", ".join(item.get("selected_agents", [])),
            "pedido": item.get("user_request", ""),
            "preview": item.get("output_preview", ""),
        }
        for item in persisted_executions
    ]
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


def _run_antese_from_payload(payload: dict[str, Any], spinner_text: str) -> None:
    """Call the Antese backend using the assembled payload."""

    try:
        with st.spinner(spinner_text):
            st.session_state["last_antese_response"] = api_request(
                "POST",
                "/antese/run",
                payload,
                timeout_seconds=get_triage_timeout_seconds(),
            )
            st.session_state["last_antese_payload"] = payload
    except RuntimeError as exc:
        st.error(f"Falha ao executar o Antese: {exc}")


def _run_antese_segmentation_preview(payload: dict[str, Any]) -> None:
    """Call the backend preview endpoint for Antese sample segmentation."""

    try:
        with st.spinner("Antese está montando o prompt de segmentação e chamando o modelo..."):
            st.session_state["last_antese_segmentation_preview"] = api_request(
                "POST",
                "/antese/samples/preview-segmentation",
                payload,
                timeout_seconds=get_triage_timeout_seconds(),
            )
    except RuntimeError as exc:
        st.error(f"Falha ao testar a segmentação: {exc}")


def render_antese_workspace() -> None:
    """Render the personalized writing workspace for Antese."""

    st.subheader("Antese")
    st.caption(
        "Workspace editorial para transformar notas em texto com briefing, perfis de estilo, cards de gênero, "
        "versionamento e feedback explícito."
    )

    try:
        antese_catalog_payload, using_fallback_catalog = load_antese_catalog_with_fallback()
    except RuntimeError as exc:
        st.error(f"Não foi possível carregar o catálogo do Antese: {exc}")
        return

    style_profiles = antese_catalog_payload.get("style_profiles", [])
    genre_cards = antese_catalog_payload.get("genre_cards", [])
    inspiration_profiles = antese_catalog_payload.get("inspiration_profiles", [])
    execution_history = antese_catalog_payload.get("executions", [])

    if not style_profiles or not genre_cards:
        st.error("O catálogo do Antese está incompleto. Verifique o bootstrap do backend e do PostgreSQL.")
        return

    if using_fallback_catalog:
        st.warning(
            "A API ativa não expôs os endpoints novos do catálogo do Antese. "
            "O frontend carregou um catálogo local de fallback. Reinicie o backend para habilitar histórico, "
            "profiles e feedback persistidos."
        )

    style_profile_options = {item["profile_id"]: item for item in style_profiles}
    genre_card_options = {item["genre_id"]: item for item in genre_cards}
    inspiration_options = {"": None, **{item["inspiration_profile_id"]: item for item in inspiration_profiles}}

    default_style_profile = next(iter(style_profile_options.keys()), "personal_default")
    default_genre_card = next(iter(genre_card_options.keys()), "brainstorm_notes")

    brief_tab, style_tab, segmentation_tab, draft_tab, samples_tab, history_tab = st.tabs(
        ["Brief", "Style", "Segmentation Test", "Draft", "Samples", "History"]
    )

    with brief_tab:
        user_request = st.text_area(
            "Pedido ao Antese",
            key="antese_user_request",
            placeholder="Ex.: Transforme estas notas em um artigo curto sobre agentes pessoais.",
            height=140,
        )
        col1, col2, col3 = st.columns(3)
        task_type = col1.selectbox(
            "Modo",
            options=["BRAINSTORM", "OUTLINE", "DRAFT", "CONSOLIDATE", "REWRITE"],
            key="antese_task_type",
        )
        goal = col2.selectbox(
            "Objetivo",
            options=["EXPLORAR", "SINTETIZAR", "INFORMAR", "PERSUADIR", "REFLETIR"],
            key="antese_goal",
            index=0,
        )
        audience = col3.text_input("Público", value="leitores gerais", key="antese_audience")
        tone_override = st.text_input("Tom", value="claro e profissional", key="antese_tone_override")
        source_notes = st.text_area(
            "Notas-fonte",
            key="antese_source_notes",
            placeholder="Cole bullets, ideias soltas, resumos, argumentos ou matéria-prima.",
            height=180,
        )
        reference_text = st.text_area(
            "Texto de referência",
            key="antese_reference_text",
            placeholder="Opcional: texto-base para consolidar ou reescrever.",
            height=140,
        )
        col4, col5 = st.columns(2)
        must_include_text = col4.text_area(
            "Must include",
            key="antese_must_include",
            placeholder="- exemplo concreto\n- ponto central\n- call to action",
            height=120,
        )
        must_avoid_text = col5.text_area(
            "Must avoid",
            key="antese_must_avoid",
            placeholder="- soar genérico\n- copiar exemplos\n- usar clichês",
            height=120,
        )

    with style_tab:
        selected_style_profile_id = st.selectbox(
            "Style profile",
            options=list(style_profile_options.keys()),
            format_func=lambda item: style_profile_options[item]["name"],
            index=list(style_profile_options.keys()).index(default_style_profile)
            if default_style_profile in style_profile_options
            else 0,
            key="antese_style_profile_id",
        )
        selected_genre_card_id = st.selectbox(
            "Genre card",
            options=list(genre_card_options.keys()),
            format_func=lambda item: genre_card_options[item]["name"],
            index=list(genre_card_options.keys()).index(default_genre_card)
            if default_genre_card in genre_card_options
            else 0,
            key="antese_genre_card_id",
        )
        selected_inspiration_id = st.selectbox(
            "Inspiration profile",
            options=list(inspiration_options.keys()),
            format_func=lambda item: "Sem inspiração extra" if not item else inspiration_options[item]["name"],
            key="antese_inspiration_profile_id",
        )

        selected_style = style_profile_options.get(selected_style_profile_id, {})
        selected_genre = genre_card_options.get(selected_genre_card_id, {})
        selected_inspiration = inspiration_options.get(selected_inspiration_id)

        style_col1, style_col2 = st.columns(2)
        with style_col1:
            st.markdown("**Traços ativos do style profile**")
            st.json(
                {
                    "voice_traits": selected_style.get("voice_traits", {}),
                    "structure_traits": selected_style.get("structure_traits", {}),
                    "rhetorical_traits": selected_style.get("rhetorical_traits", {}),
                    "lexical_traits": selected_style.get("lexical_traits", {}),
                    "dos": selected_style.get("dos", []),
                    "donts": selected_style.get("donts", []),
                }
            )
        with style_col2:
            st.markdown("**Configuração do gênero**")
            st.json(
                {
                    "primary_goal": selected_genre.get("primary_goal", ""),
                    "expected_structure": selected_genre.get("expected_structure", []),
                    "tone_defaults": selected_genre.get("tone_defaults", []),
                    "quality_checklist": selected_genre.get("quality_checklist", []),
                    "inspiration": selected_inspiration or {},
                }
            )

        with st.expander("Criar/editar style profile", expanded=False):
            if using_fallback_catalog:
                st.caption("Desabilitado enquanto o backend estiver sem os endpoints novos do Antese.")
            else:
                with st.form("antese_create_style_profile_form"):
                    profile_name = st.text_input("Nome do profile", value="Session Custom Profile")
                    owner_scope = st.selectbox("Escopo", options=["session", "personal", "workspace"], index=0)
                    voice_traits_json = st.text_area(
                        "Voice traits (JSON)",
                        value='{"formalidade": "equilibrada", "assertividade": "alta", "concretude": "alta"}',
                        height=120,
                    )
                    structure_traits_json = st.text_area(
                        "Structure traits (JSON)",
                        value='{"abertura": "direta", "transicoes": "claras"}',
                        height=120,
                    )
                    rhetorical_traits_json = st.text_area(
                        "Rhetorical traits (JSON)",
                        value='{"exemplos": "frequentes", "contraste": "moderado"}',
                        height=120,
                    )
                    lexical_traits_json = st.text_area(
                        "Lexical traits (JSON)",
                        value='{"simplicidade_vocabular": "media-alta", "preferencia_verbal": "verbos fortes"}',
                        height=120,
                    )
                    dos_text = st.text_area("Dos", value="- trazer exemplos concretos\n- manter progressao clara", height=100)
                    donts_text = st.text_area("Donts", value="- soar genérico\n- exagerar em abstrações", height=100)
                    sample_texts = st.text_area(
                        "Textos seus para extração assistida",
                        placeholder="Cole 1 ou mais textos seus aqui para o Antese inferir um viés estilístico inicial.",
                        height=180,
                    )
                    create_profile = st.form_submit_button("Salvar profile", type="primary", use_container_width=True)

                if create_profile:
                    try:
                        created_profile = api_request(
                            "POST",
                            "/antese/style-profiles",
                            {
                                "name": profile_name,
                                "owner_scope": owner_scope,
                                "voice_traits": parse_json_text(voice_traits_json, {}),
                                "structure_traits": parse_json_text(structure_traits_json, {}),
                                "rhetorical_traits": parse_json_text(rhetorical_traits_json, {}),
                                "lexical_traits": parse_json_text(lexical_traits_json, {}),
                                "dos": parse_multiline_items(dos_text),
                                "donts": parse_multiline_items(donts_text),
                                "sample_texts": [sample_texts] if sample_texts.strip() else [],
                            },
                        )
                    except RuntimeError as exc:
                        st.error(f"Falha ao salvar style profile: {exc}")
                    else:
                        st.success(f"Style profile salvo: {created_profile['name']}")
                        st.rerun()

        with st.expander("Importar text samples para memória de escrita", expanded=False):
            if using_fallback_catalog:
                st.caption("Desabilitado enquanto o backend estiver sem os endpoints novos do Antese.")
            else:
                with st.form("antese_import_samples_form"):
                    import_title = st.text_input("Título do corpus", value="Mau's Domain")
                    import_source_scope = st.selectbox("Escopo do corpus", options=["personal", "workspace", "reference"], index=0)
                    import_style_profile_id = st.selectbox(
                        "Associar ao style profile",
                        options=list(style_profile_options.keys()),
                        format_func=lambda item: style_profile_options[item]["name"],
                        index=list(style_profile_options.keys()).index(st.session_state.get("antese_style_profile_id", default_style_profile))
                        if st.session_state.get("antese_style_profile_id", default_style_profile) in style_profile_options
                        else 0,
                    )
                    import_genre_card_id = st.selectbox(
                        "Genre card associado (opcional)",
                        options=[""] + list(genre_card_options.keys()),
                        format_func=lambda item: "Sem gênero fixo" if not item else genre_card_options[item]["name"],
                    )
                    import_source_url = st.text_input(
                        "URL da fonte",
                        value="https://portfolio-digital-dl2ipti.gamma.site/",
                    )
                    import_raw_text = st.text_area(
                        "Texto bruto (use este campo se o site bloquear scraping)",
                        placeholder="Cole aqui o conteúdo exportado/copied do seu Gamma para garantir a ingestão.",
                        height=220,
                    )
                    import_metadata_json = st.text_area(
                        "Metadata extra (JSON)",
                        value='{"source_label": "gamma_maus_domain", "owner": "mau"}',
                        height=100,
                    )
                    import_segment_with_llm = st.checkbox(
                        "Segmentar conteúdo com LLM antes de salvar",
                        value=True,
                        help="Para sites como Gamma, o backend tenta separar automaticamente textos autorais, bio, trabalho e outros blocos.",
                    )
                    import_submit = st.form_submit_button("Importar corpus", type="primary", use_container_width=True)

                if import_submit:
                    try:
                        import_result = api_request(
                            "POST",
                            "/antese/samples/import",
                            {
                                "title": import_title,
                                "source_scope": import_source_scope,
                                "style_profile_id": import_style_profile_id,
                                "genre_id": import_genre_card_id or None,
                                "source_url": import_source_url.strip() or None,
                                "raw_text": import_raw_text.strip() or None,
                                "metadata": parse_json_text(import_metadata_json, {}),
                                "segment_with_llm": import_segment_with_llm,
                            },
                            timeout_seconds=get_triage_timeout_seconds(),
                        )
                    except RuntimeError as exc:
                        st.error(f"Falha ao importar text samples: {exc}")
                    else:
                        st.success(
                            f"Corpus importado com sucesso. "
                            f"{import_result['chunk_count']} chunk(s) indexado(s) para o Antese."
                        )
                        st.json(import_result)

    with segmentation_tab:
        st.markdown("**Teste do prompt de segmentação**")
        st.caption(
            "Este fluxo não salva nada. Ele só extrai os blocos, monta o prompt atual do segmentador "
            "e devolve a resposta do modelo para você validar a separação."
        )

        with st.form("antese_segmentation_preview_form"):
            preview_title = st.text_input("Título da fonte", value="Mau's Portfolio")
            preview_url = st.text_input(
                "URL da fonte",
                value="https://portfolio-digital-dl2ipti.gamma.site/",
            )
            preview_genre_id = st.selectbox(
                "Genre sugerido (opcional)",
                options=[""] + list(genre_card_options.keys()),
                format_func=lambda item: "Sem gênero fixo" if not item else genre_card_options[item]["name"],
            )
            preview_raw_text = st.text_area(
                "Texto bruto opcional",
                placeholder="Use este campo para testar a segmentação colando texto diretamente, sem buscar a URL.",
                height=180,
            )
            preview_submit = st.form_submit_button(
                "Executar teste do prompt",
                type="primary",
                use_container_width=True,
            )

        if preview_submit:
            if not preview_url.strip() and not preview_raw_text.strip():
                st.warning("Informe uma URL ou cole texto bruto para testar a segmentação.")
            else:
                _run_antese_segmentation_preview(
                    {
                        "title": preview_title.strip() or "Fonte sem título",
                        "source_url": preview_url.strip() or None,
                        "raw_text": preview_raw_text.strip() or None,
                        "genre_id": preview_genre_id or None,
                    }
                )

        preview_response = st.session_state.get("last_antese_segmentation_preview")
        if not preview_response:
            st.info("Nenhum teste de segmentação executado nesta sessão ainda.")
        else:
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Parser", preview_response.get("source_parser", "-"))
            meta_col2.metric("Blocos", preview_response.get("block_count", 0))
            meta_col3.metric("Estratégia", preview_response.get("used_strategy", "-"))

            st.markdown("**Saída normalizada do segmentador**")
            st.json(preview_response.get("normalized_samples", []))

            st.markdown("**Resposta bruta do modelo**")
            raw_output = preview_response.get("raw_llm_output")
            if raw_output:
                st.code(raw_output, language="json")
            else:
                st.caption("Sem saída bruta de LLM. O backend caiu em estratégia heurística.")

            with st.expander("Prompt montado", expanded=False):
                st.code(preview_response.get("prompt_preview", ""), language="text")

            with st.expander("Blocos extraídos enviados ao prompt", expanded=False):
                st.json(preview_response.get("blocks_preview", []))

            with st.expander("Resposta parseada", expanded=False):
                st.json(preview_response.get("parsed_response", {}))

    payload = {
        "user_request": user_request,
        "task_type": task_type,
        "source_notes": source_notes,
        "context_notes": source_notes,
        "goal": goal,
        "genre_id": st.session_state.get("antese_genre_card_id", default_genre_card),
        "style_profile_id": st.session_state.get("antese_style_profile_id", default_style_profile),
        "inspiration_profile_id": st.session_state.get("antese_inspiration_profile_id") or None,
        "audience": audience,
        "tone_override": tone_override,
        "tone": tone_override,
        "must_include": parse_multiline_items(must_include_text),
        "must_avoid": parse_multiline_items(must_avoid_text),
        "reference_text": reference_text,
        "action_label": "default",
    }

    with draft_tab:
        action_col1, action_col2 = st.columns([2, 3])
        if action_col1.button("Executar Antese", type="primary", use_container_width=True):
            if not user_request.strip():
                st.warning("Descreva um pedido ao Antese antes de executar.")
            else:
                _run_antese_from_payload(payload, "Antese está estruturando, escrevendo e revisando...")

        response = st.session_state.get("last_antese_response")
        stored_payload = st.session_state.get("last_antese_payload")
        if response:
            quick_actions = {
                "Encurtar": "encurtar",
                "Expandir": "expandir",
                "Mais pessoal": "mais_pessoal",
                "Mais técnico": "mais_tecnico",
                "Mais claro": "mais_claro",
            }
            quick_cols = action_col2.columns(len(quick_actions))
            for index, (label, action_key) in enumerate(quick_actions.items()):
                if quick_cols[index].button(label, key=f"antese_quick_{action_key}", use_container_width=True):
                    if stored_payload:
                        action_payload = dict(stored_payload)
                        extra_instruction = {
                            "encurtar": "Reescreva com menos palavras, mantendo as ideias centrais.",
                            "expandir": "Expanda com mais desenvolvimento e exemplos concretos.",
                            "mais_pessoal": "Aproxime o texto da voz pessoal do style profile.",
                            "mais_tecnico": "Aumente a precisão conceitual e técnica sem perder clareza.",
                            "mais_claro": "Simplifique formulações e torne a progressão mais explícita.",
                        }[action_key]
                        action_payload["action_label"] = action_key
                        action_payload["source_notes"] = (
                            f"{action_payload.get('source_notes', '')}\n\nInstrução editorial adicional: {extra_instruction}"
                        ).strip()
                        action_payload["reference_text"] = response.get("output_text", action_payload.get("reference_text", ""))
                        _run_antese_from_payload(action_payload, f"Antese está aplicando a ação '{label}'...")

        if not response:
            st.info("Nenhuma execução direta do Antese nesta sessão ainda.")
        else:
            st.markdown("**Brief normalizado**")
            st.json(response.get("normalized_brief", {}))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Outline**")
                st.code(response.get("outline_text", ""), language="markdown")
            with col2:
                st.markdown("**Quality report**")
                st.json(response.get("quality_report", {}))
            st.markdown("**Primeiro rascunho**")
            st.write(response.get("draft_text", ""))
            st.markdown("**Versão final**")
            st.write(response.get("output_text", ""))
            st.markdown("**Exemplos recuperados**")
            st.dataframe(
                pd.DataFrame(response.get("retrieved_examples_preview", [])),
                use_container_width=True,
                hide_index=True,
            )

    with samples_tab:
        st.markdown("**Text samples indexados**")
        st.caption(
            "Aqui você consegue auditar os textos já salvos na memória de escrita e corrigir manualmente o gênero."
        )
        sample_limit = st.slider("Quantidade de samples", min_value=20, max_value=500, value=120, step=20)
        try:
            samples_payload = api_request("GET", f"/antese/samples?limit={sample_limit}")
        except RuntimeError as exc:
            st.error(f"Não foi possível carregar os text samples: {exc}")
        else:
            samples = samples_payload.get("samples", [])
            if not samples:
                st.info("Nenhum text sample salvo ainda.")
            else:
                sample_rows = [
                    {
                        "sample_id": item.get("sample_id", ""),
                        "title": item.get("title", ""),
                        "source_scope": item.get("source_scope", ""),
                        "genre_id": item.get("genre_id", ""),
                        "sample_type": item.get("metadata", {}).get("sample_type", ""),
                        "persona_scope": item.get("metadata", {}).get("persona_scope", ""),
                        "updated_at": item.get("updated_at", ""),
                    }
                    for item in samples
                ]
                sample_table = pd.DataFrame(sample_rows)
                selection_event = st.dataframe(
                    sample_table,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                )
                if selection_event and getattr(selection_event, "selection", None):
                    selected_rows = selection_event.selection.get("rows", [])
                    if selected_rows:
                        selected_index = selected_rows[0]
                        if 0 <= selected_index < len(sample_rows):
                            st.session_state["selected_antese_sample_id"] = sample_rows[selected_index]["sample_id"]

                sample_options = [item.get("sample_id", "") for item in samples]
                selected_sample_state = st.session_state.get("selected_antese_sample_id")
                if selected_sample_state not in sample_options and sample_options:
                    selected_sample_state = sample_options[0]
                    st.session_state["selected_antese_sample_id"] = selected_sample_state

                selected_sample_id = st.selectbox(
                    "Escolha um sample para inspecionar",
                    options=sample_options,
                    format_func=lambda sample_id: next(
                        (
                            (
                                f"{row.get('title', '')} • "
                                f"{row.get('genre_id', '') or 'sem gênero'} • "
                                f"{row.get('metadata', {}).get('sample_type', 'tipo n/d')} • "
                                f"{row.get('source_scope', 'escopo n/d')}"
                            )
                            for row in samples
                            if row.get("sample_id") == sample_id
                        ),
                        sample_id,
                    ),
                    key="selected_antese_sample_id",
                )
                selected_sample = next(
                    (item for item in samples if item.get("sample_id") == selected_sample_id),
                    {},
                )

                top_col1, top_col2 = st.columns([2, 1])
                with top_col1:
                    st.markdown("**Conteúdo do sample**")
                    st.write(selected_sample.get("text_content", ""))
                with top_col2:
                    st.markdown("**Metadata**")
                    st.json(
                        {
                            "title": selected_sample.get("title", ""),
                            "genre_id": selected_sample.get("genre_id", ""),
                            "source_scope": selected_sample.get("source_scope", ""),
                            "style_profile_id": selected_sample.get("style_profile_id", ""),
                            "sample_type": selected_sample.get("metadata", {}).get("sample_type", ""),
                            "persona_scope": selected_sample.get("metadata", {}).get("persona_scope", ""),
                            "tags": selected_sample.get("metadata", {}).get("tags", []),
                            "text_fingerprint": selected_sample.get("text_fingerprint", ""),
                        }
                    )

                with st.form("antese_sample_reclassify_form"):
                    new_genre_id = st.selectbox(
                        "Reclassificar gênero",
                        options=[""] + list(genre_card_options.keys()),
                        format_func=lambda item: "Sem gênero" if not item else genre_card_options[item]["name"],
                        index=([""] + list(genre_card_options.keys())).index(selected_sample.get("genre_id", ""))
                        if selected_sample.get("genre_id", "") in ([""] + list(genre_card_options.keys()))
                        else 0,
                    )
                    reclassify_submit = st.form_submit_button(
                        "Salvar reclassificação",
                        type="primary",
                        use_container_width=True,
                    )

                if reclassify_submit:
                    try:
                        api_request(
                            "POST",
                            f"/antese/samples/{selected_sample_id}/reclassify",
                            {"genre_id": new_genre_id or None},
                        )
                    except RuntimeError as exc:
                        st.error(f"Falha ao reclassificar o sample: {exc}")
                    else:
                        st.success("Gênero do sample atualizado com sucesso.")
                        st.rerun()

    with history_tab:
        st.markdown("**Execuções recentes do Antese**")
        if using_fallback_catalog:
            st.caption("Histórico persistido indisponível enquanto a API ativa não expuser os endpoints novos.")
        if not execution_history:
            st.info("Nenhuma execução persistida do Antese ainda.")
        else:
            history_rows = [
                {
                    "execution_id": item.get("execution_id", ""),
                    "task_type": item.get("task_type", ""),
                    "genre_id": item.get("genre_id", ""),
                    "style_profile_id": item.get("style_profile_id", ""),
                    "created_at": item.get("created_at", ""),
                    "pedido": item.get("user_request", ""),
                }
                for item in execution_history
            ]
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

            selected_execution_id = st.selectbox(
                "Escolha uma execução para inspecionar",
                options=[item.get("execution_id", "") for item in execution_history],
                format_func=lambda execution_id: next(
                    (
                        f"{row.get('created_at', '')} • {row.get('task_type', '')} • {row.get('user_request', '')[:60]}"
                        for row in execution_history
                        if row.get("execution_id") == execution_id
                    ),
                    execution_id,
                ),
            )
            try:
                versions_payload = api_request("GET", f"/antese/executions/{selected_execution_id}/versions")
                feedback_payload = api_request("GET", f"/antese/feedback?execution_id={selected_execution_id}&limit=20")
            except RuntimeError as exc:
                st.error(f"Não foi possível carregar o histórico detalhado: {exc}")
            else:
                versions = versions_payload.get("versions", [])
                feedback_rows = feedback_payload.get("feedback", [])
                selected_execution = next(
                    (item for item in execution_history if item.get("execution_id") == selected_execution_id),
                    {},
                )

                st.markdown("**Execução selecionada**")
                st.json(
                    {
                        "normalized_brief": selected_execution.get("normalized_brief", {}),
                        "applied_style_profile": selected_execution.get("applied_style_profile", {}),
                        "applied_genre_card": selected_execution.get("applied_genre_card", {}),
                    }
                )

                if versions:
                    st.markdown("**Versões**")
                    version_rows = [
                        {
                            "version_id": item.get("version_id", ""),
                            "stage": item.get("stage", ""),
                            "action_label": item.get("action_label", ""),
                            "created_at": item.get("created_at", ""),
                        }
                        for item in versions
                    ]
                    st.dataframe(pd.DataFrame(version_rows), use_container_width=True, hide_index=True)
                    selected_version_id = st.selectbox(
                        "Versão para feedback",
                        options=[item.get("version_id", "") for item in versions],
                        format_func=lambda version_id: next(
                            (
                                f"{row.get('stage', '')} • {row.get('action_label', '')}"
                                for row in versions
                                if row.get("version_id") == version_id
                            ),
                            version_id,
                        ),
                    )
                    selected_version = next(
                        (item for item in versions if item.get("version_id") == selected_version_id),
                        {},
                    )
                    st.write(selected_version.get("output_text", ""))

                    with st.form("antese_feedback_form"):
                        liked = st.selectbox("Gostou dessa versão?", options=["indefinido", "sim", "nao"], index=0)
                        colf1, colf2, colf3, colf4 = st.columns(4)
                        style_score = colf1.slider("Style match", 0.0, 1.0, 0.8, 0.05)
                        usefulness_score = colf2.slider("Usefulness", 0.0, 1.0, 0.8, 0.05)
                        creativity_score = colf3.slider("Creativity", 0.0, 1.0, 0.75, 0.05)
                        faithfulness_score = colf4.slider("Faithfulness", 0.0, 1.0, 0.85, 0.05)
                        reason_tags = st.multiselect(
                            "Reason tags",
                            options=[
                                "mais_meu_estilo",
                                "muito_generico",
                                "muito_engessado",
                                "boa_estrutura",
                                "faltou_clareza",
                                "faltou_originalidade",
                                "inventou_contexto",
                            ],
                        )
                        feedback_notes = st.text_area("Notas de feedback", height=120)
                        feedback_submit = st.form_submit_button("Salvar feedback", type="primary", use_container_width=True)

                    if feedback_submit:
                        try:
                            feedback_result = api_request(
                                "POST",
                                "/antese/feedback",
                                {
                                    "execution_id": selected_execution_id,
                                    "version_id": selected_version_id,
                                    "liked": None if liked == "indefinido" else liked == "sim",
                                    "style_match_score": style_score,
                                    "usefulness_score": usefulness_score,
                                    "creativity_score": creativity_score,
                                    "faithfulness_score": faithfulness_score,
                                    "notes": feedback_notes,
                                    "reason_tags": reason_tags,
                                },
                            )
                        except RuntimeError as exc:
                            st.error(f"Falha ao salvar feedback: {exc}")
                        else:
                            st.success(f"Feedback salvo com id {feedback_result['feedback_id']}.")
                            st.rerun()

                st.markdown("**Feedbacks salvos**")
                if feedback_rows:
                    st.dataframe(pd.DataFrame(feedback_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("Nenhum feedback salvo para esta execução ainda.")


def render_enzima_workspace(status: dict[str, Any] | None, dashboard_payload: dict[str, Any]) -> None:
    """Render the Enzima-specific email operations workspace."""

    render_hero(status, dashboard_payload)
    render_metrics(dashboard_payload)
    overview_tab, executions_tab, review_tab = st.tabs(
        ["Visão Geral", "Execuções & Correções", "Revisão Manual"]
    )

    with overview_tab:
        st.divider()
        render_category_breakdowns(dashboard_payload)
        st.divider()
        render_execution_history()

    with executions_tab:
        st.divider()
        render_last_execution_results()
        st.divider()
        render_processed_email_audit()

    with review_tab:
        st.divider()
        render_review_queue()


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

    micelio_tab, enzima_tab, antese_tab = st.tabs(["Micélio", "Enzima", "Antese"])

    with micelio_tab:
        st.divider()
        render_micelio_workspace()

    with enzima_tab:
        st.divider()
        render_enzima_workspace(status, dashboard_payload)

    with antese_tab:
        st.divider()
        render_antese_workspace()


if __name__ == "__main__":
    main()
