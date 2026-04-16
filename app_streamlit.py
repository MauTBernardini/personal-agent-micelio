"""Streamlit interface for the personal email triage MVP.

This file intentionally contains presentation code only and delegates all
business logic to `core_agent.py`.
"""

from __future__ import annotations

import streamlit as st

from core_agent import (
    LLMProviderError,
    get_connection_status,
    get_dashboard_metrics,
    get_manual_review_categories,
    get_review_queue,
    manual_reclassify_email,
    run_triage_workflow,
)


st.set_page_config(
    page_title="Agente Pessoal de Triagem de E-mails",
    page_icon="📬",
    layout="wide",
)


def render_sidebar() -> None:
    """Render the Streamlit sidebar controls."""

    st.sidebar.title("Controle do Agente")

    if st.sidebar.button("Rodar Triagem", use_container_width=True, type="primary"):
        try:
            with st.spinner("Executando o fluxo LangGraph e processando a caixa de entrada..."):
                results = run_triage_workflow()

            if results:
                st.sidebar.success(f"Triagem concluída: {len(results)} novo(s) e-mail(s) processado(s).")
            else:
                st.sidebar.info("Nenhum novo e-mail para processar no provider ativo no momento.")
        except LLMProviderError as exc:
            st.sidebar.error(f"Falha no provider LLM: {exc}")
            st.error(f"Falha no provider LLM selecionado: {exc}")
        except Exception as exc:
            st.sidebar.error(f"Erro na triagem: {exc}")
            st.error(f"Erro durante a triagem: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Status das conexões")

    status = get_connection_status()
    st.sidebar.caption(f"Gmail: {status['gmail']}")
    st.sidebar.caption(f"PostgreSQL: {status['postgres']}")
    st.sidebar.caption(f"Chroma: {status['chroma']}")
    st.sidebar.caption(f"LLM Provider: {status['llm_provider']}")
    st.sidebar.caption(f"LLM Status: {status['llm_status']}")
    st.sidebar.caption(f"LLM Model: {status['llm_model']}")


def render_metrics() -> None:
    """Render the dashboard metric cards."""

    metrics = get_dashboard_metrics()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Processados Hoje", metrics["emails_processados_hoje"])
    col2.metric("Prioritários", metrics["prioritarios_detectados"])
    col3.metric("Em Revisão", metrics["fila_revisao"])
    col4.metric("Taxa de Automação", f"{metrics['taxa_automacao']}%")


def render_review_queue() -> None:
    """Render emails currently waiting for manual review."""

    st.subheader("Fila de Revisão Manual")
    st.caption(
        "Os e-mails classificados como `EM_DUVIDA` aparecem aqui para decisão humana. "
        "Cada ação atualiza o PostgreSQL e injeta aprendizado no ChromaDB."
    )

    review_queue = get_review_queue()
    categories = get_manual_review_categories()

    if not review_queue:
        st.success("Nenhum e-mail aguardando revisão manual.")
        return

    for item in review_queue:
        header = f"{item['subject']} • {item['sender']}"
        with st.expander(header, expanded=False):
            st.write(f"**Message ID:** {item['message_id']}")
            st.write(f"**Status atual:** {item['status']}")
            st.write(f"**Categoria atual:** {item['category']}")
            st.write("**Corpo do e-mail:**")
            st.write(item.get("body", "Sem corpo disponível no mock."))

            action_cols = st.columns(len(categories))
            for index, category in enumerate(categories):
                if action_cols[index].button(
                    category,
                    key=f"{item['message_id']}_{category}",
                    use_container_width=True,
                ):
                    result = manual_reclassify_email(item["message_id"], category)
                    st.success(
                        f"E-mail {result['message_id']} reclassificado manualmente para "
                        f"{result['forced_category']}."
                    )
                    st.rerun()


def main() -> None:
    """Main entry point for the Streamlit app."""

    render_sidebar()

    st.title("Agente Pessoal de Triagem de E-mails")
    st.caption(
        "MVP agentic com LangGraph, memória transacional em PostgreSQL dockerizado, "
        "memória semântica em ChromaDB e erro explícito no front quando o "
        "provider LLM selecionado não estiver operacional."
    )

    render_metrics()
    st.divider()
    render_review_queue()


if __name__ == "__main__":
    main()
