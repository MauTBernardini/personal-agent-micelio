"""Simple supervisor/router for the multi-agent MVP."""

from __future__ import annotations

from typing import Any

from email_agent.service import run_triage_and_collect
from email_agent.storage import get_recent_supervisor_executions, save_supervisor_execution
from writing_agent.service import get_antese_capabilities, run_antese


def get_available_agents() -> list[dict[str, Any]]:
    """Return the currently registered specialist agents."""

    return [
        {
            "agent_name": "ENZIMA",
            "description": "Especialista em triagem e classificação de e-mails.",
            "task_types": ["TRIAGE_EMAILS"],
        },
        get_antese_capabilities(),
    ]


def get_micelio_execution_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return persisted Micelio executions for audit and UI history."""

    return get_recent_supervisor_executions(limit=limit)


def _infer_agent_route(user_request: str, requested_agent: str | None = None) -> tuple[list[str], str, list[str]]:
    """Choose which specialist agents should handle the request."""

    if requested_agent:
        normalized_requested_agent = requested_agent.strip().upper()
        if normalized_requested_agent in {"ENZIMA", "ANTESE"}:
            return [normalized_requested_agent], "Rota solicitada explicitamente pelo usuário.", [
                f"Executar {normalized_requested_agent}.",
            ]

    normalized_request = user_request.lower()
    email_signals = ("email", "e-mail", "gmail", "inbox", "triagem", "classificar")
    writing_signals = ("escrever", "texto", "rascunho", "ideia", "brainstorm", "outline", "post", "artigo")

    has_email_signal = any(signal in normalized_request for signal in email_signals)
    has_writing_signal = any(signal in normalized_request for signal in writing_signals)

    if has_email_signal and has_writing_signal:
        return (
            ["ENZIMA", "ANTESE"],
            "O pedido mistura triagem de e-mails com produção ou consolidação de texto.",
            [
                "Executar Enzima para buscar/classificar material relevante.",
                "Passar o contexto resultante para Antese consolidar a escrita.",
            ],
        )
    if has_email_signal:
        return ["ENZIMA"], "O pedido é focado em triagem ou leitura operacional de e-mails.", [
            "Executar Enzima para processar os e-mails relevantes.",
        ]
    return ["ANTESE"], "O pedido é focado em ideação, estruturação ou escrita.", [
        "Executar Antese para gerar ou consolidar o texto solicitado.",
    ]


def run_micelio(
    user_request: str,
    requested_agent: str | None = None,
    writing_task_type: str = "BRAINSTORM",
    context_notes: str = "",
    tone: str = "claro e profissional",
    audience: str = "leitores gerais",
    email_limit: int | None = None,
) -> dict[str, Any]:
    """Route the request and execute the selected specialist agents."""

    selected_agents, reason, execution_plan = _infer_agent_route(user_request, requested_agent=requested_agent)
    agent_outputs: list[dict[str, Any]] = []

    triage_result: dict[str, Any] | None = None
    if "ENZIMA" in selected_agents:
        triage_result = run_triage_and_collect(limit=email_limit)
        agent_outputs.append(
            {
                "agent_name": "ENZIMA",
                "result": triage_result,
            }
        )

    if "ANTESE" in selected_agents:
        combined_context = context_notes.strip()
        if triage_result is not None:
            processed = triage_result.get("processed", 0)
            category_counts = triage_result.get("category_counts", [])
            category_summary = ", ".join(
                f"{row['categoria']}: {row['quantidade']}"
                for row in category_counts
                if int(row.get("quantidade", 0)) > 0
            ) or "Sem novas categorias processadas."
            triage_context = (
                f"Resultado da triagem do Enzima: {processed} e-mails processados. "
                f"Categorias: {category_summary}."
            )
            combined_context = f"{triage_context}\n\n{combined_context}".strip()

        writing_result = run_antese(
            user_request=user_request,
            task_type=writing_task_type,
            source_notes=combined_context,
            tone_override=tone,
            audience=audience,
        )
        agent_outputs.append(
            {
                "agent_name": "ANTESE",
                "result": writing_result,
            }
        )

    output_preview_parts: list[str] = []
    for item in agent_outputs:
        if item["agent_name"] == "ENZIMA":
            result = item["result"]
            output_preview_parts.append(
                f"Enzima processou {result.get('processed', 0)} e-mail(s)."
            )
        elif item["agent_name"] == "ANTESE":
            result = item["result"]
            preview_text = str(result.get("output_text", "")).strip().replace("\n", " ")
            output_preview_parts.append(
                f"Antese ({result.get('task_type', 'N/D')}): {preview_text[:180]}{'...' if len(preview_text) > 180 else ''}"
            )

    execution_id = save_supervisor_execution(
        supervisor_name="MICELIO",
        user_request=user_request,
        requested_agent=requested_agent,
        selected_agents=selected_agents,
        reason=reason,
        execution_plan=execution_plan,
        writing_task_type=writing_task_type,
        tone=tone,
        audience=audience,
        context_notes=context_notes,
        email_limit=email_limit,
        agent_outputs=agent_outputs,
        output_preview=" ".join(output_preview_parts).strip(),
    )

    return {
        "supervisor_name": "MICELIO",
        "execution_id": execution_id,
        "selected_agents": selected_agents,
        "reason": reason,
        "execution_plan": execution_plan,
        "agent_outputs": agent_outputs,
    }
