"""LangGraph workflow assembly and execution helpers."""

from __future__ import annotations

from typing import Any, Sequence, cast

from langgraph.graph import END, START, StateGraph

from email_agent.email_provider import list_email_messages, move_email_message
from email_agent.llm import classify_email_with_llm, summarize_priority_email
from email_agent.models import ALLOWED_CATEGORIES, EmailAgentState
from email_agent.storage import (
    bootstrap_services,
    get_chroma_collection,
    is_already_processed,
    save_processed_email,
    upsert_semantic_memory,
)


def retrieve_context(state: EmailAgentState) -> EmailAgentState:
    email_data = state["email_data"]
    query_text = f"{email_data.get('subject', '')}\n{email_data.get('body', '')}"
    collection = get_chroma_collection()

    result = collection.query(query_texts=[query_text], n_results=2)
    documents = result.get("documents", [[]])
    flattened = documents[0] if documents and documents[0] else []
    rag_context = "\n---\n".join(flattened) if flattened else "Sem memoria semantica relevante."
    return {**state, "rag_context": rag_context}


def classify_email(state: EmailAgentState) -> EmailAgentState:
    classification_result = classify_email_with_llm(
        email_data=state["email_data"],
        rag_context=state["rag_context"],
    )
    return {**state, "classification_result": classification_result}


def execute_action(state: EmailAgentState) -> EmailAgentState:
    email_data = state["email_data"]
    classification = state["classification_result"]
    category = str(classification["categoria"])
    status = "EM_DUVIDA" if category == "EM_DUVIDA" else "PROCESSADO"

    save_processed_email(
        message_id=str(email_data["message_id"]),
        sender=str(email_data["sender"]),
        subject=str(email_data["subject"]),
        status=status,
        category=category,
    )

    if category != "EM_DUVIDA":
        classification["move_result"] = move_email_message(str(email_data["message_id"]), category)
    return {**state, "classification_result": classification}


def dynamic_learning(state: EmailAgentState) -> EmailAgentState:
    email_data = state["email_data"]
    classification = state["classification_result"]
    summary = summarize_priority_email(email_data)
    upsert_semantic_memory(
        message_id=str(email_data["message_id"]),
        email_data=email_data,
        learned_text=summary,
        category=str(classification["categoria"]),
    )
    classification["priority_summary"] = summary
    return {**state, "classification_result": classification}


def should_run_dynamic_learning(state: EmailAgentState) -> str:
    if bool(state["classification_result"].get("is_priority", False)):
        return "dynamic_learning"
    return END


def build_email_triage_graph() -> Any:
    graph_builder = StateGraph(EmailAgentState)
    graph_builder.add_node("retrieve_context", retrieve_context)
    graph_builder.add_node("classify_email", classify_email)
    graph_builder.add_node("execute_action", execute_action)
    graph_builder.add_node("dynamic_learning", dynamic_learning)

    graph_builder.add_edge(START, "retrieve_context")
    graph_builder.add_edge("retrieve_context", "classify_email")
    graph_builder.add_edge("classify_email", "execute_action")
    graph_builder.add_conditional_edges(
        "execute_action",
        should_run_dynamic_learning,
        {"dynamic_learning": "dynamic_learning", END: END},
    )
    graph_builder.add_edge("dynamic_learning", END)
    return graph_builder.compile()


def run_triage_workflow(limit: int | None = None) -> list[dict[str, Any]]:
    bootstrap_services()
    graph = build_email_triage_graph()
    outputs: list[dict[str, Any]] = []

    for email in list_email_messages(limit=limit):
        if is_already_processed(str(email["message_id"])):
            continue

        final_state = graph.invoke(
            {"email_data": email, "rag_context": "", "classification_result": {}}
        )
        outputs.append(cast(dict[str, Any], final_state))

    return outputs


def summarize_category_counts(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {category: 0 for category in ALLOWED_CATEGORIES}
    for item in results:
        classification = cast(dict[str, Any], item.get("classification_result", {}))
        category = str(classification.get("categoria", "EM_DUVIDA")).strip().upper()
        if category not in counts:
            category = "EM_DUVIDA"
        counts[category] += 1
    return [{"categoria": category, "quantidade": counts[category]} for category in ALLOWED_CATEGORIES]

