"""LangGraph workflow assembly and execution helpers."""

from __future__ import annotations

from typing import Any, Sequence, cast

from langgraph.graph import END, START, StateGraph

from email_agent.email_provider import list_email_messages, move_email_message
from email_agent.llm import LLMProviderError, classify_email_with_llm, summarize_memory_email
from email_agent.models import EmailAgentState, FINAL_LABELS, FewShotExample
from email_agent.storage import (
    bootstrap_services,
    get_chroma_collection,
    is_already_processed,
    query_semantic_examples,
    save_processed_email,
    upsert_semantic_memory,
)


def _select_few_shots(candidates: list[FewShotExample], focus: str) -> list[FewShotExample]:
    """Choose 2 positive and 1 hard-negative examples from semantic memory."""

    if not candidates:
        return []

    positives: list[FewShotExample] = []
    for candidate in candidates:
        if len(positives) >= 2:
            break
        positives.append({**candidate, "sample_role": "POSITIVE"})

    reference = positives[0] if positives else candidates[0]
    negative: FewShotExample | None = None
    for candidate in candidates[2:]:
        if focus == "theme" and candidate["theme_category"] != reference["theme_category"]:
            negative = {**candidate, "sample_role": "NEGATIVE"}
            break
        if focus == "priority" and candidate["priority_level"] != reference["priority_level"]:
            negative = {**candidate, "sample_role": "NEGATIVE"}
            break

    if negative is None and len(candidates) >= 3:
        negative = {**candidates[-1], "sample_role": "NEGATIVE"}

    selected = positives.copy()
    if negative is not None:
        selected.append(negative)
    return selected[:3]


def retrieve_context(state: EmailAgentState) -> EmailAgentState:
    """Retrieve related semantic memory and build few-shots for both prompts."""

    email_data = state["email_data"]
    query_text = f"{email_data.get('subject', '')}\n{email_data.get('body', '')}"
    rag_context = "Sem memoria semantica relevante."
    semantic_candidates: list[FewShotExample] = []

    try:
        collection = get_chroma_collection()
        result = collection.query(query_texts=[query_text], n_results=2)
        documents = result.get("documents", [[]])
        flattened = documents[0] if documents and documents[0] else []
        rag_context = "\n---\n".join(flattened) if flattened else rag_context
        semantic_candidates = query_semantic_examples(query_text=query_text, n_results=8)
    except Exception:
        rag_context = "Sem memoria semantica relevante."
        semantic_candidates = []

    theme_few_shots = _select_few_shots(semantic_candidates, focus="theme")
    priority_few_shots = _select_few_shots(semantic_candidates, focus="priority")

    return {
        **state,
        "rag_context": rag_context,
        "theme_few_shots": theme_few_shots,
        "priority_few_shots": priority_few_shots,
    }


def classify_email(state: EmailAgentState) -> EmailAgentState:
    """Classify the e-mail theme and priority in a single LLM pass."""

    classification_result = classify_email_with_llm(
        email_data=state["email_data"],
        rag_context=state["rag_context"],
        theme_few_shots=state["theme_few_shots"],
        priority_few_shots=state["priority_few_shots"],
    )
    if str(classification_result.get("final_label", "EM_DUVIDA")) not in FINAL_LABELS:
        classification_result["final_label"] = "EM_DUVIDA"
        classification_result["requires_manual_review"] = True
        classification_result["gmail_flagged"] = False
    return {**state, "classification_result": classification_result}


def execute_action(state: EmailAgentState) -> EmailAgentState:
    """Persist classification results and label/flag the e-mail."""

    email_data = state["email_data"]
    classification = state["classification_result"]
    final_label = str(classification["final_label"])
    theme_category = str(classification["theme_category"])
    priority_level = str(classification["priority_level"])
    status = "EM_DUVIDA" if final_label == "EM_DUVIDA" else "PROCESSADO"

    save_processed_email(
        message_id=str(email_data["message_id"]),
        sender=str(email_data["sender"]),
        subject=str(email_data["subject"]),
        status=status,
        category=final_label,
        theme_category=theme_category,
        priority_level=priority_level,
        final_label=final_label,
        needs_action=bool(classification["needs_action"]),
        is_important=bool(classification["is_important"]),
        time_sensitivity=str(classification["time_sensitivity"]),
        confidence=float(classification["confidence"]),
        gmail_flagged=bool(classification["gmail_flagged"]),
    )

    if final_label != "EM_DUVIDA":
        classification["move_result"] = move_email_message(
            str(email_data["message_id"]),
            final_label,
            should_flag=bool(classification["gmail_flagged"]),
        )
    return {**state, "classification_result": classification}


def dynamic_learning(state: EmailAgentState) -> EmailAgentState:
    """Persist a compact factual memory for the classified e-mail."""

    email_data = state["email_data"]
    classification = state["classification_result"]
    summary = summarize_memory_email(email_data, classification)
    upsert_semantic_memory(
        message_id=str(email_data["message_id"]),
        email_data=email_data,
        learned_text=summary,
        theme_category=str(classification["theme_category"]),
        priority_level=str(classification["priority_level"]),
        final_label=str(classification["final_label"]),
    )
    classification["memory_summary"] = summary
    return {**state, "classification_result": classification}


def should_run_dynamic_learning(state: EmailAgentState) -> str:
    if str(state["classification_result"].get("final_label", "EM_DUVIDA")) != "EM_DUVIDA":
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


def _is_quota_error(message: str) -> bool:
    normalized_message = message.lower()
    return "quota" in normalized_message or "cota" in normalized_message or "rate limit" in normalized_message


def run_triage_batch(limit: int | None = None) -> dict[str, Any]:
    """Run the workflow and capture per-email failures without dropping the whole batch."""

    bootstrap_services()
    graph = build_email_triage_graph()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stopped_due_to_quota = False
    stopped_reason = ""

    for email in list_email_messages(limit=limit):
        if is_already_processed(str(email["message_id"])):
            continue

        try:
            final_state = graph.invoke(
                {
                    "email_data": email,
                    "rag_context": "",
                    "theme_few_shots": [],
                    "priority_few_shots": [],
                    "classification_result": {},
                }
            )
        except LLMProviderError as exc:
            error_message = str(exc)
            failures.append(
                {
                    "message_id": str(email.get("message_id", "")),
                    "sender": str(email.get("sender", "")),
                    "subject": str(email.get("subject", "")),
                    "error": error_message,
                }
            )
            if _is_quota_error(error_message):
                stopped_due_to_quota = True
                stopped_reason = error_message
                break
            continue

        results.append(cast(dict[str, Any], final_state))

    return {
        "results": results,
        "failures": failures,
        "stopped_due_to_quota": stopped_due_to_quota,
        "stopped_reason": stopped_reason,
    }


def run_triage_workflow(limit: int | None = None) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], run_triage_batch(limit=limit)["results"])


def summarize_category_counts(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {label: 0 for label in FINAL_LABELS}
    for item in results:
        classification = cast(dict[str, Any], item.get("classification_result", {}))
        final_label = str(classification.get("final_label", "EM_DUVIDA")).strip().upper()
        if final_label not in counts:
            final_label = "EM_DUVIDA"
        counts[final_label] += 1
    return [{"categoria": label, "quantidade": counts[label]} for label in FINAL_LABELS]
