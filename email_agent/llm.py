"""LLM provider abstraction and classification helpers."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from email_agent.models import (
    EMAIL_CLASSIFICATION_PROMPT,
    FINAL_LABELS,
    PRIORITY_LEVELS,
    SUMMARY_PROMPT,
    THEME_CATEGORIES,
    FewShotExample,
)
from email_agent.settings import (
    Anthropic,
    OpenAI,
    genai,
    genai_types,
    get_anthropic_model_name,
    get_gemini_fallback_model_name,
    get_gemini_model_name,
    get_llm_provider,
    get_openai_model_name,
)


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot satisfy a request."""


def _extract_sender_address(sender: str) -> str:
    match = re.search(r"<([^>]+)>", sender)
    if match:
        return match.group(1).strip().lower()
    return sender.strip().lower()


def _sender_based_classification_override(email_data: dict[str, Any]) -> dict[str, Any] | None:
    sender_raw = str(email_data.get("sender", "")).strip()
    sender_address = _extract_sender_address(sender_raw)
    sender_text = f"{sender_raw} {sender_address}".lower()

    if "linkedin" in sender_text and (
        sender_address.startswith("noreply@")
        or sender_address.startswith("jobs-noreply@")
        or "noreply" in sender_address
    ):
        return {
            "theme_category": "CARREIRA",
            "priority_level": "BAIXA",
            "needs_action": False,
            "is_important": False,
            "time_sensitivity": "BAIXA",
            "confidence": 0.99,
            "motivo_tema": "Regra determinística: remetente noreply do LinkedIn tratado como carreira.",
            "motivo_prioridade": "Regra determinística: notificações automáticas do LinkedIn entram como baixa prioridade.",
            "evidence": [
                sender_raw or "Remetente ausente.",
                str(email_data.get("subject", "")).strip() or "Assunto ausente.",
            ],
            "uncertainty_reason": "",
            "final_label": "BAIXA_CARREIRA",
            "gmail_flagged": False,
            "requires_manual_review": False,
        }

    newsletter_signals = ("newsletter", "digest", "roundup", "bulletin")
    if any(signal in sender_text for signal in newsletter_signals):
        return {
            "theme_category": "NEWSLETTER",
            "priority_level": "BAIXA",
            "needs_action": False,
            "is_important": False,
            "time_sensitivity": "BAIXA",
            "confidence": 0.97,
            "motivo_tema": "Regra determinística: remetente com sinal forte de newsletter.",
            "motivo_prioridade": "Regra determinística: newsletters entram como baixa prioridade.",
            "evidence": [
                sender_raw or "Remetente ausente.",
                str(email_data.get("subject", "")).strip() or "Assunto ausente.",
            ],
            "uncertainty_reason": "",
            "final_label": "BAIXA_NEWSLETTER",
            "gmail_flagged": False,
            "requires_manual_review": False,
        }

    return None


def _is_quota_message(raw_message: str) -> bool:
    normalized_message = raw_message.lower()
    return "resource_exhausted" in normalized_message or "quota" in normalized_message or "429" in normalized_message


def _wrap_provider_exception(provider_name: str, exc: Exception) -> LLMProviderError:
    """Convert raw SDK exceptions into user-facing provider errors."""

    raw_message = str(exc).strip() or provider_name
    normalized_message = raw_message.lower()

    if _is_quota_message(raw_message):
        retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", raw_message, flags=re.IGNORECASE)
        retry_suffix = ""
        if retry_match:
            retry_suffix = f" Tente novamente em cerca de {round(float(retry_match.group(1)))}s."
        return LLMProviderError(
            f"{provider_name} sem cota disponível no momento. Verifique billing/rate limits.{retry_suffix}"
        )

    if "503" in normalized_message or "unavailable" in normalized_message or "overloaded" in normalized_message:
        return LLMProviderError(
            f"{provider_name} indisponível temporariamente. Tente novamente em instantes."
        )

    return LLMProviderError(f"{provider_name} falhou: {raw_message}")


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(raw_text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _coerce_confidence(value: Any, default: float = 0.5) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric_value))


def _normalize_theme_result(payload: dict[str, Any]) -> dict[str, Any]:
    theme_category = str(payload.get("theme_category", "EM_DUVIDA")).strip().upper()
    if theme_category not in THEME_CATEGORIES:
        theme_category = "EM_DUVIDA"

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    normalized_evidence = [str(item).strip() for item in evidence if str(item).strip()][:3]
    if not normalized_evidence:
        normalized_evidence = ["Sem evidência explícita extraída do e-mail."]

    return {
        "theme_category": theme_category,
        "confidence": _coerce_confidence(payload.get("confidence"), default=0.45),
        "motivo_curto": str(payload.get("motivo_curto", "Tema inferido pelo classificador.")).strip()
        or "Tema inferido pelo classificador.",
        "evidence": normalized_evidence,
        "uncertainty_reason": str(payload.get("uncertainty_reason", "")).strip(),
    }


def _normalize_priority_result(payload: dict[str, Any]) -> dict[str, Any]:
    priority_level = str(payload.get("priority_level", "BAIXA")).strip().upper()
    if priority_level not in PRIORITY_LEVELS:
        priority_level = "BAIXA"

    time_sensitivity = str(payload.get("time_sensitivity", "BAIXA")).strip().upper()
    if time_sensitivity not in PRIORITY_LEVELS:
        time_sensitivity = "BAIXA"

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    normalized_evidence = [str(item).strip() for item in evidence if str(item).strip()][:3]
    if not normalized_evidence:
        normalized_evidence = ["Sem evidência operacional explícita extraída do e-mail."]

    return {
        "priority_level": priority_level,
        "needs_action": bool(payload.get("needs_action", False)),
        "is_important": bool(payload.get("is_important", False)),
        "time_sensitivity": time_sensitivity,
        "confidence": _coerce_confidence(payload.get("confidence"), default=0.45),
        "motivo_curto": str(payload.get("motivo_curto", "Prioridade inferida pelo classificador.")).strip()
        or "Prioridade inferida pelo classificador.",
        "evidence": normalized_evidence,
        "uncertainty_reason": str(payload.get("uncertainty_reason", "")).strip(),
    }


def _normalize_combined_result(payload: dict[str, Any]) -> dict[str, Any]:
    theme_result = _normalize_theme_result(
        {
            "theme_category": payload.get("theme_category", "EM_DUVIDA"),
            "confidence": payload.get("theme_confidence", payload.get("confidence", 0.45)),
            "motivo_curto": payload.get("theme_reason", payload.get("motivo_curto", "")),
            "evidence": payload.get("evidence", []),
            "uncertainty_reason": payload.get("uncertainty_reason", ""),
        }
    )
    priority_result = _normalize_priority_result(
        {
            "priority_level": payload.get("priority_level", "BAIXA"),
            "needs_action": payload.get("needs_action", False),
            "is_important": payload.get("is_important", False),
            "time_sensitivity": payload.get("time_sensitivity", "BAIXA"),
            "confidence": payload.get("priority_confidence", payload.get("confidence", 0.45)),
            "motivo_curto": payload.get("priority_reason", payload.get("motivo_curto", "")),
            "evidence": payload.get("evidence", []),
            "uncertainty_reason": payload.get("uncertainty_reason", ""),
        }
    )
    return _compose_final_classification(theme_result, priority_result)


def _compose_final_classification(theme_result: dict[str, Any], priority_result: dict[str, Any]) -> dict[str, Any]:
    theme_category = str(theme_result["theme_category"])
    priority_level = str(priority_result["priority_level"])
    theme_confidence = float(theme_result["confidence"])
    priority_confidence = float(priority_result["confidence"])

    is_uncertain = (
        theme_category == "EM_DUVIDA"
        or theme_confidence < 0.45
        or priority_confidence < 0.45
    )
    final_label = "EM_DUVIDA" if is_uncertain else f"{priority_level}_{theme_category}"
    if final_label not in FINAL_LABELS:
        final_label = "EM_DUVIDA"

    return {
        "theme_category": theme_category,
        "priority_level": priority_level,
        "needs_action": bool(priority_result["needs_action"]),
        "is_important": bool(priority_result["is_important"]),
        "time_sensitivity": str(priority_result["time_sensitivity"]),
        "confidence": round(min(theme_confidence, priority_confidence), 3),
        "motivo_tema": str(theme_result["motivo_curto"]),
        "motivo_prioridade": str(priority_result["motivo_curto"]),
        "evidence": [
            *cast(list[str], theme_result["evidence"]),
            *cast(list[str], priority_result["evidence"]),
        ][:5],
        "uncertainty_reason": (
            str(theme_result.get("uncertainty_reason", "")).strip()
            or str(priority_result.get("uncertainty_reason", "")).strip()
        ),
        "final_label": final_label,
        "gmail_flagged": final_label.startswith("ALTA_"),
        "requires_manual_review": final_label == "EM_DUVIDA",
    }


def _heuristic_theme_classifier(email_data: dict[str, Any], rag_context: str) -> dict[str, Any]:
    text = f"{email_data.get('subject', '')} {email_data.get('body', '')} {rag_context}".lower()

    if any(token in text for token in ("entrevista", "vaga", "recrut", "remote", "remota")):
        theme = "CARREIRA"
        reason = "Conteúdo de recrutamento ou entrevista."
    elif any(token in text for token in ("gcp", "postgresql", "arquitetura", "microsserv", "agentic")):
        theme = "PROJETOS_TECH"
        reason = "Assunto técnico alinhado a projetos de engenharia."
    elif any(token in text for token in ("bambu lab", "petg", "impressora 3d", "boardgame", "maker")):
        theme = "CRIATIVO_MAKER"
        reason = "Tema maker, hobby técnico ou criativo."
    elif any(token in text for token in ("curso", "turma", "inscri", "aprenda", "workshop")):
        theme = "CURSOS_APRENDIZADO"
        reason = "Convite ou conteúdo educacional."
    elif any(token in text for token in ("fatura", "cartao", "banco", "vencimento", "boleto")):
        theme = "FINANCEIRO"
        reason = "Mensagem de contexto financeiro."
    elif any(token in text for token in ("newsletter", "resumo diario", "curadoria", "digest")):
        theme = "NEWSLETTER"
        reason = "Conteúdo recorrente de newsletter."
    elif any(token in text for token in ("ganhou", "gratis", "clique agora", "prêmio", "premio")):
        theme = "SPAM"
        reason = "Padrão típico de spam promocional."
    elif "colaboração" in text or "colaboracao" in text:
        theme = "EM_DUVIDA"
        reason = "Pedido potencialmente relevante, mas ambíguo."
    else:
        theme = "OUTROS"
        reason = "Não se encaixa claramente nos temas principais."

    return _normalize_theme_result(
        {
            "theme_category": theme,
            "confidence": 0.74 if theme != "EM_DUVIDA" else 0.35,
            "motivo_curto": reason,
            "evidence": [str(email_data.get("subject", "")).strip() or "Assunto ausente."],
            "uncertainty_reason": "" if theme != "EM_DUVIDA" else "Heurística encontrou ambiguidade.",
        }
    )


def _heuristic_priority_classifier(email_data: dict[str, Any], theme_result: dict[str, Any]) -> dict[str, Any]:
    text = f"{email_data.get('subject', '')} {email_data.get('body', '')}".lower()
    theme = str(theme_result["theme_category"])

    high_signals = ("hoje", "urgente", "vencimento", "confirme", "entrevista", "prazo", "amanhã")
    medium_signals = ("inscri", "disponível", "convite", "revisar", "call", "agendar")

    needs_action = any(token in text for token in ("responder", "confirmar", "agendar", "pagar", "revisar"))
    is_important = theme in {"CARREIRA", "PROJETOS_TECH", "FINANCEIRO"} or needs_action
    if any(token in text for token in high_signals):
        priority = "ALTA"
        time_sensitivity = "ALTA"
    elif any(token in text for token in medium_signals) or is_important:
        priority = "MEDIA"
        time_sensitivity = "MEDIA"
    else:
        priority = "BAIXA"
        time_sensitivity = "BAIXA"

    if theme in {"NEWSLETTER", "SPAM"}:
        priority = "BAIXA"
        time_sensitivity = "BAIXA"
        needs_action = False
        is_important = False

    return _normalize_priority_result(
        {
            "priority_level": priority,
            "needs_action": needs_action,
            "is_important": is_important,
            "time_sensitivity": time_sensitivity,
            "confidence": 0.72 if theme != "EM_DUVIDA" else 0.38,
            "motivo_curto": "Prioridade inferida por sinais operacionais do conteúdo.",
            "evidence": [str(email_data.get("subject", "")).strip() or "Assunto ausente."],
            "uncertainty_reason": "" if theme != "EM_DUVIDA" else "Tema ainda ambíguo.",
        }
    )


def _openai_client() -> OpenAI | None:
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key)


def _gemini_client() -> Any | None:
    import os

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if genai is None or not api_key:
        return None
    return genai.Client(api_key=api_key)


def _anthropic_client() -> Anthropic | None:
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if Anthropic is None or not api_key:
        return None
    return Anthropic(api_key=api_key)


def _extract_openai_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return str(output_text).strip()

    output_items = getattr(response, "output", [])
    text_fragments: list[str] = []
    for item in output_items:
        for content in getattr(item, "content", []):
            text_value = getattr(content, "text", "")
            if text_value:
                text_fragments.append(str(text_value))
    return "\n".join(text_fragments).strip()


def _invoke_gemini_text(system_prompt: str, user_prompt: str, json_output: bool) -> str:
    client = _gemini_client()
    if client is None:
        raise LLMProviderError("Gemini indisponível: configure `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no .env.")
    if genai_types is None:
        raise LLMProviderError("SDK do Gemini indisponível: instale `google-genai` na virtualenv ativa.")

    config_kwargs: dict[str, Any] = {"system_instruction": system_prompt}
    if json_output:
        config_kwargs["response_mime_type"] = "application/json"

    config = genai_types.GenerateContentConfig(**config_kwargs)
    primary_model = get_gemini_model_name()
    fallback_model = get_gemini_fallback_model_name()

    try:
        response = client.models.generate_content(
            model=primary_model,
            contents=user_prompt,
            config=config,
        )
    except Exception as exc:
        if _is_quota_message(str(exc)) and fallback_model and fallback_model != primary_model:
            try:
                response = client.models.generate_content(
                    model=fallback_model,
                    contents=user_prompt,
                    config=config,
                )
            except Exception as fallback_exc:
                raise _wrap_provider_exception("Gemini", fallback_exc) from fallback_exc
        else:
            raise _wrap_provider_exception("Gemini", exc) from exc
    response_text = getattr(response, "text", "")
    if response_text:
        return str(response_text).strip()
    raise LLMProviderError("Gemini retornou uma resposta vazia.")


def _invoke_openai_text(system_prompt: str, user_prompt: str, json_output: bool) -> str:
    client = _openai_client()
    if client is None:
        raise LLMProviderError("OpenAI indisponível: configure `OPENAI_API_KEY` no .env e valide o billing da API.")

    text_config: dict[str, Any]
    normalized_user_prompt = user_prompt
    if json_output:
        text_config = {"format": {"type": "json_object"}}
        normalized_user_prompt = (
            "Responda em JSON valido seguindo estritamente as instrucoes.\n\n"
            f"{user_prompt}"
        )
    else:
        text_config = {"format": {"type": "text"}}

    try:
        response = client.responses.create(
            model=get_openai_model_name(),
            instructions=system_prompt,
            input=normalized_user_prompt,
            text=text_config,
        )
    except Exception as exc:
        raise _wrap_provider_exception("OpenAI", exc) from exc
    raw_output = _extract_openai_output_text(response)
    if not raw_output:
        raise LLMProviderError("OpenAI retornou uma resposta vazia.")
    return raw_output


def _invoke_anthropic_json(system_prompt: str, user_prompt: str) -> str:
    client = _anthropic_client()
    if client is None:
        raise LLMProviderError("Anthropic indisponível: configure `ANTHROPIC_API_KEY` no .env.")

    try:
        response = client.messages.create(
            model=get_anthropic_model_name(),
            max_tokens=500,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        raise _wrap_provider_exception("Anthropic", exc) from exc
    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    if not text_blocks:
        raise LLMProviderError("Anthropic retornou uma resposta vazia.")
    return "\n".join(text_blocks).strip()


def _invoke_llm_text(system_prompt: str, user_prompt: str, json_output: bool) -> str:
    provider = get_llm_provider()
    if provider == "heuristic":
        raise LLMProviderError("O provider `heuristic` nao usa chamada externa de LLM.")
    if provider == "gemini":
        return _invoke_gemini_text(system_prompt, user_prompt, json_output=json_output)
    if provider == "openai":
        return _invoke_openai_text(system_prompt, user_prompt, json_output=json_output)
    if provider == "anthropic":
        return _invoke_anthropic_json(system_prompt, user_prompt)
    raise LLMProviderError(f"LLM provider nao suportado: {provider}")


def invoke_llm_text(system_prompt: str, user_prompt: str, json_output: bool = False) -> str:
    """Public helper to reuse the configured LLM provider outside the email agent."""

    return _invoke_llm_text(system_prompt, user_prompt, json_output=json_output)


def _format_few_shots(few_shots: list[FewShotExample], focus: str) -> str:
    if not few_shots:
        return "Nenhum few-shot disponível na memória semântica ainda."

    blocks: list[str] = []
    for index, example in enumerate(few_shots, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Exemplo {index} ({example['sample_role']})",
                    f"Tema: {example['theme_category']}",
                    f"Prioridade: {example['priority_level']}",
                    f"Label final: {example['final_label']}",
                    f"Resumo: {example['document']}",
                    f"Foco para este prompt: {focus}",
                ]
            )
        )
    return "\n\n".join(blocks)


def classify_email_with_llm(
    email_data: dict[str, Any],
    rag_context: str,
    theme_few_shots: list[FewShotExample],
    priority_few_shots: list[FewShotExample],
) -> dict[str, Any]:
    override_result = _sender_based_classification_override(email_data)
    if override_result is not None:
        return override_result

    if get_llm_provider() == "heuristic":
        theme_result = _heuristic_theme_classifier(email_data, rag_context)
        priority_result = _heuristic_priority_classifier(email_data, theme_result)
        return _compose_final_classification(theme_result, priority_result)

    prompt = f"""
Few-shots para decisão de TEMA:
{_format_few_shots(theme_few_shots, focus="determinar o tema do e-mail sem copiar exemplos irrelevantes")}

Few-shots para decisão de PRIORIDADE:
{_format_few_shots(priority_few_shots, focus="determinar prioridade e natureza operacional do e-mail")}

Contexto semântico recuperado:
{rag_context or "Sem contexto semântico relevante."}

E-mail para classificar:
Remetente: {email_data.get("sender", "")}
Assunto: {email_data.get("subject", "")}
Corpo: {email_data.get("body", "")}
""".strip()

    raw_output = _invoke_llm_text(EMAIL_CLASSIFICATION_PROMPT, prompt, json_output=True)
    try:
        parsed = _safe_json_loads(raw_output)
    except Exception as exc:
        raise LLMProviderError("O provider LLM retornou um payload inválido para classificação consolidada.") from exc
    return _normalize_combined_result(parsed)


def build_factual_memory_summary(email_data: dict[str, Any], classification_result: dict[str, Any]) -> str:
    """Build a deterministic factual summary for semantic memory."""

    sender = str(email_data.get("sender", "")).strip() or "Remetente desconhecido"
    subject = str(email_data.get("subject", "")).strip() or "Sem assunto"
    theme = str(classification_result.get("theme_category", "EM_DUVIDA")).strip() or "EM_DUVIDA"
    priority = str(classification_result.get("priority_level", "BAIXA")).strip() or "BAIXA"
    final_label = str(classification_result.get("final_label", "EM_DUVIDA")).strip() or "EM_DUVIDA"
    needs_action = "sim" if bool(classification_result.get("needs_action", False)) else "não"
    body = " ".join(str(email_data.get("body", "")).split())
    body_excerpt = body[:280].strip()
    if len(body) > 280:
        body_excerpt = f"{body_excerpt}..."

    return (
        f"E-mail de {sender} com assunto '{subject}', classificado como {final_label} "
        f"(tema {theme}, prioridade {priority}, requer ação: {needs_action}). "
        f"Trecho factual: {body_excerpt or 'Sem corpo disponível.'}"
    )


def summarize_memory_email(email_data: dict[str, Any], classification_result: dict[str, Any]) -> str:
    prompt = f"""
Remetente: {email_data.get("sender", "")}
Assunto: {email_data.get("subject", "")}
Corpo: {email_data.get("body", "")}
Tema inferido: {classification_result.get("theme_category", "EM_DUVIDA")}
Prioridade inferida: {classification_result.get("priority_level", "BAIXA")}
Label final: {classification_result.get("final_label", "EM_DUVIDA")}
Necessita ação: {classification_result.get("needs_action", False)}
""".strip()

    if get_llm_provider() == "heuristic":
        return build_factual_memory_summary(email_data, classification_result)

    try:
        return _invoke_llm_text(SUMMARY_PROMPT, prompt, json_output=False)
    except LLMProviderError:
        return build_factual_memory_summary(email_data, classification_result)
