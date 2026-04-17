"""LLM provider abstraction and classification helpers."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from email_agent.models import ALLOWED_CATEGORIES, SUMMARY_PROMPT, SYSTEM_PROMPT
from email_agent.settings import (
    Anthropic,
    OpenAI,
    genai,
    genai_types,
    get_anthropic_model_name,
    get_gemini_model_name,
    get_llm_provider,
    get_openai_model_name,
)


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot satisfy a request."""


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(raw_text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _normalize_classification(payload: dict[str, Any]) -> dict[str, Any]:
    category = str(payload.get("categoria", "EM_DUVIDA")).strip().upper()
    if category not in ALLOWED_CATEGORIES:
        category = "EM_DUVIDA"

    priority_categories = {"CARREIRA_PRIORIDADE", "PROJETOS_TECH"}
    is_priority = category in priority_categories

    motivo = str(payload.get("motivo_curto", "Classificacao ajustada pelo validador local.")).strip()
    if not motivo:
        motivo = "Classificacao ajustada pelo validador local."

    return {"categoria": category, "motivo_curto": motivo, "is_priority": is_priority}


def _heuristic_classifier(email_data: dict[str, Any], rag_context: str) -> dict[str, Any]:
    text = f"{email_data.get('subject', '')} {email_data.get('body', '')} {rag_context}".lower()

    if any(token in text for token in ("entrevista", "vaga", "recrut", "remote", "remota")):
        category = "CARREIRA_PRIORIDADE"
        reason = "Conteudo de recrutamento ou entrevista de trabalho."
    elif any(token in text for token in ("gcp", "postgresql", "arquitetura", "microsserv", "agentic")):
        category = "PROJETOS_TECH"
        reason = "Assunto tecnico alinhado a projetos de engenharia."
    elif any(token in text for token in ("bambu lab", "petg", "impressora 3d", "boardgame", "maker")):
        category = "CRIATIVO_E_MAKER"
        reason = "Tema maker, hobby tecnico ou criativo."
    elif any(token in text for token in ("curso", "turma", "inscri", "aprenda", "workshop")):
        category = "CURSOS_E_APRENDIZADO"
        reason = "Convite ou conteudo educacional."
    elif any(token in text for token in ("fatura", "cartao", "banco", "vencimento", "boleto")):
        category = "PESSOAL_FINANCEIRO"
        reason = "Mensagem de contexto financeiro pessoal."
    elif any(token in text for token in ("newsletter", "resumo diario", "curadoria", "digest")):
        category = "NEWSLETTER_INFORMATIVO"
        reason = "Conteudo recorrente de newsletter."
    elif any(token in text for token in ("ganhou", "gratis", "clique agora", "prêmio", "premio")):
        category = "SPAM_LIXO"
        reason = "Padrao tipico de spam promocional."
    elif "colaboração" in text or "colaboracao" in text:
        category = "EM_DUVIDA"
        reason = "Pedido potencialmente relevante, mas ambiguo."
    else:
        category = "OUTROS"
        reason = "Nao se encaixa claramente nas categorias principais."

    return _normalize_classification(
        {"categoria": category, "motivo_curto": reason, "is_priority": category in {"CARREIRA_PRIORIDADE", "PROJETOS_TECH"}}
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

    response = client.models.generate_content(
        model=get_gemini_model_name(),
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(**config_kwargs),
    )
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

    response = client.responses.create(
        model=get_openai_model_name(),
        instructions=system_prompt,
        input=normalized_user_prompt,
        text=text_config,
    )
    raw_output = _extract_openai_output_text(response)
    if not raw_output:
        raise LLMProviderError("OpenAI retornou uma resposta vazia.")
    return raw_output


def _invoke_anthropic_json(system_prompt: str, user_prompt: str) -> str:
    client = _anthropic_client()
    if client is None:
        raise LLMProviderError("Anthropic indisponível: configure `ANTHROPIC_API_KEY` no .env.")

    response = client.messages.create(
        model=get_anthropic_model_name(),
        max_tokens=300,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
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


def classify_email_with_llm(email_data: dict[str, Any], rag_context: str) -> dict[str, Any]:
    prompt = f"""
Contexto semantico recuperado:
{rag_context or "Sem contexto previo relevante."}

E-mail para classificar:
Remetente: {email_data.get("sender", "")}
Assunto: {email_data.get("subject", "")}
Corpo: {email_data.get("body", "")}
""".strip()

    if get_llm_provider() == "heuristic":
        return _heuristic_classifier(email_data, rag_context)

    raw_output = _invoke_llm_text(SYSTEM_PROMPT, prompt, json_output=True)
    try:
        parsed = _safe_json_loads(raw_output)
    except Exception as exc:
        raise LLMProviderError(
            "O provider LLM retornou um payload invalido para classificacao JSON."
        ) from exc
    return _normalize_classification(parsed)


def summarize_priority_email(email_data: dict[str, Any]) -> str:
    prompt = f"""
Remetente: {email_data.get("sender", "")}
Assunto: {email_data.get("subject", "")}
Corpo: {email_data.get("body", "")}
""".strip()

    if get_llm_provider() == "heuristic":
        return (
            f"Resumo factual: e-mail de '{email_data.get('sender', '')}' com assunto "
            f"'{email_data.get('subject', '')}', tratado como item prioritario para acompanhamento."
        )

    return _invoke_llm_text(SUMMARY_PROMPT, prompt, json_output=False)

