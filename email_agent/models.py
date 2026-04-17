"""Shared models and static prompts for the email triage agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, TypedDict


ALLOWED_CATEGORIES: Final[tuple[str, ...]] = (
    "CARREIRA_PRIORIDADE",
    "PROJETOS_TECH",
    "CRIATIVO_E_MAKER",
    "CURSOS_E_APRENDIZADO",
    "PESSOAL_FINANCEIRO",
    "NEWSLETTER_INFORMATIVO",
    "SPAM_LIXO",
    "OUTROS",
    "EM_DUVIDA",
)

MANUAL_REVIEW_CATEGORIES: Final[tuple[str, ...]] = (
    "CARREIRA_PRIORIDADE",
    "PROJETOS_TECH",
    "CURSOS_E_APRENDIZADO",
    "OUTROS",
)

SYSTEM_PROMPT: Final[str] = """
Você é um agente de triagem de e-mails pessoal extremamente rigoroso.

Sua tarefa é classificar um único e-mail em APENAS uma das categorias abaixo:
- CARREIRA_PRIORIDADE -> is_priority: true
- PROJETOS_TECH -> is_priority: true
- CRIATIVO_E_MAKER -> is_priority: false
- CURSOS_E_APRENDIZADO -> is_priority: false
- PESSOAL_FINANCEIRO -> is_priority: false
- NEWSLETTER_INFORMATIVO -> is_priority: false
- SPAM_LIXO -> is_priority: false
- OUTROS -> is_priority: false
- EM_DUVIDA -> is_priority: false

Regras obrigatórias:
1. Responda APENAS com JSON válido.
2. O JSON DEVE conter exatamente as chaves: "categoria", "motivo_curto", "is_priority".
3. "categoria" DEVE ser uma das categorias permitidas.
4. "motivo_curto" deve ser objetivo e curto, em português do Brasil.
5. "is_priority" deve respeitar o mapeamento obrigatório acima.
6. Se houver ambiguidade real, use "EM_DUVIDA".
7. Regra de ouro: NUNCA marcar o e-mail como lido na origem.
""".strip()

SUMMARY_PROMPT: Final[str] = """
Você receberá o conteúdo de um e-mail.
Produza um resumo factual em 1 parágrafo, em português do Brasil, sem inventar nada.
""".strip()


class EmailAgentState(TypedDict):
    """LangGraph state shared between workflow nodes."""

    email_data: dict[str, Any]
    rag_context: str
    classification_result: dict[str, Any]


@dataclass(frozen=True)
class MockEmail:
    """Local representation of a Gmail message used by the stubs."""

    message_id: str
    sender: str
    subject: str
    body: str
    labels: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "labels": list(self.labels),
        }

