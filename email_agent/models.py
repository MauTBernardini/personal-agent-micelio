"""Shared models and static prompts for the email triage agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, TypedDict


THEME_CATEGORIES: Final[tuple[str, ...]] = (
    "CARREIRA",
    "PROJETOS_TECH",
    "CRIATIVO_MAKER",
    "CURSOS_APRENDIZADO",
    "FINANCEIRO",
    "NEWSLETTER",
    "SPAM",
    "OUTROS",
    "EM_DUVIDA",
)

PRIORITY_LEVELS: Final[tuple[str, ...]] = ("BAIXA", "MEDIA", "ALTA")

FINAL_LABELS: Final[tuple[str, ...]] = tuple(
    [f"{priority}_{theme}" for priority in PRIORITY_LEVELS for theme in THEME_CATEGORIES if theme != "EM_DUVIDA"]
    + ["EM_DUVIDA"]
)

MANUAL_REVIEW_CATEGORIES: Final[tuple[str, ...]] = THEME_CATEGORIES
MANUAL_REVIEW_PRIORITY_LEVELS: Final[tuple[str, ...]] = PRIORITY_LEVELS

EMAIL_CLASSIFICATION_PROMPT: Final[str] = """
Você é um classificador rigoroso de e-mails que deve decidir, em uma única passada, o tema e a prioridade operacional.

Temas permitidos:
- CARREIRA
- PROJETOS_TECH
- CRIATIVO_MAKER
- CURSOS_APRENDIZADO
- FINANCEIRO
- NEWSLETTER
- SPAM
- OUTROS
- EM_DUVIDA

Prioridades permitidas:
- BAIXA
- MEDIA
- ALTA

Critérios operacionais:
- ALTA: exige atenção prioritária, ação relevante ou sensibilidade temporal significativa.
- MEDIA: relevante, mas sem urgência alta.
- BAIXA: informativo, operacional leve ou sem necessidade clara de ação.

Instruções:
1. Use raciocínio interno, mas NÃO exponha cadeia de pensamento completa.
2. Analise tema e prioridade separadamente, ainda que na mesma resposta.
3. Use os few-shots como referência, inclusive o exemplo negativo, para evitar analogias fáceis e alucinações.
4. Baseie a decisão apenas no conteúdo do e-mail e nos few-shots fornecidos.
5. Se houver ambiguidade real sobre o tema, escolha EM_DUVIDA.
6. Responda APENAS com JSON válido.
7. O JSON deve conter exatamente:
   - "theme_category": string
   - "theme_confidence": número entre 0 e 1
   - "theme_reason": string curta em português
   - "priority_level": BAIXA, MEDIA ou ALTA
   - "priority_confidence": número entre 0 e 1
   - "priority_reason": string curta em português
   - "needs_action": boolean
   - "is_important": boolean
   - "time_sensitivity": BAIXA, MEDIA ou ALTA
   - "evidence": lista de 2 ou 3 evidências curtas ancoradas no e-mail
   - "uncertainty_reason": string curta ou vazia
8. Regra de ouro: NUNCA marcar o e-mail como lido na origem.
""".strip()

SUMMARY_PROMPT: Final[str] = """
Você receberá o conteúdo de um e-mail já classificado.
Produza um resumo factual curto, em 1 parágrafo, em português do Brasil, sem inventar nada.
Inclua o tema e o nível de prioridade inferidos quando eles estiverem presentes no contexto.
""".strip()


class FewShotExample(TypedDict):
    """Compact few-shot example retrieved from semantic memory."""

    document: str
    theme_category: str
    priority_level: str
    final_label: str
    sample_role: str
    distance: float


class EmailAgentState(TypedDict):
    """LangGraph state shared between workflow nodes."""

    email_data: dict[str, Any]
    rag_context: str
    theme_few_shots: list[FewShotExample]
    priority_few_shots: list[FewShotExample]
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
