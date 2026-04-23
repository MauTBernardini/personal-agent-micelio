"""Service layer for the Antese writing agent."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from email_agent.llm import LLMProviderError, invoke_llm_text
from email_agent.settings import get_llm_provider
from email_agent.storage import (
    get_antese_feedback_history,
    get_antese_execution_versions,
    get_antese_executions,
    get_antese_genre_cards,
    get_antese_inspiration_profiles,
    get_antese_style_profiles,
    query_writing_feedback_memory,
    query_writing_runs_memory,
    query_writing_samples,
    save_antese_execution,
    save_antese_feedback,
    save_antese_genre_card,
    save_antese_sample,
    save_antese_style_profile,
    save_antese_version,
)
from writing_agent.models import (
    ANTese_SYSTEM_PROMPT,
    CRITIQUE_PROMPT,
    DRAFT_PROMPT,
    NORMALIZE_BRIEF_PROMPT,
    PLAN_PROMPT,
    REWRITE_PROMPT,
    STYLE_REASON_TAGS,
    WRITING_GOALS,
    WRITING_TASK_TYPES,
)


def get_antese_capabilities() -> dict[str, Any]:
    """Expose the Antese writing modes used by the supervisor and frontend."""

    return {
        "agent_name": "ANTESE",
        "description": "Especialista em ideação, estruturação, escrita personalizada e revisão.",
        "task_types": list(WRITING_TASK_TYPES),
        "goals": list(WRITING_GOALS),
        "genre_cards": [row["genre_id"] for row in get_antese_genre_cards()],
    }


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_score(value: Any, default: float = 0.7) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric_value))


def _list_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _get_default_goal_for_task(task_type: str) -> str:
    mapping = {
        "BRAINSTORM": "EXPLORAR",
        "OUTLINE": "SINTETIZAR",
        "DRAFT": "INFORMAR",
        "CONSOLIDATE": "SINTETIZAR",
        "REWRITE": "SINTETIZAR",
    }
    return mapping.get(task_type, "EXPLORAR")


def _get_default_genre_for_task(task_type: str) -> str:
    mapping = {
        "BRAINSTORM": "brainstorm_notes",
        "OUTLINE": "outline",
        "DRAFT": "essay_article",
        "CONSOLIDATE": "consolidated_memo",
        "REWRITE": "rewrite_clarity",
    }
    return mapping.get(task_type, "brainstorm_notes")


def _format_style_profile(style_profile: dict[str, Any]) -> str:
    sections = [
        ("Voice traits", style_profile.get("voice_traits", {})),
        ("Structure traits", style_profile.get("structure_traits", {})),
        ("Rhetorical traits", style_profile.get("rhetorical_traits", {})),
        ("Lexical traits", style_profile.get("lexical_traits", {})),
    ]
    parts = [f"Style profile: {style_profile.get('name', 'N/D')} ({style_profile.get('owner_scope', 'N/D')})"]
    for title, payload in sections:
        traits = ", ".join(f"{key}: {value}" for key, value in payload.items())
        if traits:
            parts.append(f"{title}: {traits}")
    dos = _list_strings(style_profile.get("dos"))
    donts = _list_strings(style_profile.get("donts"))
    if dos:
        parts.append(f"Do: {'; '.join(dos)}")
    if donts:
        parts.append(f"Don't: {'; '.join(donts)}")
    return "\n".join(parts)


def _format_genre_card(genre_card: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Genre card: {genre_card.get('name', 'N/D')} ({genre_card.get('genre_id', 'N/D')})",
            f"Primary goal: {genre_card.get('primary_goal', 'N/D')}",
            f"Expected structure: {', '.join(_list_strings(genre_card.get('expected_structure')))}",
            f"Tone defaults: {', '.join(_list_strings(genre_card.get('tone_defaults')))}",
            f"Quality checklist: {', '.join(_list_strings(genre_card.get('quality_checklist')))}",
            f"Typical openings: {', '.join(_list_strings(genre_card.get('typical_openings')))}",
            f"Typical closings: {', '.join(_list_strings(genre_card.get('typical_closings')))}",
        ]
    )


def _format_inspiration_profile(inspiration_profile: dict[str, Any] | None) -> str:
    if not inspiration_profile:
        return "Sem inspiration profile adicional."
    return "\n".join(
        [
            f"Inspiration profile: {inspiration_profile.get('name', 'N/D')}",
            f"Borrow: {', '.join(_list_strings(inspiration_profile.get('traits_to_borrow')))}",
            f"Avoid: {', '.join(_list_strings(inspiration_profile.get('forbidden_behaviors')))}",
            f"Strength: {inspiration_profile.get('transformation_strength', 'media')}",
        ]
    )


def _format_retrieved_examples(
    sample_examples: list[dict[str, Any]],
    feedback_examples: list[dict[str, Any]],
    run_examples: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    if sample_examples:
        blocks.append("Exemplos de escrita recuperados:")
        for index, item in enumerate(sample_examples, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[Sample {index}] {item.get('title', 'Sem título')}",
                        f"Resumo: {item.get('editorial_summary', '')}",
                        f"Trecho: {str(item.get('document', ''))[:360]}",
                    ]
                )
            )
    if feedback_examples:
        blocks.append("Feedbacks recuperados:")
        for index, item in enumerate(feedback_examples, start=1):
            blocks.append(
                f"[Feedback {index}] Tags: {', '.join(_list_strings(item.get('reason_tags')))} | "
                f"Notas: {str(item.get('document', ''))[:220]}"
            )
    if run_examples:
        blocks.append("Execuções anteriores relacionadas:")
        for index, item in enumerate(run_examples, start=1):
            blocks.append(
                f"[Run {index}] task={item.get('task_type', '')} genre={item.get('genre_id', '')} "
                f"trecho={str(item.get('document', ''))[:220]}"
            )
    return "\n\n".join(blocks) if blocks else "Sem contexto recuperado ainda."


def _infer_style_profile_from_samples(
    sample_texts: list[str],
    profile_name: str,
    owner_scope: str,
) -> dict[str, Any]:
    joined_samples = "\n\n---\n\n".join(text.strip() for text in sample_texts if text.strip())
    if not joined_samples:
        return {
            "profile_id": uuid.uuid4().hex[:16],
            "name": profile_name,
            "owner_scope": owner_scope,
            "voice_traits": {},
            "structure_traits": {},
            "rhetorical_traits": {},
            "lexical_traits": {},
            "dos": [],
            "donts": [],
            "sample_text_ids": [],
        }

    if get_llm_provider() == "heuristic":
        return {
            "profile_id": uuid.uuid4().hex[:16],
            "name": profile_name,
            "owner_scope": owner_scope,
            "voice_traits": {
                "formalidade": "equilibrada",
                "assertividade": "alta",
                "concretude": "alta",
            },
            "structure_traits": {
                "abertura": "direta",
                "transicoes": "claras",
            },
            "rhetorical_traits": {
                "exemplos": "frequentes",
                "contraste": "moderado",
            },
            "lexical_traits": {
                "simplicidade_vocabular": "media-alta",
                "preferencia_verbal": "verbos fortes",
            },
            "dos": ["Manter clareza e progressao logica."],
            "donts": ["Nao inflar o texto sem necessidade."],
            "sample_text_ids": [],
        }

    prompt = f"""
Extraia um style profile a partir destes textos do usuário.

Nome do profile: {profile_name}
Escopo: {owner_scope}

Textos:
{joined_samples}

Responda APENAS em JSON com:
- voice_traits
- structure_traits
- rhetorical_traits
- lexical_traits
- dos
- donts
""".strip()
    raw_output = invoke_llm_text(ANTese_SYSTEM_PROMPT, prompt, json_output=True)
    parsed = _safe_json_loads(raw_output)
    return {
        "profile_id": uuid.uuid4().hex[:16],
        "name": profile_name,
        "owner_scope": owner_scope,
        "voice_traits": parsed.get("voice_traits", {}),
        "structure_traits": parsed.get("structure_traits", {}),
        "rhetorical_traits": parsed.get("rhetorical_traits", {}),
        "lexical_traits": parsed.get("lexical_traits", {}),
        "dos": _list_strings(parsed.get("dos")),
        "donts": _list_strings(parsed.get("donts")),
        "sample_text_ids": [],
    }


def get_antese_style_profiles_catalog() -> list[dict[str, Any]]:
    """Return all persisted style profiles."""

    return get_antese_style_profiles()


def create_or_update_antese_style_profile(
    profile_id: str | None,
    name: str,
    owner_scope: str,
    voice_traits: dict[str, Any] | None = None,
    structure_traits: dict[str, Any] | None = None,
    rhetorical_traits: dict[str, Any] | None = None,
    lexical_traits: dict[str, Any] | None = None,
    dos: list[str] | None = None,
    donts: list[str] | None = None,
    sample_text_ids: list[str] | None = None,
    sample_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Create or update one Antese style profile, optionally inferred from texts."""

    normalized_profile_id = (profile_id or name.lower().replace(" ", "_")).strip() or uuid.uuid4().hex[:16]
    stored_sample_ids = sample_text_ids or []
    if sample_texts:
        inferred_profile = _infer_style_profile_from_samples(sample_texts, name, owner_scope)
        voice_traits = voice_traits or inferred_profile["voice_traits"]
        structure_traits = structure_traits or inferred_profile["structure_traits"]
        rhetorical_traits = rhetorical_traits or inferred_profile["rhetorical_traits"]
        lexical_traits = lexical_traits or inferred_profile["lexical_traits"]
        dos = dos or inferred_profile["dos"]
        donts = donts or inferred_profile["donts"]
        for index, sample_text in enumerate(sample_texts, start=1):
            normalized_sample_text = sample_text.strip()
            if not normalized_sample_text:
                continue
            sample_id = save_antese_sample(
                title=f"{name} sample {index}",
                source_scope=owner_scope,
                text_content=normalized_sample_text,
                editorial_summary=normalized_sample_text[:240],
                style_profile_id=normalized_profile_id,
                genre_id=None,
                metadata={"profile_id": normalized_profile_id, "profile_name": name},
            )
            stored_sample_ids.append(sample_id)

    return save_antese_style_profile(
        profile_id=normalized_profile_id,
        name=name,
        owner_scope=owner_scope,
        voice_traits=voice_traits or {},
        structure_traits=structure_traits or {},
        rhetorical_traits=rhetorical_traits or {},
        lexical_traits=lexical_traits or {},
        dos=dos or [],
        donts=donts or [],
        sample_text_ids=stored_sample_ids,
    )


def get_antese_genre_card_catalog() -> list[dict[str, Any]]:
    """Return all available genre cards."""

    return get_antese_genre_cards()


def create_or_update_antese_genre_card(
    genre_id: str,
    name: str,
    primary_goal: str,
    expected_structure: list[str],
    tone_defaults: list[str],
    length_defaults: dict[str, Any],
    quality_checklist: list[str],
    typical_openings: list[str],
    typical_closings: list[str],
) -> dict[str, Any]:
    """Create or update a genre card."""

    return save_antese_genre_card(
        genre_id=genre_id,
        name=name,
        primary_goal=primary_goal,
        expected_structure=expected_structure,
        tone_defaults=tone_defaults,
        length_defaults=length_defaults,
        quality_checklist=quality_checklist,
        typical_openings=typical_openings,
        typical_closings=typical_closings,
    )


def get_antese_inspiration_profile_catalog() -> list[dict[str, Any]]:
    """Return all inspiration profiles."""

    return get_antese_inspiration_profiles()


def _resolve_style_profile(style_profile_id: str | None) -> dict[str, Any]:
    profiles = get_antese_style_profiles()
    if style_profile_id:
        for item in profiles:
            if item["profile_id"] == style_profile_id:
                return item
    for item in profiles:
        if item["profile_id"] == "personal_default":
            return item
    return profiles[0]


def _resolve_genre_card(genre_id: str | None, task_type: str) -> dict[str, Any]:
    cards = get_antese_genre_cards()
    normalized_genre_id = (genre_id or "").strip()
    if normalized_genre_id:
        for item in cards:
            if item["genre_id"] == normalized_genre_id:
                return item
    default_genre_id = _get_default_genre_for_task(task_type)
    for item in cards:
        if item["genre_id"] == default_genre_id:
            return item
    return cards[0]


def _resolve_inspiration_profile(inspiration_profile_id: str | None) -> dict[str, Any] | None:
    if not inspiration_profile_id:
        return None
    for item in get_antese_inspiration_profiles():
        if item["inspiration_profile_id"] == inspiration_profile_id:
            return item
    return None


def _heuristic_normalized_brief(
    task_type: str,
    user_request: str,
    source_notes: str,
    goal: str,
    genre_id: str,
    audience: str,
    tone_override: str,
    must_include: list[str],
    must_avoid: list[str],
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "goal": goal,
        "audience": audience,
        "tone": tone_override,
        "output_format": genre_id,
        "genre_id": genre_id,
        "source_notes_summary": (source_notes.strip() or user_request.strip())[:400],
        "must_include": must_include,
        "must_avoid": must_avoid,
        "key_constraints": ["Preservar fatos do material-base.", "Nao copiar exemplos literalmente."],
        "success_criteria": [
            "Texto claro e coeso.",
            "Aderencia ao gênero e ao tom.",
            "Cobertura dos pontos obrigatorios.",
        ],
    }


def _heuristic_plan(brief: dict[str, Any], genre_card: dict[str, Any]) -> str:
    sections = _list_strings(genre_card.get("expected_structure")) or ["abertura", "desenvolvimento", "fechamento"]
    lines = [f"# Outline para {brief.get('genre_id', 'texto')}", ""]
    for section in sections:
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append(f"- Desenvolver o bloco de {section} a serviço do objetivo {brief.get('goal', 'EXPLORAR')}.")
        lines.append(f"- Manter o tom {brief.get('tone', 'claro e profissional')}.")
        if brief.get("must_include"):
            lines.append(f"- Considerar: {', '.join(_list_strings(brief.get('must_include'))[:2])}.")
        lines.append("")
    return "\n".join(lines).strip()


def _heuristic_draft(
    user_request: str,
    brief: dict[str, Any],
    outline_text: str,
    source_notes: str,
    style_profile: dict[str, Any],
    inspiration_profile: dict[str, Any] | None,
) -> str:
    style_hint = ", ".join(
        [
            f"formalidade {style_profile.get('voice_traits', {}).get('formalidade', 'equilibrada')}",
            f"assertividade {style_profile.get('voice_traits', {}).get('assertividade', 'media')}",
            f"concretude {style_profile.get('voice_traits', {}).get('concretude', 'media')}",
        ]
    )
    inspiration_hint = ""
    if inspiration_profile:
        inspiration_hint = (
            f" Com inspiração {inspiration_profile.get('name', '')}, "
            f"priorize {', '.join(_list_strings(inspiration_profile.get('traits_to_borrow'))[:2])}."
        )
    body = source_notes.strip() or user_request.strip()
    return (
        f"{user_request.strip()}\n\n"
        f"Objetivo editorial: {brief.get('goal', 'EXPLORAR')} para {brief.get('audience', 'leitores gerais')}."
        f" O tom deve ser {brief.get('tone', 'claro e profissional')} com {style_hint}.{inspiration_hint}\n\n"
        f"Material-base consolidado:\n{body}\n\n"
        f"Estrutura sugerida:\n{outline_text}"
    ).strip()


def _heuristic_quality_report(
    brief: dict[str, Any],
    output_text: str,
    source_notes: str,
) -> dict[str, Any]:
    faithfulness = 0.85 if source_notes.strip() else 0.7
    creativity = 0.72 if len(output_text.split()) > 80 else 0.6
    return {
        "overall_score": round((0.82 + creativity + faithfulness) / 3, 3),
        "style_match_score": 0.78,
        "usefulness_score": 0.82,
        "creativity_score": round(creativity, 3),
        "faithfulness_score": round(faithfulness, 3),
        "strengths": [
            "Estrutura clara para seguir iterando.",
            "Mantem foco no objetivo principal.",
        ],
        "weaknesses": [
            "Pode ganhar mais acabamento estilístico.",
        ],
        "revision_actions": [
            "Tornar a abertura mais forte.",
            "Aproximar mais a voz do profile ativo.",
        ],
    }


def _heuristic_rewrite(draft_text: str, quality_report: dict[str, Any], tone_override: str) -> str:
    revision_actions = "; ".join(_list_strings(quality_report.get("revision_actions")))
    return (
        f"{draft_text}\n\n"
        f"[Versão refinada para o tom {tone_override}. Ajustes editoriais priorizados: {revision_actions}]"
    ).strip()


def _normalize_brief_with_llm(
    task_type: str,
    user_request: str,
    source_notes: str,
    goal: str,
    genre_id: str,
    audience: str,
    tone_override: str,
    must_include: list[str],
    must_avoid: list[str],
    reference_text: str,
) -> dict[str, Any]:
    if get_llm_provider() == "heuristic":
        return _heuristic_normalized_brief(
            task_type,
            user_request,
            source_notes,
            goal,
            genre_id,
            audience,
            tone_override,
            must_include,
            must_avoid,
        )

    prompt = f"""
Task type: {task_type}
Goal desejado: {goal}
Genre desejado: {genre_id}
Pedido do usuário:
{user_request}

Source notes:
{source_notes or "Sem notas adicionais."}

Reference text:
{reference_text or "Sem texto de referência."}

Audience:
{audience}

Tone override:
{tone_override}

Must include:
{json.dumps(must_include, ensure_ascii=False)}

Must avoid:
{json.dumps(must_avoid, ensure_ascii=False)}
""".strip()
    raw_output = invoke_llm_text(ANTese_SYSTEM_PROMPT + "\n\n" + NORMALIZE_BRIEF_PROMPT, prompt, json_output=True)
    parsed = _safe_json_loads(raw_output)
    return {
        "task_type": str(parsed.get("task_type", task_type)).strip().upper(),
        "goal": str(parsed.get("goal", goal)).strip().upper(),
        "audience": str(parsed.get("audience", audience)).strip() or audience,
        "tone": str(parsed.get("tone", tone_override)).strip() or tone_override,
        "output_format": str(parsed.get("output_format", genre_id)).strip() or genre_id,
        "genre_id": str(parsed.get("genre_id", genre_id)).strip() or genre_id,
        "source_notes_summary": str(parsed.get("source_notes_summary", source_notes[:400])).strip(),
        "must_include": _list_strings(parsed.get("must_include")) or must_include,
        "must_avoid": _list_strings(parsed.get("must_avoid")) or must_avoid,
        "key_constraints": _list_strings(parsed.get("key_constraints")) or ["Preservar fatos do usuário."],
        "success_criteria": _list_strings(parsed.get("success_criteria")) or ["Entregar texto útil e claro."],
    }


def _plan_with_llm(
    normalized_brief: dict[str, Any],
    style_profile: dict[str, Any],
    genre_card: dict[str, Any],
    inspiration_profile: dict[str, Any] | None,
    retrieved_context: str,
) -> str:
    if get_llm_provider() == "heuristic":
        return _heuristic_plan(normalized_brief, genre_card)

    prompt = f"""
Brief canônico:
{json.dumps(normalized_brief, ensure_ascii=False, indent=2)}

{_format_style_profile(style_profile)}

{_format_genre_card(genre_card)}

{_format_inspiration_profile(inspiration_profile)}

Contexto recuperado:
{retrieved_context}
""".strip()
    return invoke_llm_text(ANTese_SYSTEM_PROMPT + "\n\n" + PLAN_PROMPT, prompt, json_output=False)


def _draft_with_llm(
    user_request: str,
    normalized_brief: dict[str, Any],
    style_profile: dict[str, Any],
    genre_card: dict[str, Any],
    inspiration_profile: dict[str, Any] | None,
    retrieved_context: str,
    outline_text: str,
    source_notes: str,
    reference_text: str,
) -> str:
    if get_llm_provider() == "heuristic":
        return _heuristic_draft(
            user_request=user_request,
            brief=normalized_brief,
            outline_text=outline_text,
            source_notes=source_notes,
            style_profile=style_profile,
            inspiration_profile=inspiration_profile,
        )

    prompt = f"""
Pedido original:
{user_request}

Brief canônico:
{json.dumps(normalized_brief, ensure_ascii=False, indent=2)}

{_format_style_profile(style_profile)}

{_format_genre_card(genre_card)}

{_format_inspiration_profile(inspiration_profile)}

Outline:
{outline_text}

Source notes:
{source_notes or "Sem notas adicionais."}

Reference text:
{reference_text or "Sem texto de referência."}

Contexto recuperado:
{retrieved_context}
""".strip()
    return invoke_llm_text(ANTese_SYSTEM_PROMPT + "\n\n" + DRAFT_PROMPT, prompt, json_output=False)


def _critique_with_llm(
    normalized_brief: dict[str, Any],
    style_profile: dict[str, Any],
    genre_card: dict[str, Any],
    draft_text: str,
) -> dict[str, Any]:
    if get_llm_provider() == "heuristic":
        return _heuristic_quality_report(normalized_brief, draft_text, normalized_brief.get("source_notes_summary", ""))

    prompt = f"""
Brief canônico:
{json.dumps(normalized_brief, ensure_ascii=False, indent=2)}

{_format_style_profile(style_profile)}

{_format_genre_card(genre_card)}

Texto para criticar:
{draft_text}
""".strip()
    raw_output = invoke_llm_text(ANTese_SYSTEM_PROMPT + "\n\n" + CRITIQUE_PROMPT, prompt, json_output=True)
    parsed = _safe_json_loads(raw_output)
    return {
        "overall_score": _coerce_score(parsed.get("overall_score"), 0.78),
        "style_match_score": _coerce_score(parsed.get("style_match_score"), 0.74),
        "usefulness_score": _coerce_score(parsed.get("usefulness_score"), 0.8),
        "creativity_score": _coerce_score(parsed.get("creativity_score"), 0.72),
        "faithfulness_score": _coerce_score(parsed.get("faithfulness_score"), 0.75),
        "strengths": _list_strings(parsed.get("strengths")),
        "weaknesses": _list_strings(parsed.get("weaknesses")),
        "revision_actions": _list_strings(parsed.get("revision_actions")),
    }


def _rewrite_with_llm(
    normalized_brief: dict[str, Any],
    style_profile: dict[str, Any],
    genre_card: dict[str, Any],
    inspiration_profile: dict[str, Any] | None,
    draft_text: str,
    quality_report: dict[str, Any],
) -> str:
    if get_llm_provider() == "heuristic":
        return _heuristic_rewrite(draft_text, quality_report, normalized_brief.get("tone", "claro e profissional"))

    prompt = f"""
Brief canônico:
{json.dumps(normalized_brief, ensure_ascii=False, indent=2)}

{_format_style_profile(style_profile)}

{_format_genre_card(genre_card)}

{_format_inspiration_profile(inspiration_profile)}

Texto atual:
{draft_text}

Crítica estruturada:
{json.dumps(quality_report, ensure_ascii=False, indent=2)}
""".strip()
    return invoke_llm_text(ANTese_SYSTEM_PROMPT + "\n\n" + REWRITE_PROMPT, prompt, json_output=False)


def _should_use_premium_pipeline(task_type: str, genre_id: str) -> bool:
    if task_type in {"CONSOLIDATE", "REWRITE"}:
        return True
    return genre_id in {"linkedin_post", "essay_article", "newsletter"}


def run_antese(
    user_request: str,
    task_type: str = "BRAINSTORM",
    context_notes: str = "",
    tone: str = "claro e profissional",
    audience: str = "leitores gerais",
    source_notes: str | None = None,
    goal: str | None = None,
    genre_id: str | None = None,
    style_profile_id: str | None = None,
    inspiration_profile_id: str | None = None,
    tone_override: str | None = None,
    must_include: list[str] | None = None,
    must_avoid: list[str] | None = None,
    reference_text: str = "",
    action_label: str = "default",
) -> dict[str, Any]:
    """Execute a personalized writing task using the configured provider."""

    normalized_task_type = task_type.strip().upper()
    if normalized_task_type not in WRITING_TASK_TYPES:
        raise ValueError(f"Tipo de tarefa de escrita inválido: {task_type}")

    normalized_goal = (goal or _get_default_goal_for_task(normalized_task_type)).strip().upper()
    if normalized_goal not in WRITING_GOALS:
        normalized_goal = _get_default_goal_for_task(normalized_task_type)

    source_notes_value = source_notes if source_notes is not None else context_notes
    tone_value = tone_override or tone
    must_include_value = must_include or []
    must_avoid_value = must_avoid or []

    resolved_style_profile = _resolve_style_profile(style_profile_id or "personal_default")
    resolved_genre_card = _resolve_genre_card(genre_id, normalized_task_type)
    resolved_inspiration_profile = _resolve_inspiration_profile(inspiration_profile_id)

    normalized_brief = _normalize_brief_with_llm(
        task_type=normalized_task_type,
        user_request=user_request,
        source_notes=source_notes_value,
        goal=normalized_goal,
        genre_id=resolved_genre_card["genre_id"],
        audience=audience,
        tone_override=tone_value,
        must_include=must_include_value,
        must_avoid=must_avoid_value,
        reference_text=reference_text,
    )

    retrieval_query = "\n".join(
        [
            user_request,
            source_notes_value or "",
            reference_text,
            normalized_brief.get("source_notes_summary", ""),
        ]
    ).strip()
    sample_examples = query_writing_samples(
        query_text=retrieval_query,
        style_profile_id=resolved_style_profile.get("profile_id"),
        genre_id=resolved_genre_card.get("genre_id"),
        n_results=5,
    )
    feedback_examples = query_writing_feedback_memory(retrieval_query, n_results=3)
    run_examples = query_writing_runs_memory(retrieval_query, n_results=3)
    retrieved_context = _format_retrieved_examples(sample_examples, feedback_examples, run_examples)

    try:
        outline_text = _plan_with_llm(
            normalized_brief=normalized_brief,
            style_profile=resolved_style_profile,
            genre_card=resolved_genre_card,
            inspiration_profile=resolved_inspiration_profile,
            retrieved_context=retrieved_context,
        )
        draft_text = _draft_with_llm(
            user_request=user_request,
            normalized_brief=normalized_brief,
            style_profile=resolved_style_profile,
            genre_card=resolved_genre_card,
            inspiration_profile=resolved_inspiration_profile,
            retrieved_context=retrieved_context,
            outline_text=outline_text,
            source_notes=source_notes_value,
            reference_text=reference_text,
        )
        quality_report = _critique_with_llm(
            normalized_brief=normalized_brief,
            style_profile=resolved_style_profile,
            genre_card=resolved_genre_card,
            draft_text=draft_text,
        )
        if _should_use_premium_pipeline(normalized_task_type, resolved_genre_card["genre_id"]):
            final_text = _rewrite_with_llm(
                normalized_brief=normalized_brief,
                style_profile=resolved_style_profile,
                genre_card=resolved_genre_card,
                inspiration_profile=resolved_inspiration_profile,
                draft_text=draft_text,
                quality_report=quality_report,
            )
        else:
            final_text = draft_text
    except LLMProviderError:
        outline_text = _heuristic_plan(normalized_brief, resolved_genre_card)
        draft_text = _heuristic_draft(
            user_request=user_request,
            brief=normalized_brief,
            outline_text=outline_text,
            source_notes=source_notes_value,
            style_profile=resolved_style_profile,
            inspiration_profile=resolved_inspiration_profile,
        )
        quality_report = _heuristic_quality_report(normalized_brief, draft_text, source_notes_value)
        final_text = (
            _heuristic_rewrite(draft_text, quality_report, tone_value)
            if _should_use_premium_pipeline(normalized_task_type, resolved_genre_card["genre_id"])
            else draft_text
        )

    retrieved_examples_preview = [
        {
            "kind": "sample",
            "title": item.get("title", ""),
            "distance": item.get("distance", 0.0),
            "style_profile_id": item.get("style_profile_id", ""),
            "genre_id": item.get("genre_id", ""),
            "summary": item.get("editorial_summary", ""),
        }
        for item in sample_examples[:5]
    ]
    retrieved_examples_preview.extend(
        {
            "kind": "feedback",
            "reason_tags": item.get("reason_tags", []),
            "distance": item.get("distance", 0.0),
            "summary": str(item.get("document", ""))[:180],
        }
        for item in feedback_examples[:2]
    )
    retrieved_examples_preview.extend(
        {
            "kind": "run",
            "task_type": item.get("task_type", ""),
            "genre_id": item.get("genre_id", ""),
            "distance": item.get("distance", 0.0),
            "summary": str(item.get("document", ""))[:180],
        }
        for item in run_examples[:2]
    )

    execution_id = save_antese_execution(
        user_request=user_request,
        task_type=normalized_task_type,
        goal=normalized_goal,
        genre_id=resolved_genre_card["genre_id"],
        style_profile_id=resolved_style_profile.get("profile_id"),
        inspiration_profile_id=resolved_inspiration_profile.get("inspiration_profile_id") if resolved_inspiration_profile else None,
        audience=audience,
        tone_override=tone_value,
        source_notes=source_notes_value,
        must_include=must_include_value,
        must_avoid=must_avoid_value,
        reference_text=reference_text,
        normalized_brief=normalized_brief,
        applied_style_profile=resolved_style_profile,
        applied_genre_card=resolved_genre_card,
        retrieved_examples_preview=retrieved_examples_preview,
        outline_text=outline_text,
        draft_text=draft_text,
        final_text=final_text,
        quality_report=quality_report,
    )
    outline_version_id = save_antese_version(
        execution_id=execution_id,
        stage="PLAN",
        action_label="outline",
        output_text=outline_text,
        metadata={"task_type": normalized_task_type, "genre_id": resolved_genre_card["genre_id"]},
    )
    draft_version_id = save_antese_version(
        execution_id=execution_id,
        stage="DRAFT",
        action_label=action_label if action_label != "default" else "draft",
        output_text=draft_text,
        metadata={"task_type": normalized_task_type, "genre_id": resolved_genre_card["genre_id"]},
    )
    final_version_id = save_antese_version(
        execution_id=execution_id,
        stage="FINAL",
        action_label=action_label,
        output_text=final_text,
        metadata={"quality_report": quality_report},
    )

    return {
        "agent_name": "ANTESE",
        "execution_id": execution_id,
        "version_id": final_version_id,
        "outline_version_id": outline_version_id,
        "draft_version_id": draft_version_id,
        "task_type": normalized_task_type,
        "goal": normalized_goal,
        "genre_id": resolved_genre_card["genre_id"],
        "style_profile_id": resolved_style_profile.get("profile_id"),
        "inspiration_profile_id": resolved_inspiration_profile.get("inspiration_profile_id") if resolved_inspiration_profile else None,
        "tone": tone_value,
        "audience": audience,
        "source_notes": source_notes_value,
        "must_include": must_include_value,
        "must_avoid": must_avoid_value,
        "reference_text": reference_text,
        "normalized_brief": normalized_brief,
        "applied_style_profile": resolved_style_profile,
        "applied_genre_card": resolved_genre_card,
        "applied_inspiration_profile": resolved_inspiration_profile,
        "retrieved_examples_preview": retrieved_examples_preview,
        "outline_text": outline_text,
        "draft_text": draft_text,
        "output_text": final_text,
        "quality_report": quality_report,
    }


def get_antese_execution_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return persisted Antese execution history."""

    return get_antese_executions(limit=limit)


def get_antese_versions(execution_id: str) -> list[dict[str, Any]]:
    """Return all stored versions for one Antese execution."""

    return get_antese_execution_versions(execution_id)


def submit_antese_feedback(
    execution_id: str | None,
    version_id: str | None,
    chosen_version_id: str | None,
    rejected_version_id: str | None,
    liked: bool | None,
    style_match_score: float | None,
    usefulness_score: float | None,
    creativity_score: float | None,
    faithfulness_score: float | None,
    notes: str,
    reason_tags: list[str],
) -> dict[str, Any]:
    """Persist one explicit editorial feedback signal."""

    normalized_reason_tags = [tag for tag in reason_tags if tag in STYLE_REASON_TAGS]
    feedback_id = save_antese_feedback(
        execution_id=execution_id,
        version_id=version_id,
        chosen_version_id=chosen_version_id,
        rejected_version_id=rejected_version_id,
        liked=liked,
        style_match_score=style_match_score,
        usefulness_score=usefulness_score,
        creativity_score=creativity_score,
        faithfulness_score=faithfulness_score,
        notes=notes,
        reason_tags=normalized_reason_tags,
    )
    return {
        "feedback_id": feedback_id,
        "execution_id": execution_id,
        "version_id": version_id,
        "reason_tags": normalized_reason_tags,
    }


def get_antese_feedback(limit: int = 50, execution_id: str | None = None) -> list[dict[str, Any]]:
    """Return stored feedback rows for Antese."""

    return get_antese_feedback_history(execution_id=execution_id, limit=limit)
