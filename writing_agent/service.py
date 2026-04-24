"""Service layer for the Antese writing agent."""

from __future__ import annotations

import html
import json
import re
from urllib.parse import urlparse
from urllib import request as urlrequest
import uuid
from typing import Any

from email_agent.llm import LLMProviderError, invoke_llm_text
from email_agent.settings import get_llm_provider
from email_agent.storage import (
    get_antese_samples,
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
    update_antese_sample_genre,
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


def _strip_html_to_text(raw_html: str) -> str:
    """Extract readable text from simple HTML pages without extra dependencies."""

    cleaned = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style\b[^>]*>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<noscript\b[^>]*>.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p\s*>", "\n\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</div\s*>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_gamma_content_blocks(raw_html: str) -> list[dict[str, str]]:
    """Extract ordered text blocks from Gamma's embedded Next.js payload."""

    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', raw_html, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    try:
        root = payload["props"]["pageProps"]["doc"]["publishedSnapshot"]["content"]["default"]
    except (KeyError, TypeError):
        return []

    blocks: list[dict[str, str]] = []
    seen_normalized_texts: set[str] = set()

    def walk(node: Any, path: str = "root") -> None:
        if isinstance(node, dict):
            text_value = node.get("text")
            if isinstance(text_value, str):
                normalized = " ".join(text_value.split()).strip()
                if len(normalized) >= 20 and normalized not in seen_normalized_texts:
                    seen_normalized_texts.add(normalized)
                    blocks.append({"block_id": f"b{len(blocks)+1:03d}", "path": path, "text": normalized})
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(root)
    return blocks


def _merge_short_blocks(blocks: list[dict[str, str]], min_length: int = 120) -> list[dict[str, str]]:
    """Merge tiny Gamma blocks into larger editorial units before LLM segmentation."""

    if not blocks:
        return []

    merged: list[dict[str, str]] = []
    buffer_ids: list[str] = []
    buffer_paths: list[str] = []
    buffer_texts: list[str] = []

    def flush() -> None:
        if not buffer_texts:
            return
        merged.append(
            {
                "block_id": ",".join(buffer_ids),
                "path": " | ".join(buffer_paths[:3]),
                "text": "\n".join(buffer_texts).strip(),
                "source_block_ids": list(buffer_ids),
            }
        )
        buffer_ids.clear()
        buffer_paths.clear()
        buffer_texts.clear()

    for block in blocks:
        buffer_ids.append(block["block_id"])
        buffer_paths.append(block["path"])
        buffer_texts.append(block["text"])
        if len(" ".join(buffer_texts)) >= min_length:
            flush()
    flush()
    return merged


def _fetch_webpage_text(source_url: str) -> str:
    """Fetch one webpage and extract readable text for Antese samples."""

    req = urlrequest.Request(
        source_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            raw_html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(
            f"Não foi possível buscar a URL informada. O site pode exigir autenticação ou bloquear scraping: {exc}"
        ) from exc

    extracted_text = _strip_html_to_text(raw_html)
    if not extracted_text:
        raise ValueError("A URL foi carregada, mas nenhum texto legível foi extraído.")
    return extracted_text


def _fetch_webpage_payload(source_url: str) -> tuple[str, str]:
    """Fetch one webpage returning both raw HTML and a plain-text fallback."""

    req = urlrequest.Request(
        source_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            raw_html = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(
            f"Não foi possível buscar a URL informada. O site pode exigir autenticação ou bloquear scraping: {exc}"
        ) from exc

    extracted_text = _strip_html_to_text(raw_html)
    if not extracted_text:
        raise ValueError("A URL foi carregada, mas nenhum texto legível foi extraído.")
    return raw_html, extracted_text


def _chunk_text_for_samples(raw_text: str, chunk_size: int = 1400, chunk_overlap: int = 200) -> list[str]:
    """Split long text into paragraph-aware chunks for Chroma ingestion."""

    normalized_text = re.sub(r"\n{3,}", "\n\n", raw_text.strip())
    paragraphs = [paragraph.strip() for paragraph in normalized_text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue
        if current_chunk:
            chunks.append(current_chunk)
            overlap_seed = current_chunk[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current_chunk = f"{overlap_seed}\n\n{paragraph}".strip() if overlap_seed else paragraph
        else:
            start = 0
            step = max(1, chunk_size - chunk_overlap)
            while start < len(paragraph):
                end = min(len(paragraph), start + chunk_size)
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start += step
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return [chunk for chunk in chunks if chunk.strip()]


def _build_editorial_summary(text_chunk: str) -> str:
    """Build a compact editorial summary for one sample chunk."""

    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text_chunk.split()))
    summary = " ".join(sentences[:2]).strip()
    return (summary or text_chunk[:240].strip())[:320]


def _infer_genre_id_from_text(text: str) -> str | None:
    normalized = text.lower()
    mapping = {
        "poema": "poem",
        "crônica": "chronicle",
        "cronica": "chronicle",
        "conto": "short_story",
        "romance": "novel_excerpt",
        "newsletter": "newsletter",
        "artigo": "essay_article",
    }
    for token, genre_id in mapping.items():
        if token in normalized:
            return genre_id
    return None


def _coerce_score(value: Any, default: float = 0.7) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric_value))


def _segment_content_blocks_heuristically(
    title: str,
    blocks: list[dict[str, str]],
    default_genre_id: str | None,
) -> list[dict[str, Any]]:
    """Fallback segmentation when no LLM is available."""

    samples: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        sample_type = "writing_sample"
        lowered = block["text"].lower()
        if any(token in lowered for token in ("linkedin", "github", "sobre mim", "trabalho", "experiência", "experience")):
            sample_type = "profile_context"
        samples.append(
            {
                "sample_title": f"{title} — segmento {index}",
                "sample_type": sample_type,
                "genre_id": default_genre_id or _infer_genre_id_from_text(block["text"]),
                "editorial_summary": _build_editorial_summary(block["text"]),
                "persona_scope": "professional" if sample_type != "writing_sample" else "creative",
                "block_ids": block["block_id"].split(","),
                "tags": ["heuristic_segmentation"],
            }
        )
    return samples


def _build_segmentation_prompt(
    title: str,
    source_url: str | None,
    blocks: list[dict[str, str]],
    default_genre_id: str | None,
) -> str:
    """Build the segmentation prompt used to split imported content into Antese samples."""

    block_lines = []
    for block in blocks:
        block_lines.append(
            "\n".join(
                [
                    f"block_id: {block['block_id']}",
                    f"source_block_ids: {', '.join(block.get('source_block_ids', [block['block_id']]))}",
                    f"path: {block['path']}",
                    f"text: {block['text']}",
                ]
            )
        )
    return f"""
Você receberá blocos textuais extraídos de um portfólio/site pessoal.
Sua tarefa é agrupar esses blocos em unidades editoriais úteis para a memória do Antese.

Título da fonte: {title}
URL: {source_url or "sem URL"}
Genre sugerido: {default_genre_id or "nao especificado"}

Tipos permitidos para `sample_type`:
- writing_sample
- profile_context
- project_case
- work_experience
- link_hub
- other

Regras:
1. Agrupe blocos que pertençam ao mesmo texto, seção autoral ou unidade temática.
2. Se houver textos literários/autorais, classifique-os como `writing_sample`.
3. Se houver bio, descrição pessoal, atuação profissional, links e afins, use os tipos apropriados.
4. Se identificar gênero textual explícito, preencha `genre_id` com um dos valores:
   poem, chronicle, short_story, novel_excerpt, essay_article, newsletter, outline, consolidated_memo, linkedin_post
5. Não invente conteúdo nem ids de blocos.
6. Use apenas block_ids existentes.
7. Responda APENAS em JSON válido.

Formato:
{{
  "samples": [
    {{
      "sample_title": "string curta",
      "sample_type": "writing_sample|profile_context|project_case|work_experience|link_hub|other",
      "genre_id": "string ou null",
      "editorial_summary": "resumo curto",
      "persona_scope": "creative|professional|personal|mixed",
      "block_ids": ["b001", "b002"],
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Blocos:
{chr(10).join(block_lines)}
""".strip()


def _normalize_segmented_samples(
    title: str,
    blocks: list[dict[str, str]],
    default_genre_id: str | None,
    samples: list[Any],
) -> list[dict[str, Any]]:
    """Normalize the LLM segmentation payload to the internal Antese sample structure."""

    valid_block_ids = {block["block_id"] for block in blocks}
    block_aliases: dict[str, str] = {}
    for block in blocks:
        merged_block_id = block["block_id"]
        block_aliases[merged_block_id] = merged_block_id
        for source_block_id in block.get("source_block_ids", []):
            normalized_source_block_id = str(source_block_id).strip()
            if normalized_source_block_id:
                block_aliases[normalized_source_block_id] = merged_block_id

    normalized_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue
        raw_block_ids = [str(item).strip() for item in sample.get("block_ids", []) if str(item).strip()]
        block_ids: list[str] = []
        for raw_block_id in raw_block_ids:
            normalized_block_id = block_aliases.get(raw_block_id, raw_block_id)
            if normalized_block_id in valid_block_ids and normalized_block_id not in block_ids:
                block_ids.append(normalized_block_id)
        if not block_ids:
            continue
        normalized_samples.append(
            {
                "sample_title": str(sample.get("sample_title", f"{title} — segmento {index}")).strip()
                or f"{title} — segmento {index}",
                "sample_type": str(sample.get("sample_type", "other")).strip() or "other",
                "genre_id": str(sample.get("genre_id", default_genre_id or "")).strip() or default_genre_id,
                "editorial_summary": str(sample.get("editorial_summary", "")).strip(),
                "persona_scope": str(sample.get("persona_scope", "mixed")).strip() or "mixed",
                "block_ids": block_ids,
                "tags": _list_strings(sample.get("tags")) or ["llm_segmented"],
            }
        )
    return normalized_samples


def _segment_content_blocks_with_llm(
    title: str,
    source_url: str | None,
    blocks: list[dict[str, str]],
    default_genre_id: str | None,
) -> list[dict[str, Any]]:
    """Use the configured LLM to split one source into Antese-ready samples."""

    if get_llm_provider() == "heuristic":
        return _segment_content_blocks_heuristically(title, blocks, default_genre_id)

    prompt = _build_segmentation_prompt(title, source_url, blocks, default_genre_id)
    raw_output = invoke_llm_text(ANTese_SYSTEM_PROMPT, prompt, json_output=True)
    parsed = _safe_json_loads(raw_output)
    samples = parsed.get("samples", [])
    if not isinstance(samples, list) or not samples:
        return _segment_content_blocks_heuristically(title, blocks, default_genre_id)

    normalized_samples = _normalize_segmented_samples(title, blocks, default_genre_id, samples)
    return normalized_samples or _segment_content_blocks_heuristically(title, blocks, default_genre_id)


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


def get_antese_samples_catalog(limit: int = 200) -> list[dict[str, Any]]:
    """Return writing samples already stored for audit and manual reclassification."""

    return get_antese_samples(limit=limit)


def reclassify_antese_sample_genre(sample_id: str, genre_id: str | None) -> dict[str, Any]:
    """Apply a manual genre correction to one writing sample."""

    normalized_genre_id = (genre_id or "").strip() or None
    if normalized_genre_id and normalized_genre_id not in {
        "brainstorm_notes",
        "outline",
        "linkedin_post",
        "essay_article",
        "newsletter",
        "rewrite_clarity",
        "consolidated_memo",
        "poem",
        "chronicle",
        "short_story",
        "novel_excerpt",
    }:
        raise ValueError("Genre inválido para reclassificação do sample.")
    return update_antese_sample_genre(sample_id=sample_id, genre_id=normalized_genre_id)


def _build_preview_blocks_from_text(raw_text: str) -> list[dict[str, str]]:
    """Create pseudo-blocks for segmentation preview when only raw text is available."""

    chunks = _chunk_text_for_samples(raw_text=raw_text, chunk_size=700, chunk_overlap=0)
    return [
        {
            "block_id": f"b{index:03d}",
            "path": f"raw_text[{index}]",
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
        if chunk.strip()
    ]


def preview_antese_sample_segmentation(
    title: str,
    source_url: str | None = None,
    raw_text: str | None = None,
    genre_id: str | None = None,
) -> dict[str, Any]:
    """Preview the exact segmentation prompt and the LLM output without persisting anything."""

    if not (source_url or raw_text):
        raise ValueError("Informe `source_url` ou `raw_text` para testar a segmentação.")

    raw_html = ""
    source_text = raw_text.strip() if raw_text else ""
    if source_url and not source_text:
        raw_html, source_text = _fetch_webpage_payload(source_url)
    if not source_text:
        raise ValueError("Nenhum texto válido foi obtido para o preview.")

    parsed_url = urlparse(source_url or "")
    is_gamma_source = "gamma.site" in parsed_url.netloc or "gamma.app" in parsed_url.netloc
    if is_gamma_source and raw_html:
        blocks = _merge_short_blocks(_extract_gamma_content_blocks(raw_html))
        source_parser = "gamma_next_data"
    else:
        blocks = _build_preview_blocks_from_text(source_text)
        source_parser = "plain_text"

    if not blocks:
        raise ValueError("Nenhum bloco válido foi extraído para o preview do prompt.")

    prompt_preview = _build_segmentation_prompt(title, source_url, blocks, genre_id)
    if get_llm_provider() == "heuristic":
        normalized_samples = _segment_content_blocks_heuristically(title, blocks, genre_id)
        return {
            "title": title,
            "source_url": source_url,
            "source_parser": source_parser,
            "block_count": len(blocks),
            "blocks_preview": blocks,
            "prompt_preview": prompt_preview,
            "used_strategy": "heuristic",
            "raw_llm_output": None,
            "parsed_response": {"samples": normalized_samples},
            "normalized_samples": normalized_samples,
        }

    raw_output = invoke_llm_text(ANTese_SYSTEM_PROMPT, prompt_preview, json_output=True)
    parsed = _safe_json_loads(raw_output)
    samples = parsed.get("samples", []) if isinstance(parsed, dict) else []
    normalized_samples = _normalize_segmented_samples(title, blocks, genre_id, samples if isinstance(samples, list) else [])
    if not normalized_samples:
        normalized_samples = _segment_content_blocks_heuristically(title, blocks, genre_id)

    return {
        "title": title,
        "source_url": source_url,
        "source_parser": source_parser,
        "block_count": len(blocks),
        "blocks_preview": blocks,
        "prompt_preview": prompt_preview,
        "used_strategy": "llm_segmentation",
        "raw_llm_output": raw_output,
        "parsed_response": parsed,
        "normalized_samples": normalized_samples,
    }


def import_antese_text_samples(
    title: str,
    source_scope: str,
    style_profile_id: str | None = "personal_default",
    genre_id: str | None = None,
    source_url: str | None = None,
    raw_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
    segment_with_llm: bool = True,
) -> dict[str, Any]:
    """Import one personal corpus into Antese samples and Chroma memory."""

    if not (raw_text or source_url):
        raise ValueError("Informe `raw_text` ou `source_url` para importar text samples.")

    raw_html = ""
    source_text = raw_text.strip() if raw_text else ""
    if source_url and not source_text:
        raw_html, source_text = _fetch_webpage_payload(source_url)
    if not source_text:
        raise ValueError("Nenhum texto válido foi obtido para importação.")

    import_id = uuid.uuid4().hex[:12]
    normalized_metadata = dict(metadata or {})
    normalized_metadata.update(
        {
            "import_id": import_id,
            "source_url": source_url or "",
            "style_profile_id": style_profile_id or "",
            "genre_id": genre_id or "",
        }
    )

    sample_ids: list[str] = []
    segmented_units: list[dict[str, Any]] = []

    parsed_url = urlparse(source_url or "")
    is_gamma_source = "gamma.site" in parsed_url.netloc or "gamma.app" in parsed_url.netloc
    if is_gamma_source and raw_html:
        gamma_blocks = _merge_short_blocks(_extract_gamma_content_blocks(raw_html))
        if gamma_blocks:
            sample_groups = (
                _segment_content_blocks_with_llm(title, source_url, gamma_blocks, genre_id)
                if segment_with_llm
                else _segment_content_blocks_heuristically(title, gamma_blocks, genre_id)
            )
            block_map = {block["block_id"]: block for block in gamma_blocks}
            for index, sample_group in enumerate(sample_groups, start=1):
                texts = [block_map[block_id]["text"] for block_id in sample_group["block_ids"] if block_id in block_map]
                combined_text = "\n\n".join(texts).strip()
                if not combined_text:
                    continue
                effective_genre_id = sample_group.get("genre_id") or genre_id or _infer_genre_id_from_text(combined_text)
                sample_id = save_antese_sample(
                    title=str(sample_group.get("sample_title", f"{title} — segmento {index}")),
                    source_scope=source_scope,
                    text_content=combined_text,
                    editorial_summary=str(sample_group.get("editorial_summary") or _build_editorial_summary(combined_text)),
                    style_profile_id=style_profile_id if sample_group.get("sample_type") == "writing_sample" else None,
                    genre_id=effective_genre_id,
                    metadata={
                        **normalized_metadata,
                        "sample_type": sample_group.get("sample_type", "other"),
                        "persona_scope": sample_group.get("persona_scope", "mixed"),
                        "tags": sample_group.get("tags", []),
                        "source_parser": "gamma_next_data",
                        "block_ids": sample_group.get("block_ids", []),
                    },
                )
                sample_ids.append(sample_id)
                segmented_units.append(
                    {
                        "sample_id": sample_id,
                        "title": str(sample_group.get("sample_title", f"{title} — segmento {index}")),
                        "sample_type": sample_group.get("sample_type", "other"),
                        "genre_id": effective_genre_id,
                        "persona_scope": sample_group.get("persona_scope", "mixed"),
                        "block_ids": sample_group.get("block_ids", []),
                    }
                )

    if not sample_ids:
        chunks = _chunk_text_for_samples(
            raw_text=source_text,
            chunk_size=max(400, chunk_size),
            chunk_overlap=max(0, min(chunk_overlap, max(400, chunk_size) - 50)),
        )
        if not chunks:
            raise ValueError("O corpus fornecido não gerou chunks válidos para indexação.")

        for index, chunk in enumerate(chunks, start=1):
            sample_id = save_antese_sample(
                title=f"{title} — trecho {index}",
                source_scope=source_scope,
                text_content=chunk,
                editorial_summary=_build_editorial_summary(chunk),
                style_profile_id=style_profile_id,
                genre_id=genre_id,
                metadata={**normalized_metadata, "chunk_index": index, "chunk_count": len(chunks), "source_parser": "plain_text"},
            )
            sample_ids.append(sample_id)
            segmented_units.append(
                {
                    "sample_id": sample_id,
                    "title": f"{title} — trecho {index}",
                    "sample_type": "writing_sample",
                    "genre_id": genre_id,
                    "persona_scope": "mixed",
                    "block_ids": [],
                }
            )

    return {
        "import_id": import_id,
        "title": title,
        "source_scope": source_scope,
        "style_profile_id": style_profile_id,
        "genre_id": genre_id,
        "source_url": source_url,
        "chunk_count": len(sample_ids),
        "sample_ids": sample_ids,
        "text_length": len(source_text),
        "segmented_units": segmented_units,
        "source_parser": "gamma_next_data" if segmented_units and is_gamma_source else "plain_text",
    }


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
