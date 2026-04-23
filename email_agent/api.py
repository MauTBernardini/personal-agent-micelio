"""FastAPI application exposing the email triage backend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from orchestrator.service import get_available_agents, get_micelio_execution_history, run_micelio
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference
from writing_agent.service import (
    create_or_update_antese_genre_card,
    create_or_update_antese_style_profile,
    get_antese_execution_history,
    get_antese_feedback,
    get_antese_genre_card_catalog,
    get_antese_inspiration_profile_catalog,
    get_antese_style_profiles_catalog,
    get_antese_versions,
    run_antese,
    submit_antese_feedback,
)

from email_agent.llm import LLMProviderError
from email_agent.service import (
    build_dashboard_payload,
    get_connection_status,
    get_manual_review_categories,
    get_manual_review_priorities,
    get_processed_emails,
    get_review_queue,
    manual_reclassify_email,
    run_triage_and_collect,
)


class ReclassifyPayload(BaseModel):
    category: str
    priority_level: str = "MEDIA"


class MicelioPayload(BaseModel):
    user_request: str
    requested_agent: str | None = None
    writing_task_type: str = "BRAINSTORM"
    context_notes: str = ""
    tone: str = "claro e profissional"
    audience: str = "leitores gerais"
    email_limit: int | None = None


class AntesePayload(BaseModel):
    user_request: str
    task_type: str = "BRAINSTORM"
    source_notes: str = ""
    goal: str = "EXPLORAR"
    genre_id: str | None = None
    style_profile_id: str | None = None
    inspiration_profile_id: str | None = None
    audience: str = "leitores gerais"
    tone_override: str = "claro e profissional"
    must_include: list[str] = []
    must_avoid: list[str] = []
    reference_text: str = ""
    action_label: str = "default"


class StyleProfilePayload(BaseModel):
    profile_id: str | None = None
    name: str
    owner_scope: str = "session"
    voice_traits: dict[str, Any] = {}
    structure_traits: dict[str, Any] = {}
    rhetorical_traits: dict[str, Any] = {}
    lexical_traits: dict[str, Any] = {}
    dos: list[str] = []
    donts: list[str] = []
    sample_text_ids: list[str] = []
    sample_texts: list[str] = []


class GenreCardPayload(BaseModel):
    genre_id: str
    name: str
    primary_goal: str
    expected_structure: list[str] = []
    tone_defaults: list[str] = []
    length_defaults: dict[str, Any] = {}
    quality_checklist: list[str] = []
    typical_openings: list[str] = []
    typical_closings: list[str] = []


class AnteseFeedbackPayload(BaseModel):
    execution_id: str | None = None
    version_id: str | None = None
    chosen_version_id: str | None = None
    rejected_version_id: str | None = None
    liked: bool | None = None
    style_match_score: float | None = None
    usefulness_score: float | None = None
    creativity_score: float | None = None
    faithfulness_score: float | None = None
    notes: str = ""
    reason_tags: list[str] = []


app = FastAPI(
    title="Personal Agent Micelio API",
    version="0.1.0",
    description="API para triagem de e-mails, dashboard e revisão manual.",
)


@app.get("/scalar", include_in_schema=False, response_class=HTMLResponse)
def scalar_docs() -> HTMLResponse:
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, str]:
    return get_connection_status()


@app.get("/agents")
def list_agents() -> dict[str, list[dict[str, Any]]]:
    return {"agents": get_available_agents()}


@app.get("/micelio/executions")
def micelio_executions(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, list[dict[str, Any]]]:
    return {"executions": get_micelio_execution_history(limit=limit)}


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    return build_dashboard_payload()


@app.post("/micelio/orchestrate")
def micelio_orchestrate(payload: MicelioPayload) -> dict[str, Any]:
    try:
        return run_micelio(
            user_request=payload.user_request,
            requested_agent=payload.requested_agent,
            writing_task_type=payload.writing_task_type,
            context_notes=payload.context_notes,
            tone=payload.tone,
            audience=payload.audience,
            email_limit=payload.email_limit,
        )
    except LLMProviderError as exc:
        message = str(exc).lower()
        if "cota" in message or "quota" in message or "rate limit" in message:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/antese/run")
def antese_run(payload: AntesePayload) -> dict[str, Any]:
    try:
        return run_antese(
            user_request=payload.user_request,
            task_type=payload.task_type,
            source_notes=payload.source_notes,
            goal=payload.goal,
            genre_id=payload.genre_id,
            style_profile_id=payload.style_profile_id,
            inspiration_profile_id=payload.inspiration_profile_id,
            audience=payload.audience,
            tone_override=payload.tone_override,
            must_include=payload.must_include,
            must_avoid=payload.must_avoid,
            reference_text=payload.reference_text,
            action_label=payload.action_label,
        )
    except LLMProviderError as exc:
        message = str(exc).lower()
        if "cota" in message or "quota" in message or "rate limit" in message:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/antese/style-profiles")
def antese_style_profiles() -> dict[str, list[dict[str, Any]]]:
    return {"style_profiles": get_antese_style_profiles_catalog()}


@app.post("/antese/style-profiles")
def antese_create_style_profile(payload: StyleProfilePayload) -> dict[str, Any]:
    try:
        return create_or_update_antese_style_profile(
            profile_id=payload.profile_id,
            name=payload.name,
            owner_scope=payload.owner_scope,
            voice_traits=payload.voice_traits,
            structure_traits=payload.structure_traits,
            rhetorical_traits=payload.rhetorical_traits,
            lexical_traits=payload.lexical_traits,
            dos=payload.dos,
            donts=payload.donts,
            sample_text_ids=payload.sample_text_ids,
            sample_texts=payload.sample_texts,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/antese/genre-cards")
def antese_genre_cards() -> dict[str, list[dict[str, Any]]]:
    return {"genre_cards": get_antese_genre_card_catalog()}


@app.post("/antese/genre-cards")
def antese_create_genre_card(payload: GenreCardPayload) -> dict[str, Any]:
    try:
        return create_or_update_antese_genre_card(
            genre_id=payload.genre_id,
            name=payload.name,
            primary_goal=payload.primary_goal,
            expected_structure=payload.expected_structure,
            tone_defaults=payload.tone_defaults,
            length_defaults=payload.length_defaults,
            quality_checklist=payload.quality_checklist,
            typical_openings=payload.typical_openings,
            typical_closings=payload.typical_closings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/antese/inspiration-profiles")
def antese_inspiration_profiles() -> dict[str, list[dict[str, Any]]]:
    return {"inspiration_profiles": get_antese_inspiration_profile_catalog()}


@app.get("/antese/executions")
def antese_executions(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, list[dict[str, Any]]]:
    return {"executions": get_antese_execution_history(limit=limit)}


@app.get("/antese/executions/{execution_id}/versions")
def antese_execution_versions(execution_id: str) -> dict[str, list[dict[str, Any]]]:
    return {"versions": get_antese_versions(execution_id)}


@app.post("/antese/feedback")
def antese_feedback(payload: AnteseFeedbackPayload) -> dict[str, Any]:
    try:
        return submit_antese_feedback(
            execution_id=payload.execution_id,
            version_id=payload.version_id,
            chosen_version_id=payload.chosen_version_id,
            rejected_version_id=payload.rejected_version_id,
            liked=payload.liked,
            style_match_score=payload.style_match_score,
            usefulness_score=payload.usefulness_score,
            creativity_score=payload.creativity_score,
            faithfulness_score=payload.faithfulness_score,
            notes=payload.notes,
            reason_tags=payload.reason_tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/antese/feedback")
def antese_feedback_history(
    execution_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, list[dict[str, Any]]]:
    return {"feedback": get_antese_feedback(limit=limit, execution_id=execution_id)}


@app.post("/triage")
def triage(limit: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
    try:
        return run_triage_and_collect(limit=limit)
    except LLMProviderError as exc:
        message = str(exc).lower()
        if "cota" in message or "quota" in message or "rate limit" in message:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@app.get("/review-queue")
def review_queue() -> list[dict[str, Any]]:
    return get_review_queue()


@app.get("/processed-emails")
def processed_emails(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    return get_processed_emails(limit=limit)


@app.get("/manual-review-categories")
def manual_review_categories() -> dict[str, list[str]]:
    return {
        "categories": list(get_manual_review_categories()),
        "priorities": list(get_manual_review_priorities()),
    }


@app.post("/review-queue/{message_id}/reclassify")
def reclassify(message_id: str, payload: ReclassifyPayload) -> dict[str, Any]:
    try:
        return manual_reclassify_email(message_id, payload.category, payload.priority_level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/processed-emails/{message_id}/reclassify")
def reclassify_processed_email(message_id: str, payload: ReclassifyPayload) -> dict[str, Any]:
    try:
        return manual_reclassify_email(message_id, payload.category, payload.priority_level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
