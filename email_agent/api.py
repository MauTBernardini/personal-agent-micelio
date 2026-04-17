"""FastAPI application exposing the email triage backend."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from email_agent.llm import LLMProviderError
from email_agent.service import (
    build_dashboard_payload,
    get_connection_status,
    get_manual_review_categories,
    get_review_queue,
    manual_reclassify_email,
    run_triage_and_collect,
)


class ReclassifyPayload(BaseModel):
    category: str


app = FastAPI(
    title="Personal Agent Micelio API",
    version="0.1.0",
    description="API para triagem de e-mails, dashboard e revisão manual.",
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, str]:
    return get_connection_status()


@app.get("/dashboard")
def dashboard() -> dict[str, Any]:
    return build_dashboard_payload()


@app.post("/triage")
def triage(limit: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
    try:
        return run_triage_and_collect(limit=limit)
    except LLMProviderError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@app.get("/review-queue")
def review_queue() -> list[dict[str, Any]]:
    return get_review_queue()


@app.get("/manual-review-categories")
def manual_review_categories() -> dict[str, list[str]]:
    return {"categories": list(get_manual_review_categories())}


@app.post("/review-queue/{message_id}/reclassify")
def reclassify(message_id: str, payload: ReclassifyPayload) -> dict[str, Any]:
    try:
        return manual_reclassify_email(message_id, payload.category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

