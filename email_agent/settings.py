"""Configuration helpers and optional dependency loading for the email agent."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Final

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials as GoogleCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover
    GoogleAuthRequest = None  # type: ignore[assignment]
    GoogleCredentials = None  # type: ignore[assignment]
    InstalledAppFlow = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment]


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
CHROMA_PATH: Final[Path] = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME: Final[str] = "semantic_memory"
WRITING_SAMPLES_COLLECTION_NAME: Final[str] = "writing_samples_memory"
WRITING_FEEDBACK_COLLECTION_NAME: Final[str] = "writing_feedback_memory"
WRITING_RUNS_COLLECTION_NAME: Final[str] = "writing_runs_memory"
ENV_FILE_PATH: Final[Path] = BASE_DIR / ".env"

ANTHROPIC_MODEL: Final[str] = "claude-3-haiku-20240307"
OPENAI_MODEL: Final[str] = "gpt-5-mini"
GEMINI_MODEL: Final[str] = "gemini-2.5-flash-lite"
GEMINI_FALLBACK_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_DATABASE_URL: Final[str] = "postgresql://micelio:micelio@localhost:5432/email_agent"

DEFAULT_LLM_PROVIDER: Final[str] = "gemini"
DEFAULT_EMAIL_PROVIDER: Final[str] = "mock"
DEFAULT_GMAIL_USER_ID: Final[str] = "me"
DEFAULT_GMAIL_QUERY: Final[str] = ""
DEFAULT_GMAIL_MAX_RESULTS: Final[int] = 0
DEFAULT_GMAIL_PAGE_SIZE: Final[int] = 100
DEFAULT_GMAIL_AFTER_DATE: Final[str] = "today"
DEFAULT_GMAIL_REQUIRE_INBOX: Final[bool] = True
DEFAULT_GMAIL_OAUTH_PORT: Final[int] = 8765
DEFAULT_GMAIL_LABEL_PREFIX: Final[str] = "AUTO_TRIAGEM_"
DEFAULT_GMAIL_ARCHIVE_AFTER_TRIAGE: Final[bool] = False
DEFAULT_GMAIL_CREDENTIALS_FILE: Final[Path] = BASE_DIR / "gmail_credentials.json"
DEFAULT_GMAIL_TOKEN_FILE: Final[Path] = BASE_DIR / "gmail_token.json"
DEFAULT_GMAIL_AUTH_URI: Final[str] = "https://accounts.google.com/o/oauth2/auth"
DEFAULT_GMAIL_TOKEN_URI: Final[str] = "https://oauth2.googleapis.com/token"
DEFAULT_API_BASE_URL: Final[str] = "http://127.0.0.1:8000"
DEFAULT_TRIAGE_BATCH_SIZE: Final[int] = 8
GMAIL_SCOPES: Final[tuple[str, ...]] = ("https://www.googleapis.com/auth/gmail.modify",)


def load_local_env_file() -> None:
    """Load key-value pairs from a local `.env` without extra dependencies."""

    candidate_paths = [ENV_FILE_PATH, Path.cwd() / ".env"]
    env_path = next((path for path in candidate_paths if path.exists()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        current_value = os.getenv(normalized_key)
        if current_value is None or not current_value.strip():
            os.environ[normalized_key] = normalized_value


def _get_bool_env(env_name: str, default: bool) -> bool:
    """Read a boolean environment variable with a conservative parser."""

    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_llm_provider() -> str:
    """Return the configured LLM provider strategy."""

    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    return provider if provider in {"gemini", "openai", "anthropic", "heuristic"} else DEFAULT_LLM_PROVIDER


def get_gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL


def get_gemini_fallback_model_name() -> str:
    return os.getenv("GEMINI_FALLBACK_MODEL", GEMINI_FALLBACK_MODEL).strip() or GEMINI_FALLBACK_MODEL


def get_openai_model_name() -> str:
    return os.getenv("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL


def get_anthropic_model_name() -> str:
    return os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL).strip() or ANTHROPIC_MODEL


def get_email_provider() -> str:
    provider = os.getenv("EMAIL_PROVIDER", DEFAULT_EMAIL_PROVIDER).strip().lower()
    return provider if provider in {"mock", "gmail"} else DEFAULT_EMAIL_PROVIDER


def get_gmail_credentials_file() -> Path:
    return Path(
        os.getenv("GMAIL_OAUTH_CLIENT_SECRET_FILE", str(DEFAULT_GMAIL_CREDENTIALS_FILE))
    ).expanduser()


def get_gmail_project_id() -> str:
    return os.getenv("GMAIL_PROJECT_ID", "").strip()


def get_gmail_client_id() -> str:
    return os.getenv("GMAIL_CLIENT_ID", "").strip()


def get_gmail_client_secret() -> str:
    return os.getenv("GMAIL_CLIENT_SECRET", "").strip()


def get_gmail_auth_uri() -> str:
    return os.getenv("GMAIL_AUTH_URI", DEFAULT_GMAIL_AUTH_URI).strip() or DEFAULT_GMAIL_AUTH_URI


def get_gmail_token_uri() -> str:
    return os.getenv("GMAIL_TOKEN_URI", DEFAULT_GMAIL_TOKEN_URI).strip() or DEFAULT_GMAIL_TOKEN_URI


def get_gmail_token_file() -> Path:
    return Path(os.getenv("GMAIL_OAUTH_TOKEN_FILE", str(DEFAULT_GMAIL_TOKEN_FILE))).expanduser()


def get_gmail_user_id() -> str:
    return os.getenv("GMAIL_USER_ID", DEFAULT_GMAIL_USER_ID).strip() or DEFAULT_GMAIL_USER_ID


def get_gmail_query() -> str:
    return os.getenv("GMAIL_QUERY", DEFAULT_GMAIL_QUERY).strip() or DEFAULT_GMAIL_QUERY


def get_gmail_max_results(default_limit: int | None = None) -> int | None:
    if default_limit is not None:
        return default_limit
    raw_limit = os.getenv("GMAIL_MAX_RESULTS", str(DEFAULT_GMAIL_MAX_RESULTS)).strip()
    try:
        parsed_limit = int(raw_limit)
    except ValueError:
        return DEFAULT_GMAIL_MAX_RESULTS
    if parsed_limit <= 0:
        return None
    return parsed_limit


def get_gmail_page_size() -> int:
    raw_value = os.getenv("GMAIL_PAGE_SIZE", str(DEFAULT_GMAIL_PAGE_SIZE)).strip()
    try:
        return max(1, min(int(raw_value), 500))
    except ValueError:
        return DEFAULT_GMAIL_PAGE_SIZE


def get_gmail_after_date() -> str | None:
    raw_value = os.getenv("GMAIL_AFTER_DATE", DEFAULT_GMAIL_AFTER_DATE).strip()
    normalized_value = raw_value.lower()
    if not raw_value or normalized_value in {"none", "off", "all"}:
        return None
    if normalized_value == "today":
        return datetime.now().strftime("%Y/%m/%d")
    for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw_value, date_format).strftime("%Y/%m/%d")
        except ValueError:
            continue
    return datetime.now().strftime("%Y/%m/%d")


def should_require_gmail_inbox() -> bool:
    return _get_bool_env("GMAIL_REQUIRE_INBOX", DEFAULT_GMAIL_REQUIRE_INBOX)


def get_gmail_oauth_port() -> int:
    raw_port = os.getenv("GMAIL_OAUTH_PORT", str(DEFAULT_GMAIL_OAUTH_PORT)).strip()
    try:
        return int(raw_port)
    except ValueError:
        return DEFAULT_GMAIL_OAUTH_PORT


def get_gmail_label_prefix() -> str:
    return os.getenv("GMAIL_LABEL_PREFIX", DEFAULT_GMAIL_LABEL_PREFIX).strip() or DEFAULT_GMAIL_LABEL_PREFIX


def build_gmail_triage_query() -> str:
    query_parts = ["is:unread"]
    after_date = get_gmail_after_date()
    if after_date:
        query_parts.insert(0, f"after:{after_date}")
    if should_require_gmail_inbox():
        query_parts.append("in:inbox")
    extra_query = get_gmail_query()
    if extra_query:
        query_parts.append(extra_query)
    return " ".join(part for part in query_parts if part).strip()


def should_archive_after_triage() -> bool:
    return _get_bool_env("GMAIL_ARCHIVE_AFTER_TRIAGE", DEFAULT_GMAIL_ARCHIVE_AFTER_TRIAGE)


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_api_base_url() -> str:
    return os.getenv("AGENT_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_triage_batch_size() -> int:
    """Return the default number of emails processed per triage execution."""

    raw_value = os.getenv("TRIAGE_BATCH_SIZE", str(DEFAULT_TRIAGE_BATCH_SIZE)).strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_TRIAGE_BATCH_SIZE


load_local_env_file()
