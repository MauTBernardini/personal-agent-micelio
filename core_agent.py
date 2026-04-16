"""Core agent for the personal email triage MVP.

This module intentionally centralizes:
- PostgreSQL persistence
- ChromaDB semantic memory
- LLM provider abstraction (OpenAI / Anthropic / fallback)
- LangGraph orchestration
- Gmail API mocks for local testing

The Streamlit UI should only import and call public functions from here.
"""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import chromadb
from chromadb.api.models.Collection import Collection
from langgraph.graph import END, START, StateGraph

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - keeps import-time errors friendlier.
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - keeps local MVP usable without the SDK installed.
    Anthropic = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - keeps local MVP usable without the SDK installed.
    OpenAI = None  # type: ignore[assignment]

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials as GoogleCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # pragma: no cover - keeps mock mode usable without Gmail deps.
    GoogleAuthRequest = None  # type: ignore[assignment]
    GoogleCredentials = None  # type: ignore[assignment]
    InstalledAppFlow = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    HttpError = Exception  # type: ignore[assignment]


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
CHROMA_PATH: Final[Path] = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME: Final[str] = "semantic_memory"
ANTHROPIC_MODEL: Final[str] = "claude-3-haiku-20240307"
OPENAI_MODEL: Final[str] = "gpt-5-mini"
DEFAULT_DATABASE_URL: Final[str] = (
    "postgresql://micelio:micelio@localhost:5432/email_agent"
)
ENV_FILE_PATH: Final[Path] = BASE_DIR / ".env"
DEFAULT_LLM_PROVIDER: Final[str] = "openai"
DEFAULT_EMAIL_PROVIDER: Final[str] = "mock"
DEFAULT_GMAIL_USER_ID: Final[str] = "me"
DEFAULT_GMAIL_QUERY: Final[str] = "in:inbox"
DEFAULT_GMAIL_MAX_RESULTS: Final[int] = 10
DEFAULT_GMAIL_OAUTH_PORT: Final[int] = 8765
DEFAULT_GMAIL_LABEL_PREFIX: Final[str] = "AUTO_TRIAGEM_"
DEFAULT_GMAIL_ARCHIVE_AFTER_TRIAGE: Final[bool] = False
DEFAULT_GMAIL_CREDENTIALS_FILE: Final[Path] = BASE_DIR / "gmail_credentials.json"
DEFAULT_GMAIL_TOKEN_FILE: Final[Path] = BASE_DIR / "gmail_token.json"
DEFAULT_GMAIL_AUTH_URI: Final[str] = "https://accounts.google.com/o/oauth2/auth"
DEFAULT_GMAIL_TOKEN_URI: Final[str] = "https://oauth2.googleapis.com/token"
GMAIL_SCOPES: Final[tuple[str, ...]] = ("https://www.googleapis.com/auth/gmail.modify",)

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

SYSTEM_PROMPT: Final[str] = f"""
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


def load_local_env_file() -> None:
    """Load key-value pairs from a local .env file without extra dependencies.

    Existing environment variables always win, which keeps shell overrides predictable.
    """

    if not ENV_FILE_PATH.exists():
        return

    for raw_line in ENV_FILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = value.strip().strip('"').strip("'")
        os.environ.setdefault(normalized_key, normalized_value)


class EmailAgentState(TypedDict):
    """LangGraph state shared between nodes."""

    email_data: dict[str, Any]
    rag_context: str
    classification_result: dict[str, Any]


load_local_env_file()


def _get_bool_env(env_name: str, default: bool) -> bool:
    """Read a boolean environment variable with a conservative parser."""

    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_llm_provider() -> str:
    """Return the configured LLM provider strategy.

    Supported values:
    - openai
    - anthropic
    - heuristic
    """

    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    return provider if provider in {"openai", "anthropic", "heuristic"} else DEFAULT_LLM_PROVIDER


def get_openai_model_name() -> str:
    """Return the OpenAI model used for classification and summarization."""

    return os.getenv("OPENAI_MODEL", OPENAI_MODEL).strip() or OPENAI_MODEL


def get_anthropic_model_name() -> str:
    """Return the Anthropic model used when the Anthropic provider is selected."""

    return os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL).strip() or ANTHROPIC_MODEL


def get_email_provider() -> str:
    """Return the active email provider strategy.

    Supported values:
    - mock: robust local data for development
    - gmail: real Gmail API using OAuth Desktop flow
    """

    provider = os.getenv("EMAIL_PROVIDER", DEFAULT_EMAIL_PROVIDER).strip().lower()
    return provider if provider in {"mock", "gmail"} else DEFAULT_EMAIL_PROVIDER


def get_gmail_credentials_file() -> Path:
    """Return the OAuth client secret JSON path for Gmail API access."""

    return Path(
        os.getenv("GMAIL_OAUTH_CLIENT_SECRET_FILE", str(DEFAULT_GMAIL_CREDENTIALS_FILE))
    ).expanduser()


def get_gmail_project_id() -> str:
    """Return the Google Cloud project id used by the Gmail OAuth client."""

    return os.getenv("GMAIL_PROJECT_ID", "").strip()


def get_gmail_client_id() -> str:
    """Return the Gmail OAuth desktop client id."""

    return os.getenv("GMAIL_CLIENT_ID", "").strip()


def get_gmail_client_secret() -> str:
    """Return the Gmail OAuth desktop client secret."""

    return os.getenv("GMAIL_CLIENT_SECRET", "").strip()


def get_gmail_auth_uri() -> str:
    """Return the OAuth authorization endpoint for Google."""

    return os.getenv("GMAIL_AUTH_URI", DEFAULT_GMAIL_AUTH_URI).strip() or DEFAULT_GMAIL_AUTH_URI


def get_gmail_token_uri() -> str:
    """Return the OAuth token endpoint for Google."""

    return os.getenv("GMAIL_TOKEN_URI", DEFAULT_GMAIL_TOKEN_URI).strip() or DEFAULT_GMAIL_TOKEN_URI


def get_gmail_token_file() -> Path:
    """Return the token file path used to cache Gmail access/refresh tokens."""

    return Path(os.getenv("GMAIL_OAUTH_TOKEN_FILE", str(DEFAULT_GMAIL_TOKEN_FILE))).expanduser()


def get_gmail_user_id() -> str:
    """Return the Gmail API user identifier. `me` is recommended."""

    return os.getenv("GMAIL_USER_ID", DEFAULT_GMAIL_USER_ID).strip() or DEFAULT_GMAIL_USER_ID


def get_gmail_query() -> str:
    """Return the Gmail search query used to fetch candidate messages."""

    return os.getenv("GMAIL_QUERY", DEFAULT_GMAIL_QUERY).strip() or DEFAULT_GMAIL_QUERY


def get_gmail_max_results(default_limit: int | None = None) -> int:
    """Resolve Gmail maxResults from the explicit call or the environment."""

    if default_limit is not None:
        return default_limit
    raw_limit = os.getenv("GMAIL_MAX_RESULTS", str(DEFAULT_GMAIL_MAX_RESULTS)).strip()
    try:
        return max(1, min(int(raw_limit), 100))
    except ValueError:
        return DEFAULT_GMAIL_MAX_RESULTS


def get_gmail_oauth_port() -> int:
    """Return the local port used by the OAuth desktop callback server."""

    raw_port = os.getenv("GMAIL_OAUTH_PORT", str(DEFAULT_GMAIL_OAUTH_PORT)).strip()
    try:
        return int(raw_port)
    except ValueError:
        return DEFAULT_GMAIL_OAUTH_PORT


def get_gmail_label_prefix() -> str:
    """Return the label prefix used for auto-triaged emails in Gmail."""

    return os.getenv("GMAIL_LABEL_PREFIX", DEFAULT_GMAIL_LABEL_PREFIX).strip() or DEFAULT_GMAIL_LABEL_PREFIX


def should_archive_after_triage() -> bool:
    """Whether the message should leave INBOX after receiving its category label."""

    return _get_bool_env("GMAIL_ARCHIVE_AFTER_TRIAGE", DEFAULT_GMAIL_ARCHIVE_AFTER_TRIAGE)


def _ensure_gmail_dependencies() -> None:
    """Validate that the optional Google dependencies exist before real Gmail usage."""

    if GoogleCredentials is None or InstalledAppFlow is None or build is None:
        raise RuntimeError(
            "Dependências do Gmail não estão instaladas. Instale "
            "`google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
        )


def _decode_base64url(content: str) -> str:
    """Decode Gmail message payload bodies which are returned in base64url format."""

    if not content:
        return ""
    padded_content = content + "=" * (-len(content) % 4)
    return base64.urlsafe_b64decode(padded_content.encode("utf-8")).decode(
        "utf-8",
        errors="replace",
    )


def _strip_html(html_content: str) -> str:
    """Convert simple HTML e-mail bodies into readable plain text."""

    without_tags = re.sub(r"<[^>]+>", " ", html_content)
    normalized = re.sub(r"\s+", " ", unescape(without_tags)).strip()
    return normalized


def _extract_body_from_payload(payload: dict[str, Any]) -> str:
    """Traverse Gmail MIME parts and return the most useful body text."""

    mime_type = str(payload.get("mimeType", ""))
    body = cast(dict[str, Any], payload.get("body", {}))
    data = str(body.get("data", ""))

    if mime_type == "text/plain" and data:
        return _decode_base64url(data)

    parts = cast(list[dict[str, Any]], payload.get("parts", []))
    for part in parts:
        plain_candidate = _extract_body_from_payload(part)
        if plain_candidate:
            return plain_candidate

    if mime_type == "text/html" and data:
        return _strip_html(_decode_base64url(data))

    if data:
        decoded = _decode_base64url(data)
        return _strip_html(decoded) if mime_type == "text/html" else decoded

    return ""


def _get_header_value(payload: dict[str, Any], header_name: str) -> str:
    """Extract one header value from the Gmail payload."""

    headers = cast(list[dict[str, Any]], payload.get("headers", []))
    for header in headers:
        if str(header.get("name", "")).lower() == header_name.lower():
            return str(header.get("value", "")).strip()
    return ""


def _gmail_message_to_email_dict(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw Gmail API message into the internal email payload shape."""

    payload = cast(dict[str, Any], message.get("payload", {}))
    body_text = _extract_body_from_payload(payload).strip()

    return {
        "message_id": str(message.get("id", "")),
        "thread_id": str(message.get("threadId", "")),
        "sender": _get_header_value(payload, "From"),
        "subject": _get_header_value(payload, "Subject"),
        "body": body_text or str(message.get("snippet", "")),
        "labels": cast(list[str], message.get("labelIds", [])),
    }


def _build_gmail_service() -> Any:
    """Authenticate and build a Gmail API service client.

    The implementation follows Google's official Python quickstart pattern for a
    desktop application, persisting OAuth tokens locally after the first consent.
    """

    _ensure_gmail_dependencies()

    token_file = get_gmail_token_file()
    creds: Any = None

    if token_file.exists():
        creds = GoogleCredentials.from_authorized_user_file(str(token_file), list(GMAIL_SCOPES))

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())
        else:
            client_id = get_gmail_client_id()
            client_secret = get_gmail_client_secret()
            project_id = get_gmail_project_id()
            credentials_file = get_gmail_credentials_file()

            if client_id and client_secret:
                client_config = {
                    "installed": {
                        "client_id": client_id,
                        "project_id": project_id or None,
                        "auth_uri": get_gmail_auth_uri(),
                        "token_uri": get_gmail_token_uri(),
                        "client_secret": client_secret,
                        "redirect_uris": [
                            f"http://localhost:{get_gmail_oauth_port()}/",
                            "http://localhost",
                        ],
                    }
                }
                flow = InstalledAppFlow.from_client_config(
                    client_config,
                    list(GMAIL_SCOPES),
                )
            else:
                if not credentials_file.exists():
                    raise RuntimeError(
                        "Credenciais OAuth do Gmail ausentes. Defina `GMAIL_CLIENT_ID` e "
                        "`GMAIL_CLIENT_SECRET` no .env ou forneça o JSON em "
                        f"{credentials_file}."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_file),
                    list(GMAIL_SCOPES),
                )
            creds = flow.run_local_server(port=get_gmail_oauth_port(), open_browser=False)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


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


MOCK_INBOX: list[MockEmail] = [
    MockEmail(
        message_id="msg_001",
        sender="talent@remotejobs.ai",
        subject="Entrevista técnica agendada para vaga Python remoto",
        body=(
            "Olá, Mauro! Seu perfil avançou no processo para Senior Python Engineer. "
            "Temos slots na terça e quarta para entrevista técnica e queremos confirmar "
            "sua disponibilidade. A posição é 100% remota, com foco em agentes de IA."
        ),
        labels=["INBOX", "IMPORTANT"],
    ),
    MockEmail(
        message_id="msg_002",
        sender="arquitetura@cloudguild.dev",
        subject="Boas práticas de GCP + PostgreSQL para microsserviços agentic",
        body=(
            "Segue o material da guilda sobre desenho de arquitetura com GCP, filas, "
            "PostgreSQL, observabilidade e estratégias de estado para agentes autônomos."
        ),
        labels=["INBOX"],
    ),
    MockEmail(
        message_id="msg_003",
        sender="news@makerweekly.com",
        subject="PETG translúcido, Bambu Lab A1 e 12 boardgames novos da semana",
        body=(
            "A edição de hoje fala sobre tuning de impressoras 3D, filamento PETG, "
            "novos acessórios para Bambu Lab e um compilado de boardgames independentes."
        ),
        labels=["INBOX", "CATEGORY_UPDATES"],
    ),
    MockEmail(
        message_id="msg_004",
        sender="billing@banksecure.com",
        subject="Fatura do cartão disponível e alerta de vencimento",
        body=(
            "Sua fatura fechou em R$ 4.382,91. O vencimento é dia 18 e já está disponível "
            "no app. Caso precise, é possível parcelar em até 12x."
        ),
        labels=["INBOX", "CATEGORY_PERSONAL"],
    ),
    MockEmail(
        message_id="msg_005",
        sender="courses@mlschool.io",
        subject="Novo curso: LangGraph para fluxos multiagente em produção",
        body=(
            "Abrimos inscrições para a nova turma com módulos de LangGraph, avaliação, "
            "memória de longo prazo e deploy de agentes em ambientes reais."
        ),
        labels=["INBOX"],
    ),
    MockEmail(
        message_id="msg_006",
        sender="promo@supercheapcoupons.biz",
        subject="VOCÊ GANHOU!!! iPhone grátis hoje clique agora",
        body=(
            "Oferta relâmpago! Clique em 5 minutos para resgatar seu prêmio exclusivo. "
            "Sem cadastro, sem custos, totalmente grátis e urgente!!!"
        ),
        labels=["INBOX", "CATEGORY_PROMOTIONS"],
    ),
    MockEmail(
        message_id="msg_007",
        sender="community@productbuilders.org",
        subject="Convite para conversar sobre colaboração em projeto de IA aplicada",
        body=(
            "Estamos montando um grupo pequeno para discutir um possível projeto envolvendo "
            "assistentes pessoais, automação de caixa de entrada e produtos para creators. "
            "Se tiver interesse, podemos marcar uma call exploratória."
        ),
        labels=["INBOX"],
    ),
    MockEmail(
        message_id="msg_008",
        sender="digest@worldsignals.com",
        subject="Resumo diário: mercado, tecnologia, geopolítica e startups",
        body=(
            "A newsletter de hoje traz destaques do mercado financeiro, novidades em IA, "
            "movimentos de startups e uma curadoria de leituras do dia."
        ),
        labels=["INBOX", "CATEGORY_UPDATES"],
    ),
]


def get_database_url() -> str:
    """Return the PostgreSQL connection string used by the transactional layer."""

    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_postgres_connection() -> Any:
    """Return a PostgreSQL connection configured for dict-like rows."""

    if psycopg is None:
        raise RuntimeError(
            "psycopg não está instalado. Instale `psycopg[binary]` para usar o PostgreSQL."
        )
    return psycopg.connect(get_database_url(), row_factory=dict_row)


def init_postgres_db() -> None:
    """Create the processed_emails table required by the MVP in PostgreSQL."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_emails (
                message_id VARCHAR(255) PRIMARY KEY,
                sender TEXT NOT NULL,
                subject TEXT NOT NULL,
                status VARCHAR(50) NOT NULL,
                category VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_status
                ON processed_emails (status)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_emails_category
                ON processed_emails (category)
                """
            )
        connection.commit()


def get_chroma_collection() -> Collection:
    """Return the local Chroma collection used as semantic memory."""

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def bootstrap_services() -> None:
    """Initialize local persistence layers."""

    init_postgres_db()
    _ = get_chroma_collection()


def gmail_list_messages_stub(limit: int | None = None) -> list[dict[str, Any]]:
    """Mock Gmail API list call.

    The return format deliberately mirrors what a real integration layer might expose:
    a list of dictionaries with stable IDs and email metadata.
    """

    messages = [email.as_dict() for email in MOCK_INBOX]
    return messages if limit is None else messages[:limit]


def gmail_get_message_stub(message_id: str) -> dict[str, Any] | None:
    """Mock Gmail API get call."""

    for email in MOCK_INBOX:
        if email.message_id == message_id:
            return email.as_dict()
    return None


def gmail_move_message_stub(message_id: str, category: str) -> dict[str, Any]:
    """Mock Gmail API move call.

    The email is never marked as read here; only a simulated label/move result is returned.
    """

    return {
        "message_id": message_id,
        "destination_label": f"AUTO_{category}",
        "moved": True,
        "marked_as_read": False,
    }


def gmail_list_messages_real(limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch Gmail inbox messages through the official Gmail API."""

    service = _build_gmail_service()
    max_results = get_gmail_max_results(limit)

    response = (
        service.users()
        .messages()
        .list(
            userId=get_gmail_user_id(),
            maxResults=max_results,
            q=get_gmail_query(),
            labelIds=["INBOX"],
            includeSpamTrash=False,
        )
        .execute()
    )

    raw_messages = cast(list[dict[str, Any]], response.get("messages", []))
    hydrated_messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        message_id = str(raw_message.get("id", ""))
        if not message_id:
            continue
        full_message = (
            service.users()
            .messages()
            .get(userId=get_gmail_user_id(), id=message_id, format="full")
            .execute()
        )
        hydrated_messages.append(_gmail_message_to_email_dict(cast(dict[str, Any], full_message)))

    return hydrated_messages


def gmail_get_message_real(message_id: str) -> dict[str, Any] | None:
    """Retrieve one Gmail message from the authenticated account."""

    try:
        service = _build_gmail_service()
        full_message = (
            service.users()
            .messages()
            .get(userId=get_gmail_user_id(), id=message_id, format="full")
            .execute()
        )
        return _gmail_message_to_email_dict(cast(dict[str, Any], full_message))
    except HttpError:
        return None


def _ensure_gmail_label(service: Any, label_name: str) -> str:
    """Ensure the destination Gmail label exists and return its id."""

    labels_response = service.users().labels().list(userId=get_gmail_user_id()).execute()
    labels = cast(list[dict[str, Any]], labels_response.get("labels", []))
    for label in labels:
        if str(label.get("name", "")) == label_name:
            return str(label.get("id", ""))

    created_label = (
        service.users()
        .labels()
        .create(
            userId=get_gmail_user_id(),
            body={
                "name": label_name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
        )
        .execute()
    )
    return str(created_label.get("id", ""))


def gmail_move_message_real(message_id: str, category: str) -> dict[str, Any]:
    """Apply Gmail labels to represent triage without marking the email as read.

    By default the message remains unread and stays in INBOX. If
    `GMAIL_ARCHIVE_AFTER_TRIAGE=true`, the INBOX label is removed after adding the
    category label, which behaves like an archive/move.
    """

    service = _build_gmail_service()
    destination_label = f"{get_gmail_label_prefix()}{category}"
    destination_label_id = _ensure_gmail_label(service, destination_label)

    remove_label_ids: list[str] = []
    if should_archive_after_triage():
        remove_label_ids.append("INBOX")

    modified_message = (
        service.users()
        .messages()
        .modify(
            userId=get_gmail_user_id(),
            id=message_id,
            body={
                "addLabelIds": [destination_label_id],
                "removeLabelIds": remove_label_ids,
            },
        )
        .execute()
    )

    resulting_labels = cast(list[str], modified_message.get("labelIds", []))
    return {
        "message_id": message_id,
        "destination_label": destination_label,
        "moved": True,
        "archived": should_archive_after_triage(),
        "marked_as_read": False,
        "resulting_label_ids": resulting_labels,
    }


def list_email_messages(limit: int | None = None) -> list[dict[str, Any]]:
    """Unified provider entrypoint for listing inbox messages."""

    if get_email_provider() == "gmail":
        return gmail_list_messages_real(limit=limit)
    return gmail_list_messages_stub(limit=limit)


def get_email_message(message_id: str) -> dict[str, Any] | None:
    """Unified provider entrypoint for loading one message."""

    if get_email_provider() == "gmail":
        return gmail_get_message_real(message_id)
    return gmail_get_message_stub(message_id)


def move_email_message(message_id: str, category: str) -> dict[str, Any]:
    """Unified provider entrypoint for applying post-classification action."""

    if get_email_provider() == "gmail":
        return gmail_move_message_real(message_id, category)
    return gmail_move_message_stub(message_id, category)


def _safe_json_loads(raw_text: str) -> dict[str, Any]:
    """Extract a JSON object from model output, tolerating wrappers when possible."""

    try:
        return cast(dict[str, Any], json.loads(raw_text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _normalize_classification(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the model output to our strict contract."""

    category = str(payload.get("categoria", "EM_DUVIDA")).strip().upper()
    if category not in ALLOWED_CATEGORIES:
        category = "EM_DUVIDA"

    priority_categories = {"CARREIRA_PRIORIDADE", "PROJETOS_TECH"}
    is_priority = category in priority_categories

    motivo = str(payload.get("motivo_curto", "Classificacao ajustada pelo validador local.")).strip()
    if not motivo:
        motivo = "Classificacao ajustada pelo validador local."

    return {
        "categoria": category,
        "motivo_curto": motivo,
        "is_priority": is_priority,
    }


def _heuristic_classifier(email_data: dict[str, Any], rag_context: str) -> dict[str, Any]:
    """Fallback classifier used when Anthropic is unavailable.

    This keeps the MVP demonstrable locally while preserving the exact JSON contract.
    """

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
        {
            "categoria": category,
            "motivo_curto": reason,
            "is_priority": category in {"CARREIRA_PRIORIDADE", "PROJETOS_TECH"},
        }
    )


def _openai_client() -> OpenAI | None:
    """Build an OpenAI client only when the SDK and key are available."""

    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    return OpenAI(api_key=api_key)


def _anthropic_client() -> Anthropic | None:
    """Build an Anthropic client only when the SDK and key are available."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if Anthropic is None or not api_key:
        return None
    return Anthropic(api_key=api_key)


def _extract_openai_output_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API payload."""

    output_text = getattr(response, "output_text", "")
    if output_text:
        return str(output_text).strip()

    output_items = getattr(response, "output", [])
    text_fragments: list[str] = []
    for item in output_items:
        content_items = getattr(item, "content", [])
        for content in content_items:
            text_value = getattr(content, "text", "")
            if text_value:
                text_fragments.append(str(text_value))
    return "\n".join(text_fragments).strip()


def _invoke_openai_text(system_prompt: str, user_prompt: str, json_output: bool) -> str:
    """Execute an OpenAI Responses API request."""

    client = _openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable. Falling back to another strategy.")

    text_config: dict[str, Any]
    if json_output:
        text_config = {"format": {"type": "json_object"}}
    else:
        text_config = {"format": {"type": "text"}}

    response = client.responses.create(
        model=get_openai_model_name(),
        instructions=system_prompt,
        input=user_prompt,
        temperature=0,
        text=text_config,
    )
    raw_output = _extract_openai_output_text(response)
    if not raw_output:
        raise RuntimeError("OpenAI returned no text output.")
    return raw_output


def _invoke_anthropic_json(system_prompt: str, user_prompt: str) -> str:
    """Execute a raw Anthropic call and return the first text block."""

    client = _anthropic_client()
    if client is None:
        raise RuntimeError("Anthropic client unavailable. Falling back to another strategy.")

    response = client.messages.create(
        model=get_anthropic_model_name(),
        max_tokens=300,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text_blocks = [block.text for block in response.content if getattr(block, "type", "") == "text"]
    if not text_blocks:
        raise RuntimeError("Anthropic returned no text blocks.")
    return "\n".join(text_blocks).strip()


def _invoke_llm_text(system_prompt: str, user_prompt: str, json_output: bool) -> str:
    """Route one request through the configured LLM provider."""

    provider = get_llm_provider()
    if provider == "heuristic":
        raise RuntimeError("Heuristic provider does not support direct LLM invocation.")
    if provider == "openai":
        return _invoke_openai_text(system_prompt, user_prompt, json_output=json_output)
    if provider == "anthropic":
        return _invoke_anthropic_json(system_prompt, user_prompt)
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def classify_email_with_llm(email_data: dict[str, Any], rag_context: str) -> dict[str, Any]:
    """Classify one email using Anthropic, with a safe local fallback."""

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

    try:
        raw_output = _invoke_llm_text(SYSTEM_PROMPT, prompt, json_output=True)
        parsed = _safe_json_loads(raw_output)
        return _normalize_classification(parsed)
    except Exception:
        return _heuristic_classifier(email_data, rag_context)


def summarize_priority_email(email_data: dict[str, Any]) -> str:
    """Create a factual one-paragraph summary for priority emails."""

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

    try:
        return _invoke_llm_text(SUMMARY_PROMPT, prompt, json_output=False)
    except Exception:
        return (
            f"Resumo factual: e-mail de '{email_data.get('sender', '')}' com assunto "
            f"'{email_data.get('subject', '')}', tratado como item prioritario para acompanhamento."
        )


def upsert_semantic_memory(
    message_id: str,
    email_data: dict[str, Any],
    learned_text: str,
    category: str,
) -> None:
    """Persist new semantic memory into ChromaDB."""

    collection = get_chroma_collection()
    document = (
        f"Categoria: {category}\n"
        f"Remetente: {email_data.get('sender', '')}\n"
        f"Assunto: {email_data.get('subject', '')}\n"
        f"Aprendizado: {learned_text}"
    )
    collection.upsert(
        ids=[f"{message_id}-{uuid.uuid4().hex[:8]}"],
        documents=[document],
        metadatas=[
            {
                "message_id": message_id,
                "sender": str(email_data.get("sender", "")),
                "subject": str(email_data.get("subject", "")),
                "category": category,
            }
        ],
    )


def save_processed_email(message_id: str, sender: str, subject: str, status: str, category: str) -> None:
    """Persist the triage result into PostgreSQL."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
            """
            INSERT INTO processed_emails (message_id, sender, subject, status, category)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT(message_id) DO UPDATE SET
                sender = EXCLUDED.sender,
                subject = EXCLUDED.subject,
                status = EXCLUDED.status,
                category = EXCLUDED.category,
                updated_at = NOW()
            """,
            (message_id, sender, subject, status, category),
            )
        connection.commit()


def _is_already_processed(message_id: str) -> bool:
    """Check whether the email was already handled previously."""

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM processed_emails WHERE message_id = %s",
                (message_id,),
            )
            row = cursor.fetchone()
    return row is not None


def retrieve_context(state: EmailAgentState) -> EmailAgentState:
    """LangGraph node: retrieve related semantic memory from ChromaDB."""

    email_data = state["email_data"]
    query_text = f"{email_data.get('subject', '')}\n{email_data.get('body', '')}"
    collection = get_chroma_collection()

    result = collection.query(query_texts=[query_text], n_results=2)
    documents = result.get("documents", [[]])
    flattened = documents[0] if documents and documents[0] else []
    rag_context = "\n---\n".join(flattened) if flattened else "Sem memoria semantica relevante."

    return {
        **state,
        "rag_context": rag_context,
    }


def classify_email(state: EmailAgentState) -> EmailAgentState:
    """LangGraph node: classify the email through the LLM contract."""

    classification_result = classify_email_with_llm(
        email_data=state["email_data"],
        rag_context=state["rag_context"],
    )
    return {
        **state,
        "classification_result": classification_result,
    }


def execute_action(state: EmailAgentState) -> EmailAgentState:
    """LangGraph node: save result in PostgreSQL and simulate moving the email."""

    email_data = state["email_data"]
    classification = state["classification_result"]
    category = str(classification["categoria"])
    status = "EM_DUVIDA" if category == "EM_DUVIDA" else "PROCESSADO"

    save_processed_email(
        message_id=str(email_data["message_id"]),
        sender=str(email_data["sender"]),
        subject=str(email_data["subject"]),
        status=status,
        category=category,
    )

    if category != "EM_DUVIDA":
        move_result = move_email_message(str(email_data["message_id"]), category)
        classification["move_result"] = move_result

    return {
        **state,
        "classification_result": classification,
    }


def dynamic_learning(state: EmailAgentState) -> EmailAgentState:
    """LangGraph node: summarize and store priority learnings in ChromaDB."""

    email_data = state["email_data"]
    classification = state["classification_result"]
    summary = summarize_priority_email(email_data)
    upsert_semantic_memory(
        message_id=str(email_data["message_id"]),
        email_data=email_data,
        learned_text=summary,
        category=str(classification["categoria"]),
    )
    classification["priority_summary"] = summary
    return {
        **state,
        "classification_result": classification,
    }


def should_run_dynamic_learning(state: EmailAgentState) -> str:
    """Conditional router for the priority learning node."""

    if bool(state["classification_result"].get("is_priority", False)):
        return "dynamic_learning"
    return END


def build_email_triage_graph():
    """Create and compile the LangGraph state machine."""

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
        {
            "dynamic_learning": "dynamic_learning",
            END: END,
        },
    )
    graph_builder.add_edge("dynamic_learning", END)
    return graph_builder.compile()


def run_triage_workflow(limit: int | None = None) -> list[dict[str, Any]]:
    """Process inbox messages through the LangGraph workflow."""

    bootstrap_services()
    graph = build_email_triage_graph()
    outputs: list[dict[str, Any]] = []

    for email in list_email_messages(limit=limit):
        if _is_already_processed(str(email["message_id"])):
            continue

        final_state = graph.invoke(
            {
                "email_data": email,
                "rag_context": "",
                "classification_result": {},
            }
        )
        outputs.append(cast(dict[str, Any], final_state))

    return outputs


def get_dashboard_metrics() -> dict[str, Any]:
    """Compute the dashboard metrics used by Streamlit from PostgreSQL data."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM processed_emails")
            total_processed = int(cursor.fetchone()["total"])
            cursor.execute(
                "SELECT COUNT(*) AS total FROM processed_emails WHERE status = 'EM_DUVIDA'"
            )
            pending_review = int(cursor.fetchone()["total"])
            cursor.execute(
                """
                SELECT COUNT(*) AS total FROM processed_emails
                WHERE category IN ('CARREIRA_PRIORIDADE', 'PROJETOS_TECH')
                """
            )
            priority_count = int(cursor.fetchone()["total"])

    automated = max(total_processed - pending_review, 0)
    automation_rate = (automated / total_processed * 100.0) if total_processed else 0.0

    return {
        "emails_processados_hoje": total_processed,
        "prioritarios_detectados": priority_count,
        "fila_revisao": pending_review,
        "taxa_automacao": round(automation_rate, 1),
    }


def get_review_queue() -> list[dict[str, Any]]:
    """Return pending manual-review emails, enriched with the active provider payload."""

    bootstrap_services()

    with get_postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
            """
            SELECT message_id, sender, subject, status, category
            FROM processed_emails
            WHERE status = 'EM_DUVIDA'
            ORDER BY sender, subject
            """
            )
            rows = cursor.fetchall()

    queue: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(cast(dict[str, Any], row))
        email_payload = get_email_message(str(row_dict["message_id"])) or {}
        queue.append({**row_dict, "body": email_payload.get("body", "")})
    return queue


def manual_reclassify_email(message_id: str, forced_category: str) -> dict[str, Any]:
    """Update PostgreSQL and inject a manual-learning note into ChromaDB."""

    if forced_category not in MANUAL_REVIEW_CATEGORIES:
        raise ValueError(f"Categoria manual inválida: {forced_category}")

    bootstrap_services()
    email_data = get_email_message(message_id)
    if email_data is None:
        raise ValueError(f"E-mail não encontrado no provider ativo: {message_id}")

    save_processed_email(
        message_id=message_id,
        sender=str(email_data["sender"]),
        subject=str(email_data["subject"]),
        status="PROCESSADO",
        category=forced_category,
    )

    move_result = move_email_message(message_id, forced_category)

    learned_text = (
        "Aprendizado manual do revisor: este e-mail foi reclassificado para "
        f"{forced_category} após análise humana."
    )
    upsert_semantic_memory(
        message_id=message_id,
        email_data=email_data,
        learned_text=learned_text,
        category=forced_category,
    )

    return {
        "message_id": message_id,
        "forced_category": forced_category,
        "updated": True,
        "move_result": move_result,
    }


def get_connection_status() -> dict[str, str]:
    """Expose simple connection health for the Streamlit sidebar."""

    postgres_status = "OFFLINE"
    try:
        bootstrap_services()
        postgres_status = "OK"
    except Exception:
        postgres_status = "OFFLINE"

    chroma_status = "OK" if CHROMA_PATH.exists() else "OFFLINE"

    gmail_status = "MOCK"
    if get_email_provider() == "gmail":
        credentials_ready = bool(get_gmail_client_id() and get_gmail_client_secret()) or get_gmail_credentials_file().exists()
        deps_ready = build is not None and GoogleCredentials is not None
        if credentials_ready and deps_ready:
            gmail_status = "CONFIGURADO"
        elif not credentials_ready:
            gmail_status = "SEM_CREDENTIALS_JSON"
        else:
            gmail_status = "DEPENDENCIAS_PENDENTES"

    llm_provider = get_llm_provider()
    if llm_provider == "openai":
        llm_status = "CONFIGURADO" if os.getenv("OPENAI_API_KEY") else "FALLBACK_HEURISTICO"
        llm_model = get_openai_model_name()
    elif llm_provider == "anthropic":
        llm_status = "CONFIGURADO" if os.getenv("ANTHROPIC_API_KEY") else "FALLBACK_HEURISTICO"
        llm_model = get_anthropic_model_name()
    else:
        llm_status = "HEURISTICO"
        llm_model = "local"

    return {
        "gmail": gmail_status,
        "postgres": postgres_status,
        "chroma": chroma_status,
        "llm_provider": llm_provider.upper(),
        "llm_status": llm_status,
        "llm_model": llm_model,
    }


def get_manual_review_categories() -> tuple[str, ...]:
    """Expose the approved manual buttons to the UI."""

    return MANUAL_REVIEW_CATEGORIES


try:
    bootstrap_services()
except Exception:
    # Importing the module should not crash the Streamlit app before the database exists.
    pass
