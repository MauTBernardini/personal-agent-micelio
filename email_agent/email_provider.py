"""Email source integrations: mock inbox plus Gmail API access."""

from __future__ import annotations

import base64
import re
from html import unescape
from typing import Any, cast

from email_agent.models import MockEmail
from email_agent.settings import (
    GMAIL_SCOPES,
    GoogleAuthRequest,
    GoogleCredentials,
    HttpError,
    InstalledAppFlow,
    build,
    get_email_provider,
    get_gmail_after_date,
    get_gmail_auth_uri,
    get_gmail_client_id,
    get_gmail_client_secret,
    get_gmail_credentials_file,
    get_gmail_label_prefix,
    get_gmail_max_results,
    get_gmail_oauth_port,
    get_gmail_page_size,
    get_gmail_project_id,
    get_gmail_query,
    get_gmail_token_file,
    get_gmail_token_uri,
    get_gmail_user_id,
    should_archive_after_triage,
    should_require_gmail_inbox,
)
from email_agent.storage import is_already_processed


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


def _ensure_gmail_dependencies() -> None:
    if GoogleCredentials is None or InstalledAppFlow is None or build is None:
        raise RuntimeError(
            "Dependências do Gmail não estão instaladas. Instale "
            "`google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
        )


def _decode_base64url(content: str) -> str:
    if not content:
        return ""
    padded_content = content + "=" * (-len(content) % 4)
    return base64.urlsafe_b64decode(padded_content.encode("utf-8")).decode("utf-8", errors="replace")


def _strip_html(html_content: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_content)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _extract_body_from_payload(payload: dict[str, Any]) -> str:
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
    headers = cast(list[dict[str, Any]], payload.get("headers", []))
    for header in headers:
        if str(header.get("name", "")).lower() == header_name.lower():
            return str(header.get("value", "")).strip()
    return ""


def _gmail_message_to_email_dict(message: dict[str, Any]) -> dict[str, Any]:
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
                flow = InstalledAppFlow.from_client_config(client_config, list(GMAIL_SCOPES))
            else:
                if not credentials_file.exists():
                    raise RuntimeError(
                        "Credenciais OAuth do Gmail ausentes. Defina `GMAIL_CLIENT_ID` e "
                        "`GMAIL_CLIENT_SECRET` no .env ou forneça o JSON em "
                        f"{credentials_file}."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), list(GMAIL_SCOPES))

            creds = flow.run_local_server(port=get_gmail_oauth_port(), open_browser=False)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def gmail_list_messages_stub(limit: int | None = None) -> list[dict[str, Any]]:
    messages = [email.as_dict() for email in MOCK_INBOX]
    return messages if limit is None else messages[:limit]


def gmail_get_message_stub(message_id: str) -> dict[str, Any] | None:
    for email in MOCK_INBOX:
        if email.message_id == message_id:
            return email.as_dict()
    return None


def gmail_move_message_stub(message_id: str, category: str) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "destination_label": f"AUTO_{category}",
        "moved": True,
        "marked_as_read": False,
    }


def gmail_list_messages_real(limit: int | None = None) -> list[dict[str, Any]]:
    service = _build_gmail_service()
    hydrated_messages: list[dict[str, Any]] = []
    total_limit = get_gmail_max_results(limit)
    page_size = get_gmail_page_size()
    page_token: str | None = None
    triage_label_prefix = get_gmail_label_prefix()
    query_parts = ["is:unread"]
    after_date = get_gmail_after_date()
    if after_date:
        query_parts.insert(0, f"after:{after_date}")
    if should_require_gmail_inbox():
        query_parts.append("in:inbox")
    if get_gmail_query():
        query_parts.append(get_gmail_query())
    query = " ".join(part for part in query_parts if part).strip()
    require_inbox = should_require_gmail_inbox()

    while True:
        remaining = None if total_limit is None else max(total_limit - len(hydrated_messages), 0)
        if remaining == 0:
            break

        current_page_size = page_size if remaining is None else min(page_size, remaining)
        list_kwargs: dict[str, Any] = {
            "userId": get_gmail_user_id(),
            "maxResults": current_page_size,
            "q": query,
            "includeSpamTrash": False,
            "pageToken": page_token,
        }
        if require_inbox:
            list_kwargs["labelIds"] = ["INBOX"]
        response = cast(dict[str, Any], service.users().messages().list(**list_kwargs).execute())

        raw_messages = cast(list[dict[str, Any]], response.get("messages", []))
        if not raw_messages:
            break

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
            normalized_message = _gmail_message_to_email_dict(cast(dict[str, Any], full_message))
            labels = cast(list[str], normalized_message.get("labels", []))
            if any(label.startswith(triage_label_prefix) for label in labels):
                continue
            if is_already_processed(message_id):
                continue
            hydrated_messages.append(normalized_message)
            if total_limit is not None and len(hydrated_messages) >= total_limit:
                break

        if total_limit is not None and len(hydrated_messages) >= total_limit:
            break

        page_token = cast(str | None, response.get("nextPageToken"))
        if not page_token:
            break

    return hydrated_messages


def gmail_get_message_real(message_id: str) -> dict[str, Any] | None:
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
            body={"addLabelIds": [destination_label_id], "removeLabelIds": remove_label_ids},
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
    if get_email_provider() == "gmail":
        return gmail_list_messages_real(limit=limit)
    return gmail_list_messages_stub(limit=limit)


def get_email_message(message_id: str) -> dict[str, Any] | None:
    if get_email_provider() == "gmail":
        return gmail_get_message_real(message_id)
    return gmail_get_message_stub(message_id)


def move_email_message(message_id: str, category: str) -> dict[str, Any]:
    if get_email_provider() == "gmail":
        return gmail_move_message_real(message_id, category)
    return gmail_move_message_stub(message_id, category)

