# Personal Agent Micelio

## Configuração

### 1. Criar e revisar o `.env`

O projeto agora lê automaticamente um arquivo local `.env` na raiz.

Arquivo criado neste repositório:

```env
AGENT_API_BASE_URL=http://127.0.0.1:8000
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-3-haiku-20240307
EMAIL_PROVIDER=mock
POSTGRES_DB=email_agent
POSTGRES_USER=micelio
POSTGRES_PASSWORD=micelio
POSTGRES_PORT=5434
DATABASE_URL=postgresql://micelio:micelio@localhost:5434/email_agent
PGADMIN_DEFAULT_EMAIL=admin@micelio.local
PGADMIN_DEFAULT_PASSWORD=micelio_admin
PGADMIN_PORT=5052
GMAIL_OAUTH_CLIENT_SECRET_FILE=gmail_credentials.json
GMAIL_OAUTH_TOKEN_FILE=gmail_token.json
GMAIL_PROJECT_ID=
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_AUTH_URI=https://accounts.google.com/o/oauth2/auth
GMAIL_TOKEN_URI=https://oauth2.googleapis.com/token
GMAIL_USER_ID=me
GMAIL_AFTER_DATE=today
GMAIL_REQUIRE_INBOX=true
GMAIL_QUERY=
GMAIL_MAX_RESULTS=0
GMAIL_PAGE_SIZE=100
GMAIL_LABEL_PREFIX=AUTO_TRIAGEM_
GMAIL_ARCHIVE_AFTER_TRIAGE=false
GMAIL_OAUTH_PORT=8765
```

Preencha:

- `AGENT_API_BASE_URL`: URL base da FastAPI consumida pelo Streamlit.
- `LLM_PROVIDER`: `gemini`, `openai`, `anthropic` ou `heuristic`.
- `GEMINI_API_KEY`: chave da Gemini Developer API, criada no Google AI Studio.
- `GEMINI_MODEL`: modelo Gemini para classificação e resumo. Recomendação inicial: `gemini-2.5-flash-lite`.
- `GOOGLE_API_KEY`: alternativa ao `GEMINI_API_KEY`; o SDK aceita qualquer um, com precedência para `GOOGLE_API_KEY`.
- `OPENAI_API_KEY`: opcional, caso você queira manter OpenAI como provider secundário.
- `OPENAI_MODEL`: modelo OpenAI quando `LLM_PROVIDER=openai`.
- `ANTHROPIC_API_KEY`: opcional, só se você quiser usar Anthropic como provider.
- `ANTHROPIC_MODEL`: modelo Anthropic quando `LLM_PROVIDER=anthropic`.
- `EMAIL_PROVIDER`: `mock` para desenvolvimento local ou `gmail` para conta real.
- `POSTGRES_DB`: nome do banco do container.
- `POSTGRES_USER`: usuário do PostgreSQL.
- `POSTGRES_PASSWORD`: senha do PostgreSQL.
- `POSTGRES_PORT`: porta local publicada pelo Docker.
- `DATABASE_URL`: conexão usada pela aplicação Python.
- `PGADMIN_DEFAULT_EMAIL`: login web do pgAdmin.
- `PGADMIN_DEFAULT_PASSWORD`: senha web do pgAdmin.
- `PGADMIN_PORT`: porta local do pgAdmin.
- `GMAIL_OAUTH_TOKEN_FILE`: caminho do token OAuth persistido após o primeiro login.
- `GMAIL_PROJECT_ID`: `project_id` do client OAuth do Google Cloud.
- `GMAIL_CLIENT_ID`: `client_id` do OAuth Desktop App.
- `GMAIL_CLIENT_SECRET`: `client_secret` do OAuth Desktop App.
- `GMAIL_AUTH_URI`: endpoint de autorização do Google.
- `GMAIL_TOKEN_URI`: endpoint de troca/refresh de token do Google.
- `GMAIL_OAUTH_CLIENT_SECRET_FILE`: fallback opcional para usar o JSON completo em vez das variáveis.
- `GMAIL_USER_ID`: normalmente `me`.
- `GMAIL_AFTER_DATE`: data mínima do filtro do Gmail. Aceita `today`, `YYYY-MM-DD`, `YYYY/MM/DD` ou vazio para desabilitar.
- `GMAIL_REQUIRE_INBOX`: `true` para restringir a triagem à inbox; `false` para buscar em qualquer label compatível com a query.
- `GMAIL_QUERY`: filtro extra opcional anexado ao recorte padrão da triagem.
- `GMAIL_MAX_RESULTS`: limite total de e-mails por rodada; use `0` para não impor teto.
- `GMAIL_PAGE_SIZE`: tamanho de cada página buscada na Gmail API durante a paginação.
- `GMAIL_LABEL_PREFIX`: prefixo das labels criadas para classificar e-mails no Gmail.
- `GMAIL_ARCHIVE_AFTER_TRIAGE`: `true` para remover da inbox após rotular; `false` para manter na inbox.
- `GMAIL_OAUTH_PORT`: porta local usada pelo callback do OAuth Desktop.

Seleção padrão da triagem no Gmail:

- Apenas e-mails a partir de `GMAIL_AFTER_DATE` por padrão.
- Apenas e-mails não lidos.
- Apenas mensagens ainda na `INBOX` quando `GMAIL_REQUIRE_INBOX=true`.
- Exclui mensagens que já tenham label com o prefixo de triagem.
- Exclui mensagens cujo `message_id` já foi persistido no PostgreSQL como processado.
- Faz paginação no Gmail até consumir tudo ou até atingir `GMAIL_MAX_RESULTS` quando ele for maior que zero.

O `.env` foi adicionado ao `.gitignore`, então suas credenciais não entram no Git.

### 2. Subir o PostgreSQL dockerizado

O repositório agora já contém um [`docker-compose.yml`](/home/mau-bernardini/projects/personal-agent-micelio/docker-compose.yml) real que lê as variáveis do `.env`.

Conteúdo do arquivo:

```yaml
services:
  postgres:
    image: postgres:16
    container_name: personal-agent-micelio-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  pgadmin:
    image: dpage/pgadmin4:8
    container_name: personal-agent-micelio-pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_DEFAULT_EMAIL}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_DEFAULT_PASSWORD}
    ports:
      - "${PGADMIN_PORT}:80"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

Com o `.env` preenchido, rode:

```bash
docker compose up -d
```

Verifique se o container ficou saudável:

```bash
docker ps
```

### 2.1. Acessar o pgAdmin via web

Depois do `docker compose up -d`, abra:

```text
http://localhost:5052
```

Se você tiver alterado `PGADMIN_PORT`, use a porta definida no `.env`.

Login do pgAdmin:

- E-mail: valor de `PGADMIN_DEFAULT_EMAIL`
- Senha: valor de `PGADMIN_DEFAULT_PASSWORD`

Para cadastrar o servidor PostgreSQL dentro do pgAdmin, use:

- Name: `personal-agent-micelio-postgres`
- Host name/address: `postgres`
- Port: `5432`
- Maintenance database: valor de `POSTGRES_DB`
- Username: valor de `POSTGRES_USER`
- Password: valor de `POSTGRES_PASSWORD`

O host deve ser `postgres` porque o pgAdmin acessa o banco pela rede interna do Docker Compose, não por `localhost`.

### 3. Criar o ambiente Python

Use Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar dependências

Instale pelo menos os pacotes abaixo:

```bash
pip install -r requirements.txt
```

### 5. Configurar a conexão real com Gmail

Hoje o projeto suporta dois modos:

- `EMAIL_PROVIDER=mock`: usa os e-mails falsos do projeto.
- `EMAIL_PROVIDER=gmail`: usa a Gmail API real.

Como a conexão real funciona:

1. Você cria um projeto no Google Cloud.
2. Habilita a Gmail API.
3. Cria um OAuth Client do tipo `Desktop app`.
4. Copia do JSON OAuth apenas os campos `project_id`, `client_id` e `client_secret` para o `.env`.
5. Na primeira execução, o app abre um fluxo OAuth local e grava um token reutilizável em `GMAIL_OAUTH_TOKEN_FILE`.
6. Depois disso, o agente usa a Gmail API para listar mensagens, buscar o conteúdo e aplicar labels de triagem.

O recorte padrão da busca no Gmail foi redesenhado para o cenário operacional do agente:

- e-mails a partir da data mínima configurada em `GMAIL_AFTER_DATE`;
- não lidos;
- ainda na inbox quando `GMAIL_REQUIRE_INBOX=true`;
- sem label de triagem anterior;
- e ainda não registrados como processados no PostgreSQL.

`GMAIL_QUERY` agora funciona como um refinamento adicional opcional sobre esse recorte base.

O agente não marca e-mails como lidos. Quando `GMAIL_ARCHIVE_AFTER_TRIAGE=false`, ele apenas adiciona a label da categoria. Quando `true`, ele remove a label `INBOX` após classificar, simulando um arquivamento.

Passo a passo no Google Cloud:

1. Acesse o Google Cloud Console.
2. Crie ou selecione um projeto.
3. Ative a Gmail API.
4. Configure a tela de consentimento OAuth.
5. Crie um cliente OAuth 2.0 do tipo `Desktop app`.
6. Abra o JSON baixado e copie `project_id`, `client_id` e `client_secret` para o `.env`.
7. Troque `EMAIL_PROVIDER=mock` para `EMAIL_PROVIDER=gmail`.

Exemplo:

```env
EMAIL_PROVIDER=gmail
GMAIL_PROJECT_ID=seu_project_id
GMAIL_CLIENT_ID=seu_client_id.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=seu_client_secret
```

O campo `GMAIL_OAUTH_CLIENT_SECRET_FILE` continua disponível apenas como fallback opcional.

### 6. Configurar o provedor de LLM

O core foi reestruturado para usar uma camada de provider de LLM.

Opções disponíveis:

- `LLM_PROVIDER=gemini`: padrão recomendado para este MVP, começando por `gemini-2.5-flash-lite`.
- `LLM_PROVIDER=openai`: compatível, caso você queira manter OpenAI.
- `LLM_PROVIDER=anthropic`: compatível, caso você queira manter Claude em algum cenário.
- `LLM_PROVIDER=heuristic`: não chama API externa; usa apenas regras locais.

Configuração recomendada para o MVP:

```env
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=sua_chave_aqui
```

Se o provider selecionado não estiver configurado corretamente ou a chamada falhar, o sistema exibe erro no Streamlit em vez de cair automaticamente para heurística. A heurística só é usada quando `LLM_PROVIDER=heuristic`.

Como obter a chave do Gemini:

1. Acesse o Google AI Studio.
2. Vá em `Get API key` / `API keys`.
3. Crie uma chave da Gemini Developer API.
4. Cole em `GEMINI_API_KEY` no `.env`.

Documentação oficial:

- SDK Python `google-genai`: https://ai.google.dev/gemini-api/docs/downloads
- API keys: https://ai.google.dev/gemini-api/docs/api-key
- Pricing: https://ai.google.dev/pricing

### 7. Primeiro teste do core e da API

Esse teste valida se:

- o `.env` foi lido;
- o PostgreSQL está acessível;
- a tabela `processed_emails` foi criada;
- o fluxo agentic consegue processar os e-mails do provider ativo;
- a FastAPI sobe corretamente e expõe o backend para o Streamlit.

Smoke test direto do core:

```bash
python3 -c "from core_agent import run_triage_workflow; print(run_triage_workflow())"
```

Resultado esperado:

- uma lista com os e-mails processados;
- registros criados no PostgreSQL;
- documentos adicionados ao ChromaDB para itens prioritários.

Se `EMAIL_PROVIDER=gmail`, a primeira execução deve pedir consentimento OAuth.

Para subir o backend HTTP:

```bash
uvicorn api_server:app --reload
```

Você pode validar rapidamente a API em:

```text
http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

### 8. Subir a interface Streamlit

Com a API já rodando em outro terminal:

```bash
streamlit run app_streamlit.py
```

Ao abrir a interface, valide:

- a sidebar mostra `PostgreSQL: OK`;
- a sidebar mostra `Gmail: MOCK` ou `Gmail: CONFIGURADO`;
- a sidebar mostra o provider LLM ativo, o status e o modelo;
- o botão `Rodar Triagem` processa os e-mails do provider ativo via FastAPI;
- as métricas sobem após a execução;
- a seção de categorização mostra o acumulado e a última execução;
- a `Fila de Revisão Manual` exibe itens `EM_DUVIDA`, quando houver;
- reclassificações manuais atualizam o PostgreSQL e alimentam o ChromaDB.

## Fluxo de teste recomendado

### Teste 1: sem API externa

Deixe `EMAIL_PROVIDER=mock`, `LLM_PROVIDER=heuristic` e rode o Streamlit.

Você deve ver:

- triagem funcionando com o classificador heurístico local;
- experiência completa de ponta a ponta sem depender da API externa.

### Teste 2: com Gemini

Adicione sua chave real:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.5-flash-lite
```

Reinicie o Streamlit e rode a triagem novamente.

Você deve ver:

- classificação baseada no provider Gemini;
- resumos factuais para e-mails prioritários;
- aprendizado semântico persistido no ChromaDB.

### Teste 3: com OpenAI opcional

Se quiser alternar de provider:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sua_chave_aqui
OPENAI_MODEL=gpt-5-mini
```

### Teste 4: com Anthropic opcional

Se quiser alternar de provider:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sua_chave_aqui
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

O comportamento funcional permanece o mesmo; muda apenas o backend de inferência.

### Teste 5: com Gmail real

Configure:

```env
EMAIL_PROVIDER=gmail
GMAIL_PROJECT_ID=seu_project_id
GMAIL_CLIENT_ID=seu_client_id.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=seu_client_secret
```

Execute novamente o teste do core ou o Streamlit.

Você deve ver:

- consentimento OAuth na primeira execução;
- criação de `gmail_token.json` após autenticar;
- leitura real da inbox via Gmail API;
- criação de labels como `AUTO_TRIAGEM_CARREIRA_PRIORIDADE`;
- nenhum e-mail marcado como lido pelo agente.

## O que o sistema cria localmente

- `processed_emails` no PostgreSQL
- `chroma_db/` com a memória semântica local
- `gmail_token.json` após autenticação Gmail real
- `__pycache__/` ao executar Python

Esses artefatos temporários locais relevantes já estão cobertos no `.gitignore`.

## Observações

- O `core_agent.py` cria a tabela `processed_emails` automaticamente no primeiro boot.
- O agente nunca marca o e-mail como lido no mock.
- O projeto agora suporta Gmail real, mantém o modo `mock` e abstrai o provedor de LLM para evitar acoplamento direto com uma única API.
