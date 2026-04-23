"""Static prompts, defaults and typed helpers for the Antese writing agent."""

from __future__ import annotations

from typing import Any, Final, TypedDict


WRITING_TASK_TYPES: Final[tuple[str, ...]] = (
    "BRAINSTORM",
    "OUTLINE",
    "DRAFT",
    "CONSOLIDATE",
    "REWRITE",
)

WRITING_GOALS: Final[tuple[str, ...]] = (
    "INFORMAR",
    "PERSUADIR",
    "REFLETIR",
    "SINTETIZAR",
    "EXPLORAR",
)

STYLE_REASON_TAGS: Final[tuple[str, ...]] = (
    "mais_meu_estilo",
    "muito_generico",
    "muito_engessado",
    "boa_estrutura",
    "faltou_clareza",
    "faltou_originalidade",
    "inventou_contexto",
)

GENRE_CARD_IDS: Final[tuple[str, ...]] = (
    "brainstorm_notes",
    "outline",
    "linkedin_post",
    "essay_article",
    "newsletter",
    "rewrite_clarity",
    "consolidated_memo",
)

DEFAULT_STYLE_PROFILE_IDS: Final[tuple[str, ...]] = (
    "personal_default",
    "workspace_specific",
    "temporary_session_profile",
)


class NormalizedBrief(TypedDict):
    """Canonical description of a writing request."""

    task_type: str
    goal: str
    audience: str
    tone: str
    output_format: str
    genre_id: str
    source_notes_summary: str
    must_include: list[str]
    must_avoid: list[str]
    key_constraints: list[str]
    success_criteria: list[str]


class QualityReport(TypedDict):
    """Compact editorial evaluation returned by the critique step."""

    overall_score: float
    style_match_score: float
    usefulness_score: float
    creativity_score: float
    faithfulness_score: float
    strengths: list[str]
    weaknesses: list[str]
    revision_actions: list[str]


DEFAULT_STYLE_PROFILES: Final[list[dict[str, Any]]] = [
    {
        "profile_id": "personal_default",
        "name": "Personal Default",
        "owner_scope": "personal",
        "voice_traits": {
            "formalidade": "equilibrada",
            "densidade": "media",
            "assertividade": "alta",
            "calor": "moderado",
            "ironia": "baixa",
            "abstracao": "media",
            "concretude": "alta",
        },
        "structure_traits": {
            "abertura": "direta com contexto",
            "cadencia": "progressiva",
            "tamanho_paragrafo": "medio",
            "uso_listas": "situacional",
            "transicoes": "claras",
        },
        "rhetorical_traits": {
            "perguntas_retóricas": "raras",
            "exemplos": "frequentes",
            "analogias": "pontuais",
            "contraste": "forte",
            "sintese_final": "pratica",
        },
        "lexical_traits": {
            "simplicidade_vocabular": "alta",
            "termos_tecnicos": "moderados",
            "repeticao_aceitavel": "baixa",
            "preferencia_verbal": "verbos fortes",
            "preferencia_adjetivos": "contidos",
        },
        "dos": [
            "Explique conceitos complexos de forma clara.",
            "Use exemplos concretos para ancorar ideias.",
            "Mantenha progressao logica entre blocos.",
        ],
        "donts": [
            "Nao floreie sem necessidade.",
            "Nao exagere em jargoes ou abstrações vazias.",
            "Nao copie literalmente exemplos de referencia.",
        ],
        "sample_text_ids": [],
    },
    {
        "profile_id": "workspace_specific",
        "name": "Workspace Specific",
        "owner_scope": "workspace",
        "voice_traits": {
            "formalidade": "profissional",
            "densidade": "media-alta",
            "assertividade": "alta",
            "calor": "moderado",
            "ironia": "baixa",
            "abstracao": "media",
            "concretude": "alta",
        },
        "structure_traits": {
            "abertura": "situacao + contexto",
            "cadencia": "analitica",
            "tamanho_paragrafo": "medio",
            "uso_listas": "frequente",
            "transicoes": "fortes",
        },
        "rhetorical_traits": {
            "perguntas_retóricas": "raras",
            "exemplos": "frequentes",
            "analogias": "moderadas",
            "contraste": "moderado",
            "sintese_final": "objetiva",
        },
        "lexical_traits": {
            "simplicidade_vocabular": "media",
            "termos_tecnicos": "altos quando pertinentes",
            "repeticao_aceitavel": "baixa",
            "preferencia_verbal": "precisa",
            "preferencia_adjetivos": "contidos",
        },
        "dos": [
            "Conecte a escrita ao contexto do projeto.",
            "Use linguagem operacional quando fizer sentido.",
        ],
        "donts": [
            "Nao deixe o texto genérico.",
            "Nao perca o fio argumentativo em digressoes.",
        ],
        "sample_text_ids": [],
    },
    {
        "profile_id": "temporary_session_profile",
        "name": "Temporary Session Profile",
        "owner_scope": "session",
        "voice_traits": {
            "formalidade": "adaptativa",
            "densidade": "media",
            "assertividade": "media",
            "calor": "moderado",
            "ironia": "baixa",
            "abstracao": "media",
            "concretude": "media",
        },
        "structure_traits": {
            "abertura": "adaptativa",
            "cadencia": "flexivel",
            "tamanho_paragrafo": "medio",
            "uso_listas": "sob demanda",
            "transicoes": "claras",
        },
        "rhetorical_traits": {
            "perguntas_retóricas": "moderadas",
            "exemplos": "moderados",
            "analogias": "moderadas",
            "contraste": "moderado",
            "sintese_final": "curta",
        },
        "lexical_traits": {
            "simplicidade_vocabular": "media",
            "termos_tecnicos": "adaptativos",
            "repeticao_aceitavel": "media",
            "preferencia_verbal": "equilibrada",
            "preferencia_adjetivos": "moderados",
        },
        "dos": [
            "Adaptar a voz ao contexto corrente.",
        ],
        "donts": [
            "Nao cristalizar uma voz indevida para todas as tarefas.",
        ],
        "sample_text_ids": [],
    },
]

DEFAULT_GENRE_CARDS: Final[list[dict[str, Any]]] = [
    {
        "genre_id": "brainstorm_notes",
        "name": "Brainstorm Notes",
        "primary_goal": "EXPLORAR",
        "expected_structure": ["framing", "angles", "examples", "next steps"],
        "tone_defaults": ["exploratorio", "energico", "claro"],
        "length_defaults": {"target": "curto a medio", "paragraphs": "4-8"},
        "quality_checklist": [
            "Variedade de angulos",
            "Evitar redundancia",
            "Conectar ideias ao objetivo",
        ],
        "typical_openings": ["Mapa inicial de ideias", "Alguns caminhos possiveis"],
        "typical_closings": ["Caminho mais promissor", "Proximo experimento"],
    },
    {
        "genre_id": "outline",
        "name": "Outline",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["title", "hook", "sections", "closing"],
        "tone_defaults": ["estruturado", "direto"],
        "length_defaults": {"target": "curto", "sections": "4-7"},
        "quality_checklist": [
            "Sequencia logica",
            "Cobertura do objetivo",
            "Secoes acionaveis",
        ],
        "typical_openings": ["Tema e tese", "Ponto de partida"],
        "typical_closings": ["Fechamento", "CTA ou sintese final"],
    },
    {
        "genre_id": "linkedin_post",
        "name": "LinkedIn Post",
        "primary_goal": "PERSUADIR",
        "expected_structure": ["hook", "story/insight", "takeaways", "closing"],
        "tone_defaults": ["profissional", "humano", "assertivo"],
        "length_defaults": {"target": "curto a medio", "paragraphs": "5-10"},
        "quality_checklist": [
            "Abertura forte",
            "Valor pratico",
            "Cadencia boa para leitura em feed",
        ],
        "typical_openings": ["Aprendi isso da pior forma", "Tem uma armadilha comum"],
        "typical_closings": ["Pergunta ao leitor", "Sintese com aprendizado"],
    },
    {
        "genre_id": "essay_article",
        "name": "Essay Article",
        "primary_goal": "REFLETIR",
        "expected_structure": ["hook", "context", "argument", "examples", "conclusion"],
        "tone_defaults": ["ensaistico", "reflexivo", "claro"],
        "length_defaults": {"target": "medio a longo", "paragraphs": "8-16"},
        "quality_checklist": [
            "Tese clara",
            "Progressao argumentativa",
            "Conclusao memoravel",
        ],
        "typical_openings": ["Existe uma tensao central", "Nos ultimos anos"],
        "typical_closings": ["O ponto central e", "Talvez a pergunta correta seja"],
    },
    {
        "genre_id": "newsletter",
        "name": "Newsletter",
        "primary_goal": "INFORMAR",
        "expected_structure": ["opening", "highlights", "commentary", "closing"],
        "tone_defaults": ["curatorial", "caloroso", "preciso"],
        "length_defaults": {"target": "medio", "blocks": "4-8"},
        "quality_checklist": [
            "Clareza editorial",
            "Bom ritmo",
            "Comentario proprio, nao so links",
        ],
        "typical_openings": ["Na edicao de hoje", "Aqui vai um recorte util"],
        "typical_closings": ["Se isso te ajudou", "Na proxima edicao"],
    },
    {
        "genre_id": "rewrite_clarity",
        "name": "Rewrite Clarity",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["preserve core", "improve flow", "tighten wording"],
        "tone_defaults": ["claro", "preciso"],
        "length_defaults": {"target": "sem alongar demais", "paragraphs": "preservar estrutura base"},
        "quality_checklist": [
            "Preservar sentido",
            "Melhorar clareza",
            "Evitar inflacao verbal",
        ],
        "typical_openings": ["Versao revisada", "Reescrita com mais clareza"],
        "typical_closings": ["Ideia central preservada", "Fechamento mais nítido"],
    },
    {
        "genre_id": "consolidated_memo",
        "name": "Consolidated Memo",
        "primary_goal": "SINTETIZAR",
        "expected_structure": ["context", "points", "decisions", "next steps"],
        "tone_defaults": ["objetivo", "profissional"],
        "length_defaults": {"target": "curto a medio", "paragraphs": "4-8"},
        "quality_checklist": [
            "Consolidacao fiel das notas",
            "Separacao entre fatos e interpretacao",
            "Proximos passos claros",
        ],
        "typical_openings": ["Resumo consolidado", "Sintese do material-base"],
        "typical_closings": ["Encaminhamentos", "Proximos passos"],
    },
]

DEFAULT_INSPIRATION_PROFILES: Final[list[dict[str, Any]]] = [
    {
        "inspiration_profile_id": "essayistic",
        "name": "Mais ensaistico",
        "traits_to_borrow": [
            "cadencia mais contemplativa",
            "transicoes mais suaves",
            "tese com desdobramento reflexivo",
        ],
        "forbidden_behaviors": [
            "imitar trechos famosos",
            "soar academicamente opaco",
        ],
        "transformation_strength": "media",
    },
    {
        "inspiration_profile_id": "aphoristic",
        "name": "Mais aforistico",
        "traits_to_borrow": [
            "frases mais condensadas",
            "fechamentos mais cortantes",
            "contraste mais visivel",
        ],
        "forbidden_behaviors": [
            "fragmentar demais o texto",
            "sacrificar clareza por efeito",
        ],
        "transformation_strength": "baixa",
    },
    {
        "inspiration_profile_id": "journalistic",
        "name": "Mais jornalistico",
        "traits_to_borrow": [
            "abertura mais informativa",
            "sequencia factual forte",
            "clareza e economia verbal",
        ],
        "forbidden_behaviors": [
            "adotar neutralidade artificial",
            "remover a voz do usuario por completo",
        ],
        "transformation_strength": "media",
    },
    {
        "inspiration_profile_id": "intimate_reflective",
        "name": "Mais intimo/reflexivo",
        "traits_to_borrow": [
            "proximidade com o leitor",
            "transicoes mais subjetivas",
            "fechamento com introspeccao",
        ],
        "forbidden_behaviors": [
            "sentimentalismo excessivo",
            "autoexposicao sem funcao",
        ],
        "transformation_strength": "media",
    },
]

ANTese_SYSTEM_PROMPT: Final[str] = """
Você é Antese, um agente especialista em escrita personalizada.

Seu trabalho é transformar material-base em texto forte, útil e com voz reconhecível.

Princípios editoriais:
- criatividade nasce na ideação e na estrutura, não na invenção de fatos;
- estilo é um viés orientador, não uma regra rígida;
- gênero define a forma; o perfil de estilo define a voz;
- preserve fatos e restrições do usuário;
- não copie literalmente os exemplos recuperados;
- não imite autores de forma literal: converta referências em traços estilísticos explícitos.

Você pode operar em etapas diferentes do pipeline editorial:
- NORMALIZE_BRIEF
- PLAN
- DRAFT
- CRITIQUE
- REWRITE

Sempre respeite o formato de saída exigido em cada prompt.
Responda em português do Brasil, salvo instrução explícita em contrário.
""".strip()

NORMALIZE_BRIEF_PROMPT: Final[str] = """
Transforme o pedido do usuário em um briefing canônico.

Regras:
1. Não invente fatos ausentes.
2. Se o usuário mencionar um autor, traduza isso em traços estilísticos e nunca em imitação literal.
3. Preencha lacunas com defaults conservadores.
4. Responda APENAS em JSON válido.

Campos obrigatórios do JSON:
- task_type
- goal
- audience
- tone
- output_format
- genre_id
- source_notes_summary
- must_include
- must_avoid
- key_constraints
- success_criteria
""".strip()

PLAN_PROMPT: Final[str] = """
Você está na etapa editorial PLAN.

Produza um outline/argument map útil para escrever bem depois.
O outline deve:
- organizar o melhor caminho argumentativo;
- aproveitar o material-base sem copiá-lo;
- refletir o gênero e o style profile como viés, sem engessar;
- explicitar hook, progressão e fechamento quando fizer sentido.

Formato de saída:
- Markdown
- Título
- Estrutura por seções
- 1 ou 2 bullets de intenção para cada seção
""".strip()

DRAFT_PROMPT: Final[str] = """
Você está na etapa editorial DRAFT.

Escreva a melhor primeira versão possível.

Regras:
- preserve fatos e pontos obrigatórios;
- use o style profile como viés;
- use o genre card como forma;
- mantenha criatividade na formulação e nas conexões;
- não copie frases dos exemplos recuperados;
- não mencione o pipeline, os perfis ou a crítica.
""".strip()

CRITIQUE_PROMPT: Final[str] = """
Você está na etapa editorial CRITIQUE.

Avalie o texto gerado de forma estruturada, sem expor chain-of-thought.

Responda APENAS em JSON com:
- overall_score (0 a 1)
- style_match_score (0 a 1)
- usefulness_score (0 a 1)
- creativity_score (0 a 1)
- faithfulness_score (0 a 1)
- strengths (lista curta)
- weaknesses (lista curta)
- revision_actions (lista curta e acionável)
""".strip()

REWRITE_PROMPT: Final[str] = """
Você está na etapa editorial REWRITE.

Reescreva o texto usando a crítica estruturada.

Regras:
- preserve a ideia central;
- preserve os pontos obrigatórios;
- melhore clareza, aderência ao estilo e estrutura;
- mantenha originalidade;
- não copie os exemplos recuperados;
- entregue apenas o texto final.
""".strip()

