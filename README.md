# Research Agent — Multi-Agent System

Мультиагентна дослідницька система: Supervisor координує трьох спеціалізованих суб-агентів (Planner → Researcher → Critic) за патерном evaluator-optimizer, з людським затвердженням (HITL) перед збереженням фінального звіту.

**Еволюція проєкту (git-теги):**
- `lesson-3-complete` — базовий Research Agent на LangChain `create_agent`
- `lesson-4-complete` — власний ReAct loop (без агентних абстракцій), Gemini Interactions API
- `lesson-5-complete` — RAG: hybrid retrieval (FAISS + BM25 + RRF) з cross-encoder reranking
- `lesson-8-complete` (поточна) — мультиагентна оркестрація, structured output, HITL

## Архітектура

```
research-agent/
├── main.py              # REPL з обробкою interrupt/resume
├── supervisor.py         # Supervisor + agent-as-tool обгортки (plan/research/critique)
├── agents/
│   ├── planner.py        # Planner Agent → структурований ResearchPlan
│   ├── research.py       # Research Agent → знахідки з цитатами
│   └── critic.py         # Critic Agent → структурований CritiqueResult
├── schemas.py            # Pydantic-моделі ResearchPlan, CritiqueResult
├── tools.py               # web_search, read_url, knowledge_search, write_report, save_report
├── retriever.py           # Hybrid retrieval + reranking (з lesson-5)
├── ingest.py               # PDF → chunks → embeddings → FAISS (з lesson-5)
├── config.py               # Settings + системні промпти всіх 4 агентів
├── data/                   # PDF для індексації
├── index/                  # FAISS-індекс (генерується, не в git)
└── output/                 # Збережені звіти (не в git)
```

### Потік виконання

```
User → Supervisor
         │
         ├─ plan(request)      → Planner  → ResearchPlan (goal, queries, sources, format)
         ├─ research(...)      → Researcher → знахідки (web + knowledge_base, з цитатами)
         ├─ critique(...)      → Critic    → CritiqueResult (verdict, gaps, revision_requests)
         │     │
         │     ├─ REVISE → research() ще раз з revision_requests (до MAX_REVISION_ROUNDS разів)
         │     └─ APPROVE → формування фінального звіту
         │
         └─ save_report(...) ── HITL interrupt ──► людина: approve / edit / reject
```

Кожен суб-агент (`Planner`, `Researcher`, `Critic`) — це **окремий `create_agent`**, обгорнутий у звичайну Python-функцію (`plan`/`research`/`critique`), яку Supervisor викликає як tool. Обгортки серіалізують структурований вивід (`ResearchPlan`/`CritiqueResult`) у JSON-рядок, щоб Supervisor міг прочитати результат і прийняти рішення про наступний крок.

### Human-in-the-Loop (HITL)

`save_report` захищено `HumanInTheLoopMiddleware(interrupt_on={"save_report": True})` — граф зупиняється перед фактичним записом файлу і чекає рішення:
- **approve** — зберегти як є
- **edit** — людина дає текстовий фідбек; Supervisor доопрацьовує звіт і знову викликає `save_report` (новий interrupt)
- **reject** — відхилити з поясненням

⚠️ **Технічна деталь, відмінна від прикладу в завданні:** офіційний приклад показує `edit` через `{"type": "edit", "edited_action": {"feedback": ...}}`. У встановленій версії (`langchain==1.3.14`) `edit` вимагає **прямої заміни аргументів tool** (`name` + `args`), а не довільного тексту. Натомість `main.py` реалізує "дай фідбек і перероби" через **`{"type": "reject", "message": ...}`** — `reject` коректно сигналізує Supervisor'у, що виклик не відбувся і потрібно спробувати ще раз, тоді як `respond` (четвертий тип рішення) для цього **не підходить**: він підміняє результат tool текстом фідбеку, змушуючи модель хибно вважати, що `save_report` уже виконався (перевірено емпірично — файл не зберігався, хоча агент рапортував про успіх).

## Встановлення

1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt` (тягне `torch` через `sentence-transformers` — може зайняти кілька хвилин)
3. Отримай **Google API ключ**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → "Create API key" → "Create API key in new project"
4. Скопіюй `.env.example` в `.env`, встав ключ
5. Поклади PDF в `./data/`, побудуй індекс: `python ingest.py`
6. `python main.py`

## Про вибір моделі: чому не `gemini-3.6-flash`

Спершу проєкт використовував `gemini-3.6-flash`, але ця (preview) модель має вкрай суворий безкоштовний ліміт — **20 запитів/день**, незалежно від того, скільки різних Google-акаунтів/проєктів створено (ліміт прив'язаний до моделі, не лише до проєкту). Мультиагентний прогін (Planner + Researcher + Critic, можливо кілька раундів revise) легко витрачає 15-20+ викликів **за один запит користувача** — тобто практично весь денний ліміт за раз.

**Рішення:** `gemini-3.5-flash-lite` — стабільна (GA, не preview) модель з набагато щедрішим лімітом, достатнім для розробки й тестування мультиагентних систем.

## Технічні нюанси, виявлені під час розробки

- **`device="cpu"` для `sentence-transformers`/`CrossEncoder`** (`retriever.py`): на Apple Silicon PyTorch за замовчуванням намагається використати MPS (Metal), що зависає намертво при виклику з фонового потоку — а LangGraph виконує tool calls саме в таких потоках. Примусовий CPU вирішує це ціною невеликої втрати швидкості (непомітно на нашому маленькому корпусі).
- **`read_url` використовує `requests.get(..., timeout=10)`**, а не `trafilatura.fetch_url()` напряму: вбудований таймаут `trafilatura` реалізований через `signal`, який **не працює поза головним потоком** — тому зависання на "мовчазних" серверах не переривалось. `trafilatura.extract()` лишається для парсингу вже завантаженого HTML.
- **`threading.Lock()` навколо lazy-ініціалізації `Retriever`** (`tools.py`): паралельні tool calls (наприклад, кілька `knowledge_search` одночасно від різних суб-агентів) без блокування спричиняли одночасне завантаження кількох копій моделей embeddings у різних потоках — нестабільно на macOS (`malloc` crashes).
- **`.with_retry()` несумісний з `create_agent`**: обгортання моделі в retry-логіку ламає `.bind_tools()`, необхідний для tool calling. Ретраї на рівні LLM-клієнта тут не застосовуються.

## Обмеження

- **`reject` завжди веде до повторної спроби**, а не до справжнього скасування — `SUPERVISOR_SYSTEM_PROMPT` трактує будь-яке відхилення як "доопрацюй і спробуй знову". Немає жорсткого способу сказати "остаточно відмінити" без виходу з програми.
- Цикл затвердження (`approve`/`edit`/`reject`) після `critique` APPROVE **не обмежений** кількістю спроб (на відміну від `research`↔`critique`, обмеженого `MAX_REVISION_ROUNDS`) — теоретично можна відхиляти нескінченно.
- Безкоштовна квота Gemini (навіть на `flash-lite`) обмежена — активне тестування мультиагентного циклу може вичерпати денний ліміт.
- `knowledge_search` покриває лише 3 документи (RAG, LangChain, LLM) — на інші теми `Planner`/`Researcher` покладаються виключно на `web_search`.

## Тестування циклу REVISE

Логіка REVISE (Critic → gaps → Researcher повторно з revision_requests) реалізована в `SUPERVISOR_SYSTEM_PROMPT` і перевірена ізольовано: прямий виклик `Critic Agent` на навмисно неповному дослідженні коректно повернув `verdict="REVISE"` з конкретними `gaps`/`revision_requests` (див. розробницький лог). У живих прогонах через повний `Supervisor` (кілька спроб з різними темами) Critic послідовно давав `APPROVE` з першої спроби — Research Agent виявився достатньо ретельним, щоб не тригерити REVISE органічно в рамках доступного тестового бюджету (безкоштовна квота Gemini). Механізм REVISE присутній у коді й системному промпті, готовий спрацювати, коли Critic дійсно виявить прогалини.