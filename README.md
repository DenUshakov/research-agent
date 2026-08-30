# Research Agent — MCP + A2A Multi-Agent System

Мультиагентна дослідницька система, де tools винесені в окремі MCP-сервери, а суб-агенти (Planner/Researcher/Critic) — в окремі A2A-сервери. Supervisor — локальний оркестратор, що делегує роботу через A2A-протокол і отримує tools (`save_report`) через MCP.

**Еволюція проєкту (git-теги):** `lesson-3` (LangChain `create_agent`) → `lesson-4` (власний ReAct loop) → `lesson-5` (RAG) → `lesson-8` (мультиагентність в одному процесі) → `lesson-10` (поточна: розподілена архітектура через MCP+A2A).

## Архітектура

```
User (main.py, REPL)
  │
  ▼
Supervisor (локальний create_agent)
  │
  ├── delegate_to_planner   ──► A2A (8903) ──► Planner Agent    ──► MCP (8901, SearchMCP)
  ├── delegate_to_researcher ──► A2A (8904) ──► Research Agent  ──► MCP (8901, SearchMCP)
  ├── delegate_to_critic    ──► A2A (8905) ──► Critic Agent     ──► MCP (8901, SearchMCP)
  │
  └── save_report_tool       ──► MCP (8902, ReportMCP) ── HITL gated
```

**6 окремих процесів**, кожен свій термінал:
```bash
python ingest.py                    # 1. побудувати RAG-індекс (одноразово)
python -m mcp_servers.search_mcp    # 2. SearchMCP, порт 8901
python -m mcp_servers.report_mcp    # 3. ReportMCP, порт 8902
python a2a_servers.py               # 4. Planner+Researcher+Critic, порти 8903-8905
python main.py                      # 5. Supervisor REPL (запускати останнім)
```

### MCP-сервери (`mcp_servers/`)

- **SearchMCP** (8901) — `web_search_tool`, `read_url_tool`, `knowledge_search_tool` (обгортки над `tools.py`); resource `resource://knowledge-base-stats`.
- **ReportMCP** (8902) — `save_report_tool`; resource `resource://output-dir`.
- Обидва прогрівають `Retriever`/RAG-модель **синхронно, в головному потоці, до старту сервера** (`preload_retriever()`) — критично важливо, див. "Технічні нюанси" нижче.

### A2A-сервери (`a2a_servers.py`, один процес, три сервери паралельно)

Кожен суб-агент (Planner/Researcher/Critic) — окремий A2A-сервер зі своєю Agent Card (`/.well-known/agent-card.json`), що отримує tools із SearchMCP через `langchain_mcp_adapters.MultiServerMCPClient`. Спільний клас `LangChainAgentExecutor` обгортає будь-якого `create_agent`-агента в A2A `AgentExecutor` (лінива ініціалізація через `asyncio.Lock`).

### Supervisor (`supervisor.py`)

Локальний `create_agent` з чотирма tools: три асинхронні обгортки-делегати (`delegate_to_planner/researcher/critic`, кожна робить A2A `send_message` виклик до відповідного сервера) плюс `save_report_tool`, отриманий через MCP з `ReportMCP`. HITL захищає саме `save_report_tool`.

## Встановлення й запуск

1. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. Скопіюй `.env.example` в `.env`, встав ключ ([aistudio.google.com/apikey](https://aistudio.google.com/apikey) → "Create API key in new project")
3. `python ingest.py` — будує RAG-індекс з `./data/`
4. У **чотирьох окремих терміналах** (кожен з активованим venv), по одній команді:
```bash
   python -m mcp_servers.search_mcp
   python -m mcp_servers.report_mcp
   python a2a_servers.py
```
5. У **п'ятому** терміналі: `python main.py`

## Про версію `a2a-sdk`: чому код відрізняється від офіційних прикладів

Практично **весь** API `a2a-sdk` (1.1.2), з яким ми зіткнулись, розходиться з документацією й туторіалами (навіть офіційними), знайденими в пошуку — судячи з усього, це наслідок швидкого розвитку протоколу (v1.0 вийшов у березні 2026). Конкретні розбіжності:

| Документація/туторіали показують | Реально працює (1.1.2) |
|---|---|
| `A2AStarletteApplication(agent_card=..., http_handler=...)` | `Starlette(routes=create_agent_card_routes(agent_card=card) + create_jsonrpc_routes(request_handler=handler, rpc_url="/"))` |
| `from a2a.utils import new_agent_text_message` | `from a2a.helpers import new_text_message` |
| `AgentCard(url="http://...", ...)` (просте поле) | `AgentCard(supported_interfaces=[AgentInterface(url=..., protocol_binding=TransportProtocol.JSONRPC, protocol_version="1.0")])` — `AgentCard` тепер protobuf-повідомлення, не Pydantic |
| `DefaultRequestHandler(agent_executor=..., task_store=...)` | Той самий + обов'язковий `agent_card=card` |
| `create_client(url)` + `client.send_message(message)` | `create_client(url, client_config=ClientConfig(httpx_client=httpx.AsyncClient(timeout=180)))` + `client.send_message(SendMessageRequest(message=message))` — потрібен явний `httpx.AsyncClient` з довгим таймаутом, інакше запит обривається за замовчуванням через ~10с (`resolver_http_kwargs` впливає лише на завантаження Agent Card, не на сам запит) |

Кожна з цих розбіжностей знайдена методом реального запуску й читання traceback — а не здогадкою. Якщо `a2a-sdk` оновиться, ці деталі можуть знову змінитись; перевіряй `python -c "help(a2a.client.create_client)"` та подібне перед довірою до будь-якого туторіалу.

## Технічні нюанси (специфічно для розподіленої архітектури)

- **`preload_retriever()` перед стартом кожного MCP/A2A сервера.** FastMCP і A2A виконують синхронні tools у фонових потоках (не в головному) — перше "ліниве" завантаження `SentenceTransformer`/`CrossEncoder` в такому потоці спричиняло **segmentation fault** на macOS (Apple Silicon), не просто зависання. Прогрів моделі синхронно, в головному потоці, до старту async-сервера, усуває проблему повністю.
- **`threading.Lock()` / `asyncio.Lock()` для лінивої ініціалізації** — і `Retriever` (у `tools.py`), і кожен `LangChainAgentExecutor` (в `a2a_servers.py`) використовують подвійну перевірку (double-checked locking), щоб паралельні запити не створювали кілька копій важких моделей одночасно.
- **`read_url` через `requests`, не `trafilatura.fetch_url`** — вбудований таймаут `trafilatura` побудований на `signal`, який не працює поза головним потоком.
- **`main.py` повністю асинхронний** (`asyncio.run(main())`, `await supervisor.ainvoke(...)`) — Supervisor використовує `delegate_to_*` tools, які самі `async def` (роблять мережеві A2A-виклики), тож `create_agent` реєструє їх лише як async-tools; синхронний `.invoke()` впаде з `NotImplementedError: StructuredTool does not support sync invocation`.
- **`edit` у HITL реалізовано через `reject` + повідомлення**, не через `respond` (успадковано з lesson-8 — `respond` підміняє результат tool замість повторного виклику).

## Обмеження

- Успадковано з lesson-8: `reject` завжди веде до повторної спроби, немає справжнього скасування; цикл затвердження не обмежений кількістю раундів.
- Всі 6 серверів мають запускатись у правильному порядку і залишатись живими одночасно — немає єдиного health-check чи orchestration-скрипта; якщо один сервер впав, наступний виклик до нього просто зависне/впаде з таймаутом.
- `httpx.AsyncClient(timeout=180)` — фіксований, доволі великий таймаут для A2A-делегування; на повільнішій мережі чи важчому запиті цього може не вистачити.
- Безкоштовна квота Gemini (`gemini-3.5-flash-lite`) — мультиагентна система через MCP+A2A робить ще більше LLM-викликів на запит, ніж lesson-8 (кожен A2A-виклик — окремий процес з окремим `create_agent`).

## Тестування циклу REVISE (успадковано з lesson-8)

REVISE-логіка перевірена ізольовано: прямий A2A-виклик до Critic Agent на навмисно неповному дослідженні коректно повертає `verdict="REVISE"` з конкретними `gaps`. У **чотирьох** живих прогонах через повний конвеєр (три через MCP+A2A у lesson-10, один через прямі виклики в lesson-8) Critic послідовно давав `APPROVE` з першої спроби — Planner формує достатньо конкретний план, а Researcher достатньо ретельно його виконує, щоб не залишати прогалин, які Critic вважає значущими. Механізм REVISE (Supervisor → Researcher з `revision_requests`, до `MAX_REVISION_ROUNDS` разів) присутній у коді й системному промпті, готовий спрацювати, коли Critic дійсно виявить недоліки.

## Виправлення за фідбеком (post-lesson-10)

- **Жорсткий ліміт REVISE у коді**, не лише в промпті: `supervisor.py` веде лічильник `_revision_state["count"]`, що скидається на кожен новий запит користувача (`reset_revision_counter()` в `main.py`). Коли Critic повертає `REVISE` понад `MAX_REVISION_ROUNDS` разів, `delegate_to_critic` **примусово підміняє** вердикт на `APPROVE` в самому тексті відповіді — Supervisor фізично не бачить нескінченного REVISE, незалежно від власних намірів LLM. Перевірено ізольовано (`MAX_REVISION_ROUNDS=0` → перший REVISE одразу форсується в APPROVE).
- **`reject` тепер остаточно скасовує**, а не запускає нову спробу. Розрізнення відбувається через префікс повідомлення, яке Supervisor бачить після відхилення tool: `"CANCELLED_BY_USER: ..."` (від `reject` у REPL) означає зупинитись і повідомити користувача без повторного виклику `save_report_tool`; `"Revise and try again: ..."` (від `edit`) означає доопрацювати і спробувати знову. `SUPERVISOR_SYSTEM_PROMPT` явно описує цю різницю.