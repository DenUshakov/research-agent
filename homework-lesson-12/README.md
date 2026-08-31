# Homework Lesson 12: Langfuse Observability

Langfuse tracing, session/user tracking, prompt management та LLM-as-a-Judge online evaluation для мультиагентної дослідницької системи (розширення `homework-lesson-11`, тобто коду з тегу `lesson-8-complete`).

## Що реалізовано

### 1. Tracing
`@observe(name="mas-user-request")` на функції `handle_request` у `main.py` створює корінь трейсу на кожен запит користувача. `propagate_attributes()` всередині розповсюджує `session_id`/`user_id`/`tags` на всі вкладені виклики через OpenTelemetry-контекст. Кожен `.invoke()` (Supervisor, а також Planner/Researcher/Critic всередині `supervisor.py`) отримує свіжий `CallbackHandler()` — оскільки Langfuse v4 читає **поточний OTel-контекст** при створенні хендлера, усі вкладені виклики автоматично зв'язуються в одне дерево, навіть коли відбуваються в різних функціях/файлах.

Перевірено: одне дерево `mas-user-request` → `LangGraph` (Supervisor) → `tools` → `plan`/`research`/`critique` → вкладений `LangGraph` (суб-агент) → `model`/`tools` (`web_search`, `knowledge_search`).

### 2. Session та User tracking
`session_id = uuid.uuid4()` генерується один раз на запуск `main()` — усі запити в межах однієї сесії CLI групуються в Langfuse Sessions. `user_id` — статичний рядок (`"den"`).

### 3. Prompt Management
Усі 4 системні промпти (`planner-system-prompt`, `researcher-system-prompt`, `critic-system-prompt`, `supervisor-system-prompt`) створено в Langfuse UI з label `production`. `config.py` більше не містить жодного хардкодженого тексту промпту — усі завантажуються через:
```python
_langfuse.get_prompt(name, label="production").compile(**variables)
```
`supervisor-system-prompt` параметризований через `{{max_revision_rounds}}` (template variable), підставляється з `settings.max_revision_rounds`.

### 4. LLM-as-a-Judge
Два evaluator'и, налаштовані в Langfuse UI (LLM-as-a-Judge → Evaluators), обидва на моделі `gemini-3.5-flash-lite` через власне LLM Connection (`google-ai-studio` адаптер, окремий від нашого `.env`-ключа — Langfuse робить власні виклики для оцінки):

- **Relevance** (numeric, 0-1) — чи відповідь стосується запиту користувача
- **Task Completion** (boolean) — чи підтверджує відповідь успішне завершення дослідження й збереження звіту

Обидва застосовані до **нових трейсів** (scope: reuse configured filters, 100% sampling). Перевірено на 4+ живих запитах — обидва score з'являються автоматично, з розгорнутим поясненням від LLM-судді, за 1-2 хвилини після завершення трейсу.

## Технічні нюанси, виявлені під час налаштування

- **US vs EU регіон Langfuse Cloud** — `us.cloud.langfuse.com` і `cloud.langfuse.com` (EU) це різні бекенди з несумісними ключами; `LANGFUSE_BASE_URL` в `.env` мусить точно відповідати тому домену, де реєструвався акаунт.
- **`get_client()` читає `os.environ` напряму**, не через Pydantic Settings — тому `config.py` починається з явного `load_dotenv()` **до** будь-яких інших імпортів, щоб гарантувати, що `LANGFUSE_*` змінні є в середовищі процесу, коли будь-який модуль (включно з `langfuse.get_client()`) до них звертається.
- **Pydantic `Settings` з `extra="ignore"`** — без цього `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` у `.env` викликають `ValidationError: Extra inputs are not permitted`, бо ці поля не потрібні (і не мають бути) в нашому власному класі `Settings` — Langfuse читає їх сам.
- **Langfuse Evaluators мають власне LLM Connection**, окреме від ключа нашого застосунку. За замовчуванням пропонує застарілу `gemini-2.5-flash`; довелось вимкнути "Enable default models" і вручну додати `gemini-3.5-flash-lite` як custom model name.
- **`CallbackHandler()` варто створювати заново в кожній точці виклику `.invoke()`**, а не як єдиний глобальний singleton — оскільки він захоплює поточний trace-контекст у момент створення; спільний екземпляр між різними запитами користувача змішав би сесії.

## Запуск

```bash
cd homework-lesson-12
python main.py
```

Кожен запит автоматично трейситься в Langfuse. Перевірити результати:
- **Tracing → Traces** — дерево викликів кожного запуску
- **Sessions** — групування по сесії CLI
- **Prompts** — системні промпти агентів
- **Evaluators** / **Scores** — автоматичні оцінки нових трейсів

## Скріншоти (`screenshots/`)

1. `01-trace-tree.png` — повне дерево трейсу з вкладеними суб-агентами
2. `02-sessions.png` — Sessions tab з кількома трейсами в одній сесії
3. `03-prompts.png` — Prompt Management з 4 промптами агентів
4. `04-scores.png` — Scores tab з автоматичними оцінками Relevance/Task Completion