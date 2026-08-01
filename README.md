# Research Agent

Агент, який отримує питання від користувача, самостійно шукає інформацію в інтернеті через набір інструментів, і генерує структурований Markdown-звіт. Підтримує зв'язний діалог у межах сесії.

**Версія lesson-4:** ReAct-цикл реалізований власноруч, без агентних абстракцій фреймворків (`create_react_agent`, `AgentExecutor` тощо) — напряму через Gemini Interactions API. Попередня версія (на LangChain `create_agent`) зафіксована в git під тегом `lesson-3-complete`.

## Архітектура

```
research-agent/
├── main.py       # Interactive REPL loop, керує history (пам'ять) вручну
├── agent.py      # Власний ReAct loop: виклик моделі → tool calls → результати → повтор
├── tools.py      # web_search, read_url, write_report + їх JSON Schema декларації
├── config.py     # Налаштування (Pydantic Settings, читає .env)
├── prompts.py    # System prompt (Few-Shot, Chain-of-Thought, Self-Reflection)
├── requirements.txt
├── example_output/
│   └── report.md
└── output/       # Згенеровані звіти (не в git)
```

### Як працює власний ReAct loop

Замість того, щоб покладатись на фреймворк, `agent.py` реалізує цикл напряму:

1. Надсилає в модель поточну історію (`history`) + описи tools (JSON Schema) + системний промпт.
2. Модель повертає список "кроків" (`steps`) — це може бути текстова відповідь або `function_call` (запит на виклик tool).
3. Якщо є `function_call` — код сам знаходить відповідну Python-функцію (`TOOL_FUNCTIONS`), викликає її з переданими аргументами, і додає результат назад у `history` як `function_result`.
4. Цикл повторюється, поки модель не дасть фінальну текстову відповідь без запитів на tool calls, або поки не вичерпається ліміт ітерацій (`MAX_ITERATIONS`).

Використовується **Gemini Interactions API** (`client.interactions.create`) у **stateless-режимі** (`store=False`) — це означає, що вся історія розмови (`history`) зберігається і передається вручну на клієнті, без серверного `MemorySaver` чи подібних механізмів. Кожен виклик передає повний `history` заново.

**Tools** (описані в `tools.py`):
- `web_search(query)` — пошук в інтернеті через DuckDuckGo (`ddgs`), повертає короткі сніпети.
- `read_url(url)` — читає повний текст сторінки (`trafilatura`), обрізаний до `MAX_TOOL_RESULT_CHARS` символів (context engineering).
- `write_report(filename, content)` — зберігає фінальний Markdown-звіт у `output/`. Захищений від path traversal (`os.path.basename` + примусове розширення `.md`).

**Обробка помилок реалізована на двох незалежних рівнях:**
- Помилка всередині конкретного tool (наприклад, недоступний URL) — ловиться в самому tool і повертається як текстове повідомлення, яке модель бачить і на яке реагує.
- Помилка виклику самої моделі (мережа, rate limit, недійсний ключ) — ловиться навколо `client.interactions.create()` в `agent.py`, не викликаючи краху процесу.

**Логування:** кожен tool call і його результат виводяться в консоль (`🔧 Tool call: ...` / `📎 Result: ...`).

**Ліміт кроків:** `MAX_ITERATIONS` (з `.env`) обмежує кількість ітерацій циклу в `agent.py`, щоб агент не міг зациклитись нескінченно.

### System Prompt

`prompts.py` застосовує кілька технік промпт-інжинірингу:
- **Chain-of-Thought** — агента явно просять проговорювати міркування (`Thought`) перед кожною дією.
- **Few-Shot** — у промпт вбудований повний приклад бажаної поведінки (Thought → Action → Observation × N → Final Answer).
- **Self-Reflection** — перед `write_report` агент має перевірити повноту й несуперечність зібраної інформації.
- **Явні обмеження поведінки** — окремий розділ "чого НЕ робити" (не вигадувати факти, не викликати `write_report` передчасно, не повторювати ідентичні запити).

## Встановлення

1. Клонуй репозиторій і перейди в папку проєкту.

2. Створи та активуй віртуальне середовище (Python 3.10+):
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

3. Встанови залежності:
```bash
pip install -r requirements.txt
```

4. Отримай **Google API ключ** (безкоштовно, без карти) на [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → **"Create API key"** → **"Create API key in new project"** (це автоматично вмикає потрібний Gemini API для проєкту; ручне привʼязування до існуючого проєкту іноді призводить до помилки `403 SERVICE_DISABLED`).

5. Створи файл `.env` в корені проєкту (за зразком `.env.example`):
```
GOOGLE_API_KEY=твій_ключ_тут
MODEL_NAME=gemini-3.6-flash
MAX_ITERATIONS=10
MAX_TOOL_RESULT_CHARS=8000
OUTPUT_DIR=output
```

> Назва актуальної моделі Gemini може змінитись — перевіряй в [документації Google AI](https://ai.google.dev/gemini-api/docs/changelog), якщо отримаєш помилку `NOT_FOUND`.

## Запуск

```bash
python main.py
```

Приклад використання:
```
You: Порівняй Docker Swarm, Kubernetes та Nomad для оркестрації контейнерів

🔧 Tool call: web_search({'query': 'Docker Swarm vs Kubernetes comparison'})
📎 Result: [...]
🔧 Tool call: web_search({'query': 'Kubernetes vs Nomad comparison'})
📎 Result: [...]
🔧 Tool call: read_url({'url': 'https://...'})
📎 Result: [...]
🔧 Tool call: write_report({'filename': 'orchestrators_comparison.md', ...})
📎 Result: Звіт успішно збережено: output/orchestrators_comparison.md

Agent: Звіт збережено. Основні висновки: ...
```

Для виходу — введи `exit` або `quit`.

## Обмеження

- Памʼять (`history`) зберігається лише в оперативній памʼяті процесу — при перезапуску `main.py` історія розмови втрачається.
- Безкоштовний рівень Gemini API має ліміт запитів на хвилину (5–20, залежно від віку проєкту/акаунту) — при активному тестуванні легко натрапити на `429 Too Many Requests`. Це очікувана, а не критична помилка — агент коректно обробляє її, не завершуючи процес аварійно.
- Follow-up питання іноді відповідаються на основі контексту попередніх кроків розмови без повторного пошуку — це очікувана поведінка ReAct-агента, коли наявної в `history` інформації достатньо.
- Попередню версію на LangChain `create_agent` можна переглянути через `git checkout lesson-3-complete`.