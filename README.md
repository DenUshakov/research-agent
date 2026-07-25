# Research Agent

Агент на LangChain, який отримує питання від користувача, самостійно шукає інформацію в інтернеті через набір інструментів, і генерує структурований Markdown-звіт. Підтримує зв'язний діалог у межах сесії.

## Архітектура

```
research-agent/
├── main.py       # Interactive REPL loop
├── agent.py      # Збірка агента (модель, tools, checkpointer)
├── tools.py      # web_search, read_url, write_report
├── config.py     # Налаштування (Pydantic Settings, читає .env)
├── prompts.py    # System prompt агента
├── requirements.txt
├── example_output/
│   └── report.md
└── output/       # Сюди зберігаються згенеровані звіти (не в git)
```

Агент побудований на `create_agent` (LangChain, на базі LangGraph). Це ReAct-агент: LLM сам вирішує, які tools викликати і в якій послідовності, поки не набере достатньо інформації для фінального звіту.

**Tools:**
- `web_search(query)` — пошук в інтернеті через DuckDuckGo (`ddgs`), повертає короткі сніпети (title, url, snippet).
- `read_url(url)` — читає повний текст сторінки (`trafilatura`), обрізаний до `MAX_TOOL_RESULT_CHARS` символів, щоб не забивати контекстне вікно.
- `write_report(filename, content)` — зберігає фінальний Markdown-звіт у папку `output/`.

**Памʼять:** реалізована через `InMemorySaver` (LangGraph checkpointer), прив'язаний до `thread_id`. Агент памʼятає всі попередні повідомлення в межах однієї сесії CLI — можна ставити уточнюючі запитання без повторення контексту.

**Ліміт кроків:** контролюється через `recursion_limit` (значення з `MAX_ITERATIONS` в `.env`), щоб агент не міг зациклитись.

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

4. Отримай **Google API ключ** (безкоштовно, без карти) на [aistudio.google.com](https://aistudio.google.com) → "Get API Key".

5. Створи файл `.env` в корені проєкту (за зразком `.env.example`):
```
GOOGLE_API_KEY=твій_ключ_тут
MODEL_NAME=gemini-3.6-flash
MAX_ITERATIONS=10
MAX_TOOL_RESULT_CHARS=8000
```

> Назва актуальної моделі Gemini може змінитись — перевіряй в [документації Google AI](https://ai.google.dev/gemini-api/docs/changelog), якщо отримаєш помилку `NOT_FOUND`.

## Запуск

```bash
python main.py
```

Приклад використання:
```
You: Порівняй три підходи до побудови RAG: naive, sentence-window та parent-child retrieval
Agent: [шукає інформацію по кожному підходу, читає релевантні сторінки, формує звіт]
       Звіт збережено у output/rag_approaches_comparison.md
```

Для виходу — введи `exit` або `quit`.

## Обмеження

- Памʼять зберігається лише в оперативній памʼяті процесу — при перезапуску `main.py` історія розмови втрачається.
- Follow-up питання іноді відповідаються на основі контексту попередніх кроків розмови без повторного пошуку — це очікувана поведінка ReAct-агента, коли він вважає наявну інформацію достатньою.