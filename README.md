# Research Agent

Агент, який отримує питання від користувача, самостійно шукає інформацію — в інтернеті та в локальній базі знань — і генерує структурований Markdown-звіт із посиланнями на джерела. Підтримує зв'язний діалог у межах сесії.

**Версія lesson-5:** додано RAG-підсистему — локальну базу знань з hybrid retrieval (semantic + BM25) і cross-encoder reranking, доступну агенту через новий tool `knowledge_search`. Попередні версії зафіксовані в git тегами `lesson-3-complete` (LangChain `create_agent`) і `lesson-4-complete` (власний ReAct loop, лише web-джерела).

## Архітектура

```
research-agent/
├── main.py       # Interactive REPL loop, керує history (пам'ять) вручну
├── agent.py      # Власний ReAct loop: виклик моделі → tool calls → результати → повтор
├── tools.py      # web_search, read_url, write_report, knowledge_search + JSON Schema
├── retriever.py  # Hybrid retrieval (FAISS + BM25, RRF fusion) + cross-encoder reranking
├── ingest.py     # Pipeline: PDF → сторінки → чанки → embeddings → FAISS індекс на диску
├── config.py     # Налаштування (Pydantic Settings, читає .env)
├── prompts.py    # System prompt (Few-Shot, Chain-of-Thought, Self-Reflection, цитування джерел)
├── requirements.txt
├── data/         # PDF документи для індексації (вхід для ingest.py)
├── index/        # Згенерований FAISS-індекс + метадані чанків (не в git)
├── example_output/
│   ├── report.md
│   └── console_log.txt
└── output/       # Згенеровані звіти (не в git)
```

### RAG-підсистема

**1. Ingestion (`ingest.py`, запускається окремо командою `python ingest.py`):**
- Читає всі PDF з `./data/`, витягує текст **по сторінках** (`pypdf`) — це зберігає можливість пізніше вказати точний номер сторінки як джерело.
- Розбиває текст кожної сторінки на чанки (`langchain-text-splitters`, `RecursiveCharacterTextSplitter`, `CHUNK_SIZE`/`CHUNK_OVERLAP` з `.env`).
- Обчислює embeddings для кожного чанка (`sentence-transformers`, модель `all-MiniLM-L6-v2`, локально, без API).
- Зберігає векторний індекс (`faiss.IndexFlatIP`, cosine similarity через нормалізовані вектори) і метадані чанків (`chunks.pkl`) в `./index/` — індекс перезавантажується напряму з диска, без повторного обчислення embeddings.

**2. Retrieval (`retriever.py`, клас `Retriever`):**
- **Semantic search** — FAISS шукає найближчі за cosine similarity чанки.
- **BM25 search** (`rank-bm25`) — лексичний пошук за збігом слів, сильний на точних термінах і назвах, яких semantic search може не вловити.
- **Reciprocal Rank Fusion (RRF)** — об'єднує обидва ранжовані списки за позицією (не за прямим скором — BM25-скор і cosine similarity живуть у несумісних шкалах).
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — точніше, але повільніше повторне ранжування вже звуженого пулу кандидатів; фінальний `top_k` (`KNOWLEDGE_SEARCH_TOP_K` з `.env`) віддається агенту.

**3. Tool (`tools.py`, `knowledge_search`):**
- Обгортка над `Retriever`, з лінивою ініціалізацією (модель embeddings і reranker завантажуються лише при першому реальному виклику, не при імпорті модуля).
- В описі tool (JSON Schema) явно перелічені теми, які покриває поточна база знань — це допомагає моделі вирішувати, коли викликати `knowledge_search`, а коли одразу `web_search`.

### Агент і промпт

Успадковано з lesson-4 (власний ReAct loop через Gemini Interactions API, `history` list як ручна пам'ять, `trim_history` для контролю росту контексту, retry з backoff на помилках моделі) — детальніше в коментарях коду `agent.py`.

`prompts.py` доповнено інструкцією стратегії вибору джерела: **спочатку `knowledge_search`** для тем локальної бази (RAG, LangChain, LLM), **потім `web_search`** для решти або якщо локальний пошук не дав достатньо — і явною вимогою позначати джерело (файл+сторінка або URL) для кожного ключового твердження у фінальному звіті.

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
> `sentence-transformers` тягне за собою `torch` — встановлення може зайняти кілька хвилин. Перший запуск `ingest.py`/`retriever.py` додатково завантажить дві моделі (embeddings ~90МБ, reranker ~90МБ) з Hugging Face — вони кешуються локально, повторні запуски вже офлайн.

4. Отримай **Google API ключ** (безкоштовно, без карти) на [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → **"Create API key"** → **"Create API key in new project"**.

5. Створи `.env` (за зразком `.env.example`):
```
GOOGLE_API_KEY=твій_ключ_тут
MODEL_NAME=gemini-3.6-flash
MAX_ITERATIONS=10
MAX_TOOL_RESULT_CHARS=8000
MAX_HISTORY_CHARS=40000
OUTPUT_DIR=output
DATA_DIR=data
INDEX_DIR=index
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
KNOWLEDGE_SEARCH_TOP_K=5
```

6. Поклади PDF-документи в `./data/` і побудуй індекс:
```bash
python ingest.py
```

## Запуск

```bash
python main.py
```

Приклад використання (питання про тему з локальної бази):
```
You: Що таке RAG і як він доповнює LLM?

🔧 Tool call: knowledge_search({'query': '...'})
📎 Result: [retrieval-augmented-generation.pdf, стор. 1] ...

🔧 Tool call: web_search({'query': '...'})   # для аспектів поза базою знань
📎 Result: [...]

🔧 Tool call: write_report({...})
📎 Result: Звіт успішно збережено: output/rag_overview.md

Agent: Звіт збережено. Основні висновки: ...
```

Для виходу — введи `exit` або `quit`.

## Обмеження

- Памʼять (`history`) живе лише в оперативній памʼяті процесу — при перезапуску `main.py` втрачається.
- Безкоштовний рівень Gemini API має ліміт 5-20 запитів/хв (залежить від акаунту) — при активному тестуванні легко натрапити на `429`. Агент робить до 3 повторних спроб з backoff, але може вичерпати їх при затяжному вичерпанні квоти.
- Локальна база знань наразі містить лише 3 документи (про RAG, LangChain, LLM) — `knowledge_search` поверне нерелевантні чанки на будь-яке інше питання, якщо агент його все ж викличе; коректна робота залежить від того, що системний промпт явно перелічує охоплені теми.
- Індекс (`index/`) і embeddings — локальні, без API. Сам агент (LLM) все ще залежить від мережі й Gemini API.
- Попередні версії доступні через `git checkout lesson-3-complete` / `lesson-4-complete`.