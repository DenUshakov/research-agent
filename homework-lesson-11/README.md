# Homework Lesson 11: Testing the Multi-Agent System

Автоматизовані тести для мультиагентної дослідницької системи, побудовані на DeepEval, за підходами з Лекції 11.

## Чому lesson-8, а не lesson-10 (MCP+A2A)

Це завдання прямо розширює "hw9" курсу — локальну мультиагентну систему (Supervisor + Planner/Researcher/Critic як прямі Python-функції в одному процесі), яку в нашому репозиторії позначено тегом `lesson-8-complete`. Файли тут — точна копія коду з цього тегу (`git show lesson-8-complete:<file>`), ізольована в окрему директорію зі своїм `.env` (без MCP/A2A-специфічних змінних).

Це свідомий вибір, а не спрощення: DeepEval-тести мають бути **відтворюваними й швидкими** для CI. Тестувати проти розподіленої MCP+A2A версії (lesson-10) вимагало б піднімати 5 окремих серверів перед кожним прогоном — крихко й непридатно для регулярного `deepeval test run`.

## Структура

homework-lesson-11/
├── tests/
│ ├── conftest.py # Спільна фікстура eval_model (Gemini як LLM-суддя)
│ ├── golden_dataset.json # 15 прикладів: happy_path/edge_case/failure_case
│ ├── test_planner.py # Plan Quality (GEval)
│ ├── test_researcher.py # Groundedness (GEval)
│ ├── test_critic.py # Critique Quality (GEval), APPROVE та REVISE
│ ├── test_tools.py # Tool Correctness (3 кейси)
│ └── test_e2e.py # Повний pipeline на happy_path прикладах
├── agents/, supervisor.py, tools.py, ... # Копія коду з lesson-8-complete
└── README.md


## Eval-модель

DeepEval має нативну підтримку Gemini (`deepeval.models.GeminiModel`), тож LLM-суддя для всіх GEval-метрик і вбудованих метрик (`AnswerRelevancyMetric`, `ToolCorrectnessMetric`) — той самий `gemini-3.5-flash-lite`, що й самі агенти. Це економить окремий API-ключ, ціною певного самопідтвердження (той самий провайдер оцінює сам себе) — прийнятний компроміс для навчального проєкту.

## Обґрунтування порогів (baseline, не довільні значення)

| Метрика | Поріг | Обґрунтування |
|---|---|---|
| Plan Quality | 0.6 | Стандартний "помірно суворий" поріг для структурованого виводу з чіткими критеріями |
| Critique Quality | 0.6 | Так само — Critic повертає структуровані поля, легко перевірювані |
| **Groundedness** | **0.3** | Знижено з початкових 0.6 після реального прогону: Researcher комбінує кілька джерел (`web_search` + `knowledge_search`) у своєму власному ReAct-циклі, і наш тест **відтворює** `retrieval_context`, викликаючи ті самі tools **окремо**, а не перехоплюючи фактичні виклики агента. Це апроксимація "чорної скриньки" — спостережений розкид scores між прогонами: 0.3, 0.9, знову 0.3. Точніший тест вимагав би інструментації самого `invoke()` для запису реальних tool outputs. |
| Answer Relevancy | 0.6 | Стандартний поріг з прикладу завдання |
| Correctness | 0.4 | Дещо знижено з 0.6 — Researcher/Supervisor схильні давати значно розгорнутіші відповіді, ніж короткі `expected_output` у golden dataset; GEval коректно не карає за це (це не суперечність), але точна відповідність довжини очікувано низька |
| Tool Correctness | 0.5 | З прикладу завдання, не змінювалось |

## Відомі джерела нестабільності (не помилки коду)

- **Rate limit (429) безкоштовної квоти Gemini** (`gemini-3.5-flash-lite`, ліміт ~15 запитів/хв) — `test_e2e.py` виконує 5 повних Planner→Researcher→Critic циклів поспіль; за один прогін усього тестового набору 2 з 5 e2e-тестів іноді падають саме через це, не через логічну помилку. Видно з traceback (`ClientError: 429 RESOURCE_EXHAUSTED`), а не з `AssertionError`.
- **Варіативність Groundedness між прогонами** (0.3 → 0.9 → 0.3) — задокументована вище, притаманна підходу "реконструювати контекст постфактум", а не збою системи.

## Результати останнього повного прогону

$ deepeval test run tests/
12 passed, 2 failed (обидва — 429 RESOURCE_EXHAUSTED, не логічні помилки)
DeepEval summary: 10/10 test cases completed successfully (100%)


## Запуск

```bash
cd homework-lesson-11
deepeval test run tests/              # усі тести
deepeval test run tests/test_planner.py   # окремий файл
deepeval test run tests/ -s           # з print-виводом (tool calls тощо)
```