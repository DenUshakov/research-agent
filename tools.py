import os

from ddgs import DDGS
import trafilatura

from config import settings


def web_search(query: str) -> list[dict]:
    """Шукає інформацію в інтернеті через DuckDuckGo."""
    try:
        raw_results = DDGS().text(query, max_results=5)
    except Exception as e:
        return [{"error": f"Помилка пошуку: {e}"}]

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw_results
    ]


def read_url(url: str) -> str:
    """Завантажує сторінку за URL і повертає її основний текст."""

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as e:
        return f"Помилка завантаження сторінки: {e}"

    if downloaded is None:
        return f"Не вдалося завантажити сторінку за адресою {url} (недоступна або невалідний URL)."

    text = trafilatura.extract(downloaded)
    if not text:
        return f"Сторінку {url} завантажено, але не вдалося витягти текстовий вміст."

    limit = settings.max_tool_result_chars
    if len(text) > limit:
        text = text[:limit] + f"\n\n[...обрізано, повний текст був {len(text)} символів...]"
    return text


def write_report(filename: str, content: str) -> str:
    """Зберігає фінальний Markdown-звіт у файл у директорії output/."""
    safe_name = os.path.basename(filename)
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    output_dir = settings.output_dir
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, safe_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Звіт успішно збережено: {os.path.abspath(path)}"


# --- JSON Schema декларації для tool calling API ---

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Шукає інформацію в інтернеті через DuckDuckGo. Повертає короткі сніпети "
            "(title, url, snippet) — цього достатньо, щоб зрозуміти релевантність джерела, "
            "але не для глибокого аналізу. Для повного тексту сторінки використовуй read_url."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Пошуковий запит"},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "read_url",
        "description": (
            "Завантажує сторінку за URL і повертає її основний текст (без меню, реклами, "
            "футерів). Використовуй, коли web_search знайшов релевантну сторінку і потрібні "
            "деталі, яких немає в сніпеті."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Повна URL-адреса сторінки"},
            },
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "write_report",
        "description": (
            "Зберігає фінальний Markdown-звіт у файл. Викликай лише після того, як зібрав "
            "достатньо інформації і сформував повний текст звіту."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Назва файлу, напр. 'report.md'"},
                "content": {"type": "string", "description": "Повний текст звіту у Markdown"},
            },
            "required": ["filename", "content"],
        },
    },
]



TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}