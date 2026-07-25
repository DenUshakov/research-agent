import os
from langchain_core.tools import tool


@tool
def write_report(filename: str, content: str) -> str:
    """Зберігає фінальний Markdown-звіт у файл у директорії output/.

    Args:
        filename: назва файлу (наприклад, "rag_comparison.md")
        content: повний текст звіту у форматі Markdown

    Returns:
        Підтвердження з повним шляхом до збереженого файлу.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    abs_path = os.path.abspath(path)
    return f"Звіт успішно збережено: {abs_path}"

from ddgs import DDGS
from config import settings


@tool
def web_search(query: str) -> list[dict]:
    """Шукає інформацію в інтернеті через DuckDuckGo.

    Повертає короткі сніпети сторінок (title, url, snippet) — цього достатньо,
    щоб зрозуміти релевантність джерела, але не для глибокого аналізу.
    Для повного тексту сторінки використовуй read_url.

    Args:
        query: пошуковий запит

    Returns:
        Список результатів, кожен з полями title, url, snippet.
        У разі помилки — список з одним елементом, що описує помилку.
    """
    try:
        raw_results = DDGS().text(query, max_results=5)
    except Exception as e:
        return [{"error": f"Помилка пошуку: {e}"}]

    results = []
    for r in raw_results:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        })
    return results

import trafilatura


@tool
def read_url(url: str) -> str:
    """Завантажує сторінку за URL і повертає її основний текст (без меню, реклами, футерів).

    Використовуй, коли web_search знайшов релевантну сторінку і потрібні деталі,
    яких немає в короткому сніпеті.

    Args:
        url: повна URL-адреса сторінки (наприклад, "https://example.com/article")

    Returns:
        Текст сторінки, обрізаний до розумного ліміту символів.
        У разі помилки (сторінка недоступна, невалідний URL тощо) — повідомлення про помилку.
    """
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