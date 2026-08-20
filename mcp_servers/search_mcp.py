import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastmcp import FastMCP

from tools import web_search, read_url, knowledge_search, preload_retriever

mcp = FastMCP("SearchMCP")


@mcp.tool()
def web_search_tool(query: str) -> list[dict]:
    """Шукає інформацію в інтернеті через DuckDuckGo. Повертає короткі сніпети (title, url, snippet)."""
    return web_search(query)


@mcp.tool()
def read_url_tool(url: str) -> str:
    """Завантажує сторінку за URL і повертає її основний текст."""
    return read_url(url)


@mcp.tool()
def knowledge_search_tool(query: str) -> str:
    """Шукає інформацію в локальній базі знань (проіндексовані PDF документи про RAG, LangChain, LLM)."""
    return knowledge_search(query)


@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> dict:
    """Кількість документів у базі знань та дата останнього оновлення індексу."""
    import os
    import pickle

    from config import settings

    chunks_path = os.path.join(settings.index_dir, "chunks.pkl")
    if not os.path.exists(chunks_path):
        return {"error": "Індекс не побудовано. Запусти python ingest.py"}

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    sources = sorted(set(c["source"] for c in chunks))
    mtime = os.path.getmtime(chunks_path)

    import datetime
    return {
        "total_chunks": len(chunks),
        "documents": sources,
        "last_updated": datetime.datetime.fromtimestamp(mtime).isoformat(),
    }


if __name__ == "__main__":
    print("Прогріваю Retriever (завантаження моделей embeddings/reranker)...")
    preload_retriever()
    print("Готово. Запускаю сервер...")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8901, path="/mcp")