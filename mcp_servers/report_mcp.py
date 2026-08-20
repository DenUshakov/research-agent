from fastmcp import FastMCP

from tools import save_report

mcp = FastMCP("ReportMCP")


@mcp.tool()
def save_report_tool(filename: str, content: str) -> str:
    """Зберігає фінальний Markdown-звіт у файл у директорії output/."""
    return save_report(filename, content)


@mcp.resource("resource://output-dir")
def output_dir_info() -> dict:
    """Шлях до директорії збережених звітів та список уже збережених файлів."""
    import os

    from config import settings

    output_dir = settings.output_dir
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".md"))

    return {
        "path": os.path.abspath(output_dir),
        "saved_reports": files,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8902, path="/mcp")