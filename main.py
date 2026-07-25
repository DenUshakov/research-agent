from agent import build_agent
from config import settings


def extract_text(content) -> str:
    """Витягує чистий текст з content, який може бути рядком або списком блоків (Gemini)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def main():
    agent = build_agent()
    config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": settings.max_iterations,
    }

    print("Research Agent. Введи питання (або 'exit' для виходу).\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            last_message = result["messages"][-1]
            print(f"\nAgent: {extract_text(last_message.content)}\n")  # ← ЗМІНЕНО
        except Exception as e:
            print(f"\n[Помилка виконання агента: {e}]\n")


if __name__ == "__main__":
    main()