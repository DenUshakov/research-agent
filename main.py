from agent import build_client, run_agent


def main():
    client = build_client()
    history = []

    print("Research Agent (власний ReAct loop). Введи питання (або 'exit' для виходу).\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        history.append({
            "type": "user_input",
            "content": [{"type": "text", "text": user_input}],
        })

        answer = run_agent(client, history)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()