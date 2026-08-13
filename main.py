from langgraph.types import Command

from supervisor import build_supervisor


def print_interrupt(interrupt_value):
    action = interrupt_value["action_requests"][0]
    args = action["args"]
    print("\n" + "=" * 60)
    print("⏸️  ACTION REQUIRES APPROVAL")
    print("=" * 60)
    print(f"  Tool:  {action['name']}")
    print(f"  File:  {args.get('filename', '?')}")
    print(f"  Preview:\n{args.get('content', '')[:500]}...\n")


def main():
    supervisor = build_supervisor()
    thread_id = "cli-session"
    config = {"configurable": {"thread_id": thread_id}}

    print("Multi-Agent Research System. Введи запит (або 'exit' для виходу).\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Цикл обробки interrupt: може повторюватись кілька разів при "edit"
        while "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            print_interrupt(interrupt_value)

            decision = input("👉 approve / edit / reject: ").strip().lower()

            if decision == "approve":
                resume = {"decisions": [{"type": "approve"}]}
            elif decision == "edit":
                feedback = input("✏️  Your feedback: ").strip()
                resume = {"decisions": [{"type": "reject", "message": f"Revise and try again: {feedback}"}]}
            elif decision == "reject":
                reason = input("Причина відхилення (Enter — без причини): ").strip()
                resume = {"decisions": [{"type": "reject", "message": reason or "Rejected by user"}]}
            else:
                print("Невідома команда, спробуй ще раз.")
                continue

            result = supervisor.invoke(Command(resume=resume), config=config)

        last_message = result["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        print(f"\n✅ Agent: {content}\n")


if __name__ == "__main__":
    main()