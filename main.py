import asyncio

from langgraph.types import Command

from supervisor import build_supervisor


def extract_text(content) -> str:
    """Витягує чистий текст з content, який може бути рядком або списком блоків (Gemini)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def print_interrupt(interrupt_value):
    action = interrupt_value["action_requests"][0]
    args = action["args"]
    print("\n" + "=" * 60)
    print("⏸️  ACTION REQUIRES APPROVAL")
    print("=" * 60)
    print(f"  Tool:  {action['name']}")
    print(f"  File:  {args.get('filename', '?')}")
    print(f"  Preview:\n{args.get('content', '')[:500]}...\n")


async def main():
    supervisor = await build_supervisor()
    thread_id = "cli-session"
    config = {"configurable": {"thread_id": thread_id}}

    print("Multi-Agent Research System (MCP + A2A). Введи запит (або 'exit' для виходу).\n")

    while True:
        user_input = await asyncio.to_thread(input, "You: ")
        user_input = user_input.strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        while "__interrupt__" in result:
            interrupt_value = result["__interrupt__"][0].value
            print_interrupt(interrupt_value)

            decision = (await asyncio.to_thread(input, "👉 approve / edit / reject: ")).strip().lower()

            if decision == "approve":
                resume = {"decisions": [{"type": "approve"}]}
            elif decision == "edit":
                feedback = (await asyncio.to_thread(input, "✏️  Your feedback: ")).strip()
                resume = {"decisions": [{"type": "reject", "message": f"Revise and try again: {feedback}"}]}
            elif decision == "reject":
                reason = (await asyncio.to_thread(input, "Причина відхилення (Enter — без причини): ")).strip()
                resume = {"decisions": [{"type": "reject", "message": reason or "Rejected by user"}]}
            else:
                print("Невідома команда, спробуй ще раз.")
                continue

            result = await supervisor.ainvoke(Command(resume=resume), config=config)

        last_message = result["messages"][-1]
        print(f"\n✅ Agent: {extract_text(last_message.content)}\n")


if __name__ == "__main__":
    asyncio.run(main())