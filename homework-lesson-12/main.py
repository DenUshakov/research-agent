import uuid

from langgraph.types import Command
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler

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


@observe(name="mas-user-request")
def handle_request(supervisor, user_input: str, session_id: str, user_id: str, thread_id: str):
    """Обробляє один запит користувача повністю (включно з HITL-циклом) під одним Langfuse-трейсом."""
    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        tags=["multi-agent-research"],
    ):
        config = {"configurable": {"thread_id": thread_id}}
        langfuse_handler = CallbackHandler()

        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={**config, "callbacks": [langfuse_handler]},
        )

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

            langfuse_handler = CallbackHandler()
            result = supervisor.invoke(
                Command(resume=resume),
                config={**config, "callbacks": [langfuse_handler]},
            )

        last_message = result["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        return content


def main():
    supervisor = build_supervisor()
    thread_id = "cli-session"
    session_id = str(uuid.uuid4())  # одна сесія на весь запуск REPL
    user_id = "den"  # постав своє ім'я/ідентифікатор

    print("Multi-Agent Research System. Введи запит (або 'exit' для виходу).\n")
    print(f"(Langfuse session_id: {session_id})\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        answer = handle_request(supervisor, user_input, session_id, user_id, thread_id)
        print(f"\n✅ Agent: {answer}\n")


if __name__ == "__main__":
    main()