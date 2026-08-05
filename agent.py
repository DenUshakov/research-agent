import json
import time

from google import genai

from config import settings
from prompts import SYSTEM_PROMPT
from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS


def build_client():
    return genai.Client(api_key=settings.google_api_key)


def trim_history(history: list, max_chars: int) -> None:
    """Обрізає найстаріші function_result кроки, якщо history перевищує бюджет символів.

    Мутує history на місці. Не чіпає user_input і текстові відповіді моделі —
    тільки стискає давні результати tools, які найменш цінні для поточного кроку.
    """
    total = sum(len(json.dumps(item, ensure_ascii=False)) for item in history)

    idx = 0
    while total > max_chars and idx < len(history):
        item = history[idx]
        if item.get("type") == "function_result":
            for block in item.get("result", []):
                text = block.get("text", "")
                if len(text) > 200:
                    old_len = len(text)
                    block["text"] = text[:200] + f"...[обрізано для економії контексту, було {old_len} символів]"
                    total -= (old_len - len(block["text"]))
        idx += 1


def run_agent(client, history: list) -> str:
    """Запускає власний ReAct-цикл над history (список кроків Interactions API)."""
    for iteration in range(settings.max_iterations):
        trim_history(history, settings.max_history_chars)

        interaction = None
        last_error = None
        for attempt in range(3):
            try:
                interaction = client.interactions.create(
                    model=settings.model_name,
                    store=False,
                    input=history,
                    tools=TOOL_DECLARATIONS,
                    system_instruction=SYSTEM_PROMPT,
                )
                break
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"⚠️ Помилка виклику моделі (спроба {attempt + 1}/3): {e}. Повтор через {wait}с...")
                time.sleep(wait)
        else:
            return f"[Помилка виклику моделі після кількох спроб: {last_error}]"

        steps = interaction.steps or []
        for step in steps:
            history.append(step.model_dump())

        function_call_steps = [s for s in steps if s.type == "function_call"]

        if not function_call_steps:
            return interaction.output_text

        for fc_step in function_call_steps:
            print(f"\n🔧 Tool call: {fc_step.name}({fc_step.arguments})")

            func = TOOL_FUNCTIONS.get(fc_step.name)
            if func is None:
                result = f"Помилка: невідомий інструмент '{fc_step.name}'"
            else:
                try:
                    result = func(**fc_step.arguments)
                except Exception as e:
                    result = f"Помилка виконання інструменту {fc_step.name}: {e}"

            result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            preview = result_text[:200] + ("..." if len(result_text) > 200 else "")
            print(f"📎 Result: {preview}")

            history.append({
                "type": "function_result",
                "name": fc_step.name,
                "call_id": fc_step.id,
                "result": [{"type": "text", "text": result_text}],
            })

    return "[Досягнуто ліміту ітерацій — не вдалось отримати фінальну відповідь]"