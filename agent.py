import json

from google import genai

from config import settings
from prompts import SYSTEM_PROMPT
from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS


def build_client():
    return genai.Client(api_key=settings.google_api_key)


def run_agent(client, history: list) -> str:
    """Запускає власний ReAct-цикл над history (список кроків Interactions API).

    history мутується на місці — так реалізується "памʼять": виклик за викликом
    список поповнюється кроками користувача, моделі та результатами tools.
    Повертає фінальний текст відповіді.
    """
    for iteration in range(settings.max_iterations):
        try:
            interaction = client.interactions.create(
                model=settings.model_name,
                store=False,
                input=history,
                tools=TOOL_DECLARATIONS,
                system_instruction=SYSTEM_PROMPT,
            )
        except Exception as e:
            return f"[Помилка виклику моделі: {e}]"

        for step in interaction.steps:
            history.append(step.model_dump())

        function_call_steps = [s for s in interaction.steps if s.type == "function_call"]

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