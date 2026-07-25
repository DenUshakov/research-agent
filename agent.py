from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from config import settings
from prompts import SYSTEM_PROMPT
from tools import web_search, read_url, write_report


def build_agent():
    model = init_chat_model(
        f"google_genai:{settings.model_name}",
        api_key=settings.google_api_key,
    )

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[web_search, read_url, write_report],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent