from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from config import settings, CRITIC_SYSTEM_PROMPT
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search


def build_critic_agent():

    model = init_chat_model(
    f"google_genai:{settings.model_name}",
    api_key=settings.google_api_key,
    )

    return create_agent(
        model=model,
        tools=[web_search, read_url, knowledge_search],
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
    )