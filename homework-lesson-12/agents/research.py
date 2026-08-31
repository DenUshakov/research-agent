from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from config import settings, RESEARCHER_SYSTEM_PROMPT
from tools import web_search, read_url, knowledge_search


def build_research_agent():
    model = init_chat_model(
    f"google_genai:{settings.model_name}",
    api_key=settings.google_api_key,
    )
    return create_agent(
        model=model,
        tools=[web_search, read_url, knowledge_search],
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
    )