from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from config import settings, PLANNER_SYSTEM_PROMPT
from schemas import ResearchPlan
from tools import web_search, knowledge_search


def build_planner_agent():
    model = init_chat_model(
    f"google_genai:{settings.model_name}",
    api_key=settings.google_api_key,
    )
    return create_agent(
        model=model,
        tools=[web_search, knowledge_search],
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
    )