from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings, RESEARCHER_SYSTEM_PROMPT


async def build_research_agent():
    client = MultiServerMCPClient(
        {
            "search": {
                "transport": "streamable_http",
                "url": settings.search_mcp_url,
            },
        }
    )
    tools = await client.get_tools()

    model = init_chat_model(
        f"google_genai:{settings.model_name}",
        api_key=settings.google_api_key,
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
    )