import asyncio
import json
import httpx

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from a2a.client import create_client, ClientConfig
from a2a.helpers import new_text_message
from a2a.types import SendMessageRequest

from config import settings, SUPERVISOR_SYSTEM_PROMPT


async def _delegate(url: str, text: str) -> str:
    """Надсилає текстове повідомлення A2A-агенту за URL і повертає текст відповіді."""
    httpx_client = httpx.AsyncClient(timeout=180.0)
    config = ClientConfig(httpx_client=httpx_client)
    client = await create_client(url, client_config=config)
    message = new_text_message(text=text)
    request = SendMessageRequest(message=message)
    async for event in client.send_message(request):
        parts = event.message.parts
        return "\n".join(p.text for p in parts if p.text)
    return ""


async def delegate_to_planner(request: str) -> str:
    """Надсилає запит Planner Agent (через A2A) для декомпозиції у структурований план дослідження."""
    print(f"\n[Supervisor --A2A--> Planner]\n🔧 delegate_to_planner({request[:80]!r})")
    url = f"http://127.0.0.1:{settings.planner_a2a_port}"
    result = await _delegate(url, request)
    print(f"  📎 {result[:150]}...")
    return result


async def delegate_to_researcher(request: str) -> str:
    """Надсилає завдання Research Agent (через A2A) для збору знахідок."""
    print(f"\n[Supervisor --A2A--> Researcher]\n🔧 delegate_to_researcher({request[:80]!r})")
    url = f"http://127.0.0.1:{settings.researcher_a2a_port}"
    result = await _delegate(url, request)
    print(f"  📎 Findings ({len(result)} chars)")
    return result


async def delegate_to_critic(findings: str) -> str:
    """Надсилає знахідки Critic Agent (через A2A) для незалежної оцінки якості."""
    print(f"\n[Supervisor --A2A--> Critic]\n🔧 delegate_to_critic(...)")
    url = f"http://127.0.0.1:{settings.critic_a2a_port}"
    result = await _delegate(url, findings)
    print(f"  📎 {result[:200]}...")
    return result

async def _get_search_mcp_tools():
    client = MultiServerMCPClient(
        {"report": {"transport": "streamable_http", "url": settings.report_mcp_url}}
    )
    return await client.get_tools()


async def build_supervisor():
    tools = await _get_search_mcp_tools()

    model = init_chat_model(
        f"google_genai:{settings.model_name}",
        api_key=settings.google_api_key,
    )
    return create_agent(
        model=model,
        tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, *tools],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"save_report_tool": True}),
        ],
        checkpointer=InMemorySaver(),
    )