import asyncio

import uvicorn
from starlette.applications import Starlette
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, AgentInterface
from a2a.helpers import new_text_message
from a2a.utils import TransportProtocol
from config import settings
from agents.planner import build_planner_agent
from agents.research import build_research_agent
from agents.critic import build_critic_agent


def _extract_text(content) -> str:
    """Витягує чистий текст з content, який може бути рядком або списком блоків (Gemini)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


class LangChainAgentExecutor(AgentExecutor):
    """Обгортає create_agent-агента (Planner/Researcher/Critic) в A2A AgentExecutor."""

    def __init__(self, build_fn, to_text):
        self._build_fn = build_fn
        self._to_text = to_text
        self._agent = None
        self._lock = asyncio.Lock()

    async def _get_agent(self):
        if self._agent is None:
            async with self._lock:
                if self._agent is None:
                    self._agent = await self._build_fn()
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        agent = await self._get_agent()
        result = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
        text = self._to_text(result)
        await event_queue.enqueue_event(new_text_message(text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel не підтримується")


def _planner_to_text(result) -> str:
    import json
    p = result["structured_response"]
    return json.dumps(
        {
            "goal": p.goal,
            "search_queries": p.search_queries,
            "sources_to_check": p.sources_to_check,
            "output_format": p.output_format,
        },
        ensure_ascii=False,
    )


def _researcher_to_text(result) -> str:
    return _extract_text(result["messages"][-1].content)


def _critic_to_text(result) -> str:
    import json
    c = result["structured_response"]
    return json.dumps(
        {
            "verdict": c.verdict,
            "is_fresh": c.is_fresh,
            "is_complete": c.is_complete,
            "is_well_structured": c.is_well_structured,
            "strengths": c.strengths,
            "gaps": c.gaps,
            "revision_requests": c.revision_requests,
        },
        ensure_ascii=False,
    )


def _build_app(name: str, description: str, skill_id: str, port: int, executor: AgentExecutor) -> Starlette:
    skill = AgentSkill(
        id=skill_id,
        name=name,
        description=description,
        tags=[skill_id],
        examples=[],
    )
    interface = AgentInterface(
        url=f"http://127.0.0.1:{port}/",
        protocol_binding=TransportProtocol.JSONRPC,
        protocol_version="1.0",
    )
    card = AgentCard(
        name=name,
        description=description,
        supported_interfaces=[interface],
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = create_agent_card_routes(agent_card=card)
    routes += create_jsonrpc_routes(request_handler=handler, rpc_url="/")
    return Starlette(routes=routes)


async def main():
    from tools import preload_retriever
    print("Прогріваю Retriever...")
    preload_retriever()
    print("Готово. Стартую A2A-сервери...")

    planner_app = _build_app(
        "Planner Agent", "Декомпозує запит у структурований план дослідження",
        "planning", settings.planner_a2a_port,
        LangChainAgentExecutor(build_planner_agent, _planner_to_text),
    )
    researcher_app = _build_app(
        "Research Agent", "Виконує дослідження за планом через web та локальну базу знань",
        "research", settings.researcher_a2a_port,
        LangChainAgentExecutor(build_research_agent, _researcher_to_text),
    )
    critic_app = _build_app(
        "Critic Agent", "Незалежно оцінює якість дослідження",
        "critique", settings.critic_a2a_port,
        LangChainAgentExecutor(build_critic_agent, _critic_to_text),
    )

    configs = [
        uvicorn.Config(planner_app, host="127.0.0.1", port=settings.planner_a2a_port, log_level="info"),
        uvicorn.Config(researcher_app, host="127.0.0.1", port=settings.researcher_a2a_port, log_level="info"),
        uvicorn.Config(critic_app, host="127.0.0.1", port=settings.critic_a2a_port, log_level="info"),
    ]
    servers = [uvicorn.Server(c) for c in configs]
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())