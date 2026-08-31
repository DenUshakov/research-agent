import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langfuse.langchain import CallbackHandler

from config import settings, SUPERVISOR_SYSTEM_PROMPT
from agents.planner import build_planner_agent
from agents.research import build_research_agent
from agents.critic import build_critic_agent
from tools import save_report

_planner = None
_researcher = None
_critic = None


def _get_planner():
    global _planner
    if _planner is None:
        _planner = build_planner_agent()
    return _planner


def _get_researcher():
    global _researcher
    if _researcher is None:
        _researcher = build_research_agent()
    return _researcher


def _get_critic():
    global _critic
    if _critic is None:
        _critic = build_critic_agent()
    return _critic


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def plan(request: str) -> str:
    """Декомпозує запит користувача у структурований план дослідження (ResearchPlan): ціль, конкретні пошукові запити, джерела для перевірки, формат виводу. Викликай ПЕРШИМ для будь-якого дослідницького запиту."""
    print(f"\n[Supervisor → Planner]\n🔧 plan({request[:80]!r})")
    langfuse_handler = CallbackHandler()
    result = _get_planner().invoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"callbacks": [langfuse_handler]},
    )
    p = result["structured_response"]
    print(f"  📎 ResearchPlan(goal={p.goal!r}, queries={p.search_queries})")
    return json.dumps(
        {
            "goal": p.goal,
            "search_queries": p.search_queries,
            "sources_to_check": p.sources_to_check,
            "output_format": p.output_format,
        },
        ensure_ascii=False,
    )


def research(request: str) -> str:
    """Виконує дослідження за планом або за конкретним завданням доопрацювання від Critic. Повертає зібрані знахідки текстом, з позначками джерел."""
    print(f"\n[Supervisor → Researcher]\n🔧 research({request[:80]!r})")
    langfuse_handler = CallbackHandler()
    result = _get_researcher().invoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"callbacks": [langfuse_handler]},
    )
    content = _extract_text(result["messages"][-1].content)
    print(f"  📎 Findings collected ({len(content)} chars)")
    return content


def critique(findings: str) -> str:
    """Незалежно оцінює якість дослідження (findings) за freshness/completeness/structure. Повертає структурований CritiqueResult (verdict, gaps, revision_requests) як текст."""
    print(f"\n[Supervisor → Critic]\n🔧 critique(...)")
    langfuse_handler = CallbackHandler()
    result = _get_critic().invoke(
        {"messages": [{"role": "user", "content": findings}]},
        config={"callbacks": [langfuse_handler]},
    )
    c = result["structured_response"]
    print(f"  📎 CritiqueResult(verdict={c.verdict}, gaps={c.gaps})")
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


def build_supervisor():
    model = init_chat_model(
        f"google_genai:{settings.model_name}",
        api_key=settings.google_api_key,
    )
    return create_agent(
        model=model,
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
        ],
        checkpointer=InMemorySaver(),
    )