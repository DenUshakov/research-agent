import pytest
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

from agents.planner import build_planner_agent
from agents.research import build_research_agent


def _extract_tool_calls(messages) -> list[ToolCall]:
    """Витягує список реально викликаних tools з історії повідомлень LangGraph."""
    calls = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                calls.append(ToolCall(name=tc["name"]))
    return calls


@pytest.fixture(scope="module")
def tool_metric(eval_model):
    return ToolCorrectnessMetric(threshold=0.5, model=eval_model)


def test_planner_calls_search_tools(tool_metric):
    """Planner отримує запит → має викликати пошукові інструменти для розвідки."""
    agent = build_planner_agent()
    request = "Compare naive RAG vs sentence-window retrieval"
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})

    tools_called = _extract_tool_calls(result["messages"])

    test_case = LLMTestCase(
        input=request,
        actual_output="(structured plan, see test_planner.py)",
        tools_called=tools_called,
        expected_tools=[ToolCall(name="web_search"), ToolCall(name="knowledge_search")],
    )
    tool_metric.measure(test_case)
    print(f"\nPlanner tools called: {[t.name for t in tools_called]}, tool correctness score: {tool_metric.score}")

    assert len(tools_called) > 0, "Planner should call at least one search tool for exploration"


def test_researcher_uses_tools_per_plan(tool_metric):
    """Researcher отримує запит → має використати пошукові інструменти згідно з задачею."""
    agent = build_research_agent()
    request = "Research: what is RAG, using both web search and the local knowledge base"
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})

    tools_called = _extract_tool_calls(result["messages"])
    called_names = {t.name for t in tools_called}

    print(f"\nResearcher tools called: {called_names}")
    assert len(tools_called) > 0, "Researcher should call at least one tool"
    assert called_names.issubset({"web_search", "read_url", "knowledge_search"}), (
        f"Researcher called unexpected tools: {called_names}"
    )


def test_researcher_prefers_knowledge_base_for_rag_topic(tool_metric):
    """Для теми, що явно є в локальній базі знань (RAG), Researcher має скористатись knowledge_search."""
    agent = build_research_agent()
    request = "Research the concept of Retrieval-Augmented Generation using the local knowledge base"
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})

    tools_called = _extract_tool_calls(result["messages"])
    called_names = {t.name for t in tools_called}

    print(f"\nTools called: {called_names}")
    assert "knowledge_search" in called_names, (
        "Researcher should use knowledge_search for a topic covered by the local knowledge base"
    )