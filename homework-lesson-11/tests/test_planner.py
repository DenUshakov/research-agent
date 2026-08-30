import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import build_planner_agent


def _run_planner(request: str):
    agent = build_planner_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["structured_response"]


@pytest.fixture(scope="module")
def plan_quality(eval_model):
    return GEval(
        name="Plan Quality",
        evaluation_steps=[
            "Check that the plan contains specific search queries (not vague)",
            "Check that sources_to_check includes relevant sources for the topic",
            "Check that the output_format matches what the user asked for",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=eval_model,
        threshold=0.6,
    )


def test_plan_quality_rag_topic(plan_quality):
    request = "Compare naive RAG vs sentence-window retrieval"
    plan = _run_planner(request)

    actual_output = (
        f"goal: {plan.goal}\n"
        f"search_queries: {plan.search_queries}\n"
        f"sources_to_check: {plan.sources_to_check}\n"
        f"output_format: {plan.output_format}"
    )

    test_case = LLMTestCase(input=request, actual_output=actual_output)
    assert_test(test_case, [plan_quality])


def test_plan_has_specific_queries(plan_quality):
    request = "What are the main components of a RAG pipeline?"
    plan = _run_planner(request)

    assert len(plan.search_queries) >= 2, "Plan should decompose into multiple search queries"
    assert all(len(q) > 10 for q in plan.search_queries), "Queries should not be trivially short/vague"

    actual_output = (
        f"goal: {plan.goal}\n"
        f"search_queries: {plan.search_queries}\n"
        f"sources_to_check: {plan.sources_to_check}\n"
        f"output_format: {plan.output_format}"
    )
    test_case = LLMTestCase(input=request, actual_output=actual_output)
    assert_test(test_case, [plan_quality])