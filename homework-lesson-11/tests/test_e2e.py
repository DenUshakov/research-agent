import json
import os

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import build_planner_agent
from agents.research import build_research_agent
from agents.critic import build_critic_agent


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _run_full_pipeline(request: str) -> str:
    """Спрощений повний pipeline: Planner → Researcher → Critic → фінальний текст.

    На відміну від справжнього Supervisor (де це робить LLM через tool calls),
    тут ми оркеструємо виклики напряму в Python — це прогнозованіше і швидше
    для тестового набору, зберігаючи ту саму логічну послідовність.
    """
    planner = build_planner_agent()
    plan_result = planner.invoke({"messages": [{"role": "user", "content": request}]})
    plan = plan_result["structured_response"]

    researcher = build_research_agent()
    research_prompt = (
        f"Research the following, addressing all these queries: {plan.search_queries}. "
        f"Goal: {plan.goal}"
    )
    research_result = researcher.invoke({"messages": [{"role": "user", "content": research_prompt}]})
    findings = _extract_text(research_result["messages"][-1].content)

    critic = build_critic_agent()
    critic_result = critic.invoke(
        {"messages": [{"role": "user", "content": f"Original request: {request}\n\nFindings: {findings}"}]}
    )
    critique = critic_result["structured_response"]

    # Для e2e-тесту повертаємо знахідки + позначку вердикту Critic, а не повний write-up звіту
    # (написання фінального Markdown — робота Supervisor, тут перевіряємо саму дослідницьку якість).
    return f"{findings}\n\n[Critic verdict: {critique.verdict}]"


@pytest.fixture(scope="module")
def golden_dataset():
    path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def answer_relevancy(eval_model):
    return AnswerRelevancyMetric(threshold=0.6, model=eval_model)


@pytest.fixture(scope="module")
def correctness(eval_model):
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict 'expected output'",
            "Penalize omission of critical details",
            "Different wording of the same concept is acceptable",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=eval_model,
        threshold=0.4,
    )


# Прогонимо лише happy_path приклади через повний pipeline — edge/failure cases
# по своїй природі не завжди дають "дослідницьку" відповідь, придатну для
# Correctness/AnswerRelevancy порівняння з expected_output.
@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_golden_dataset_happy_path(index, golden_dataset, answer_relevancy, correctness):
    happy_path_examples = [d for d in golden_dataset if d["category"] == "happy_path"]
    example = happy_path_examples[index]

    actual_output = _run_full_pipeline(example["input"])

    test_case = LLMTestCase(
        input=example["input"],
        actual_output=actual_output,
        expected_output=example["expected_output"],
    )
    assert_test(test_case, [answer_relevancy, correctness])