import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import build_critic_agent


def _run_critic(findings: str):
    agent = build_critic_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": findings}]})
    return result["structured_response"]


@pytest.fixture(scope="module")
def critique_quality(eval_model):
    return GEval(
        name="Critique Quality",
        evaluation_steps=[
            "Check that the critique identifies specific issues, not vague complaints",
            "Check that revision_requests are actionable (researcher can act on them)",
            "If verdict is APPROVE, gaps list should be empty or contain only minor items",
            "If verdict is REVISE, there must be at least one revision_request",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=eval_model,
        threshold=0.6,
    )


def test_critique_approve_on_solid_research(critique_quality):
    findings = (
        "Original request: What is RAG?\n\n"
        "Findings: RAG (Retrieval-Augmented Generation) combines a retriever and an LLM. "
        "The retriever encodes the query, searches a vector database for similar chunks, "
        "and passes them to the LLM as context. This reduces hallucinations and allows "
        "the system to use up-to-date or domain-specific knowledge without retraining. "
        "Key components: chunking, embeddings, vector store, retriever, generator."
    )
    result = _run_critic(findings)

    assert result.verdict == "APPROVE", f"Expected APPROVE for solid research, got {result.verdict}"

    actual_output = (
        f"verdict: {result.verdict}\ngaps: {result.gaps}\n"
        f"revision_requests: {result.revision_requests}\nstrengths: {result.strengths}"
    )
    test_case = LLMTestCase(input=findings, actual_output=actual_output)
    assert_test(test_case, [critique_quality])


def test_critique_revise_on_sparse_research(critique_quality):
    findings = "Original request: Explain RAG in full detail with architecture, benefits, and limitations.\n\nFindings: RAG is a technique."
    result = _run_critic(findings)

    assert result.verdict == "REVISE", f"Expected REVISE for sparse research, got {result.verdict}"
    assert len(result.revision_requests) >= 1, "REVISE verdict must include at least one revision_request"

    actual_output = (
        f"verdict: {result.verdict}\ngaps: {result.gaps}\n"
        f"revision_requests: {result.revision_requests}\nstrengths: {result.strengths}"
    )
    test_case = LLMTestCase(input=findings, actual_output=actual_output)
    assert_test(test_case, [critique_quality])