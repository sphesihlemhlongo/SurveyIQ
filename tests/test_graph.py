"""Integration tests for end-to-end LangGraph execution."""

from langchain_core.messages import HumanMessage
from src.agent import build_survey_agent_graph


def test_graph_refusal_path():
    """Verify end-to-end refusal path skips tool execution."""
    graph = build_survey_agent_graph()
    state = {
        "messages": [HumanMessage(content="Who is the worker who scored 1 on question 4 in District Court?")],
        "step_count": 0,
        "metadata_log": [],
    }
    result = graph.invoke(state)
    assert "Refusal:" in result["messages"][-1].content
    assert result["step_count"] == 0
    assert any(log.get("type") == "router_refusal" for log in result["metadata_log"])


def test_graph_multi_hop_comparison():
    """Verify multi-hop execution: executes 2 queries and synthesizes comparison."""
    graph = build_survey_agent_graph()
    state = {
        "messages": [HumanMessage(content="Compare Question 01 between Human Services and District Court in 2024 May")],
        "step_count": 0,
        "metadata_log": [],
    }
    result = graph.invoke(state)
    assert result["step_count"] >= 2
    assert any(term in result["messages"][-1].content for term in ("Comparative Synthesis", "Comparison", "Comparative", "compare", "Comparing"))
    assert "Human Services" in result["messages"][-1].content
    assert "District Court" in result["messages"][-1].content


def test_graph_privacy_guardrail_integration():
    """Verify privacy guardrail inside graph logs privacy_block and returns fallback answer."""
    graph = build_survey_agent_graph()
    state = {
        "messages": [HumanMessage(content="What did the manager in Assessor-Treasurer's Office score in 2021 May on Question 01?")],
        "step_count": 0,
        "metadata_log": [],
    }
    result = graph.invoke(state)
    assert any(log.get("type") == "privacy_block" for log in result["metadata_log"])
    assert any(term in result["messages"][-1].content.upper() for term in ("PRIVACY GUARDRAIL", "PRIVACY", "CONFIDENTIAL", "SUPPRESS"))

