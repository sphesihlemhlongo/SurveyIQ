"""Unit tests for deterministic loop safety and operational budget cap."""

from langchain_core.messages import HumanMessage
from src.agent import build_survey_agent_graph, LOOP_SAFETY_BUDGET
from src.mock_llm import DeterministicSurveyLLM


def test_loop_safety_budget_exhaustion_terminates_graph():
    """Verify that when tool iterations reach LOOP_SAFETY_BUDGET (4), graph terminates with exact error."""
    runaway_llm = DeterministicSurveyLLM(mode="runaway")
    graph = build_survey_agent_graph(llm=runaway_llm)

    state = {
        "messages": [HumanMessage(content="Loop continuously without completing")],
        "step_count": 0,
        "metadata_log": [],
    }

    result = graph.invoke(state)

    # Validate terminal message
    final_message = result["messages"][-1].content
    assert "Terminal error: Operational budget is exhausted" in final_message
    assert f"step count >= {LOOP_SAFETY_BUDGET}" in final_message

    # Validate step count is at least the budget cap
    assert result["step_count"] >= LOOP_SAFETY_BUDGET

    # Validate terminal log entry
    terminal_logs = [log for log in result["metadata_log"] if log.get("type") == "terminal_budget_exhausted"]
    assert len(terminal_logs) == 1
    assert terminal_logs[0]["step_count"] >= LOOP_SAFETY_BUDGET
