"""Unit tests for the dedicated router node and refusal boundaries."""

import pytest
from langchain_core.messages import HumanMessage
from src.router import classify_query, router_node, route_decision, REFUSAL_INDIVIDUAL, REFUSAL_RAW_DATA, REFUSAL_OUT_OF_SCOPE


@pytest.mark.parametrize(
    "query,expected_category",
    [
        ("Who is the employee who answered 1 on question 4 in District Court?", "individual_targeting"),
        ("What did Bob in Human Services say in his survey?", "individual_targeting"),
        ("Tell me about Alice's responses in 2023", "individual_targeting"),
        ("Find the individual who gave Strongly Disagree on question 2", "individual_targeting"),
        ("Show me the specific person who complained about their supervisor", "individual_targeting"),
    ],
)
def test_router_rejects_individual_targeting(query: str, expected_category: str):
    decision, reason, category = classify_query(query)
    assert decision == "refuse"
    assert category == expected_category
    assert reason == REFUSAL_INDIVIDUAL


@pytest.mark.parametrize(
    "query,expected_category",
    [
        ("Show me the raw rows for 2024 May", "raw_data_request"),
        ("Dump all records from survey_responses table", "raw_data_request"),
        ("Give me row 42 from the dataset", "raw_data_request"),
        ("Export unaggregated data for Human Services", "raw_data_request"),
        ("Download every single row in CSV format", "raw_data_request"),
    ],
)
def test_router_rejects_raw_records(query: str, expected_category: str):
    decision, reason, category = classify_query(query)
    assert decision == "refuse"
    assert category == expected_category
    assert reason == REFUSAL_RAW_DATA


@pytest.mark.parametrize(
    "query,expected_category",
    [
        ("What is the average salary in Human Services?", "out_of_scope"),
        ("How much bonus was paid to directors in 2023?", "out_of_scope"),
        ("What is the county's total annual budget for 2024?", "out_of_scope"),
        ("Who is the president of the United States?", "out_of_scope"),
        ("Write a Python script to sort a list", "out_of_scope"),
    ],
)
def test_router_rejects_out_of_scope(query: str, expected_category: str):
    decision, reason, category = classify_query(query)
    assert decision == "refuse"
    assert category == expected_category
    assert reason == REFUSAL_OUT_OF_SCOPE


@pytest.mark.parametrize(
    "query",
    [
        "What is the average agreement score for Question 01 in Human Services in 2024 May?",
        "Compare Question 01 between Human Services and District Court in 2023 May",
        "What was the respondent count for Managers across all departments in 2022 May?",
        "Show the distribution of answers for Question 02 in Planning & Public Works",
    ],
)
def test_router_approves_valid_aggregate_queries(query: str):
    decision, reason, category = classify_query(query)
    assert decision == "answer"
    assert reason is None
    assert category is None


def test_router_node_state_mutation():
    """Verify router_node updates messages and metadata_log correctly on refusal."""
    state = {
        "messages": [HumanMessage(content="Who is the worker that gave 1 in Juvenile Court?")],
        "step_count": 0,
        "metadata_log": [],
    }
    result = router_node(state)
    assert "messages" in result
    assert "Refusal" in result["messages"][0].content
    assert result["metadata_log"][0]["type"] == "router_refusal"

    merged_state = {**state, **result}
    decision = route_decision(merged_state)
    assert decision == "end"
