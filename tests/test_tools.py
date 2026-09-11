"""Unit tests for query tool wrapper, tenacity fault tolerance, and privacy guardrails."""

import unittest.mock as mock
import pytest
from src.query import AggregateResult, QueryError
from src.tools import (
    execute_safe_query,
    _query_with_retry,
    _normalize_role,
    K_ANONYMITY_THRESHOLD,
)


def test_tenacity_retry_recovers_from_transient_query_errors():
    """Verify tenacity decorator retries on QueryError up to 5 attempts and succeeds."""
    mock_call = mock.MagicMock()
    # Fail 3 times with QueryError, then succeed
    mock_call.side_effect = [
        QueryError("simulated flaky cache failure"),
        QueryError("simulated timeout"),
        QueryError("simulated rate limit"),
        AggregateResult(
            respondent_count=50,
            mean_score=3.5,
            distribution={"Agree": 30, "Strongly Agree": 20},
            filters_applied={"department": "Human Services"},
        ),
    ]

    with mock.patch("src.tools.query_aggregate", mock_call):
        result = _query_with_retry(department="Human Services")
        assert result.respondent_count == 50
        assert mock_call.call_count == 4


def test_tenacity_retry_raises_after_max_attempts():
    """Verify tenacity exhausts after 5 failed attempts."""
    mock_call = mock.MagicMock()
    mock_call.side_effect = QueryError("persistent backend failure")

    with mock.patch("src.tools.query_aggregate", mock_call):
        with pytest.raises(QueryError):
            _query_with_retry(department="Human Services")
        assert mock_call.call_count == 5


def test_k_anonymity_suppression_and_dynamic_fallback():
    """Verify single-person cohort (Assessor-Treasurer Manager, n=1) is suppressed and generalized to department (n=45)."""
    result = execute_safe_query(
        department="Assessor-Treasurer's Office",
        role="Manager",
        year="2021 May",
        question="01. I know what is expected of me at work.",
    )

    assert result.status == "privacy_fallback"
    assert result.respondent_count == 45
    assert result.filters_applied["role"] is None  # role filter stripped
    assert result.filters_applied["department"] == "Assessor-Treasurer's Office"
    assert result.privacy_note is not None
    assert "PRIVACY GUARDRAIL TRIGGERED" in result.privacy_note
    assert result.metadata_event is not None
    assert result.metadata_event["type"] == "privacy_block"
    assert result.metadata_event["fallback_action"] == "stripped role filter to broader department level"


def test_long_format_single_respondent_trap():
    """In long format, question=None for a 1-person cohort (Economic Dev 2021 May) yields 17 rows.
    
    Verify that our tool detects effective_respondents = 17 // 17 = 1 (< 10) and triggers dynamic fallback.
    """
    result = execute_safe_query(
        department="Economic Development",
        year="2021 May",
        question=None,
    )
    # Department itself has only 1 respondent in 2021 May, so it generalizes across all years
    assert result.status == "privacy_fallback"
    assert result.respondent_count >= K_ANONYMITY_THRESHOLD
    assert result.filters_applied["year"] is None  # year filter stripped to aggregate across waves
    assert result.metadata_event["type"] == "privacy_block"


def test_role_normalization_across_waves():
    """Verify role normalization handles Staff in 2019/2020 and Staff Member in 2021-2024."""
    assert _normalize_role("Staff", "2024 May") == "Staff Member"
    assert _normalize_role("Staff Member", "2019 May") == "Staff"
    assert _normalize_role("Manager", "2024 May") == "Manager"
    assert _normalize_role(None, "2024 May") is None
