"""Fault-tolerant, privacy-preserving query wrapper around src.query.query_aggregate."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.query import AggregateResult, QueryError, query_aggregate

logger = logging.getLogger(__name__)

K_ANONYMITY_THRESHOLD = 10
QUESTIONS_PER_SURVEY_WAVE = 17


@dataclass
class SafeQueryResult:
    """Safe aggregate result returned to the agent and user."""
    status: str  # "success", "privacy_fallback", "suppressed_no_fallback", "zero_results"
    respondent_count: int
    mean_score: float | None
    distribution: dict[str, int]
    filters_applied: dict[str, str | None]
    privacy_note: str | None = None
    original_filters: dict[str, str | None] | None = None
    metadata_event: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@retry(
    retry=retry_if_exception_type(QueryError),
    wait=wait_exponential(multiplier=0.05, min=0.05, max=0.8),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _query_with_retry(
    question: str | None = None,
    year: str | None = None,
    department: str | None = None,
    role: str | None = None,
) -> AggregateResult:
    """Executes query_aggregate with native tenacity retry for transient QueryErrors.
    
    The underlying query_aggregate logic in src/query.py is imported and called as-is.
    """
    return query_aggregate(
        question=question,
        year=year,
        department=department,
        role=role,
    )


def _normalize_role(role: str | None, year: str | None) -> str | None:
    """Normalize role naming across survey waves (Staff vs Staff Member)."""
    if role is None:
        return None
    r_lower = role.strip().lower()
    if r_lower in ("staff", "staff member"):
        if year in ("2019 May", "2020 Jun"):
            return "Staff"
        elif year in ("2021 May", "2022 May", "2023 May", "2024 May"):
            return "Staff Member"
    return role


def execute_safe_query(
    question: str | None = None,
    year: str | None = None,
    department: str | None = None,
    role: str | None = None,
) -> SafeQueryResult:
    """Queries survey aggregates with strict k-anonymity privacy and dynamic fallback.
    
    1. Retries on QueryError up to 5 times with exponential backoff.
    2. Intercepts respondent_count. If respondent_count is between 1 and 9 (k-anonymity violation),
       exact microdata is suppressed to prevent differencing attacks.
    3. Automatically executes Dynamic Fallback: strips the most specific filter (role),
       and re-queries the broader department.
    """
    role = _normalize_role(role, year)
    initial_filters = {"question": question, "year": year, "department": department, "role": role}

    # Attempt primary query
    res = _query_with_retry(question=question, year=year, department=department, role=role)

    # If role returned 0 but was 'Staff'/'Staff Member', try alternative synonym
    if res.respondent_count == 0 and role in ("Staff", "Staff Member"):
        alt_role = "Staff" if role == "Staff Member" else "Staff Member"
        alt_res = _query_with_retry(question=question, year=year, department=department, role=alt_role)
        if alt_res.respondent_count > 0:
            res = alt_res
            role = alt_role
            initial_filters["role"] = role

    # Check for empty result
    if res.respondent_count == 0:
        return SafeQueryResult(
            status="zero_results",
            respondent_count=0,
            mean_score=None,
            distribution={},
            filters_applied=res.filters_applied,
            privacy_note=None,
            original_filters=initial_filters,
        )

    # Assess k-anonymity violation:
    # 1. Exact count between 1 and 9 for a specific question (or overall)
    # 2. Long-format trap: if question is None and count <= (9 * 17) respondents
    is_k_anon_violation = (1 <= res.respondent_count <= 9)
    if not is_k_anon_violation and question is None:
        # If question is None, 1 respondent generates 17 rows
        effective_respondents = res.respondent_count / QUESTIONS_PER_SURVEY_WAVE
        if 0 < effective_respondents < K_ANONYMITY_THRESHOLD:
            is_k_anon_violation = True

    if not is_k_anon_violation:
        # Safe aggregate meets k-anonymity threshold (k >= 10)
        return SafeQueryResult(
            status="success",
            respondent_count=res.respondent_count,
            mean_score=res.mean_score,
            distribution=res.distribution,
            filters_applied=res.filters_applied,
            original_filters=initial_filters,
        )

    # Privacy Guardrail Triggered: 1 <= respondent_count <= 9
    logger.warning(
        f"Privacy block: k-anonymity violation for {initial_filters} (count={res.respondent_count} < {K_ANONYMITY_THRESHOLD})"
    )

    # DYNAMIC FALLBACK: Strip the most specific filter (role -> department level)
    fallback_role = None
    fallback_filters = {"question": question, "year": year, "department": department, "role": fallback_role}
    
    # Re-query broader department
    fallback_res = _query_with_retry(
        question=question,
        year=year,
        department=department,
        role=fallback_role,
    )

    # Check if department-level is also small (e.g., Economic Development with 1 respondent in 2021 May)
    is_fallback_k_viol = (1 <= fallback_res.respondent_count <= 9)
    if not is_fallback_k_viol and question is None:
        effective_dept_resp = fallback_res.respondent_count / QUESTIONS_PER_SURVEY_WAVE
        if 0 < effective_dept_resp < K_ANONYMITY_THRESHOLD:
            is_fallback_k_viol = True

    if is_fallback_k_viol:
        # Department itself is too small in this wave; fallback to department across ALL years
        wave_fallback_year = None
        broad_res = _query_with_retry(
            question=question,
            year=wave_fallback_year,
            department=department,
            role=None,
        )
        if broad_res.respondent_count >= K_ANONYMITY_THRESHOLD:
            metadata_event = {
                "type": "privacy_block",
                "reason": f"k-anonymity violation (subgroup count={res.respondent_count}, dept wave count={fallback_res.respondent_count})",
                "original_filters": initial_filters,
                "fallback_action": "stripped role and year filters (generalized to department across all waves)",
                "final_filters": broad_res.filters_applied,
                "final_respondent_count": broad_res.respondent_count,
            }
            note = (
                f"[PRIVACY GUARDRAIL TRIGGERED]: The requested specific segment ({initial_filters}) has fewer than "
                f"{K_ANONYMITY_THRESHOLD} respondents (count={res.respondent_count}). To prevent re-identification and differencing "
                f"attacks, exact scores are suppressed. Generalized across all survey waves for department '{department}' "
                f"(n={broad_res.respondent_count})."
            )
            return SafeQueryResult(
                status="privacy_fallback",
                respondent_count=broad_res.respondent_count,
                mean_score=broad_res.mean_score,
                distribution=broad_res.distribution,
                filters_applied=broad_res.filters_applied,
                privacy_note=note,
                original_filters=initial_filters,
                metadata_event=metadata_event,
            )
        else:
            # Department even across all waves has < 10 respondents: hard suppression
            metadata_event = {
                "type": "privacy_block",
                "reason": f"k-anonymity violation (insufficient cohort size n={res.respondent_count} even at broader scope)",
                "original_filters": initial_filters,
                "fallback_action": "full_suppression",
            }
            note = (
                f"[PRIVACY GUARDRAIL]: Cohort has fewer than {K_ANONYMITY_THRESHOLD} respondents. "
                "Data suppressed to maintain absolute privacy."
            )
            return SafeQueryResult(
                status="suppressed_no_fallback",
                respondent_count=0,
                mean_score=None,
                distribution={},
                filters_applied=initial_filters,
                privacy_note=note,
                original_filters=initial_filters,
                metadata_event=metadata_event,
            )

    # Standard dynamic fallback succeeded at department level
    metadata_event = {
        "type": "privacy_block",
        "reason": f"k-anonymity violation (respondent_count={res.respondent_count} < {K_ANONYMITY_THRESHOLD})",
        "original_filters": initial_filters,
        "fallback_action": "stripped role filter to broader department level",
        "final_filters": fallback_res.filters_applied,
        "final_respondent_count": fallback_res.respondent_count,
    }
    privacy_note = (
        f"[PRIVACY GUARDRAIL TRIGGERED]: Subgroup '{role}' in department '{department}' contains only "
        f"{res.respondent_count} respondent(s) (< {K_ANONYMITY_THRESHOLD} threshold). Exact microdata suppressed "
        f"to prevent differencing attacks. Dynamically generalized to the broader department level: "
        f"'{department}' ({fallback_res.respondent_count} respondents)."
    )

    return SafeQueryResult(
        status="privacy_fallback",
        respondent_count=fallback_res.respondent_count,
        mean_score=fallback_res.mean_score,
        distribution=fallback_res.distribution,
        filters_applied=fallback_res.filters_applied,
        privacy_note=privacy_note,
        original_filters=initial_filters,
        metadata_event=metadata_event,
    )


@tool
def safe_query_tool(
    question: str | None = None,
    year: str | None = None,
    department: str | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Query aggregate survey statistics safely over the HR employee engagement dataset.
    
    Args:
        question: Exact statement text (e.g. '01. I know what is expected of me at work.'), or None for all statements.
        year: Survey wave (e.g. '2024 May', '2023 May', '2022 May', '2021 May', '2020 Jun', '2019 May'), or None for all waves.
        department: Department name (e.g. 'Human Services', 'District Court', 'Assessor-Treasurer\\'s Office'), or None for all.
        role: Role filter (e.g. 'Staff', 'Staff Member', 'Manager', 'Supervisor', 'Lead', 'Director'), or None for all.
        
    Returns:
        A dictionary containing aggregate respondent_count, mean_score, distribution, filters_applied, and any privacy fallback notes.
    """
    result = execute_safe_query(
        question=question,
        year=year,
        department=department,
        role=role,
    )
    return result.to_dict()
