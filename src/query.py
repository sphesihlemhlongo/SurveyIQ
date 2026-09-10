"""The single tool available to your system for touching the dataset.

This is implemented for you — do not rewrite the aggregation logic. Import
and use it. You may wrap it, but the underlying computation should stay
as-is, since part of what's being evaluated is how your system behaves
around this exact tool, not a tool you wrote yourself.

Design constraint you must respect regardless of how you call this: no
raw individual survey rows may reach the LLM. This function returns
aggregates. Whether an aggregate can, in practice, describe a single
person is something you need to work out for yourself.
"""

from dataclasses import dataclass

import random

import pandas as pd

from data.loader import load_survey

_DF = None


def _df() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = load_survey()
    return _DF


@dataclass
class AggregateResult:
    respondent_count: int
    mean_score: float | None
    distribution: dict[str, int]
    filters_applied: dict[str, str | None]


class QueryError(Exception):
    """Raised when the query could not be completed."""


_call_counter = {"n": 0}


def query_aggregate(
    question: str | None = None,
    year: str | None = None,
    department: str | None = None,
    role: str | None = None,
) -> AggregateResult:
    """Return an aggregate over the survey, filtered by the given fields.

    Args:
        question:   exact question text, or None for all questions
        year:       survey wave, e.g. "2024 May", or None for all waves
        department: department name, or None for all departments
        role:       role name, or None for all roles

    Returns:
        An AggregateResult.

    Raises:
        QueryError: the query could not be completed. This happens under
            real, if unstated, conditions — your system needs to decide
            what to do when it happens, not assume it won't. Do not
            design against the specific mechanism below; design against
            a tool that fails sometimes, the way a real dependency would.
    """
    _call_counter["n"] += 1

    # Simulates a real-world flaky dependency (e.g. a downstream cache or
    # rate-limited store). Non-deterministic and unseeded on purpose —
    # this isn't something you can read your way around by counting calls.
    if random.random() < 0.12:
        raise QueryError("aggregate store temporarily unavailable")

    df = _df()
    filters = {"question": question, "year": year, "department": department, "role": role}

    mask = pd.Series(True, index=df.index)
    if question is not None:
        mask &= df["Question"] == question
    if year is not None:
        mask &= df["Year"] == year
    if department is not None:
        mask &= df["Department"] == department
    if role is not None:
        mask &= df["Role"] == role

    subset = df[mask]
    n = len(subset)

    if n == 0:
        return AggregateResult(
            respondent_count=0, mean_score=None, distribution={}, filters_applied=filters
        )

    mean = float(subset["Answer_Numeric"].mean())
    dist = subset["Answer_Text"].value_counts(dropna=True).to_dict()

    return AggregateResult(
        respondent_count=n,
        mean_score=mean,
        distribution=dist,
        filters_applied=filters,
    )
