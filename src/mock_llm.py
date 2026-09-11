"""Deterministic local LLM mock for offline evaluation, unit testing, and simulation."""

from __future__ import annotations

import json
import re
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


DEPARTMENTS = [
    "Assessor-Treasurer's Office",
    "Assigned Council",
    "Auditor's Office",
    "Clerk of Superior Court",
    "Communications Office",
    "Council's Office",
    "County Executive's Office",
    "District Court",
    "Economic Development",
    "Emergency Management",
    "Exec Office & Directors",
    "Executive's Office",
    "Facilities Management",
    "Family Justice Center",
    "Finance",
    "Human Resources",
    "Human Services",
    "Information Technology",
    "Juvenile Court",
    "Medical Examiner",
    "Parks and Recreation",
    "Planning & Public Works",
    "Prosecuting Attorney's Office",
    "Sheriff's Department",
    "Superior Court",
]

ROLES = ["Director", "Manager", "Supervisor", "Lead", "Staff", "Staff Member"]

YEARS = ["2019 May", "2020 Jun", "2021 May", "2022 May", "2023 May", "2024 May"]


def _extract_query_params(text: str) -> dict[str, str | None]:
    t = text.lower()
    dept = None
    for d in DEPARTMENTS:
        if d.lower() in t:
            dept = d
            break

    role = None
    for r in ROLES:
        if re.search(rf"\b{re.escape(r.lower())}\b", t):
            role = r
            break

    year = None
    for y in YEARS:
        if y.lower() in t or y[:4] in t:
            year = y
            break

    question = None
    if "question 01" in t or "question 1" in t or "expected of me" in t:
        question = "01. I know what is expected of me at work."
    elif "question 02" in t or "question 2" in t:
        question = "02. I have the materials and equipment I need to do my work right."
    elif "question 03" in t or "question 3" in t:
        question = "03. At work, I have the opportunity to do what I do best every day."
    elif "question 04" in t or "question 4" in t:
        question = "04. In the last seven days, I have received recognition or praise for doing good work."

    return {"department": dept, "role": role, "year": year, "question": question}


class DeterministicSurveyLLM:
    """Simulates Claude 3.5 Sonnet tool-calling behavior for the HR survey domain."""

    def __init__(self, mode: str = "normal"):
        self.mode = mode

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        # Find the original user question
        user_query = ""
        tool_messages: list[ToolMessage] = []
        for m in messages:
            if hasattr(m, "type") and m.type == "human" or type(m).__name__ in ("HumanMessage", "ChatMessage"):
                user_query = m.content if isinstance(m.content, str) else str(m.content)
            elif isinstance(m, ToolMessage) or (hasattr(m, "type") and m.type == "tool"):
                tool_messages.append(m)

        uq_lower = user_query.lower()

        # Mode: runaway (used for testing loop safety budget cap)
        if self.mode == "runaway":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "safe_query_tool",
                        "args": {"department": "Human Services", "year": "2024 May"},
                        "id": f"call_runaway_{len(tool_messages) + 1}",
                    }
                ],
            )

        # Multi-hop comparison scenario:
        # e.g., comparing Human Services and District Court
        if "compare" in uq_lower and ("human services" in uq_lower and "district court" in uq_lower):
            if len(tool_messages) == 0:
                # Hop 1: Query first department
                return AIMessage(
                    content="Querying first department: Human Services...",
                    tool_calls=[
                        {
                            "name": "safe_query_tool",
                            "args": {
                                "question": "01. I know what is expected of me at work.",
                                "department": "Human Services",
                                "year": "2024 May",
                            },
                            "id": "call_multihop_1",
                        }
                    ],
                )
            elif len(tool_messages) == 1:
                # Hop 2: Retain first result and query second department
                return AIMessage(
                    content="Context retained from Human Services. Querying second department: District Court...",
                    tool_calls=[
                        {
                            "name": "safe_query_tool",
                            "args": {
                                "question": "01. I know what is expected of me at work.",
                                "department": "District Court",
                                "year": "2024 May",
                            },
                            "id": "call_multihop_2",
                        }
                    ],
                )
            else:
                # Hop 3: Synthesize comparative answer from both tool results
                res1 = json.loads(tool_messages[0].content)
                res2 = json.loads(tool_messages[1].content)
                score1 = res1.get('mean_score') or 0.0
                score2 = res2.get('mean_score') or 0.0
                return AIMessage(
                    content=(
                        f"Comparative Synthesis for Question 01 (2024 May):\n"
                        f"- Human Services: mean score = {score1:.2f}, n = {res1.get('respondent_count')}\n"
                        f"- District Court: mean score = {score2:.2f}, n = {res2.get('respondent_count')}\n"
                        f"Comparison: Human Services scored higher in employee role clarity than District Court."
                    )
                )

        # Standard or small cohort lookup
        if len(tool_messages) == 0:
            extracted = _extract_query_params(user_query)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "safe_query_tool",
                        "args": extracted,
                        "id": "call_lookup_1",
                    }
                ],
            )
        else:
            res = json.loads(tool_messages[0].content)
            mean_score = res.get('mean_score')
            score_formatted = f"{mean_score:.2f}" if mean_score is not None else "N/A"
            privacy_note = res.get("privacy_note")
            if privacy_note:
                return AIMessage(
                    content=(
                        f"Analysis: {privacy_note}\n"
                        f"Generalized aggregate: mean = {score_formatted} "
                        f"(n={res.get('respondent_count')})."
                    )
                )
            return AIMessage(
                content=(
                    f"Survey Analysis Result:\n"
                    f"- Respondent Count: {res.get('respondent_count')}\n"
                    f"- Mean Score: {score_formatted}\n"
                    f"- Distribution: {res.get('distribution')}"
                )
            )

    def bind_tools(self, tools: Any) -> DeterministicSurveyLLM:
        return self
