"""Dedicated router node implementing hard refusal boundaries for individual targeting, raw rows, and out-of-scope queries."""

from __future__ import annotations

import re
from typing import Any, Literal
from langchain_core.messages import AIMessage

from src.state import AgentState

# Refusal categories
REFUSAL_INDIVIDUAL = (
    "Refusal: The query targets an individual employee or attempts to identify specific person-level responses. "
    "To protect employee privacy and maintain absolute confidentiality, individual identity queries are strictly prohibited."
)

REFUSAL_RAW_DATA = (
    "Refusal: The query requests raw survey rows, microdata, or individual database records. "
    "This system operates strictly on aggregate statistics; raw data extraction is prohibited."
)

REFUSAL_OUT_OF_SCOPE = (
    "Refusal: The requested information is out of scope. The dataset contains employee engagement survey responses "
    "(satisfaction, feedback, and workplace culture statements across Pierce County departments from 2019 to 2024), "
    "and does not contain salary, compensation, personal records, external facts, or unrelated operational data."
)

# Regex heuristics for out-of-scope queries
OUT_OF_SCOPE_PATTERNS = [
    r"\b(salary|salaries|compensation|pay\s+rate|wage|bonus|benefits|health\s+insurance)\b",
    r"\b(budget|financial\s+statement|revenue|expenditure|tax\s+revenue)\b",
    r"\b(weather|president|capital\s+of|write\s+a\s+poem|solve\s+math|python|script|recipe)\b",
    r"\b(hiring\s+date|termination|fired|social\s+security|ssn|home\s+address|phone\s+number)\b",
]

# Regex heuristics for raw rows / microdata extraction
RAW_DATA_PATTERNS = [
    r"\braw\s+(\w+\s+)?(rows?|records?|data|table|csv|export)\b",
    r"\bunaggregated(\s+(\w+\s+)?(data|records?|rows?))?\b",
    r"\bmicrodata\b",
    r"\b(export|dump|download|extract)\s+.*(rows?|dataset|database|records?|data|csv|table)\b",
    r"\b(show|display|give)\s+me\s+(row\s+\d+|the\s+first\s+\d+\s+rows|unaggregated\s+data|microdata)\b",
    r"\bevery\s+single\s+row\b",
    r"\bunfiltered\s+individual\s+records\b",
]

# Regex heuristics for individual targeting
INDIVIDUAL_PATTERNS = [
    r"\bwho\s+(is|was|gave|answered|said|scored|rated)\b",
    r"\b(who is|who said|who gave|who answered|tell me about)\s+(the\s+)?(person|employee|individual|name|worker|boss)\b",
    r"\b(find|identify|track|reveal|locate)\s+(the\s+)?(individual|person|respondent|employee|worker|who)\b",
    r"\b(what did\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s+say)\b",
    r"\b(bob|alice|john|jane|sarah|mike|david|smith|doe)\b",
    r"\bshow\s+me\s+(the\s+)?specific\s+person('s)?\b",
    r"\bwhich\s+specific\s+(person|employee|worker|respondent)\b",
    r"\bidentify\s+who\b",
]


def classify_query(query: str) -> tuple[Literal["answer", "refuse"], str | None, str | None]:
    """Classifies incoming user query into 'answer' or 'refuse'.
    
    Checks out-of-scope and raw-data boundaries before individual targeting.
    
    Returns:
        (decision, refusal_reason_text, refusal_category)
    """
    q = query.strip().lower()

    # 1. Check out of scope boundary
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "refuse", REFUSAL_OUT_OF_SCOPE, "out_of_scope"

    # 2. Check raw data boundary
    for pat in RAW_DATA_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "refuse", REFUSAL_RAW_DATA, "raw_data_request"

    # 3. Check individual targeting boundary
    for pat in INDIVIDUAL_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "refuse", REFUSAL_INDIVIDUAL, "individual_targeting"

    return "answer", None, None


def router_node(state: AgentState) -> dict[str, Any]:
    """Dedicated router node analyzing the incoming user query.
    
    If the query violates privacy boundaries (targeting individuals, asking for raw records,
    or out-of-scope information), returns a hard refusal immediately and halts execution.
    """
    messages = state.get("messages", [])
    if not messages:
        refusal = "Refusal: Empty query received."
        return {
            "messages": [AIMessage(content=refusal)],
            "metadata_log": [{"type": "router_refusal", "reason": refusal, "category": "empty_query"}],
        }

    last_user_message = ""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human" or type(m).__name__ in ("HumanMessage", "ChatMessage"):
            last_user_message = m.content if isinstance(m.content, str) else str(m.content)
            break

    if not last_user_message:
        last_user_message = messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)

    decision, refusal_text, category = classify_query(last_user_message)

    if decision == "refuse":
        refusal_msg = refusal_text or "Refusal: The request violates privacy or scope constraints."
        return {
            "messages": [AIMessage(content=refusal_msg)],
            "metadata_log": [
                {
                    "type": "router_refusal",
                    "reason": refusal_msg,
                    "category": category,
                    "query": last_user_message,
                }
            ],
        }

    # Safe to proceed to agent node
    return {
        "metadata_log": [
            {
                "type": "router_approval",
                "category": "aggregate_query",
                "query": last_user_message,
            }
        ]
    }


def route_decision(state: AgentState) -> Literal["agent_node", "end"]:
    """Conditional routing function after router_node."""
    metadata = state.get("metadata_log", [])
    for entry in reversed(metadata):
        if entry.get("type") == "router_refusal":
            return "end"
        if entry.get("type") == "router_approval":
            return "agent_node"
    return "agent_node"
