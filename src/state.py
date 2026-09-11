"""State definitions for the LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """The central state of the LangGraph workflow.
    
    Attributes:
        messages: The sequence of chat messages, appended deterministically via operator.add.
        step_count: Current count of tool execution iterations (budget cap enforcement).
        metadata_log: Audit trail tracking privacy blocks, dynamic fallbacks, tool calls, and retries.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    step_count: int
    metadata_log: Annotated[list[dict[str, Any]], operator.add]


class PrivacyBlockRecord(TypedDict, total=False):
    """Log entry when a k-anonymity privacy boundary is intercepted."""
    type: str  # "privacy_block"
    reason: str
    original_filters: dict[str, str | None]
    respondent_count: int
    fallback_action: str
    fallback_filters: dict[str, str | None]


class ToolExecutionRecord(TypedDict, total=False):
    """Log entry for an executed query tool call."""
    type: str  # "tool_execution"
    tool_name: str
    arguments: dict[str, Any]
    respondent_count: int
    mean_score: float | None
    filters_applied: dict[str, str | None]
    privacy_fallback_triggered: bool
