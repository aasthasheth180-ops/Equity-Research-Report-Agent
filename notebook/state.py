# state.py — owned by Lead
# Single source of truth for what flows through the LangGraph graph.
# Everyone imports ReportState from here. No one else edits this file.

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ReportState(TypedDict, total=False):
    # ── message history (short-term memory, same pattern as Week7) ──────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── tool plumbing (same ToolCall / ToolResult pattern as Week7) ──────────
    tool_call:   Optional[Any]   # ToolCall  (from tools.py)
    tool_result: Optional[Any]   # ToolResult (from tools.py)
    attempts:    int

    # ── report fields ────────────────────────────────────────────────────────
    ticker:          str
    sections_done:   Dict[str, str]   # section_name -> markdown content
    current_section: str              # section currently being written
    final_report:    str              # fully assembled markdown at the end
    tool_calls:      int              # tool calls made for current section (loop guard)
