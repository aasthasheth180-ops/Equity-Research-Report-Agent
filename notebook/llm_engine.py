# llm_engine.py — owned by Person B
#
# Responsibilities:
#   1. OpenRouter / GPT-4o client  (same setup as Week7)
#   2. llm_generate()              called by node_llm in orchestrator
#   3. REPORT_SECTIONS + SECTION_PROMPTS  — report structure config
#   4. build_system_prompt()       injects section task + tool schemas
#
# Interface the orchestrator relies on:
#   llm_generate(messages)            -> str
#   build_system_prompt(ticker, section) -> str
#   REPORT_SECTIONS                   -> List[str]

import json
import os
from typing import Dict, List

from openai import OpenAI
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)

# ── Client ───────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID           = "openai/gpt-4o"

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://colab.research.google.com",
        "X-Title":      "Equity Research RAG",
    },
)


# ── Message formatting (verbatim from Week7) ─────────────────────────────────

def _format_chat(messages: List[BaseMessage]) -> List[Dict]:
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system",    "content": str(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user",      "content": str(m.content)})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": str(m.content)})
        elif isinstance(m, ToolMessage):
            out.append({"role": "user",      "content": f"[tool_output]\n{m.content}"})
        else:
            out.append({"role": "user",      "content": str(m.content)})
    return out


# ── LLM call ─────────────────────────────────────────────────────────────────

def llm_generate(messages: List[BaseMessage]) -> str:
    """Single entry point for all LLM calls. Returns raw assistant text."""
    response = _client.chat.completions.create(
        model=MODEL_ID,
        messages=_format_chat(messages),
        temperature=0.1,
        max_tokens=1500,
    )
    return (response.choices[0].message.content or "").strip()


# ── Report section configuration ─────────────────────────────────────────────

REPORT_SECTIONS: List[str] = [
    "investment_thesis",      # lead with the call — sell-side standard
    "executive_summary",
    "business_overview",
    "financial_performance",
    "valuation",
    "industry_analysis",
    "risks",
]

SECTION_PROMPTS: Dict[str, str] = {
    "investment_thesis": (
        "Open with a one-line verdict: '[Rating] | PT $X | X% upside | 12-month horizon'. "
        "Then write three labeled subsections: ## Bull Case (3 specific catalysts with metrics), "
        "## Bear Case (3 specific risks with triggers), ## Catalysts (named events + expected timing). "
        "Use 'we expect', 'we forecast', 'we believe'. Never use 'could', 'might', 'may'. "
        "Every bullet must contain at least one number."
    ),
    "executive_summary": (
        "Do NOT open with a company description. Lead with the rating and price target. "
        "Cover: (1) recommendation + PT + upside, (2) the single most important financial metric "
        "that supports the thesis, (3) the key near-term catalyst. 100 words max."
    ),
    "business_overview": (
        "Focus on what is CHANGING, not what is stable. Describe the three revenue segments "
        "with their approximate revenue contribution. Identify the one segment driving incremental "
        "growth. Mention one structural competitive advantage that peers cannot easily replicate."
    ),
    "financial_performance": (
        "Lead with the most recent annual revenue and YoY growth rate. Then cover: "
        "net income margin trend, EPS trajectory, ROE vs cost of equity, and balance sheet leverage. "
        "Include at least 4 specific numbers. End with one forward-looking sentence on what "
        "we expect for the next fiscal year."
    ),
    "valuation": (
        "Call run_dcf_valuation first to get the FCFE-based price target. "
        "Then call fetch_financials for market multiples. "
        "Structure: (1) DCF intrinsic value with key assumptions (WACC/cost of equity, terminal growth), "
        "(2) P/E and P/B vs JPM and MS, (3) reconcile to final 12-month PT. "
        "State the upside/downside explicitly in percentage terms."
    ),
    "industry_analysis": (
        "State the industry's current cycle position (early/mid/late expansion or contraction). "
        "Name the top 3 competitors with one differentiating metric each. "
        "Identify the single biggest regulatory or macro tailwind/headwind for the next 12 months."
    ),
    "risks": (
        "List exactly 5 risks. For each: name, H/M/L severity, the specific metric or event "
        "that would signal the risk is materializing, and the estimated EPS/PT impact if it does. "
        "Use a markdown table: | Risk | Severity | Trigger | PT Impact |"
    ),
}

# ── Tool schema registry ──────────────────────────────────────────────────────
# tools.py calls register_tool_schemas() once at import time.
# Keeps the import direction one-way: llm_engine never imports tools.

_tool_schemas: str = "{}"

def register_tool_schemas(schemas_json: str) -> None:
    global _tool_schemas
    _tool_schemas = schemas_json


# ── System prompt builder ─────────────────────────────────────────────────────

def build_system_prompt(ticker: str, section: str) -> str:
    task = SECTION_PROMPTS.get(section, "Write this section professionally.")
    return f"""You are a senior sell-side equity research analyst at a bulge-bracket firm writing a report on {ticker}.

CURRENT SECTION: {section.replace('_', ' ').title()}
TASK: {task}

TOOLS AVAILABLE:
{_tool_schemas}

OUTPUT CONTRACTS:

CONTRACT A – tool call (when you need data):
{{"type":"tool","name":"<tool_name>","arguments":{{...}}}}

CONTRACT B – final section content (after you have sufficient data):
{{"type":"final","section":"{section}","content":"<markdown text>"}}

ANALYST VOICE RULES (non-negotiable):
- Use: "we expect", "we forecast", "we believe", "we rate", "we see"
- Never use: "could", "might", "may", "potentially", "it appears", "it is worth noting"
- Never open with a company description — the reader knows the company
- Every paragraph must contain at least one specific number with units
- Lead with the most important insight, not with background or context
- Do not use filler: "overall", "it should be noted", "importantly", "needless to say"

DATA RULES:
- Never invent numbers — all figures must come from tool results
- For valuation: call run_dcf_valuation first, then fetch_financials for multiples
- For risks: call fetch_market_risk_data to ground market risk metrics
- For all other sections: call retrieve_context first, then fetch_financials if needed

FORMAT RULES:
- One JSON per response. No markdown fences. No prose outside the JSON.
- Content: 150-300 words of well-structured markdown.
"""
