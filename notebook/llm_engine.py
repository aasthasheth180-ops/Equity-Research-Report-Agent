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
#MODEL_ID           = "openai/gpt-4o"
MODEL_ID            ="google/gemini-2.5-flash"


_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180.0,
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
    _client.api_key = os.getenv("OPENROUTER_API_KEY", "")
    response = _client.chat.completions.create(
        model=MODEL_ID,
        messages=_format_chat(messages),
        temperature=0.1,
        max_tokens=1500,
        timeout=180,
    )
    return (response.choices[0].message.content or "").strip()


# ── Report section configuration ─────────────────────────────────────────────

REPORT_SECTIONS = [
    "executive_summary",
    "business_overview",
    "industry_analysis",
    "financial_performance",
    "investment_thesis",      # lead with the call — sell-side standard
    "valuation",
    "risks",
    "references"
]


SECTION_PROMPTS = {
    "executive_summary":
        "Write a professional executive summary. Do NOT include a section heading. "
        "Include: (1) a key figures table with Current Share Price (as of today's date), TTM P/E, Forward P/E, Market Cap, 52-week range, dividend yield, beta; "
        "(2) a 150-word summary paragraph with Buy/Hold/Sell rating, current price, 12-month price target and % upside — "
        "do NOT label this paragraph with a heading like 'Investment Thesis'; "
        "(3) one sentence on the core risk. "
        "State where the bank stands in the market by assets and G-SIB designation. "
        "Include primary regulatory oversights (Fed, OCC, FDIC). "
        "Never use generic phrases like 'Goldman is a leading bank'. "
        "When quoting a number always include the year/quarter it refers to.",

    "business_overview":
        "Do NOT include a section heading. Do NOT include Porter's Five Forces. "
        "Produce a markdown table with five columns: Segment | Revenue $B | % of Total | "
        "Margin | Key Revenue Driver. Cover all segments from the most recent Annual Report. "
        "Follow with one sentence per segment explaining how it makes money. "
        "Cover: primary revenue lines with margins, key products/platforms, geographic "
        "concentration, competitive moat rating (Wide/Narrow/None) with one-line justification.",


    "industry_analysis":
        "Cover: total addressable market size ($B) and CAGR, top 3-5 competitors with "
        "market share and key differentiators, regulatory environment (Basel III, CCAR "
        "for banks), macro tailwinds and headwinds. Include a competitive positioning "
        "summary table with key metrics vs peers.",

    "financial_performance":
        "Analyze the last 2 years of financials. Cover: revenue growth YoY with drivers, "
        "gross/EBIT/net margin trends, EPS growth, FCF margin, key balance sheet metrics "
        "(debt/equity, current ratio, cash position). Compare margins to sector peers. "
        "Flag any deteriorating trends explicitly.",


    "investment_thesis":
        "Structure as: (1) Bull case — 3 catalysts with specific triggers and upside scenario "
        "price target; (2) Bear case — 3 risks with downside scenario price target; "
        "(3) Base case — most likely outcome with 12-month price target; "
        "(4) Final recommendation: Buy/Hold/Sell with conviction level (High/Medium/Low) "
        "and one-sentence rationale. End with expected total return including dividend yield.",


    "valuation":
        "Produce a blended valuation with three methods weighted equally (33% each): "
        "(1) FCFE DCF — call run_dcf tool, show 5-year projection table, terminal value, "
        "intrinsic price per share; "
        "(2) P/B multiple — compare to sector median P/B, derive implied price; "
        "(3) ROE/Cost of equity implied P/B — use Gordon Growth model. "
        "Show sensitivity table (cost of equity vs terminal growth). "
        "State final blended price target and % upside/downside clearly.",

    "risks":
        "List exactly 5 risks with: risk name, 2-sentence description, severity (H/M/L), "
        "probability (H/M/L), and one mitigant. Focus on banking-specific risks: regulatory capital, "
        "credit/loan loss risk, interest rate risk, trading revenue volatility,geopolitical/macro risk, "
        "operational/technology risk.",

    "references":
        "List all data sources used in this report. Format as a numbered markdown list. "
        "Include: (1) regulatory filings — FR Y-9C and FFIEC 102 with filing date; "
        "(2) market data source — Yahoo Finance with access date; "
        "(3) Goldman Sachs Annual Report — state the year; "
        "(4) any peer comparison data sources. "
        "Format each entry as: [Number]. Source Name — Description — Date accessed.",

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
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")
    task = SECTION_PROMPTS.get(section, "Write this section professionally.")
    return f"""You are a senior sell-side equity research analyst at a bulge-bracket firm writing a report on {ticker}.
TODAY'S DATE: {today} — use this as the "as of" date for all current market prices and data.

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
- Do NOT start content with a ## section heading — the heading is added automatically.
"""
