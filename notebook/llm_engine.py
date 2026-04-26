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
        max_tokens=4000,
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
        "Include: (1) a key figures table with Current Share Price (as of today's date), 12 month expected target, TTM P/E, Forward P/E, TTM P/B, Market Cap, 52-week range, dividend yield, beta. Include As-of date clearly at the top of the table. "
        "After the table, write 2-3 sentences interpreting the key figures — what the valuation multiples signal relative to peers and whether the stock looks cheap or expensive. "
        "(2) a 150-word analyst commentary paragraph with Buy/Hold/Sell rating, current price, 12-month price target and % upside — do NOT label this paragraph with a heading; "
        "(3) one sentence on the single most important risk to the thesis. "
        "State where the bank stands in the market by assets and G-SIB designation. "
        "Include primary regulatory oversights (Fed, OCC, FDIC). "
        "Never use generic phrases like 'Goldman is a leading bank'. "
        "When quoting a number always include the year/quarter it refers to.",

    "business_overview":
        "Do NOT include a section heading. Do NOT include Porter's Five Forces. "
        "Call retrieve_context three times before writing anything: "
        "(1) query='business segments revenue margin annual report', "
        "(2) query='geographic revenue breakdown regions', "
        "(3) query='competitive positioning moat products platforms'. "
        "All figures must come from the retrieved annual report context — never invent numbers. "
        "Structure as follows:\n\n"
        "SEGMENT TABLE: Produce a markdown table with columns: "
        "Segment | Revenue $B | % of Total | Margin | Key Revenue Driver. "
        "Use exact figures from the annual report. Cover every segment.\n\n"
        "SEGMENT DESCRIPTIONS: For each segment write 2-3 sentences on how it makes money, "
        "key products or platforms, and its competitive position.\n\n"
        "GEOGRAPHIC CONCENTRATION: One paragraph with revenue split by region "
        "(Americas, EMEA, Asia-Pacific) sourced from the annual report.\n\n"
        "COMPETITIVE MOAT: One paragraph assigning Wide/Narrow/None with a one-sentence "
        "justification grounded in switching costs, balance sheet scale, or franchise value. "
        "End with one analyst sentence on the overall implication for the investment thesis.",



    "industry_analysis":
        "Do NOT include a section heading. Do NOT use markdown tables or pipe characters. "
        "Write entirely in prose paragraphs. Structure as three parts:\n\n"
        "PART 1 — WHERE THE INDUSTRY STANDS TODAY: "
        "Write 2-3 paragraphs on the current state of the global banking and capital markets industry. "
        "Cover: the post-rate-hike environment and its effect on net interest margins, "
        "the IB fee wallet recovery trajectory (Dealogic data), equity and FICC trading revenue pools, "
        "AUM growth in asset and wealth management, and the competitive pressure from fintech and private credit. "
        "Every sentence must contain at least one specific number with a source cited inline.\n\n"
        "PART 2 — WHERE THE INDUSTRY IS HEADED (3-5 YEAR OUTLOOK): "
        "Write 2-3 paragraphs on structural tailwinds and headwinds. "
        "Cover: rate normalization impact on NIM, IB pipeline recovery, AI-driven cost reduction, "
        "Basel III endgame capital requirements, the rise of tokenized assets and digital infrastructure, "
        "and consolidation in wealth management. "
        "Use forward-looking language: 'we expect', 'we forecast', 'we project'. "
        "Include at least two specific CAGR forecasts with sources.\n\n"
        "PART 3 — TOTAL ADDRESSABLE MARKET BY SEGMENT: "
        "For each of the following segments, write one sentence stating the TAM in $B, "
        "the 5-year CAGR forecast, and the named source: "
        "Global IB Fee Wallet, Global Equities Trading, Global FICC Trading, "
        "Global Asset & Wealth Management AUM, Prime Brokerage, "
        "Transaction Banking, Digital Assets / Tokenized Securities.",

    "financial_performance":
        "Do NOT include a section heading. "
        "Call get_income_statement_projections first, then get_balance_sheet_projections. "
        "Each tool returns a 'markdown_table' field and an 'assumptions_table' field. "
        "COPY THESE FIELDS VERBATIM — do not reformat, do not reconstruct the table yourself. "
        "Structure the section as follows:\n\n"
        "**Projected Income Statement**\n"
        "<paste markdown_table from get_income_statement_projections verbatim>\n\n"
        "2-3 sentences on revenue drivers and margin trend.\n\n"
        "**Projected Balance Sheet**\n"
        "<paste markdown_table from get_balance_sheet_projections verbatim>\n\n"
        "2-3 sentences on balance sheet growth and capital adequacy.\n\n"
        "**Key Projection Assumptions**\n"
        "<paste assumptions_table from get_income_statement_projections verbatim>\n\n"
        "<paste assumptions_table from get_balance_sheet_projections verbatim>",


    "investment_thesis":
        "Structure as: (1) Bull case — 3 catalysts with specific triggers and upside scenario "
        "price target; (2) Bear case — 3 risks with downside scenario price target; "
        "(3) Base case — most likely outcome with 12-month price target; "
        "(4) Final recommendation: Buy/Hold/Sell with conviction level (High/Medium/Low) "
        "and one-sentence rationale. End with expected total return including dividend yield.",


    "valuation":
        "Do NOT include a section heading. Do NOT use markdown tables or pipe characters. "
        "Call run_dcf_valuation first, then fetch_financials. Write entirely in prose. "
        "Structure as four numbered paragraphs:\n\n"
        "1. FCFE DCF MODEL: State the intrinsic price target and % upside/downside. "
        "Describe the 5-year FCFE projections in prose (Year, Net Income $B, FCFE $B, PV $B — "
        "write as sentences, not a table). State terminal FCFE, terminal value, PV of terminal value, "
        "total equity value, and price per share. List key assumptions inline.\n\n"
        "2. PRICE-TO-BOOK (P/B) MULTIPLE: State current TTM P/B, sector median P/B (use 1.5x for "
        "bulge bracket peers), book value per share, and derived implied price target.\n\n"
        "3. ROE/COST OF EQUITY IMPLIED P/B (GORDON GROWTH): Show the formula P/B = (ROE - g) / "
        "(Ke - g) with the actual values substituted. State implied P/B, implied price target, "
        "and a brief sensitivity comment on how the price moves if Ke shifts +/-1%.\n\n"
        "4. BLENDED PRICE TARGET: State the equal-weighted (33% each) blended price target, "
        "% upside or downside from current price, and final Buy/Hold/Sell rating with conviction level. "
        "A valuation chart is inserted separately — do not describe one here.",

    "risks":
        "Do NOT include a section heading. "
        "List exactly 5 risks. For each risk write: bold risk name, 2-sentence description of the mechanism "
        "and how it specifically affects this company, severity (H/M/L), probability (H/M/L), and one mitigant. "
        "Focus on: regulatory capital adequacy, credit/loan loss, interest rate sensitivity, "
        "trading revenue volatility, geopolitical/macro risk. "
        "After all 5 risks, write a concluding analyst paragraph (3-4 sentences) that: "
        "(1) identifies the single most material risk to the base case price target, "
        "(2) explains under what conditions it would be triggered, "
        "(3) states how much downside it represents to the price target.",

    "references":
        "Produce a structured reference list with exactly four numbered top-level categories. "
        "Use TODAY'S DATE for all 'Accessed' dates. "
        "Format precisely as shown below — do NOT deviate from this structure:\n\n"
        "1. **Regulatory Filings:**\n"
        "   - FR Y-9C — The [Company] Group, Inc. — [Most recent quarter-end date] (Accessed [today])\n"
        "   - FFIEC 102 — The [Company] Group, Inc. — [Most recent quarter-end date] (Accessed [today])\n\n"
        "2. **Market Data Source:**\n"
        "   - Yahoo Finance — [TICKER] Stock Data (Accessed [today])\n\n"
        "3. **[Company] Annual Report:**\n"
        "   - The [Company] Group, Inc. [Year] Form 10-K (Accessed [today])\n\n"
        "4. **Peer Comparison Data Sources:**\n"
        "   - Dealogic — Global Investment Banking Fee Wallet Data (Cited [year])\n"
        "   - McKinsey Global Banking Pools — Global IB Fee Wallet Data, Global Transaction Banking Data (Cited [year])\n"
        "   - Coalition Greenwich — Global Equities Trading Revenue Pool, Global FICC Trading Revenue Pool (Cited [year])\n"
        "   - PwC Asset & Wealth Management Outlook — Global Asset & Wealth Management AUM Data (Cited [year])\n"
        "   - FDIC — U.S. Commercial & Retail Banking Data (Cited [year])\n"
        "   - Federal Reserve — U.S. Commercial & Retail Banking Data (Cited [year])\n"
        "   - Oliver Wyman — Global Prime Brokerage Revenue Pool Data (Cited [year])\n"
        "   - SWIFT — Global Transaction Banking Data (Cited [year])\n"
        "   - World Economic Forum — Digital Assets / Tokenized Securities Data (Cited [year])\n"
        "   - BIS (Bank for International Settlements) — Digital Assets / Tokenized Securities Data (Cited [year])\n\n"
        "Replace all [placeholders] with the correct values for the specific company and ticker. "
        "Bold source names using **Source Name** format. Use an em-dash (—) to separate source from description.",

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
- For business_overview: call retrieve_context 3x (segments, geography, competitive) — annual report is primary source
- For financial_performance: call get_income_statement_projections first, then get_balance_sheet_projections
- For valuation: call run_dcf_valuation first, then fetch_financials for multiples
- For risks: call fetch_market_risk_data to ground market risk metrics
- For all other sections: call retrieve_context first, then fetch_financials if needed

FORMAT RULES:
- One JSON per response. No markdown fences. No prose outside the JSON.
- Content: 150-600 words of well-structured markdown (financial_performance may be longer due to tables).
- Do NOT start content with a ## section heading — the heading is added automatically.

TABLE RULES (non-negotiable):
- Every table MUST be followed by at least 3 sentences of analyst commentary.
- Commentary must explain: (1) what the key trend is, (2) what is driving it, (3) what it means for the investment thesis.
- A table without commentary is not acceptable — it is the analyst's job to interpret numbers, not just present them.
- Never end a section with a table. Always end with a concluding analyst sentence.
"""