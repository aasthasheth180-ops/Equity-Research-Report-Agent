# tools.py — owned by Person C
#
# Responsibilities:
#   1. ToolCall / ToolResult Pydantic models  (same pattern as Week7)
#   2. Tool implementations: retrieve_context, fetch_financials, fetch_market_risk_data
#   3. run_tool() with tenacity retries  — single entry point used by node_tool
#   4. TOOL_SCHEMAS injected into llm_engine at import time
#
# Interface the orchestrator relies on:
#   ToolCall, ToolResult
#   run_tool(name, arguments) -> ToolResult

import json
from typing import Any, Dict, Literal, Optional

import yfinance as yf
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_store import rag_store
from tools_dcf import run_dcf, get_balance_sheet_projections, get_income_statement_projections
import llm_engine


# ── Pydantic models (same pattern as Week7) ───────────────────────────────────

class ToolCall(BaseModel):
    name: Literal["retrieve_context", "fetch_financials", "fetch_market_risk_data", "run_dcf_valuation", "get_balance_sheet_projections", "get_income_statement_projections"]
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name:   str
    ok:     bool
    result: Any
    error:  Optional[str] = None


# ── Tool implementations ──────────────────────────────────────────────────────

def _retrieve_context(query: str, k: int = 5) -> str:
    chunks = rag_store.retrieve(query, k=k)
    return "\n\n---\n\n".join(chunks) if chunks else "No relevant context found."


def _fetch_financials(ticker: str) -> Dict:
    info = yf.Ticker(ticker).info
    return {
        "ticker":           ticker,
        "company_name":     info.get("longName", ""),
        "sector":           info.get("sector", ""),
        "industry":         info.get("industry", ""),
        "market_cap":       info.get("marketCap"),
        "price":            info.get("currentPrice"),
        "pe_ratio":         info.get("trailingPE"),
        "pb_ratio":         info.get("priceToBook"),
        "eps":              info.get("trailingEps"),
        "revenue":          info.get("totalRevenue"),
        "net_income":       info.get("netIncomeToCommon"),
        "roe":              info.get("returnOnEquity"),
        "roa":              info.get("returnOnAssets"),
        "debt_to_equity":   info.get("debtToEquity"),
        "dividend_yield":   info.get("dividendYield"),
        "52w_high":         info.get("fiftyTwoWeekHigh"),
        "52w_low":          info.get("fiftyTwoWeekLow"),
        "analyst_target":   info.get("targetMeanPrice"),
        "recommendation":   info.get("recommendationKey"),
        "business_summary": (info.get("longBusinessSummary") or "")[:600],
    }


def _fetch_market_risk_data(ticker: str, date: str, csv_folder: str) -> str:
    """
    Load FR Y-9C + FFIEC 102 CSVs, ingest into rag_store, return summary string.
    Side effect: the DataFrame is also indexed so retrieve_context can find it.
    """
    from data_loader import build_market_risk_df
    df = build_market_risk_df(ticker, date, csv_folder)
    rag_store.ingest_dataframe(df, description=f"{ticker} {date}")
    lines = [f"Market Risk Data for {ticker} as of {date}:"]
    for idx, row in df.iterrows():
        lines.append(f"  {idx}: {row['Amount $millions']}  (source: {row['Source']})")
    return "\n".join(lines)


_TOOLS = {
    "retrieve_context":               _retrieve_context,
    "fetch_financials":               _fetch_financials,
    "fetch_market_risk_data":         _fetch_market_risk_data,
    "run_dcf_valuation":              run_dcf,
    "get_balance_sheet_projections":  get_balance_sheet_projections,
    "get_income_statement_projections": get_income_statement_projections,
}


# ── Tool runner (tenacity retries, same pattern as Week7) ─────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def run_tool(name: str, arguments: Dict) -> ToolResult:
    if name not in _TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    result = _TOOLS[name](**arguments)
    return ToolResult(name=name, ok=True, result=result)


# ── Schema registration ───────────────────────────────────────────────────────
# Called once here at import time so llm_engine.build_system_prompt()
# always has the current schema. Import order: orchestrator imports tools,
# which triggers this line, which calls llm_engine. No circular deps.

TOOL_SCHEMAS = json.dumps({
    "retrieve_context": {
        "description": "Retrieve relevant text chunks from ingested research documents and regulatory filings",
        "arguments": {
            "query": "string — what to search for",
            "k":     "int (optional, default 5) — number of chunks to return",
        }
    },
    "fetch_financials": {
        "description": "Fetch live financial metrics from Yahoo Finance",
        "arguments": {
            "ticker": "string — e.g. 'GS'",
        }
    },
    "fetch_market_risk_data": {
        "description": "Load regulatory VaR, SvaR, RWA and qualitative risk metrics from FR Y-9C and FFIEC 102 filings",
        "arguments": {
            "ticker":     "string — e.g. 'GS'",
            "date":       "string — YYYYMMDD e.g. '20250930'",
            "csv_folder": "string — path to folder containing FRY9C_*.csv and FFIEC102_*.csv",
        }
    },
    "run_dcf_valuation": {
        "description": "Run a bank FCFE DCF model. Returns intrinsic price target, upside %, FCFE projections, and key assumptions. Use this for the valuation section.",
        "arguments": {
            "ticker": "string — e.g. 'GS'",
        }
    },
    "get_balance_sheet_projections": {
        "description": "Returns the current annual balance sheet and 3-year forward projections (total assets, loans, deposits, equity). Use for financial_performance and valuation sections.",
        "arguments": {
            "ticker": "string — e.g. 'GS'",
        }
    },
    "get_income_statement_projections": {
        "description": "Returns the current annual income statement and 3-year forward projections (revenue, expenses, net income, EPS). Use for financial_performance and investment_thesis sections.",
        "arguments": {
            "ticker": "string — e.g. 'GS'",
        }
    },
}, indent=2)

llm_engine.register_tool_schemas(TOOL_SCHEMAS)
