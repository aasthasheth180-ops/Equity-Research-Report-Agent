# Autonomous Equity Research Agent: Architecture and Methodology

**Course:** Large Language Models — Spring 2026
**Author:** Manoj Kumar Matala, Satish
**Date:** April 2026

---

## Abstract

This report documents an autonomous equity research agent capable of generating
institutional-quality sell-side research reports for U.S. financial firms.
The system combines a LangGraph-based agentic orchestrator, a retrieval-augmented
generation (RAG) pipeline, a bank-specific FCFE discounted cash flow model, and
live financial data from Yahoo Finance and regulatory filings.  A single call to
`run_report(ticker)` produces a structured eight-section markdown report — including
projected income statements, balance sheet forecasts, and a valuation football-field
chart — in under five minutes.

---

## 1. Introduction

Sell-side equity research is labour-intensive: a single initiating coverage report
takes an analyst team two to three weeks.  The work is highly structured — the same
eight sections appear in virtually every bulge-bracket report — yet requires synthesis
of heterogeneous sources: regulatory filings, annual reports, earnings transcripts,
and live market prices.

Naive LLM prompting fails for financial research for three reasons: (1) models
hallucinate numbers, (2) markdown tables embedded in JSON strings corrupt parsing, and
(3) bank valuation (FCFE-based DCF) requires precise multi-step computation.

The agent resolves all three: tools ground every figure in real data; Python
pre-generates all table-containing sections so they never pass through a JSON string;
and a purpose-built bank DCF engine produces analytically correct valuations — zero
invented numbers, sell-side structural standards.

---

## 2. System Architecture

The agent is implemented as a **LangGraph `StateGraph`** — a directed graph where each
node is a Python function and edges are conditional routing functions.  State is a
typed dictionary that flows through the graph unchanged except by explicit node updates.

### 2.1 State Schema

| Field            | Type          | Purpose                                        |
|------------------|---------------|------------------------------------------------|
| `ticker`         | str           | Company symbol (e.g. "GS")                     |
| `section`        | str           | Report section currently being written         |
| `messages`       | List[Message] | Conversation history passed to the LLM         |
| `sections_done`  | Dict[str,str] | Completed sections (markdown string per key)   |
| `tool_calls`     | int           | Running count; caps at 12 to prevent loops     |
| `chart_path`     | str           | File path of the generated valuation chart     |

### 2.2 Graph Nodes

| Node       | Responsibility                                                      |
|------------|---------------------------------------------------------------------|
| `plan`     | Initialise state; inject system prompt for the current section      |
| `llm`      | Call Gemini 2.5 Flash via OpenRouter; return raw assistant text     |
| `parse`    | Extract CONTRACT A (tool call) or CONTRACT B (final content) JSON  |
| `tool`     | Execute the requested tool; append result to message history        |
| `save`     | Write completed section to `sections_done`; advance to next section |
| `compile`  | Assemble all sections into one markdown document; write to disk     |

Routing is conditional: after `parse`, the graph goes to `tool` if the LLM
requested data, or to `save` if it returned a final section.  After `save`, it
loops back to `plan` for the next section, or proceeds to `compile` when all eight
sections are complete.

### 2.3 Pre-generation Pattern

Sections containing markdown tables (`financial_performance`, `valuation`) are
generated entirely in Python before the graph is invoked.  Python functions return
clean multi-line markdown placed directly into `sections_done`; the LLM never touches
them.  The remaining six sections are written by the LLM using a structured JSON
output contract (CONTRACT A = tool call, CONTRACT B = final markdown).

---

## 3. RAG Pipeline and Data Sources

### 3.1 Document Ingestion

The `RagStore` class wraps a FAISS flat-index vector store with a
`sentence-transformers` encoder (`all-MiniLM-L6-v2`, 384 dimensions).  Documents
are ingested at startup from three sources:

- **Annual Report (PDF):** Chunked at 200-word boundaries with 20-word overlap.
  The 10-K / annual report is the primary source for business segment descriptions,
  geographic revenue breakdowns, and competitive positioning.
- **Earnings Call Transcripts (PDF):** Management commentary on guidance, margin
  trends, and capital allocation.  Same chunking strategy as the annual report.
- **Regulatory Filings (CSV):** FR Y-9C (consolidated holding company) and FFIEC 102
  (market risk) data loaded via `build_market_risk_df()` and ingested as structured
  rows, giving the RAG store access to VaR, Stressed VaR, and RWA figures.

### 3.2 Retrieval

Each `retrieve_context(query, k)` call encodes the query, performs a FAISS
`search` for the top-*k* cosine-nearest chunks, and returns them concatenated with
`---` separators.  The LLM is instructed to call this tool three times for the
`business_overview` section (queries for segments, geography, and competitive moat)
before writing any text, ensuring all figures trace back to the annual report.

### 3.3 Live Market Data

`fetch_financials(ticker)` pulls current price, P/E, P/B, ROE, dividend yield,
52-week range, and analyst consensus from Yahoo Finance via `yfinance`.  The field
lookup is intentionally defensive — each metric is tried under five to seven
alternative key names (yfinance field names vary across versions) with a final
fallback to the `.info` dictionary.

---

## 4. Financial Modelling Engine

### 4.1 Bank FCFE DCF

Banks cannot be valued with a standard unlevered DCF because interest expense is a
revenue item, not a financing cost.  The correct approach discounts **Free Cash Flow
to Equity (FCFE)**, where regulatory capital reinvestment replaces capital expenditure:

> **FCFE = Net Income − ΔCET1 Capital**

CET1 capital is the Common Equity Tier 1 buffer required by Basel III:

> **CET1 Capital = Total Assets × RWA% × CET1 Ratio**

The model projects five years of FCFE, discounts at the cost of equity (Ke = 14%,
derived from CAPM), and adds a terminal value via the Gordon Growth Model
(terminal growth g = 1%).  All base financials — total assets, book equity, net
income, shares outstanding — are sourced live from Yahoo Finance at run time.

### 4.2 Balance Sheet Projections

`get_balance_sheet_projections()` fetches the most recent annual balance sheet and
produces a three-year forward view structured in three sections: **Assets** (Total
Assets, Net Loans, Investment Portfolio), **Liabilities** (Total Deposits, Long-Term
Debt, Total Liabilities), and **Equity** (Total Equity, Book Value per Share).

Key growth assumptions: total assets at 2.5% CAGR; net loans and investment portfolio
at 4.0%; deposits at 3.5%; long-term debt at 1.5% (conservative for IB-heavy firms
that are not deposit-funded).  Equity grows via retained earnings: prior equity plus
net income multiplied by the retention ratio (1 − payout ratio).

### 4.3 Income Statement Projections

`get_income_statement_projections()` projects net revenue, non-interest expense,
provision for credit losses, pre-tax income, tax, net income, and EPS over three years.
Revenue grows at 5.0% per year (blended IB fees, net interest income, asset management).
The efficiency ratio (non-interest expense / net revenue) is held at 65%.  Provisions
are modelled as 0.30% of projected net loans — appropriate for a benign credit environment.

### 4.4 Blended Valuation

Three price targets are computed and equally weighted (33⅓% each):

1. **FCFE DCF** — intrinsic value from five-year free cash flow model.
2. **P/B Multiple** — sector median P/B of 1.5× applied to current book value per share.
3. **Gordon Growth Implied P/B** — `P/B = (ROE − g) / (Ke − g)` with current ROE, Ke = 14%, g = 1%.

The blended target is displayed in a **football-field chart** (horizontal bar chart)
showing sensitivity ranges for each method alongside the current price and 52-week range.
Sensitivity is analytically derived: DCF and Gordon Growth ranges are computed at Ke ± 1%;
the P/B range spans 1.3× – 1.7× sector median.

---

## 5. Results and Conclusion

### 5.1 Report Output

A completed report contains eight sections totalling approximately 3,000–4,500 words:
executive summary (with key figures table), business overview (segment table and
geographic breakdown), industry analysis (TAM by segment, 3–5 year outlook), financial
performance (projected IS and BS with assumptions), investment thesis (bull/base/bear
cases), valuation (all three methods in prose), risks (five rated risks with mitigants),
and references.  Total wall-clock time for a Goldman Sachs report is approximately
3–5 minutes on a standard laptop.

### 5.2 Limitations and Future Work

The primary limitation is data availability: Goldman Sachs is an investment bank, so
deposit and net-loan line items are near-zero in Yahoo Finance's schema; the investment
portfolio is proxied from residual assets.  Second, `industry_analysis` cites
forward-looking forecasts that cannot be cross-checked against retrieved context.
Third, the cost of equity is fixed at 14%; a live CAPM estimate (current beta,
real risk-free rate) would improve precision.

Planned enhancements: (1) peer comparison table via `fetch_financials` on five sector
peers; (2) live CAPM module pulling the 10-year Treasury yield at run time;
(3) citation tracking that tags each LLM sentence with the RAG chunk it drew from.

### 5.3 Conclusion

The agent demonstrates that a structured agentic pipeline — LLMs handle prose
synthesis; Python handles all numerical computation — can reliably produce
institutional-quality financial research.  The key architectural insight: tool
pre-generation and JSON-safe section routing eliminate the two failure modes that
plague naive LLM document generation — hallucinated numbers and corrupted table
formatting — producing a zero-hallucination, sell-side-standard research report.

---

*Generated by the Autonomous Equity Research Agent — Spring 2026 LLM Project*
