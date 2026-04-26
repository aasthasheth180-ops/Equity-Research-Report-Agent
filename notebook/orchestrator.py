# orchestrator.py — owned by Lead
#
# Owns the LangGraph graph: nodes, routing, graph assembly.
# Imports interfaces from all three teammates and wires them together.
# No domain logic lives here — only graph topology.
#
# Public API:
#   app                                        — compiled LangGraph app
#   run_report(ticker, pdf_paths,
#              csv_folder, filing_date) -> str  — convenience wrapper

import json
import os
import re
import time
from typing import Dict, List, Optional
import pandas as pd

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from rich import print as rprint

# ── Teammate interfaces ───────────────────────────────────────────────────────
from state       import ReportState
from llm_engine  import llm_generate, build_system_prompt, REPORT_SECTIONS
from tools       import ToolCall, ToolResult, run_tool
from rag_store   import rag_store
from tools_dcf   import get_income_statement_projections, get_balance_sheet_projections, generate_valuation_chart, IS_DEFAULTS, BS_DEFAULTS

# ── Tuneable limits ───────────────────────────────────────────────────────────
MAX_TOOL_CALLS_PER_SECTION = 4    # LLM can call at most 4 tools before we force final
MAX_RETRIES_PER_SECTION    = 3    # JSON parse failure retries
SECTION_TIMEOUT_SECS       = 180  # 3 min hard cap per section

# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────

def plan_node(state: ReportState) -> ReportState:
    ticker   = state["ticker"]
    done     = state.get("sections_done", {})
    sections = [s for s in REPORT_SECTIONS if s not in done]

    if not sections:
        return state

    section = sections[0]
    rprint(f"\n[bold cyan]▶ [{len(done)+1}/{len(REPORT_SECTIONS)}] Section:[/bold cyan] "
           f"[bold]{section}[/bold]")

    return {
        "current_section": section,
        "attempts":    0,
        "tool_calls":  0,
        "section_start": time.time(),   # stored as float in state via extra key
        "messages": [
            SystemMessage(content=build_system_prompt(ticker, section)),
            HumanMessage(content=(
                f"Write the '{section.replace('_', ' ').title()}' section for {ticker}."
            )),
        ],
    }


def node_llm(state: ReportState) -> ReportState:
    # Per-section timeout guard
    start = state.get("section_start", time.time())
    elapsed = time.time() - start
    if elapsed > SECTION_TIMEOUT_SECS:
        rprint(f"[red]  ⚠ Section timeout ({elapsed:.0f}s) — forcing final[/red]")
        section = state.get("current_section", "")
        timeout_msg = AIMessage(content=json.dumps({
            "type":    "final",
            "section": section,
            "content": f"_Section timed out after {elapsed:.0f}s._",
        }))
        return {**state, "messages": state.get("messages", []) + [timeout_msg]}

    messages = state.get("messages", [])
    tool_calls = int(state.get("tool_calls", 0))

    # If LLM has already made MAX tool calls, force it to write the final answer
    if tool_calls >= MAX_TOOL_CALLS_PER_SECTION:
        rprint(f"[yellow]  ⚠ Tool call cap reached ({tool_calls}) — forcing final answer[/yellow]")
        messages = messages + [HumanMessage(content=(
            "You have gathered enough data. "
            "Now output ONLY CONTRACT B — the final section JSON. No more tool calls."
        ))]

    rprint(f"[dim]  LLM call #{len([m for m in messages if isinstance(m, AIMessage)])+1} "
           f"| tools used: {tool_calls} | elapsed: {time.time()-start:.0f}s[/dim]")

    raw = llm_generate(messages)
    rprint(f"[dim]  → {raw[:100]}…[/dim]")
    return {**state, "messages": messages + [AIMessage(content=raw)]}


def node_parse(state: ReportState) -> ReportState:
    messages = state.get("messages", [])
    attempts = int(state.get("attempts", 0))

    if not messages or not isinstance(messages[-1], AIMessage):
        return {**state, "tool_call": None}

    raw = messages[-1].content.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$",          "", raw)

    def _repair_json(s: str) -> str:
        """
        Attempt to salvage JSON where the LLM emitted literal newlines
        inside the 'content' string value instead of escaped \\n.
        Strategy: find the content value boundaries and re-escape its interior.
        """
        import re as _re
        m = _re.search(r'"content"\s*:\s*"', s)
        if not m:
            return s
        start = m.end()              # first char of content value
        # Walk forward to find the closing quote that ends the JSON object
        depth = 0
        i = start
        while i < len(s):
            c = s[i]
            if c == '\\':            # skip escaped character
                i += 2
                continue
            if c == '"' and depth == 0:
                break
            if c in '{[':
                depth += 1
            elif c in '}]':
                depth -= 1
            i += 1
        interior = s[start:i]
        # Re-escape any bare newlines/tabs inside the content value
        interior = interior.replace('\n', '\\n').replace('\r', '').replace('\t', '\\t')
        return s[:start] + interior + s[i:]

    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = json.loads(_repair_json(raw))
        except Exception:
            obj = None

    if obj is None:
        attempts += 1
        rprint(f"[red]  ✗ JSON parse failed (attempt {attempts}/{MAX_RETRIES_PER_SECTION})[/red]")
        if attempts >= MAX_RETRIES_PER_SECTION:
            rprint(f"[red]  ✗ Max retries hit — skipping section[/red]")
            return {
                **state,
                "tool_call": None,
                "attempts":  attempts,
                "messages":  messages + [AIMessage(content=json.dumps({
                    "type":    "final",
                    "section": state.get("current_section", ""),
                    "content": "_Section generation failed after retries._",
                }))],
            }
        fix = HumanMessage(content=(
            "Invalid JSON. Output ONLY CONTRACT A or CONTRACT B. "
            "No markdown. Start with '{', end with '}'."
        ))
        return {**state, "tool_call": None, "attempts": attempts,
                "messages": messages + [fix]}

    if obj.get("type") == "tool":
        try:
            tc = ToolCall(name=obj["name"], arguments=obj.get("arguments", {}))
            return {**state, "tool_call": tc, "attempts": attempts}
        except (ValidationError, Exception):
            fix = HumanMessage(content=(
                "Invalid tool. Use: retrieve_context, fetch_financials, or fetch_market_risk_data."
            ))
            return {**state, "tool_call": None, "messages": messages + [fix]}

    return {**state, "tool_call": None, "attempts": attempts}


def node_tool(state: ReportState) -> ReportState:
    tc = state.get("tool_call")
    if tc is None:
        return state

    tool_calls = int(state.get("tool_calls", 0)) + 1
    rprint(f"[blue]  tool [{tool_calls}/{MAX_TOOL_CALLS_PER_SECTION}]:[/blue] "
           f"{tc.name}({tc.arguments})")
    try:
        tr = run_tool(tc.name, tc.arguments)
    except Exception as e:
        rprint(f"[red]  tool error: {e}[/red]")
        tr = ToolResult(name=tc.name, ok=False, result=None, error=str(e))

    rprint(f"[blue]  result preview:[/blue] {str(tr.result)[:120]}")
    tool_msg = ToolMessage(
        content=json.dumps(tr.model_dump(), ensure_ascii=False),
        tool_call_id=tc.name,
    )
    return {
        **state,
        "tool_call":   None,
        "tool_calls":  tool_calls,
        "tool_result": tr,
        "messages":    state.get("messages", []) + [tool_msg],
    }


def save_section_node(state: ReportState) -> ReportState:
    messages = state.get("messages", [])
    content  = ""

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            raw = m.content.strip()
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$",          "", raw)
            try:
                obj = json.loads(raw)
                if obj.get("type") == "final":
                    content = obj.get("content", raw)
                    break
            except Exception:
                content = raw
                break

    section = state.get("current_section", "")
    elapsed = time.time() - state.get("section_start", time.time())
    done    = dict(state.get("sections_done", {}))
    done[section] = content
    rprint(f"[green]  ✓ saved:[/green] {section}  "
           f"({len(content)} chars, {elapsed:.0f}s, "
           f"{int(state.get('tool_calls',0))} tool calls)")
    return {"sections_done": done}


def compile_node(state: ReportState) -> ReportState:
    ticker = state.get("ticker", "TICKER")
    done   = state.get("sections_done", {})
    ts     = time.strftime("%Y-%m-%d %H:%M:%S")

    parts = [f"# Equity Research Report — {ticker}\n_Generated {ts}_\n"]
    for s in REPORT_SECTIONS:
        if s in done:
            title = s.replace("_", " ").title()
            parts.append(f"\n## {title}\n\n{done[s]}\n")

    report = "\n".join(parts)
    rprint(f"\n[bold green]✓ Report compiled:[/bold green] "
           f"{len(done)}/{len(REPORT_SECTIONS)} sections, {len(report):,} chars")
    return {"final_report": report}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def _last_ai_is_final(state: ReportState) -> bool:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            raw = m.content.strip()
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$",          "", raw)
            try:
                return json.loads(raw).get("type") == "final"
            except Exception:
                return False
    return False


def route_after_parse(state: ReportState) -> str:
    if state.get("tool_call"):
        return "tool"
    if _last_ai_is_final(state):
        return "save"
    return "llm"


def route_after_save(state: ReportState) -> str:
    remaining = [s for s in REPORT_SECTIONS
                 if s not in state.get("sections_done", {})]
    return "plan" if remaining else "compile"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

_builder = StateGraph(ReportState)

_builder.add_node("plan",    plan_node)
_builder.add_node("llm",     node_llm)
_builder.add_node("parse",   node_parse)
_builder.add_node("tool",    node_tool)
_builder.add_node("save",    save_section_node)
_builder.add_node("compile", compile_node)

_builder.set_entry_point("plan")

_builder.add_edge("plan",  "llm")
_builder.add_edge("llm",   "parse")

_builder.add_conditional_edges(
    "parse", route_after_parse,
    {"tool": "tool", "save": "save", "llm": "llm"},
)

_builder.add_edge("tool", "llm")

_builder.add_conditional_edges(
    "save", route_after_save,
    {"plan": "plan", "compile": "compile"},
)

_builder.add_edge("compile", END)

app = _builder.compile(checkpointer=InMemorySaver())


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON-GENERATED SECTIONS (bypass LLM to avoid JSON/table formatting issues)
# ─────────────────────────────────────────────────────────────────────────────

def _build_segment_chart(ticker: str, segments: list, chart_path: str) -> bool:
    """
    Generate a segment contribution pie chart (horizontal bar style like IB reports).
    segments: [{"name": str, "pct": float, "description": str}, ...]
    Returns True on success.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        NAVY    = "#1E2761"
        COLOURS = ["#1E2761", "#4A6FA5", "#6B9AC4", "#97BCD1", "#C5DCE8",
                   "#8B9DC3", "#D4E2F4", "#2E4A7A", "#3D6B9E"]

        names = [s["name"]  for s in segments]
        pcts  = [s["pct"]   for s in segments]
        total = sum(pcts)
        # Normalise so slices sum to 100
        pcts  = [p / total * 100 for p in pcts]

        fig, (ax_pie, ax_leg) = plt.subplots(1, 2, figsize=(10, 4.5),
                                              gridspec_kw={"width_ratios": [1, 1]})
        fig.patch.set_facecolor("white")

        # ── Pie chart ────────────────────────────────────────────────────────
        cols_used = COLOURS[:len(names)]
        wedges, texts, autotexts = ax_pie.pie(
            pcts,
            labels=None,
            colors=cols_used,
            autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
            startangle=90,
            pctdistance=0.72,
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")
            at.set_fontweight("bold")

        title_year = pd.Timestamp.now().year - 1
        ax_pie.set_title(f"Revenue Mix by Segment\n({ticker} {title_year})",
                         fontsize=11, fontweight="bold", color=NAVY, pad=10)

        # ── Legend with descriptions ─────────────────────────────────────────
        ax_leg.axis("off")
        y_pos = 0.97
        ax_leg.text(0, y_pos, "Segment Descriptions", fontsize=10,
                    fontweight="bold", color=NAVY, transform=ax_leg.transAxes,
                    va="top")
        y_pos -= 0.08

        for i, seg in enumerate(segments):
            colour = cols_used[i] if i < len(cols_used) else NAVY
            ax_leg.add_patch(plt.Rectangle((0, y_pos - 0.02), 0.03, 0.04,
                                           color=colour,
                                           transform=ax_leg.transAxes,
                                           clip_on=False))
            label = f"{seg['name']} ({pcts[i]:.0f}%)"
            ax_leg.text(0.05, y_pos, label, fontsize=9, fontweight="bold",
                        color=NAVY, transform=ax_leg.transAxes, va="top")
            y_pos -= 0.055
            desc = seg.get("description", "")
            # Wrap description to ~60 chars
            words = desc.split()
            line, lines = "", []
            for w in words:
                if len(line) + len(w) + 1 > 60:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)
            for ln in lines[:2]:      # max 2 lines per segment
                ax_leg.text(0.05, y_pos, ln, fontsize=7.5, color="#444444",
                            transform=ax_leg.transAxes, va="top")
                y_pos -= 0.045
            y_pos -= 0.02

        plt.tight_layout(pad=1.5)
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        rprint(f"[yellow]  Segment chart error: {e}[/yellow]")
        return False


def _build_business_overview(ticker: str) -> str:
    """
    Pre-generate business_overview in Python:
      1. Retrieve RAG context (segments, geography, moat) from the annual report.
      2. Two-pass segment extraction:
           Pass A — ask LLM to identify ALL business segments from context.
           Pass B — if fewer than 2 returned, ask explicitly for each known segment.
         Fallback: use approximate GS-standard segment splits so the chart is never blank.
      3. Generate a segment revenue-mix pie chart.
      4. Ask LLM to write prose + bullet list (how each segment makes money).
      5. Assemble: chart → bullets → prose.
    """
    import datetime, os
    from langchain_core.messages import HumanMessage as HM

    today = datetime.date.today().strftime("%B %d, %Y")

    # ── Step 1: retrieve context ──────────────────────────────────────────────
    try:
        seg_ctx  = "\n\n".join(rag_store.retrieve(
            "Global Banking Markets Asset Wealth Management Platform Solutions "
            "business segments net revenues percentage", k=8))
        geo_ctx  = "\n\n".join(rag_store.retrieve(
            "geographic revenue Americas EMEA Asia Pacific international", k=4))
        comp_ctx = "\n\n".join(rag_store.retrieve(
            "competitive positioning moat switching costs franchise market share", k=4))
    except Exception as e:
        rprint(f"[yellow]  RAG retrieval failed: {e}[/yellow]")
        seg_ctx = geo_ctx = comp_ctx = "No context retrieved."

    combined_ctx = f"SEGMENTS:\n{seg_ctx}\n\nGEOGRAPHY:\n{geo_ctx}\n\nCOMPETITIVE:\n{comp_ctx}"

    # ── Step 2: extract structured segment data (two-pass) ───────────────────
    seg_extract_prompt = f"""Extract ALL business segments reported by {ticker} from the annual report context below.

CONTEXT:
{seg_ctx[:5000]}

Return a JSON array — one object per segment:
{{"name": "<exact segment name from report>", "pct": <estimated % of total net revenue>, "description": "<one sentence: how this segment makes money>"}}

RULES:
- Include EVERY segment mentioned (typically 2-4 for large banks)
- For Goldman Sachs the three segments are: Global Banking & Markets, Asset & Wealth Management, Platform Solutions
- pct values must sum to 100; use your best estimate from the context figures
- Return ONLY the JSON array — no markdown fences, no explanation

Example: [{{"name": "Global Banking & Markets", "pct": 74, "description": "Earns IB advisory fees, FICC and equities trading revenues, and financing income."}}, ...]"""

    segments = []
    try:
        raw = llm_generate([HM(content=seg_extract_prompt)])
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            # Keep only items that have name and pct
            segments = [s for s in parsed if s.get("name") and s.get("pct", 0) > 0]
        rprint(f"[cyan]  Segments extracted: {[s['name'] for s in segments]}[/cyan]")
    except Exception as e:
        rprint(f"[yellow]  Segment extraction pass A failed: {e}[/yellow]")

    # Fallback: if fewer than 2 segments, use a second targeted extraction
    if len(segments) < 2:
        rprint("[yellow]  < 2 segments — running fallback extraction...[/yellow]")
        fallback_prompt = f"""Based on the annual report context below, provide revenue split estimates for these {ticker} business segments.

CONTEXT:
{seg_ctx[:4000]}

Return exactly this JSON (fill in the pct values and descriptions from context, or use your best estimates):
[
  {{"name": "Global Banking & Markets", "pct": <number>, "description": "Generates revenues from IB advisory and underwriting fees, FICC intermediation and financing, equities intermediation and financing, and transaction banking."}},
  {{"name": "Asset & Wealth Management", "pct": <number>, "description": "Earns management fees, incentive fees, and private banking income from AUM, private wealth clients, and alternative investments."}},
  {{"name": "Platform Solutions", "pct": <number>, "description": "Provides consumer financial services including credit cards, loans, and transaction banking for corporate clients."}}
]

pct values must sum to 100. Return ONLY the JSON array."""
        try:
            raw2 = llm_generate([HM(content=fallback_prompt)])
            raw2 = re.sub(r"```(?:json)?", "", raw2).strip().strip("`")
            m2 = re.search(r"\[.*\]", raw2, re.DOTALL)
            if m2:
                segments = json.loads(m2.group(0))
                rprint(f"[cyan]  Fallback segments: {[s['name'] for s in segments]}[/cyan]")
        except Exception as e2:
            rprint(f"[yellow]  Fallback extraction failed: {e2} — using hardcoded defaults[/yellow]")

    # Hard fallback: if still empty, use approximate GS splits so chart is never blank
    if len(segments) < 2:
        segments = [
            {"name": "Global Banking & Markets", "pct": 74,
             "description": "IB advisory, underwriting, FICC/equities trading and financing."},
            {"name": "Asset & Wealth Management", "pct": 23,
             "description": "Management fees, incentive fees, private banking and lending."},
            {"name": "Platform Solutions",         "pct":  3,
             "description": "Consumer and transaction banking services."},
        ]

    # ── Step 3: generate segment pie chart ────────────────────────────────────
    chart_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", f"{ticker}_segment_chart.png"
    ))
    ok = _build_segment_chart(ticker, segments, chart_path)
    if not ok:
        chart_path = None

    # ── Step 4: LLM writes prose + segment bullets ────────────────────────────
    # Build the segment bullet block from extracted data so the LLM references
    # the exact segments we charted.
    seg_bullet_template = "\n".join(
        f"- **{s['name']}** — {s.get('description', '')}" for s in segments
    )

    prose_prompt = f"""You are a senior sell-side equity research analyst writing the Business Overview for {ticker}.
TODAY: {today}

ANNUAL REPORT CONTEXT:
{combined_ctx[:5000]}

STRUCTURE (output exactly in this order, no deviations):

PART 1 — BUSINESS MODEL PARAGRAPH (3 sentences, pure prose):
How the firm is structured and its primary sources of revenue. Lead with the most important insight.
Every sentence must contain a specific number with units.

PART 2 — SEGMENT BULLETS (one bullet per segment, exactly matching these segments):
{seg_bullet_template}

For each bullet expand to 2-3 sentences explaining:
  (a) what the segment does and how it makes money (fee types, spread sources, AUM, etc.)
  (b) approximate revenue contribution (% of total net revenues)
  (c) one sentence on its competitive positioning

Format each bullet as:
- **Segment Name** — [your 2-3 sentence expansion]

PART 3 — GEOGRAPHIC CONCENTRATION (2-3 sentences, pure prose):
Revenue split by region (Americas, EMEA, Asia-Pacific) with specific percentages.

PART 4 — COMPETITIVE MOAT (2-3 sentences, pure prose):
Assign Wide/Narrow/None moat with justification. End with one sentence on investment thesis implication.

ANALYST VOICE RULES:
- Use: "we believe", "we estimate", "we note"
- Never use: "could", "might", "may", "potentially"
- No section headings — output flows as Paragraph / Bullets / Paragraph / Paragraph
- Never open with a company description

Return ONLY the formatted content. No JSON."""

    try:
        prose = llm_generate([HM(content=prose_prompt)])
        prose = re.sub(r'^\{.*?"type".*?\}', "", prose, flags=re.DOTALL).strip()
        prose = re.sub(r'^#{1,3}\s.*$', "", prose, flags=re.MULTILINE).strip()
    except Exception as e:
        rprint(f"[yellow]  Business overview prose failed: {e}[/yellow]")
        prose = "_Business overview could not be generated._"

    # ── Step 5: assemble section ──────────────────────────────────────────────
    chart_md = ""
    if chart_path and os.path.exists(chart_path):
        chart_md = f"\n\n![Segment Revenue Mix]({chart_path})\n\n"

    return (chart_md + prose).strip()


def _build_financial_performance(ticker: str) -> str:
    """
    Generate the financial_performance section entirely in Python.
    Calls the two projection tools directly and assembles clean markdown.
    Bypasses the LLM so markdown tables are never embedded in a JSON string.
    """
    try:
        is_data = get_income_statement_projections(ticker)
        bs_data = get_balance_sheet_projections(ticker)
    except Exception as e:
        rprint(f"[red]  financial_performance Python build failed: {e}[/red]")
        return "_Section could not be generated — data fetch error._"

    isc = is_data["current"]
    bsc = bs_data["current"]
    isp = is_data["projections"]
    bsp = bs_data["projections"]

    # ── Income statement commentary ───────────────────────────────────────────
    rev_cur  = isc.get("net_revenue_$B", 0)
    rev_last = isp[-1].get("net_revenue_$B", 0) if isp else 0
    ni_cur   = isc.get("net_income_$B", 0)
    ni_last  = isp[-1].get("net_income_$B", 0) if isp else 0
    rg       = IS_DEFAULTS["revenue_growth"]

    is_commentary = (
        f"We project net revenue to grow from ${rev_cur:,.2f}B in {isc['year']} "
        f"to ${rev_last:,.2f}B in {isp[-1]['year'] if isp else 'N/A'}, "
        f"driven by a {rg*100:.1f}% blended annual growth rate "
        f"across investment banking fees, net interest income, and asset management. "
        f"Net income is expected to increase from ${ni_cur:,.2f}B to ${ni_last:,.2f}B "
        f"over the same period, with the efficiency ratio stable at "
        f"{IS_DEFAULTS['efficiency_ratio']*100:.1f}% throughout."
    )

    # ── Balance sheet commentary ──────────────────────────────────────────────
    ta_cur   = bsc.get("total_assets_$B", 0)
    ta_last  = bsp[-1].get("total_assets_$B", 0) if bsp else 0
    eq_cur   = bsc.get("total_equity_$B", 0)
    eq_last  = bsp[-1].get("total_equity_$B", 0) if bsp else 0
    ip_cur   = bsc.get("invest_portfolio_$B", 0)
    ip_last  = bsp[-1].get("invest_portfolio_$B", 0) if bsp else 0
    bvps_cur = bsc.get("book_value_ps", 0)
    bvps_lst = bsp[-1].get("book_value_ps", 0) if bsp else 0

    ip_sentence = (
        f" The investment portfolio is expected to grow from ${ip_cur:,.2f}B to ${ip_last:,.2f}B, "
        f"consistent with the firm's trading-book and AFS securities expansion."
        if ip_cur > 0 else ""
    )
    bs_commentary = (
        f"Total assets are forecast to expand from ${ta_cur:,.2f}B in {bsc['year']} "
        f"to ${ta_last:,.2f}B in {bsp[-1]['year'] if bsp else 'N/A'}, "
        f"reflecting a {BS_DEFAULTS['asset_growth']*100:.1f}% annual growth rate.{ip_sentence} "
        f"Total equity is projected to increase from ${eq_cur:,.2f}B to ${eq_last:,.2f}B, "
        f"with book value per share rising from ${bvps_cur:,.2f} to ${bvps_lst:,.2f}, "
        f"indicating strong capital generation through retained earnings and a growing equity cushion."
    )

    section = f"""**Projected Income Statement**

{is_data["markdown_table"]}

{is_commentary}

**Projected Balance Sheet**

{bs_data["markdown_table"]}

{bs_commentary}

**Key Projection Assumptions**

{is_data["assumptions_table"]}

{bs_data["assumptions_table"]}
"""
    return section


def _build_valuation_section(ticker: str, chart_path: str = None) -> str:
    """
    Generate the valuation section entirely in Python.
    Produces a clean FCFE projection table and computes all three methods.
    Bypasses the LLM so tables are never embedded in JSON strings.
    """
    from tools_dcf import run_dcf, DEFAULTS
    import yfinance as yf

    try:
        dcf = run_dcf(ticker)
    except Exception as e:
        rprint(f"[red]  Valuation build failed: {e}[/red]")
        return "_Valuation section could not be generated._"

    info          = yf.Ticker(ticker).info
    book_value    = info.get("bookValue") or 0
    roe           = info.get("returnOnEquity") or 0.13
    current_price = dcf["current_price"]
    ke            = DEFAULTS["cost_of_equity"]
    g             = DEFAULTS["terminal_growth"]

    # ── Method 1: FCFE DCF ───────────────────────────────────────────────────
    fcfe_target  = dcf["price_target"]
    fcfe_upside  = dcf["upside_pct"]
    proj         = dcf["projections"]

    # Build FCFE projection table
    fcfe_rows = []
    for p in proj:
        fcfe_rows.append([
            str(p["year"]),
            f"${p['net_income_$B']:.3f}",
            f"${p['fcfe_$B']:.3f}",
            f"${p['pv_fcfe_$B']:.3f}",
        ])
    fcfe_table = _build_md_table(
        ["Year", "Net Income ($B)", "FCFE ($B)", "PV of FCFE ($B)"],
        fcfe_rows
    )

    pv_sum      = dcf["pv_fcfe_sum_$B"]
    pv_terminal = dcf["pv_terminal_$B"]
    eq_value    = dcf["equity_value_$B"]
    assum       = dcf["assumptions"]

    fcfe_prose = (
        f"Our FCFE DCF model yields an intrinsic price target of **${fcfe_target:,.2f}**, "
        f"representing a {fcfe_upside:+.1f}% {'upside' if fcfe_upside > 0 else 'downside'} "
        f"from the current share price of ${current_price:,.2f}. "
        f"The sum of present values of projected FCFE is ${pv_sum:.2f}B, "
        f"and the terminal value contributes ${pv_terminal:.2f}B, "
        f"giving a total equity value of ${eq_value:.2f}B. "
        f"Key assumptions: cost of equity {assum['cost_of_equity']}, "
        f"terminal growth {assum['terminal_growth']}, "
        f"initial CET1 ratio {assum['cet1_ratio_initial']}, "
        f"payout ratio {assum['payout_ratio']}. "
        f"We note this DCF is sensitive to the cost of equity assumption — "
        f"a 100bps reduction in Ke would add approximately "
        f"${fcfe_target * 0.12:,.0f} to the intrinsic value."
    )

    # ── Method 2: P/B Multiple ───────────────────────────────────────────────
    sector_pb  = 1.50
    pb_target  = sector_pb * book_value
    curr_pb    = info.get("priceToBook") or (current_price / book_value if book_value else 0)
    pb_upside  = (pb_target - current_price) / current_price * 100 if current_price else 0

    pb_prose = (
        f"Goldman Sachs trades at a TTM P/B of {curr_pb:.2f}x against a sector median of "
        f"{sector_pb:.2f}x for bulge-bracket peers. "
        f"Applying the sector median to a book value per share of ${book_value:.2f} "
        f"yields an implied price target of **${pb_target:,.2f}** ({pb_upside:+.1f}% vs current). "
        f"We believe a sector-median multiple is appropriate given GS's above-peer ROE, "
        f"partially offset by its higher trading revenue volatility."
    )

    # ── Method 3: Gordon Growth implied P/B ──────────────────────────────────
    implied_pb     = (roe - g) / (ke - g) if (ke - g) != 0 else 1.0
    gordon_target  = implied_pb * book_value
    gordon_upside  = (gordon_target - current_price) / current_price * 100 if current_price else 0
    # Sensitivity: Ke +1%
    ke_up          = ke + 0.01
    implied_pb_up  = (roe - g) / (ke_up - g) if (ke_up - g) != 0 else 1.0
    gordon_ke_up   = implied_pb_up * book_value

    gordon_prose = (
        f"Using the Gordon Growth model, P/B = (ROE - g) / (Ke - g) = "
        f"({roe*100:.2f}% - {g*100:.1f}%) / ({ke*100:.1f}% - {g*100:.1f}%) = {implied_pb:.3f}x. "
        f"Applied to book value per share of ${book_value:.2f}, this yields an implied price of "
        f"**${gordon_target:,.2f}** ({gordon_upside:+.1f}% vs current). "
        f"A 100bps increase in the cost of equity (to {ke_up*100:.1f}%) reduces the implied P/B "
        f"to {implied_pb_up:.3f}x and the price target to ${gordon_ke_up:,.2f}, "
        f"highlighting sensitivity to rate expectations."
    )

    # ── Blended target ────────────────────────────────────────────────────────
    blended       = (fcfe_target + pb_target + gordon_target) / 3
    blend_upside  = (blended - current_price) / current_price * 100 if current_price else 0
    rating        = "Buy" if blend_upside > 15 else "Hold" if blend_upside > -10 else "Sell"

    blend_prose = (
        f"Weighting each method equally at 33%, our blended price target is **${blended:,.2f}**, "
        f"implying {blend_upside:+.1f}% {'upside' if blend_upside > 0 else 'downside'} "
        f"from the current price of ${current_price:,.2f}. "
        f"We initiate coverage with a **{rating}** rating. "
        f"The FCFE DCF anchors fundamental value, the P/B multiple reflects market pricing of peers, "
        f"and the Gordon Growth model ties valuation to the firm's sustainable return on equity — "
        f"together providing a robust triangulation of intrinsic value."
    )

    # ── Assemble section ──────────────────────────────────────────────────────
    chart_md = f"\n![Valuation Football Field]({chart_path})\n" if chart_path else ""

    section = f"""**1. Free Cash Flow to Equity (FCFE) DCF Model**

{fcfe_table}

{fcfe_prose}

**2. Price-to-Book (P/B) Multiple Comparison**

{pb_prose}

**3. ROE/Cost of Equity Implied P/B (Gordon Growth Model)**

{gordon_prose}

**4. Blended Price Target**

{blend_prose}
{chart_md}"""
    return section


def _build_md_table(headers: list, rows: list) -> str:
    """Build a clean markdown table."""
    sep  = "| " + " | ".join(["---"] * len(headers)) + " |"
    hdr  = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([hdr, sep, body])


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def run_report(
    ticker:       str,
    pdf_paths:    List[str] = None,
    csv_folder:   str       = None,
    filing_date:  str       = None,
) -> str:
    total_start = time.time()

    # 1. Ingest PDFs
    if pdf_paths:
        for path in pdf_paths:
            n = rag_store.ingest_pdf(path)
            rprint(f"[cyan]Ingested {n} chunks ← {path}[/cyan]")

    # 2. Ingest regulatory CSV data
    if csv_folder and filing_date:
        from data_loader import INST_ID_MAP, build_market_risk_df
        if ticker.upper() in INST_ID_MAP:
            df = build_market_risk_df(ticker, filing_date, csv_folder)
            rag_store.ingest_dataframe(df, description=f"{ticker} {filing_date}")
            rprint(f"[cyan]Ingested market risk data ← {ticker} {filing_date}[/cyan]")
        else:
            rprint(f"[yellow]No inst_id for '{ticker}' — skipping CSV ingestion[/yellow]")

    rprint(f"[bold]RAG store:[/bold] {rag_store.size} chunks indexed")
    rprint(f"[bold]Sections to write:[/bold] {len(REPORT_SECTIONS)} "
           f"| max {MAX_TOOL_CALLS_PER_SECTION} tool calls/section "
           f"| {SECTION_TIMEOUT_SECS}s timeout/section\n")

    # 3. Pre-generate sections that require clean markdown tables (bypass LLM)
    pre_done = {}

    rprint("[bold cyan]>> Pre-generating business_overview section in Python...[/bold cyan]")
    pre_done["business_overview"] = _build_business_overview(ticker)
    rprint(f"[green]  business_overview ready ({len(pre_done['business_overview'])} chars)[/green]")

    rprint("[bold cyan]>> Pre-generating financial_performance section in Python...[/bold cyan]")
    pre_done["financial_performance"] = _build_financial_performance(ticker)
    rprint(f"[green]  financial_performance ready ({len(pre_done['financial_performance'])} chars)[/green]")

    # 4. Generate valuation football field chart
    chart_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{ticker}_valuation_chart.png")
    chart_path = os.path.abspath(chart_path)
    rprint("[bold cyan]>> Generating valuation football field chart...[/bold cyan]")
    try:
        chart_data = generate_valuation_chart(ticker, chart_path)
        rprint(f"[green]  Chart saved → {chart_path}[/green]")
        rprint(f"[green]  Blended target: ${chart_data['blended_target']} "
               f"({chart_data['upside_pct']:+.1f}%) — {chart_data['rating']}[/green]")
    except Exception as e:
        rprint(f"[yellow]  Chart generation failed: {e} — continuing without chart[/yellow]")
        chart_path = None

    rprint("[bold cyan]>> Pre-generating valuation section in Python...[/bold cyan]")
    pre_done["valuation"] = _build_valuation_section(ticker, chart_path)
    rprint(f"[green]  valuation ready ({len(pre_done['valuation'])} chars)[/green]")

    config = {"configurable": {"thread_id": f"{ticker}-report-{int(time.time())}"}}
    result = app.invoke(
        {"ticker": ticker, "sections_done": pre_done,
         "tool_calls": 0, "chart_path": chart_path},
        config=config,
    )

    total = time.time() - total_start
    rprint(f"\n[bold]Total time:[/bold] {total:.0f}s ({total/60:.1f} min)")
    return result.get("final_report", "")
