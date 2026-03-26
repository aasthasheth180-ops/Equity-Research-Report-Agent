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
import re
import time
from typing import Dict, List, Optional

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

    try:
        obj = json.loads(raw)
    except Exception:
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

    config = {"configurable": {"thread_id": f"{ticker}-report-{int(time.time())}"}}
    result = app.invoke(
        {"ticker": ticker, "sections_done": {}, "tool_calls": 0},
        config=config,
    )

    total = time.time() - total_start
    rprint(f"\n[bold]Total time:[/bold] {total:.0f}s ({total/60:.1f} min)")
    return result.get("final_report", "")
