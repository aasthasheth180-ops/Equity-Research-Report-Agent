# tools_dcf.py — owned by Person C
#
# Bank financial projection models.
# Methodology from Valuation-FCFE sheet, Financial_Model_Banking_MK.xlsx
#
# Core mechanic (DCF):
#   Banks treat regulatory capital as working capital reinvestment.
#   FCFE = Net Income - Change in CET1 Capital
#   CET1 Capital = Total Assets x RWA% x CET1 Ratio
#
# Interface tools.py relies on:
#   run_dcf(ticker, **optional_overrides)              -> Dict
#   get_balance_sheet_projections(ticker, ...)         -> Dict
#   get_income_statement_projections(ticker, ...)      -> Dict

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional


# ── Default assumptions (Citigroup / GS baseline from spreadsheet) ────────────

DEFAULTS = {
    "asset_growth":        0.0094,   # 0.94% -- total asset CAGR (Assumption #9)
    "rwa_pct":             0.52,     # 52%   -- risk weight % of total assets
    "cet1_ratio_initial":  0.1363,   # 13.63% -- CET1 ratio years 1-2
    "cet1_ratio_terminal": 0.1663,   # 16.63% -- CET1 ratio years 3-5 (Basel III buffer)
    "cet1_step_year":      3,        # year when ratio steps up
    "roe":                 0.0633,   # 6.33%  -- return on equity (TTM)
    "payout_ratio":        0.3487,   # 34.87% -- dividend payout ratio
    "cost_of_equity":      0.14,     # 14%    -- CAPM cost of equity
    "terminal_growth":     0.01,     # 1%     -- long-run nominal GDP growth
    "years":               5,
}


# ── Core FCFE engine ──────────────────────────────────────────────────────────

def run_dcf(
    ticker:               str,
    asset_growth:         float = None,
    rwa_pct:              float = None,
    cet1_ratio_initial:   float = None,
    cet1_ratio_terminal:  float = None,
    cet1_step_year:       int   = None,
    roe:                  float = None,
    payout_ratio:         float = None,
    cost_of_equity:       float = None,
    terminal_growth:      float = None,
    years:                int   = None,
) -> Dict:
    """
    Bank FCFE DCF model.
    All parameters default to DEFAULTS dict -- pass overrides as needed.
    Fetches base financials live from yfinance.

    Returns intrinsic price target, upside %, FCFE projections, and sensitivity.
    """

    # ── Apply defaults ────────────────────────────────────────────────────────
    ag   = asset_growth        or DEFAULTS["asset_growth"]
    rwa  = rwa_pct             or DEFAULTS["rwa_pct"]
    ce1i = cet1_ratio_initial  or DEFAULTS["cet1_ratio_initial"]
    ce1t = cet1_ratio_terminal or DEFAULTS["cet1_ratio_terminal"]
    csy  = cet1_step_year      or DEFAULTS["cet1_step_year"]
    roe_ = roe                 or DEFAULTS["roe"]
    pay  = payout_ratio        or DEFAULTS["payout_ratio"]
    ke   = cost_of_equity      or DEFAULTS["cost_of_equity"]
    tg   = terminal_growth     or DEFAULTS["terminal_growth"]
    n    = years               or DEFAULTS["years"]

    retention = 1 - pay

    # ── Fetch base financials from yfinance ───────────────────────────────────
    info              = yf.Ticker(ticker).info
    total_assets_0    = (info.get("totalAssets")           or 0) / 1e9   # $B
    book_equity_0     = (info.get("bookValue")             or 0) * \
                        (info.get("sharesOutstanding")     or 1) / 1e9   # $B
    net_income_0      = (info.get("netIncomeToCommon")     or 0) / 1e9   # $B
    shares_b          = (info.get("sharesOutstanding")     or 1) / 1e9   # billions
    current_price     = info.get("currentPrice", 0)

    # Override ROE default with live value if not explicitly passed in
    if roe is None:
        roe_ = info.get("returnOnEquity") or DEFAULTS["roe"]

    # ── Base year CET1 capital ────────────────────────────────────────────────
    cet1_prev = total_assets_0 * rwa * ce1i

    # ── Year-by-year projections ──────────────────────────────────────────────
    projections  = []
    pv_fcfe_sum  = 0.0
    book_equity  = book_equity_0
    net_income   = net_income_0

    for t in range(1, n + 1):
        # Total assets and RWA
        total_assets = total_assets_0 * (1 + ag) ** t
        rwa_t        = total_assets * rwa

        # CET1 ratio steps up at cet1_step_year
        cet1_ratio = ce1t if t >= csy else ce1i
        cet1_curr  = rwa_t * cet1_ratio

        # Change in regulatory capital = the "capex" reinvestment for a bank
        delta_cet1 = cet1_curr - cet1_prev

        # Net income: prior book equity x ROE
        net_income = book_equity * roe_

        # Book equity grows by retained earnings
        book_equity = book_equity + net_income * retention

        # FCFE = Net income - regulatory capital reinvestment
        fcfe = net_income - delta_cet1

        # Present value
        pv = fcfe / (1 + ke) ** t
        pv_fcfe_sum += pv

        projections.append({
            "year":            2024 + t,
            "total_assets_$B": round(total_assets, 2),
            "rwa_$B":          round(rwa_t, 2),
            "cet1_ratio":      f"{cet1_ratio*100:.2f}%",
            "cet1_capital_$B": round(cet1_curr, 2),
            "delta_cet1_$B":   round(delta_cet1, 2),
            "net_income_$B":   round(net_income, 3),
            "fcfe_$B":         round(fcfe, 3),
            "pv_fcfe_$B":      round(pv, 3),
        })

        cet1_prev = cet1_curr

    # ── Terminal value ────────────────────────────────────────────────────────
    terminal_fcfe  = projections[-1]["fcfe_$B"] * (1 + tg)
    terminal_value = terminal_fcfe / (ke - tg)
    pv_terminal    = terminal_value / (1 + ke) ** n

    # ── Equity value and price target ─────────────────────────────────────────
    equity_value = pv_fcfe_sum + pv_terminal
    price_target = equity_value / shares_b if shares_b > 0 else 0
    upside_pct   = (price_target - current_price) / current_price * 100 \
                   if current_price > 0 else 0

    return {
        "ticker":              ticker,
        "current_price":       round(current_price, 2),
        "price_target":        round(price_target, 2),
        "upside_pct":          round(upside_pct, 1),
        "equity_value_$B":     round(equity_value, 2),
        "pv_fcfe_sum_$B":      round(pv_fcfe_sum, 2),
        "pv_terminal_$B":      round(pv_terminal, 2),
        "terminal_fcfe_$B":    round(terminal_fcfe, 3),
        "rating": (
            "Buy"  if upside_pct >  15 else
            "Hold" if upside_pct > -10 else
            "Sell"
        ),
        "assumptions": {
            "asset_growth":        f"{ag*100:.2f}%",
            "rwa_pct":             f"{rwa*100:.1f}%",
            "cet1_ratio_initial":  f"{ce1i*100:.2f}%",
            "cet1_ratio_terminal": f"{ce1t*100:.2f}%",
            "cet1_step_year":      csy,
            "roe":                 f"{roe_*100:.2f}%",
            "payout_ratio":        f"{pay*100:.2f}%",
            "cost_of_equity":      f"{ke*100:.1f}%",
            "terminal_growth":     f"{tg*100:.1f}%",
            "projection_years":    n,
            "base_total_assets_$B": round(total_assets_0, 2),
            "base_book_equity_$B":  round(book_equity_0, 3),
            "shares_outstanding_B": round(shares_b, 3),
        },
        "projections": projections,
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _get(df: pd.DataFrame, *keys: str, col: int = 0, scale: float = 1e9) -> Optional[float]:
    """
    Try each key against the DataFrame index in order.
    Returns the first non-null, non-zero value / scale.
    Returns None if all keys miss -- lets caller chain with an info-dict fallback.
    yfinance field names change across versions; providing multiple variants is intentional.
    """
    for key in keys:
        try:
            val = df.loc[key].iloc[col]
            if pd.notna(val) and float(val) != 0:
                return float(val) / scale
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _v(df_val: Optional[float], info_val, scale: float = 1e9) -> float:
    """Return df_val if truthy, else info_val / scale, else 0.0."""
    if df_val is not None:
        return df_val
    if info_val:
        return float(info_val) / scale
    return 0.0


def _fmt(val: float, prefix: str = "", suffix: str = "") -> str:
    """Format a float cleanly: 0.0 -> 'N/A', else prefix + 2dp + suffix."""
    if val == 0.0:
        return "N/A"
    return f"{prefix}{val:,.2f}{suffix}"


def _build_table(headers: list, rows: list) -> str:
    """Build a clean markdown table from headers and rows (list of lists)."""
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return "\n".join([header_row, sep, body])


# ── Balance Sheet Assumptions ─────────────────────────────────────────────────
#
#   asset_growth          : 2.5%  -- moderate expansion in lending + trading book
#   loan_growth           : 4.0%  -- faster than total assets; core banking growth driver
#   deposit_growth        : 3.5%  -- deposits grow with loan demand; slight lag
#   long_term_debt_growth : 1.5%  -- conservative; GS is IB-heavy, not deposit-funded
#   payout_ratio          : 34.87% -- consistent with DCF assumption
#   roe                   : live from yfinance; fallback 13%
#   years                 : 3 -- near-term visibility horizon for balance sheet

BS_DEFAULTS = {
    "asset_growth":           0.025,   # 2.5%  -- total asset CAGR
    "loan_growth":            0.040,   # 4.0%  -- loan portfolio CAGR
    "deposit_growth":         0.035,   # 3.5%  -- deposit base CAGR
    "long_term_debt_growth":  0.015,   # 1.5%  -- long-term debt CAGR
    "payout_ratio":           0.3487,  # 34.87% -- dividend payout ratio
    "roe":                    0.13,    # 13%   -- fallback ROE
    "years":                  2,       # 2-year horizon keeps table width clean
}


def get_balance_sheet_projections(
    ticker:                str,
    asset_growth:          float = None,
    loan_growth:           float = None,
    deposit_growth:        float = None,
    long_term_debt_growth: float = None,
    payout_ratio:          float = None,
    roe:                   float = None,
    years:                 int   = None,
) -> Dict:
    """
    Current balance sheet (most recent annual) + 3-year forward projections.

    Data priority for current year:
      1. yfinance balance_sheet DataFrame (multiple key variants tried)
      2. yfinance .info dict
      3. 0.0 only as absolute last resort

    Projection logic:
      - Total Assets      : grown at asset_growth rate
      - Net Loans         : grown at loan_growth rate (faster -- core business)
      - Total Deposits    : grown at deposit_growth rate
      - Long-Term Debt    : grown at long_term_debt_growth rate
      - Total Equity      : prior equity + net income x retention ratio
      - Net Income        : prior equity x ROE (consistent with DCF)
      - Total Liabilities : Total Assets - Total Equity (balance sheet identity)

    All dollar figures in $B.
    """

    # ── Apply defaults ────────────────────────────────────────────────────────
    ag   = asset_growth          or BS_DEFAULTS["asset_growth"]
    lg   = loan_growth           or BS_DEFAULTS["loan_growth"]
    dg   = deposit_growth        or BS_DEFAULTS["deposit_growth"]
    ltdg = long_term_debt_growth or BS_DEFAULTS["long_term_debt_growth"]
    pay  = payout_ratio          or BS_DEFAULTS["payout_ratio"]
    n    = years                 or BS_DEFAULTS["years"]
    retention = 1 - pay

    # ── Fetch live data ───────────────────────────────────────────────────────
    tk   = yf.Ticker(ticker)
    info = tk.info
    bs   = tk.balance_sheet      # rows = line items, cols = annual dates (newest first)

    roe_ = roe or info.get("returnOnEquity") or BS_DEFAULTS["roe"]
    current_year = pd.Timestamp.now().year - 1

    # ── Current balance sheet -- robust multi-key lookups ─────────────────────
    shares_out = (info.get("sharesOutstanding") or 1) / 1e9  # billions

    total_assets_0 = _v(
        _get(bs, "Total Assets", "TotalAssets"),
        info.get("totalAssets")
    )
    cash_0 = _v(
        _get(bs, "Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments",
             "Cash And Short Term Investments"),
        info.get("totalCash")
    )
    net_loans_0 = _v(
        _get(bs, "Net Loan", "Net Loans", "Gross Loans", "Loans",
             "Loans And Leases", "Net Receivables"),
        info.get("netReceivables")
    )
    # Investment portfolio: trading book + AFS securities (IB-heavy firms like GS carry large)
    invest_portfolio_0 = _v(
        _get(bs, "Available For Sale Securities",
             "Investments And Advances", "Investment Securities",
             "Financial Assets Designatedat Fair Value Through Profit Or Loss",
             "Trading Securities", "Securities"),
        None
    )
    # If still missing, estimate: total assets minus loans minus cash as a proxy
    if invest_portfolio_0 == 0.0 and total_assets_0 > 0:
        invest_portfolio_0 = max(0.0, total_assets_0 - net_loans_0 - cash_0) * 0.60

    total_deposits_0 = _v(
        _get(bs, "Total Deposits", "Deposits", "Customer Deposits",
             "Interest Bearing Deposits", "Total Deposits And Short Term Borrowings"),
        None
    )
    long_term_debt_0 = _v(
        _get(bs, "Long Term Debt", "Long Term Debt And Capital Lease Obligation",
             "Long Term Debt Non Current"),
        info.get("longTermDebt")
    )
    total_debt_0 = _v(
        _get(bs, "Total Debt", "Short Long Term Debt Total"),
        None
    ) or long_term_debt_0
    total_equity_0 = _v(
        _get(bs, "Stockholders Equity", "Common Stock Equity",
             "Total Equity Gross Minority Interest", "Tangible Book Value"),
        (info.get("bookValue") or 0) * (info.get("sharesOutstanding") or 0)
    )
    total_liab_0 = _v(
        _get(bs, "Total Liabilities Net Minority Interest",
             "Total Liab Net Minority Interest", "Total Liabilities"),
        None
    ) or (total_assets_0 - total_equity_0)

    book_value_ps_0 = (total_equity_0 / shares_out) if shares_out > 0 else 0
    leverage_0      = (total_assets_0 / total_equity_0) if total_equity_0 > 0 else 0

    # Debug print so the user can see what was fetched each run
    print(f"[BS fetch] Total Assets: ${total_assets_0:.1f}B | Cash: ${cash_0:.1f}B | "
          f"Net Loans: ${net_loans_0:.1f}B | Invest Portfolio: ${invest_portfolio_0:.1f}B | "
          f"Deposits: ${total_deposits_0:.1f}B | LT Debt: ${long_term_debt_0:.1f}B | "
          f"Equity: ${total_equity_0:.1f}B | BVPS: ${book_value_ps_0:.2f}")

    current = {
        "year":                    current_year,
        "total_assets_$B":         round(total_assets_0, 2),
        "net_loans_$B":            round(net_loans_0, 2),
        "invest_portfolio_$B":     round(invest_portfolio_0, 2),
        "total_deposits_$B":       round(total_deposits_0, 2),
        "long_term_debt_$B":       round(long_term_debt_0, 2),
        "total_debt_$B":           round(total_debt_0, 2),
        "total_liabilities_$B":    round(total_liab_0, 2),
        "total_equity_$B":         round(total_equity_0, 2),
        "book_value_ps":           round(book_value_ps_0, 2),
        "leverage_ratio":          round(leverage_0, 1),
        "data_source":             "yfinance annual balance sheet",
    }

    # ── 3-year projections ────────────────────────────────────────────────────
    projections = []
    equity      = total_equity_0

    for t_yr in range(1, n + 1):
        year             = current_year + t_yr
        total_assets     = total_assets_0      * (1 + ag)   ** t_yr
        net_loans        = net_loans_0         * (1 + lg)   ** t_yr
        invest_portfolio = invest_portfolio_0  * (1 + lg)   ** t_yr  # grows with lending activity
        deposits         = total_deposits_0    * (1 + dg)   ** t_yr
        lt_debt          = long_term_debt_0    * (1 + ltdg) ** t_yr
        net_income       = equity * roe_
        equity           = equity + net_income * retention
        total_liab       = total_assets - equity
        bvps             = equity / shares_out if shares_out > 0 else 0

        projections.append({
            "year":                 year,
            "total_assets_$B":      round(total_assets, 2),
            "net_loans_$B":         round(net_loans, 2),
            "invest_portfolio_$B":  round(invest_portfolio, 2),
            "total_deposits_$B":    round(deposits, 2),
            "long_term_debt_$B":    round(lt_debt, 2),
            "total_liabilities_$B": round(total_liab, 2),
            "total_equity_$B":      round(equity, 2),
            "book_value_ps":        round(bvps, 2),
            "net_income_$B":        round(net_income, 3),
        })

    # ── Pre-formatted markdown table — structured with Asset/Liability/Equity sections ──
    p = projections
    years_hdr  = [f"Year+{i+1} ({p[i]['year']})" for i in range(n)]
    blank_cols = [""] * (n + 1)  # current + projection columns
    bs_headers = ["Metric", f"Current ({current_year})"] + years_hdr

    def brow(label, key, prefix="$"):
        """Build a data row; show N/A if value is 0 in both current and projections."""
        cur_val = current.get(key, 0.0)
        # Skip the row entirely if current value is 0 and we have no projection data
        cur_str = _fmt(cur_val, prefix=prefix)
        proj_strs = []
        for pr in p:
            pval = pr.get(key, 0.0)
            proj_strs.append(_fmt(pval, prefix=prefix))
        return [label, cur_str] + proj_strs

    def section_header(title):
        """Bold section divider row spanning all columns."""
        return [f"**{title}**"] + blank_cols

    bs_rows = [
        section_header("ASSETS"),
        brow("Total Assets ($B)",          "total_assets_$B"),
        brow("Net Loans ($B)",             "net_loans_$B"),
        brow("Investment Portfolio ($B)",  "invest_portfolio_$B"),
        section_header("LIABILITIES"),
        brow("Total Deposits ($B)",        "total_deposits_$B"),
        brow("Long-Term Debt ($B)",        "long_term_debt_$B"),
        brow("Total Liabilities ($B)",     "total_liabilities_$B"),
        section_header("EQUITY"),
        brow("Total Equity ($B)",          "total_equity_$B"),
        brow("Book Value per Share ($)",   "book_value_ps"),
    ]

    # Drop any row where ALL value columns are N/A (keeps table clean for IB firms with 0 deposits)
    def all_na(row):
        return all(v in ("N/A", "$N/A", "") for v in row[1:])

    bs_rows = [r for r in bs_rows if not (r[0].startswith("**") is False and all_na(r))]

    bs_table = _build_table(bs_headers, bs_rows)

    assump_rows = [
        ["Total Asset Growth",        f"{ag*100:.1f}% per year",   "Moderate balance sheet expansion"],
        ["Net Loan Growth",           f"{lg*100:.1f}% per year",   "Faster than assets; core banking growth driver"],
        ["Investment Portfolio Growth",f"{lg*100:.1f}% per year",  "Grows with lending/trading activity at loan rate"],
        ["Deposit Growth",            f"{dg*100:.1f}% per year",   "Grows in line with loan demand"],
        ["Long-Term Debt Growth",     f"{ltdg*100:.1f}% per year", "Conservative; GS is IB-heavy, not deposit-funded"],
        ["Payout Ratio",              f"{pay*100:.2f}%",           "Consistent with DCF assumption"],
        ["ROE",                       f"{roe_*100:.2f}%",          "Live from yfinance; drives retained earnings"],
        ["Equity Growth",             "Retained earnings method",  "Prior equity + Net income x (1 - payout ratio)"],
        ["Total Liabilities",         "Balance sheet identity",    "Total Assets - Total Equity"],
    ]
    assump_table = _build_table(["Assumption", "Value", "Rationale"], assump_rows)

    return {
        "ticker":         ticker,
        "current":        current,
        "projections":    projections,
        "markdown_table": bs_table,
        "assumptions_table": assump_table,
    }


# ── Income Statement Assumptions ──────────────────────────────────────────────
#
#   revenue_growth    : 5.0%  -- blended IB fee + NII + asset mgmt CAGR
#   nim_bps_change    : 0 bps -- NIM held flat; rates assumed to plateau
#   efficiency_ratio  : 65%   -- non-interest expense / net revenue (GS ~65% historical)
#   provision_rate    : 0.30% -- provision for credit losses / net loans (benign environment)
#   tax_rate          : 21%   -- U.S. statutory corporate rate; GS effective ~20-22%
#   years             : 3

IS_DEFAULTS = {
    "revenue_growth":   0.050,   # 5.0%  -- blended revenue CAGR
    "nim_bps_change":   0,       # 0 bps -- NIM held flat YoY
    "efficiency_ratio": 0.650,   # 65%   -- non-interest expense / net revenue
    "provision_rate":   0.003,   # 0.30% -- provision for credit losses / net loans
    "tax_rate":         0.210,   # 21%   -- effective tax rate
    "years":            2,       # 2-year horizon keeps table width clean
}


def get_income_statement_projections(
    ticker:           str,
    revenue_growth:   float = None,
    nim_bps_change:   int   = None,
    efficiency_ratio: float = None,
    provision_rate:   float = None,
    tax_rate:         float = None,
    years:            int   = None,
) -> Dict:
    """
    Current income statement (most recent annual) + 3-year forward projections.

    Data priority for current year:
      1. yfinance income_stmt DataFrame (multiple key variants tried)
      2. yfinance .info dict
      3. 0.0 only as absolute last resort

    Projection logic:
      - Net Revenue       : grown at revenue_growth rate each year
      - Non-Int. Expense  : Net Revenue x efficiency_ratio
      - Provision         : provision_rate x projected net loans
      - Pre-tax Income    : Net Revenue - Non-Int. Expense - Provision
      - Tax               : Pre-tax Income x tax_rate
      - Net Income        : Pre-tax Income x (1 - tax_rate)
      - EPS               : Net Income / shares outstanding

    All dollar figures in $B except EPS.
    """

    # ── Apply defaults ────────────────────────────────────────────────────────
    rg  = revenue_growth   or IS_DEFAULTS["revenue_growth"]
    nim = nim_bps_change   if nim_bps_change is not None else IS_DEFAULTS["nim_bps_change"]
    er  = efficiency_ratio or IS_DEFAULTS["efficiency_ratio"]
    pr  = provision_rate   or IS_DEFAULTS["provision_rate"]
    tr  = tax_rate         or IS_DEFAULTS["tax_rate"]
    n   = years            or IS_DEFAULTS["years"]

    # ── Fetch live data ───────────────────────────────────────────────────────
    tk   = yf.Ticker(ticker)
    info = tk.info
    inc  = tk.income_stmt    # rows = line items, cols = annual dates (newest first)
    bs   = tk.balance_sheet

    shares_b     = (info.get("sharesOutstanding") or 1) / 1e9
    current_year = pd.Timestamp.now().year - 1

    # ── Current income statement -- robust multi-key lookups ──────────────────

    # Net Revenue: banks report "Total Revenue"; fallback to info dict
    net_revenue_0 = _v(
        _get(inc, "Total Revenue", "Net Revenue", "Operating Revenue",
             "Total Net Revenue", "Revenue"),
        info.get("totalRevenue")
    )

    # Non-Interest / Operating Expense
    op_expense_0 = _v(
        _get(inc, "Non Interest Expense", "Operating Expense", "Total Expenses",
             "Selling General Administrative", "Total Operating Expenses"),
        None
    )
    # If still missing, derive from efficiency ratio applied to actual revenue
    if op_expense_0 == 0.0 and net_revenue_0 > 0:
        op_expense_0 = net_revenue_0 * IS_DEFAULTS["efficiency_ratio"]

    # Pre-tax Income
    pretax_0 = _v(
        _get(inc, "Pretax Income", "Pre Tax Income", "Income Before Tax",
             "Operating Income", "EBIT"),
        None
    )

    # Tax Provision
    tax_0 = _v(
        _get(inc, "Tax Provision", "Income Tax Expense", "Tax Expense"),
        None
    )

    # Net Income
    net_income_0 = _v(
        _get(inc, "Net Income", "Net Income Common Stockholders",
             "Net Income Including Noncontrolling Interests"),
        info.get("netIncomeToCommon")
    )

    # Interest Expense
    interest_exp_0 = _v(
        _get(inc, "Interest Expense", "Net Interest Income", "Interest Expense Non Operating"),
        None
    )

    # Net Loans (for provision projection -- consistent with balance sheet)
    net_loans_0 = _v(
        _get(bs, "Net Loan", "Net Loans", "Gross Loans", "Loans",
             "Loans And Leases", "Net Receivables"),
        info.get("netReceivables")
    )

    eps_0 = net_income_0 / shares_b if shares_b > 0 else 0
    eff_ratio_actual = (op_expense_0 / net_revenue_0) if net_revenue_0 > 0 else er
    provision_0 = net_loans_0 * pr  # apply assumed rate to current net loans

    current = {
        "year":                      current_year,
        "net_revenue_$B":            round(net_revenue_0, 2),
        "interest_expense_$B":       round(interest_exp_0, 2),
        "non_interest_expense_$B":   round(op_expense_0, 2),
        "provision_$B":              round(provision_0, 3),
        "pretax_income_$B":          round(pretax_0, 2),
        "tax_provision_$B":          round(tax_0, 2),
        "net_income_$B":             round(net_income_0, 2),
        "eps_$":                     round(eps_0, 2),
        "efficiency_ratio":          f"{eff_ratio_actual*100:.1f}%",
        "data_source":               "yfinance annual income statement",
    }

    # ── 3-year projections ────────────────────────────────────────────────────
    projections = []
    net_revenue = net_revenue_0
    net_loans   = net_loans_0

    for t_yr in range(1, n + 1):
        year        = current_year + t_yr
        net_revenue = net_revenue * (1 + rg)
        net_loans   = net_loans   * (1 + BS_DEFAULTS["loan_growth"])

        non_int_exp = net_revenue * er
        provision   = net_loans   * pr
        pretax      = net_revenue - non_int_exp - provision
        tax         = pretax * tr
        net_income  = pretax - tax
        eps         = net_income / shares_b if shares_b > 0 else 0

        projections.append({
            "year":                    year,
            "net_revenue_$B":          round(net_revenue, 2),
            "non_interest_expense_$B": round(non_int_exp, 2),
            "provision_$B":            round(provision, 3),
            "pretax_income_$B":        round(pretax, 2),
            "tax_provision_$B":        round(tax, 2),
            "net_income_$B":           round(net_income, 2),
            "eps_$":                   round(eps, 2),
            "efficiency_ratio":        f"{er*100:.1f}%",
        })

    # ── Pre-formatted markdown table (LLM copies verbatim) ───────────────────
    p = projections
    years_hdr = [f"Year+{i+1} ({p[i]['year']})" for i in range(n)]
    is_headers = ["Metric", f"Current ({current_year})"] + years_hdr

    def irow(label, cur_key, proj_key, prefix="$"):
        cur = _fmt(current[cur_key], prefix=prefix)
        proj_vals = [_fmt(pr[proj_key], prefix=prefix) for pr in p]
        return [label, cur] + proj_vals

    # Key lines only — Non-Interest Expense and Tax Provision are derivable
    # from Efficiency Ratio and Tax Rate shown in the assumptions table.
    is_rows = [
        irow("Net Revenue ($B)",   "net_revenue_$B",   "net_revenue_$B"),
        irow("Pre-Tax Income ($B)","pretax_income_$B", "pretax_income_$B"),
        irow("Net Income ($B)",    "net_income_$B",    "net_income_$B"),
        irow("EPS ($)",            "eps_$",            "eps_$"),
        ["Efficiency Ratio",
         current["efficiency_ratio"]] + [pr["efficiency_ratio"] for pr in p],
    ]
    is_table = _build_table(is_headers, is_rows)

    assump_rows = [
        ["Revenue Growth",      f"{rg*100:.1f}% per year",    "Blended IB fees + NII + asset management CAGR"],
        ["NIM Change",          f"{nim} bps per year",        "NIM held flat; rates assumed to plateau"],
        ["Efficiency Ratio",    f"{er*100:.1f}%",             "Non-interest expense / net revenue; GS historical avg"],
        ["Provision Rate",      f"{pr*100:.2f}% of loans",    "Benign credit environment assumed"],
        ["Tax Rate",            f"{tr*100:.1f}%",             "U.S. statutory rate; GS effective ~20-22%"],
        ["Loan Growth",         f"{BS_DEFAULTS['loan_growth']*100:.1f}% per year", "Consistent with balance sheet projection"],
        ["Shares Outstanding",  f"{shares_b:.3f}B shares",   "Used to compute EPS; sourced from yfinance"],
    ]
    assump_table = _build_table(["Assumption", "Value", "Rationale"], assump_rows)

    return {
        "ticker":            ticker,
        "current":           current,
        "projections":       projections,
        "markdown_table":    is_table,
        "assumptions_table": assump_table,
    }

# ── Valuation Football Field Chart ────────────────────────────────────────────

def generate_valuation_chart(ticker: str, output_path: str) -> Dict:
    """
    Compute all three valuation methods in Python and generate a
    football field chart saved to output_path.

    Methods:
      1. FCFE DCF       : run_dcf()
      2. P/B Multiple   : sector median P/B (1.5x) x book value per share
      3. Gordon Growth  : P/B = (ROE - g) / (Ke - g) x book value per share

    Chart shows each price target vs current price and 52-week range.
    Returns a dict with all computed values for use in report commentary.
    """

    # ── 1. FCFE DCF ───────────────────────────────────────────────────────────
    dcf = run_dcf(ticker)
    fcfe_target  = dcf["price_target"]
    current_price = dcf["current_price"]

    # ── 2. Fetch market data ──────────────────────────────────────────────────
    info          = yf.Ticker(ticker).info
    book_value    = info.get("bookValue") or 0          # per share
    roe           = info.get("returnOnEquity") or 0.13
    week52_low    = info.get("fiftyTwoWeekLow") or current_price * 0.6
    week52_high   = info.get("fiftyTwoWeekHigh") or current_price * 1.1

    # ── 3. P/B Multiple (sector median = 1.5x for bulge bracket) ─────────────
    sector_pb     = 1.50
    pb_target     = sector_pb * book_value

    # ── 4. Gordon Growth implied P/B ─────────────────────────────────────────
    ke            = DEFAULTS["cost_of_equity"]   # 14%
    g             = DEFAULTS["terminal_growth"]  # 1%
    implied_pb    = (roe - g) / (ke - g) if (ke - g) != 0 else 1.0
    gordon_target = implied_pb * book_value

    # ── 5. Blended price target (equal weight) ────────────────────────────────
    blended_target = (fcfe_target + pb_target + gordon_target) / 3
    upside_pct     = (blended_target - current_price) / current_price * 100

    # ── Compute meaningful ranges (sensitivity-based, not arbitrary %) ────────
    # FCFE: Ke sensitivity +/-1%
    dcf_lo = run_dcf(ticker, cost_of_equity=ke + 0.01)["price_target"]
    dcf_hi = run_dcf(ticker, cost_of_equity=ke - 0.01)["price_target"]

    # P/B: sector range 1.3x – 1.7x
    pb_lo = 1.30 * book_value
    pb_hi = 1.70 * book_value

    # Gordon Growth: Ke sensitivity +/-1%
    gg_lo = ((roe - g) / (ke + 0.01 - g)) * book_value if (ke + 0.01 - g) != 0 else gordon_target
    gg_hi = ((roe - g) / (ke - 0.01 - g)) * book_value if (ke - 0.01 - g) != 0 else gordon_target

    # Blended: min/max of all three methods
    blend_lo = min(dcf_lo, pb_lo, gg_lo)
    blend_hi = max(dcf_hi, pb_hi, gg_hi)

    # ── 6. Build football field chart (style matching IB standard) ────────────
    NAVY = "#1E2761"
    BAR  = "#17307a"

    # Bottom to top order (52-week at bottom, blended at top)
    rows = [
        ("52-Week Range",    week52_low,   week52_high,  None),
        ("Blended Target",   blend_lo,     blend_hi,     blended_target),
        ("Gordon Growth P/B",gg_lo,        gg_hi,        gordon_target),
        ("P/B Multiple",     pb_lo,        pb_hi,        pb_target),
        ("FCFE DCF",         dcf_lo,       dcf_hi,       fcfe_target),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (label, lo, hi, mid) in enumerate(rows):
        # Thick solid bar
        ax.barh(i, hi - lo, left=lo, height=0.35,
                color=BAR, alpha=0.90, zorder=3)
        # Endpoint dots
        ax.plot([lo, hi], [i, i], "o", color="black",
                markersize=7, zorder=4)
        # Endpoint labels
        ax.text(lo - 2, i, f"{lo:,.0f}",
                ha="right", va="center", fontsize=8.5, color="black")
        ax.text(hi + 2, i, f"{hi:,.0f}",
                ha="left",  va="center", fontsize=8.5, color="black")
        # Midpoint marker (price target)
        if mid is not None:
            ax.plot(mid, i, "D", color="white",
                    markersize=7, zorder=5, markeredgecolor=BAR, markeredgewidth=1.5)

    # Current price: shaded band + dashed line
    band_w = current_price * 0.025
    ax.axvspan(current_price - band_w, current_price + band_w,
               color="#f4a460", alpha=0.25, zorder=2)
    import datetime
    today_str = datetime.date.today().strftime("%m/%d/%y")
    ax.axvline(current_price, color=BAR, linewidth=2.5,
               linestyle="--", zorder=6,
               label=f"Current share price {today_str} = {current_price:,.2f}")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("Share Price (USD)", fontsize=10)
    ax.set_title(f"{ticker} — Valuation Ranges by Methodology",
                 fontsize=13, fontweight="bold", color=NAVY, pad=14)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.set_xlim(min(week52_low, dcf_lo, pb_lo, gg_lo) * 0.88,
                max(week52_high, dcf_hi, pb_hi, gg_hi) * 1.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "ticker":          ticker,
        "current_price":   current_price,
        "fcfe_target":     round(fcfe_target, 2),
        "pb_target":       round(pb_target, 2),
        "gordon_target":   round(gordon_target, 2),
        "blended_target":  round(blended_target, 2),
        "upside_pct":      round(upside_pct, 1),
        "week52_low":      round(week52_low, 2),
        "week52_high":     round(week52_high, 2),
        "book_value":      round(book_value, 2),
        "roe":             f"{roe*100:.2f}%",
        "sector_pb":       sector_pb,
        "implied_pb":      round(implied_pb, 3),
        "rating":          ("Buy"  if upside_pct > 15 else
                            "Hold" if upside_pct > -10 else "Sell"),
        "chart_path":      output_path,
    }
