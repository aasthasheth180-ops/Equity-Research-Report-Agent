# dcf_tool.py — owned by Person C
#
# Bank FCFE valuation model.
# Methodology from Valuation-FCFE sheet, Financial_Model_Banking_MK.xlsx
#
# Core mechanic:
#   Banks treat regulatory capital as working capital reinvestment.
#   FCFE = Net Income − Change in CET1 Capital
#   CET1 Capital = Total Assets × RWA% × CET1 Ratio
#
# Interface tools.py relies on:
#   run_dcf(ticker, **optional_overrides) -> Dict

import yfinance as yf
from typing import Dict, Optional


# ── Default assumptions (Citigroup / GS baseline from spreadsheet) ────────────

DEFAULTS = {
    "asset_growth":        0.0094,   # 0.94% — total asset CAGR (Assumption #9)
    "rwa_pct":             0.52,     # 52%   — risk weight % of total assets
    "cet1_ratio_initial":  0.1363,   # 13.63% — CET1 ratio years 1-2
    "cet1_ratio_terminal": 0.1663,   # 16.63% — CET1 ratio years 3-5 (Basel III buffer)
    "cet1_step_year":      3,        # year when ratio steps up
    "roe":                 0.0633,   # 6.33%  — return on equity (TTM)
    "payout_ratio":        0.3487,   # 34.87% — dividend payout ratio
    "cost_of_equity":      0.14,     # 14%    — CAPM cost of equity
    "terminal_growth":     0.01,     # 1%     — long-run nominal GDP growth
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
    All parameters default to DEFAULTS dict — pass overrides as needed.
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

        # Net income: prior book equity × ROE
        net_income = book_equity * roe_

        # Book equity grows by retained earnings
        book_equity = book_equity + net_income * retention

        # FCFE = Net income − regulatory capital reinvestment
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