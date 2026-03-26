# data_loader.py — owned by Person C
#
# Wraps the Goldman Sachs regulatory data pipeline.
# Source: market_risk.ipynb — functions extracted verbatim.
# Hardcoded paths and inst_id lookup moved to config constants below.
#
# Interface the orchestrator relies on:
#   build_market_risk_df(ticker, date, csv_folder) -> pd.DataFrame
#   INST_ID_MAP: Dict[str, str]

import os
from typing import Dict, Tuple

import pandas as pd


# ── Institution config ────────────────────────────────────────────────────────
# Maps Yahoo Finance ticker → FFIEC / Fed inst_id

INST_ID_MAP: Dict[str, str] = {
    "GS":  "2380443",   # Goldman Sachs  ← active for this project
    "JPM": "1039502",   # JPMorgan Chase  (add CSVs to enable)
    "BAC": "1073757",   # Bank of America (add CSVs to enable)
    "WFC": "1120754",   # Wells Fargo     (add CSVs to enable)
    "C":   "1951350",   # Citigroup       (add CSVs to enable)
}

# Qualitative data from annual reports / 10-K.
# Keyed by inst_id. Add a block for each new bank you onboard.
_QUAL_DATA: Dict[str, Dict] = {
    "2380443": {   # Goldman Sachs — from 2024 Annual Report
        "RWA Approach":             "IMA",
        "Internal Risk Management": "95% 1-day VaR",
        "Return Weight Approach":   "Exponential Weightage",
        "Historical Window":        "5 years",
        "Recent Annual Report Date": "12/31/2024",
    },
}


# ── Core pipeline (verbatim from market_risk.ipynb) ──────────────────────────

def create_fry9c_database(input_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FR Y-9C and FFIEC102 CSVs from input_folder.
    Filename convention: FRY9C_<inst_id>_<date>.csv
                         FFIEC102_<inst_id>_<date>.csv
    Returns (fry9c_df, ffiec102_df).
    """
    fry9c_data    = pd.DataFrame()
    ffiec102_data = pd.DataFrame()

    for file in os.listdir(input_folder):
        if not file.lower().endswith(".csv"):
            continue
        file_path = os.path.join(input_folder, file)
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        try:
            parts   = file.replace(".csv", "").split("_")
            inst_id = parts[1] if len(parts) > 1 else ""
            date    = parts[2] if len(parts) > 2 else ""
        except Exception:
            inst_id = date = ""

        df["inst_id"] = inst_id
        df["date"]    = date

        if file.upper().startswith("FRY9C"):
            fry9c_data = pd.concat([fry9c_data, df], ignore_index=True)
        elif file.upper().startswith("FFIEC102"):
            ffiec102_data = pd.concat([ffiec102_data, df], ignore_index=True)

    return fry9c_data, ffiec102_data


def market_risk_analysis(
    fry9c_df: pd.DataFrame,
    ffiec102_df: pd.DataFrame,
    inst_id: str,
    date: str,
) -> pd.DataFrame:
    """
    Extract market risk metrics for one institution / reporting date.
    Returns a DataFrame indexed by metric name:
      columns: Amount $millions | Source | date | Institution Name
    """
    fry9c_df    = fry9c_df[(fry9c_df["inst_id"] == inst_id) & (fry9c_df["date"] == date)]
    ffiec102_df = ffiec102_df[(ffiec102_df["inst_id"] == inst_id) & (ffiec102_df["date"] == date)]

    codes_y9c = {
        "BHCK2170": "Total Assets",
        "BHCK3545": "Total Trading Assets",
        "BHCKA220": "Total Trading Revenue",
        "BHCKS581": "Market Risk RWA",
    }
    codes_ffiec = {
        "MRRRS298": "Most recent Regulatory 10-day 99% VaR",
        "MRRRS366": "Stress Period Start Date",
        "MRRRS302": "Most recent SvaR",
        "MRRRS300": "Multiplicative factor",
    }

    # FR Y-9C metrics
    y9c_data = fry9c_df[fry9c_df["ItemName"].isin(codes_y9c.keys())]
    results  = dict(zip(
        y9c_data["ItemName"].map(codes_y9c),
        pd.to_numeric(y9c_data["Value"], errors="coerce") / 1000,
    ))
    sources = {label: "FR Y-9C" for label in results}

    # FFIEC 102 metrics
    ffiec_data = ffiec102_df[ffiec102_df["ItemName"].isin(codes_ffiec.keys())]
    for _, row in ffiec_data.iterrows():
        label = codes_ffiec[row["ItemName"]]
        results[label] = (
            row["Value"]
            if label == "Stress Period Start Date"
            else pd.to_numeric(row["Value"], errors="coerce") / 1000
        )
        sources[label] = "FFIEC 102"

    # Build base table
    final_table = pd.DataFrame.from_dict(results, orient="index", columns=["Amount $millions"])
    for idx in final_table.index:
        val = final_table.loc[idx, "Amount $millions"]
        if not isinstance(val, str):
            final_table.loc[idx, "Amount $millions"] = float(val)

    final_table["Source"] = final_table.index.map(sources)
    final_table["date"]   = date

    inst_name_vals = fry9c_df[fry9c_df["ItemName"] == "Institution Name"]["Value"].values
    inst_name = inst_name_vals[0] if len(inst_name_vals) > 0 else "Unknown"
    final_table["Institution Name"] = inst_name

    # Qualitative rows from annual report
    qual = _QUAL_DATA.get(inst_id)
    if qual:
        qual_rows = pd.DataFrame({
            "Amount $millions": list(qual.values()),
            "Source":           "Annual Report",
            "date":             date,
            "Institution Name": inst_name,
        }, index=qual.keys())
        final_table = pd.concat([final_table, qual_rows])

    return final_table


# ── Public entry point (called by orchestrator) ───────────────────────────────

def build_market_risk_df(ticker: str, date: str, csv_folder: str) -> pd.DataFrame:
    """
    High-level entry point.
      ticker:     Yahoo Finance ticker e.g. 'GS'
      date:       YYYYMMDD string    e.g. '20250930'
      csv_folder: folder containing FRY9C_*.csv and FFIEC102_*.csv
    """
    inst_id = INST_ID_MAP.get(ticker.upper())
    if inst_id is None:
        raise ValueError(
            f"No inst_id mapped for '{ticker}'. Add it to INST_ID_MAP in data_loader.py."
        )
    fry9c_df, ffiec102_df = create_fry9c_database(csv_folder)
    return market_risk_analysis(fry9c_df, ffiec102_df, inst_id, date)
