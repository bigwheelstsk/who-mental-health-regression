"""
WHO Mental Health & Suicide Rate — Data Collection Script (v2, fault-tolerant)
Group Project: Regression Analysis (QM Course)

Run this in Google Colab or any Python environment with internet access.
Pulls live data from WHO's GHO API, with a World Bank fallback for the
health-expenditure control variable (WHO's old NHA-prefixed codes are
legacy and often return empty results).
"""

import requests
import pandas as pd
import time

GHO_BASE = "https://ghoapi.azureedge.net/api/"
WB_BASE = "https://api.worldbank.org/v2/country/all/indicator/"

# WHO GHO indicators (confirmed live in the current indicator catalog)
GHO_INDICATORS = {
    "MH_12": "suicide_rate",             # Dependent variable
    "MH_6":  "psychiatrists_per_100k",   # Predictor
    "MH_18": "mh_units_gen_hospitals",   # Predictor
    "MH_4":  "govt_mh_expenditure_pct",  # Predictor
    "SA_0000001688": "alcohol_consumption_L",  # Control (country-level; SA_0000001737 is regional-only, always empty)
    # Modern Global Health Expenditure Database code (replaces legacy NHAGGHEGDP,
    # confirmed present in WHO's current indicator catalog).
    "GHED_GGHE-DGDP_SHA2011": "govt_health_exp_pct_gdp",  # Control
}
# World Bank is now only a fallback, used automatically if the GHED code above
# ever returns empty for some reason.
WB_INDICATOR = ("SH.XPD.CHEX.GD.ZS", "govt_health_exp_pct_gdp")

def fetch_gho_indicator(code):
    """Pull one indicator from WHO's GHO API. Returns None if empty/unavailable."""
    url = f"{GHO_BASE}{code}"
    print(f"Fetching {code} (WHO GHO) ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    records = resp.json().get("value", [])
    if not records:
        print(f"  WARNING: {code} returned no data -- skipping this indicator.")
        return None
    df = pd.DataFrame(records)
    if "SpatialDimType" not in df.columns:
        print(f"  WARNING: {code} response missing expected fields -- skipping.")
        return None
    df = df[df["SpatialDimType"] == "COUNTRY"]
    df = df[["SpatialDim", "TimeDim", "NumericValue"]]
    df.columns = ["country", "year", code]
    df = df.groupby(["country", "year"], as_index=False)[code].mean()
    return df

def fetch_worldbank_indicator(code):
    """Pull one indicator from the World Bank API (JSON, paginated)."""
    print(f"Fetching {code} (World Bank) ...")
    all_rows = []
    page = 1
    while True:
        url = f"{WB_BASE}{code}?format=json&per_page=1000&page={page}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if len(payload) < 2 or not payload[1]:
            break
        all_rows.extend(payload[1])
        if page >= payload[0]["pages"]:
            break
        page += 1
    if not all_rows:
        print(f"  WARNING: {code} returned no data from World Bank.")
        return None
    df = pd.DataFrame([
        {"country": r["countryiso3code"], "year": int(r["date"]), code: r["value"]}
        for r in all_rows if r["value"] is not None and r["countryiso3code"]
    ])
    return df

def main():
    merged = None
    gdp_col_pulled = False

    for code, name in GHO_INDICATORS.items():
        df = fetch_gho_indicator(code)
        if df is None:
            continue
        df = df.rename(columns={code: name})
        if name == "govt_health_exp_pct_gdp":
            gdp_col_pulled = True
        merged = df if merged is None else merged.merge(df, on=["country", "year"], how="outer")
        time.sleep(1)

    # Only fall back to World Bank if the GHED indicator above didn't come through
    if not gdp_col_pulled:
        wb_code, wb_name = WB_INDICATOR
        print("GHED indicator unavailable -- falling back to World Bank ...")
        df = fetch_worldbank_indicator(wb_code)
        if df is not None:
            df = df.rename(columns={wb_code: wb_name})
            merged = df if merged is None else merged.merge(df, on=["country", "year"], how="outer")

    if merged is None or merged.empty:
        raise RuntimeError("No data was successfully pulled from any source. Check your internet connection.")

    merged = merged.sort_values(["country", "year"]).reset_index(drop=True)
    merged.to_csv("who_mental_health_suicide_panel.csv", index=False)

    print("\nDone. Saved who_mental_health_suicide_panel.csv")
    print(f"Shape: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print("\nColumns pulled successfully:", [c for c in merged.columns if c not in ("country", "year")])
    print("\nMissing values per column:")
    print(merged.isna().sum())
    print("\nYear range:", merged["year"].min(), "-", merged["year"].max())
    print("Countries:", merged["country"].nunique())

if __name__ == "__main__":
    main()