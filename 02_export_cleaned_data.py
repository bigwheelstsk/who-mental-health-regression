"""
WHO Mental Health & Suicide Rate — Cleaned Data Export
Group Project: Regression Analysis (QM Course)

Reproduces the exact cleaning step app.py runs before fitting the
regression (forward/backward-fill missing predictors within each
country, then drop any row still missing the target or a predictor),
and writes the result to an Excel workbook for inspection / submission.

Run with: python 02_export_cleaned_data.py
Output:   who_mental_health_suicide_panel_cleaned.xlsx
"""

import pandas as pd

CSV_PATH = "who_mental_health_suicide_panel.csv"
OUT_PATH = "who_mental_health_suicide_panel_cleaned.xlsx"
TARGET = "suicide_rate"
PREDICTORS = [
    "psychiatrists_per_100k",
    "mh_units_gen_hospitals",
    "govt_mh_expenditure_pct",
    "govt_health_exp_pct_gdp",
    "alcohol_consumption_L",
]

def main():
    raw = pd.read_csv(CSV_PATH)

    df = raw.sort_values(["country", "year"]).copy()
    fill_cols = [c for c in PREDICTORS if c != "alcohol_consumption_L"]
    df.loc[:, fill_cols] = df.groupby("country")[fill_cols].transform(lambda s: s.ffill().bfill())
    clean = df.dropna(subset=[TARGET] + PREDICTORS).reset_index(drop=True)

    coverage = pd.DataFrame({
        "predictor": PREDICTORS,
        "countries_reporting": [raw[raw[c].notna()].country.nunique() for c in PREDICTORS],
        "total_countries": raw.country.nunique(),
    })

    summary = pd.DataFrame({
        "metric": [
            "raw rows", "raw countries",
            "cleaned rows", "cleaned countries",
            "cleaned year range",
        ],
        "value": [
            len(raw), raw.country.nunique(),
            len(clean), clean.country.nunique(),
            f"{clean.year.min()}-{clean.year.max()}",
        ],
    })

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        clean.to_excel(writer, sheet_name="cleaned_data", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        coverage.to_excel(writer, sheet_name="predictor_coverage", index=False)

    print(f"Saved {OUT_PATH}")
    print(f"Cleaned rows: {len(clean)} across {clean.country.nunique()} countries "
          f"({clean.year.min()}-{clean.year.max()})")

if __name__ == "__main__":
    main()
