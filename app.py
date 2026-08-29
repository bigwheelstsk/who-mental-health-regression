"""
WHO Mental Health & Suicide Rate — Local App Server
Group Project: Regression Analysis (QM Course)

Run with: python app.py
Then open: http://localhost:5000

This loads who_mental_health_suicide_panel.csv (from 01_pull_who_data.py),
fits the regression fresh every time the server starts, and serves both
the model (as JSON) and the interactive frontend. No manual copy-pasting
of coefficients needed -- the browser always reflects your real, current data.
"""

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import statsmodels.api as sm
import os

app = Flask(__name__, static_folder=".")

CSV_PATH = "who_mental_health_suicide_panel.csv"
TARGET = "suicide_rate"
PREDICTORS = [
    "psychiatrists_per_100k",
    "mh_units_gen_hospitals",
    "govt_mh_expenditure_pct",
    "govt_health_exp_pct_gdp",
    "alcohol_consumption_L",
]

STATE = {"model": None, "ranges": None, "df": None}

def load_and_fit():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"\n\n'{CSV_PATH}' not found.\n"
            "Run '01_pull_who_data.py' first to download real WHO data,\n"
            "then restart this server.\n"
        )
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["country", "year"])
    df.loc[:, PREDICTORS] = df.groupby("country")[PREDICTORS].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=[TARGET] + PREDICTORS)

    if len(df) < 15:
        raise ValueError(
            f"Only {len(df)} usable rows after cleaning -- too few to fit a "
            "reliable model. Check 01_pull_who_data.py ran successfully."
        )

    X = sm.add_constant(df[PREDICTORS])
    y = df[TARGET]
    model = sm.OLS(y, X).fit()

    ranges = {}
    for p in PREDICTORS:
        ranges[p] = {
            "min": float(df[p].min()),
            "mean": float(df[p].mean()),
            "max": float(df[p].max()),
        }

    STATE["model"] = model
    STATE["ranges"] = ranges
    STATE["df"] = df
    print(f"\nModel fitted on {len(df)} country-year rows.")
    print(f"R-squared: {model.rsquared:.3f}\n")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/model")
def get_model():
    model = STATE["model"]
    return jsonify({
        "intercept": float(model.params["const"]),
        "coefficients": {p: float(model.params[p]) for p in PREDICTORS},
        "pValues": {p: float(model.pvalues[p]) for p in PREDICTORS},
        "rSquared": float(model.rsquared),
        "adjRSquared": float(model.rsquared_adj),
        "nObservations": int(model.nobs),
        "ranges": STATE["ranges"],
    })

@app.route("/api/data")
def get_data():
    df = STATE["df"]
    cols = PREDICTORS + [TARGET, "country", "year"]
    return jsonify(df[cols].round(3).to_dict(orient="records"))

@app.route("/api/predict", methods=["POST"])
def predict():
    values = request.get_json()
    model = STATE["model"]
    y = model.params["const"]
    for p in PREDICTORS:
        y += model.params[p] * float(values.get(p, STATE["ranges"][p]["mean"]))
    return jsonify({"prediction": max(0.0, float(y))})

if __name__ == "__main__":
    load_and_fit()
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
else:
    # Also fit when imported by a production server like gunicorn
    load_and_fit()