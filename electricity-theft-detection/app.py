"""Flask web app: Smart Electricity Theft Detection System.

Run:  python train.py   (once)   then   python app.py
"""
import io
import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for

from features import FEATURES, FEATURE_LABELS, change_percent

app = Flask(__name__)
app.secret_key = "change-me-for-production"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "predictions.db")
THRESHOLD = 0.5

# ----------------------------------------------------------- load trained models
try:
    META = json.load(open(os.path.join(BASE, "models", "meta.json")))
    MODELS = {k: joblib.load(os.path.join(BASE, "models", f"{k}.joblib")) for k in META["results"]}
except FileNotFoundError:
    raise SystemExit("Models not found. Run  python train.py  first.")

MODEL_NAMES = {k: v["name"] for k, v in META["results"].items()}
MEDIANS = META["feature_medians"]


def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            ts TEXT, source TEXT, model TEXT, suspicious INTEGER, probability REAL)""")


def log(source, model, flags, probs):
    with sqlite3.connect(DB) as c:
        ts = datetime.now().isoformat(timespec="seconds")
        c.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)",
                      [(ts, source, model, int(f), float(p)) for f, p in zip(flags, probs)])


def risk_level(p):
    return "High" if p >= 0.8 else "Medium" if p >= THRESHOLD else "Low"


def predict_frame(df, model_key):
    model = MODELS[model_key]
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return (proba >= THRESHOLD).astype(int), proba


def prepare(df):
    """Fill optional/missing columns so a partial input still works."""
    df = df.copy()
    for col in ["prev_month", "curr_month", "avg_consumption"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "change_pct" not in df.columns:
        df["change_pct"] = np.nan
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    computed = change_percent(df["prev_month"].fillna(0), df["curr_month"].fillna(0))
    df["change_pct"] = df["change_pct"].fillna(pd.Series(computed, index=df.index))
    df["curr_to_avg"] = df["curr_month"] / df["avg_consumption"].clip(lower=1e-6)
    # Variability: if not supplied, estimate it from the three monthly figures
    # (a sharp drop implies uneven usage), never lower than the typical value.
    months = df[["prev_month", "curr_month", "avg_consumption"]]
    cv_est = (months.std(axis=1, ddof=0) / months.mean(axis=1).clip(lower=1e-6)).clip(upper=1.5)
    cv_est = np.maximum(cv_est, MEDIANS["cv"])
    if "cv" not in df.columns:
        df["cv"] = np.nan
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce").fillna(cv_est)
    if "zero_days_pct" not in df.columns:
        df["zero_days_pct"] = np.nan
    df["zero_days_pct"] = pd.to_numeric(df["zero_days_pct"], errors="coerce").fillna(MEDIANS["zero_days_pct"])
    df = df.dropna(subset=["prev_month", "curr_month", "avg_consumption"])
    if (df[["prev_month", "curr_month", "avg_consumption"]] < 0).any().any():
        raise ValueError("Consumption values cannot be negative.")
    return df


# ------------------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template("index.html", meta=META)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result, form = None, {}
    if request.method == "POST":
        form = request.form.to_dict()
        model_key = form.get("model", META["best_model"])
        if model_key not in MODELS:
            model_key = META["best_model"]
        try:
            row = {k: [form.get(k, "").strip() or np.nan] for k in FEATURES}
            df = prepare(pd.DataFrame(row))
            if df.empty:
                raise ValueError("Please fill in previous, current and average consumption.")
            flag, proba = predict_frame(df, model_key)
            log("manual", model_key, flag, proba)
            r = df.iloc[0]
            result = {
                "suspicious": bool(flag[0]),
                "probability": round(float(proba[0]) * 100, 1),
                "risk": risk_level(proba[0]),
                "model": MODEL_NAMES[model_key],
                "change_pct": round(float(r["change_pct"]), 1),
            }
        except ValueError as e:
            flash(str(e), "error")
    return render_template("predict.html", result=result, form=form, labels=FEATURE_LABELS,
                           model_names=MODEL_NAMES, best=META["best_model"], batch=None)


@app.route("/predict/upload", methods=["POST"])
def upload():
    model_key = request.form.get("model", META["best_model"])
    if model_key not in MODELS:
        model_key = META["best_model"]
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        flash("Please upload a .csv file.", "error")
        return redirect(url_for("predict"))
    try:
        df = prepare(pd.read_csv(f))
        if df.empty:
            raise ValueError("No usable rows found in the file.")
        flag, proba = predict_frame(df, model_key)
        log("upload", model_key, flag, proba)
    except Exception as e:
        flash(f"Could not process file: {e}", "error")
        return redirect(url_for("predict"))

    out = df.copy()
    out["prediction"] = np.where(flag == 1, "Suspicious", "Normal")
    out["probability_%"] = (proba * 100).round(1)
    out["risk"] = [risk_level(p) for p in proba]
    out = out.sort_values("probability_%", ascending=False)
    batch = {
        "total": len(out), "suspicious": int(flag.sum()), "normal": int((1 - flag).sum()),
        "model": MODEL_NAMES[model_key],
        "rows": out.head(100).round(2).to_dict("records"),
        "columns": list(out.columns),
        "csv": out.round(3).to_csv(index=False),
    }
    session["batch"] = batch
    return redirect(url_for("dashboard"))


@app.route("/download", methods=["POST"])
def download():
    return Response(request.form["csv"], mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.route("/sample.csv")
def sample():
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(10):
        avg = rng.uniform(150, 450); prev = avg * rng.uniform(0.9, 1.1)
        rows.append([round(prev), round(prev * rng.uniform(0.9, 1.1)), round(avg)])
    for _ in range(5):   # drop in use
        avg = rng.uniform(150, 450); prev = avg * rng.uniform(0.9, 1.1)
        rows.append([round(prev), round(prev * rng.uniform(0.1, 0.4)), round(avg)])
    df = pd.DataFrame(rows, columns=["prev_month", "curr_month", "avg_consumption"])
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_input.csv"})


@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB) as c:
        total, susp = c.execute("SELECT COUNT(*), COALESCE(SUM(suspicious),0) FROM predictions").fetchone()
    live = {"total": total, "suspicious": int(susp), "normal": total - int(susp)}
    batch = session.pop("batch", None)
    return render_template("dashboard.html", meta=META, names=MODEL_NAMES, live=live, batch=batch)


init_db()

if __name__ == "__main__":
    app.run(debug=True)
