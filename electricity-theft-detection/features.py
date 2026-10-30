"""Feature engineering shared by training and the web app.

The SGCC dataset stores one row per customer with ~1,000 daily readings.
The website, however, asks for simple monthly figures. This module turns a
daily series into the same six monthly-level features the website uses, so
the model trained offline and the model used online see identical inputs.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "prev_month",        # total units in the month before the latest month
    "curr_month",        # total units in the latest month
    "avg_consumption",   # average monthly units over the whole period
    "change_pct",        # % change from previous to current month
    "curr_to_avg",       # current month / average month (1.0 = typical)
    "cv",                # coefficient of variation (std / mean) of daily use
    "zero_days_pct",     # % of days with zero consumption
]

FEATURE_LABELS = {
    "prev_month": "Previous month consumption (units)",
    "curr_month": "Current month consumption (units)",
    "avg_consumption": "Average monthly consumption (units)",
    "change_pct": "Consumption change (%)",
    "curr_to_avg": "Current / average consumption",
    "cv": "Variability of daily use (std/mean)",
    "zero_days_pct": "Zero-consumption days (%)",
}


def change_percent(prev, curr):
    prev = np.asarray(prev, dtype=float)
    curr = np.asarray(curr, dtype=float)
    out = (curr - prev) / np.maximum(prev, 1e-6) * 100.0
    return np.clip(out, -100.0, 500.0)


def from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """daily: rows = customers, columns = consecutive days (oldest -> newest)."""
    x = daily.apply(pd.to_numeric, errors="coerce")
    raw = x.values.astype(float)
    zero_pct = np.nanmean(np.where(np.isnan(raw), np.nan, (raw == 0).astype(float)), axis=1) * 100

    x = x.interpolate(axis=1, limit_direction="both").fillna(0)
    arr = np.clip(x.values.astype(float), 0, None)

    curr = arr[:, -30:].sum(axis=1)
    prev = arr[:, -60:-30].sum(axis=1)
    avg = arr.mean(axis=1) * 30
    cv = arr.std(axis=1) / np.maximum(arr.mean(axis=1), 1e-6)

    return pd.DataFrame({
        "prev_month": prev,
        "curr_month": curr,
        "avg_consumption": avg,
        "change_pct": change_percent(prev, curr),
        "curr_to_avg": curr / np.maximum(avg, 1e-6),
        "cv": cv,
        "zero_days_pct": zero_pct,
    })
