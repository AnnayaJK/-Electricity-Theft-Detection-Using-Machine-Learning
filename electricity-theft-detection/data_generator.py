"""Synthetic stand-in for the SGCC dataset (same shape: one row per customer,
one column per day, plus a FLAG column: 1 = theft, 0 = normal).

Used ONLY when data/data.csv (the real SGCC file) is not present, so the
project always runs. Replace it with the real dataset for your final report.
"""
import numpy as np
import pandas as pd


def generate(n_customers=2500, days=360, theft_ratio=0.12, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    series, flags = [], []

    for _ in range(n_customers):
        base = rng.lognormal(2.3, 0.6)                       # typical kWh/day
        season = 1 + 0.25 * np.sin(2 * np.pi * (t + rng.integers(0, 60)) / 365)
        weekly = 1 + 0.08 * np.sin(2 * np.pi * t / 7)
        noise = rng.normal(1, rng.uniform(0.08, 0.25), days)
        s = base * season * weekly * noise

        theft = rng.random() < theft_ratio
        if theft:
            kind = rng.integers(0, 4)
            start = rng.integers(days // 3, days - 60)
            if kind == 0:      # sudden permanent drop (meter tampering)
                s[start:] *= rng.uniform(0.15, 0.6)
            elif kind == 1:    # random zero-reading days (meter bypass)
                mask = rng.random(days) < rng.uniform(0.15, 0.4)
                mask[:start] = False
                s[mask] = 0
            elif kind == 2:    # erratic, reduced usage
                s[start:] *= rng.uniform(0.3, 0.7) * np.clip(rng.normal(1, 0.4, days - start), 0.2, 1.6)
            else:              # usage capped at a low ceiling
                s[start:] = np.minimum(s[start:], base * rng.uniform(0.3, 0.5))
        else:
            # honest customers also have odd patterns (holiday, outages)
            if rng.random() < 0.08:
                v = rng.integers(0, days - 30)
                s[v:v + rng.integers(15, 30)] *= 0.2
            if rng.random() < 0.05:
                s[rng.random(days) < 0.05] = 0

        s = np.clip(s, 0, None)
        s[rng.random(days) < 0.01] = np.nan                  # missing readings
        series.append(s)
        flags.append(int(theft))

    df = pd.DataFrame(series, columns=[f"day_{i+1}" for i in range(days)])
    df.insert(0, "CONS_NO", [f"C{i:05d}" for i in range(n_customers)])
    df["FLAG"] = flags
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate().to_csv("data/synthetic_sgcc.csv", index=False)
    print("Saved data/synthetic_sgcc.csv")
