"""Train and compare four classifiers, then save models, metrics and charts.

Run:  python train.py
Uses data/data.csv (real SGCC) if present, otherwise synthetic data.
"""
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_generator import generate
from features import FEATURES, from_daily

SEED = 42
REAL_DATA = "data/data.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

# ---------------------------------------------------------------- 1. load data
if os.path.exists(REAL_DATA):
    raw = pd.read_csv(REAL_DATA)
    source = "SGCC dataset (data/data.csv)"
    raw = raw.drop_duplicates()
else:
    raw = generate()
    source = "SYNTHETIC data (SGCC file not found)"
print("Data source:", source)

label = raw["FLAG"].astype(int)
daily = raw.drop(columns=[c for c in ["CONS_NO", "FLAG"] if c in raw.columns])
if not os.path.exists(REAL_DATA):
    pass
else:
    # SGCC columns are dates; make sure they are in chronological order
    try:
        daily = daily[sorted(daily.columns, key=pd.to_datetime)]
    except Exception:
        pass

X = from_daily(daily)
y = label.values
print(f"Records: {len(X)} | theft: {y.sum()} ({y.mean():.1%})")

# monthly consumption profile (for the dashboard chart)
arr = daily.apply(pd.to_numeric, errors="coerce").interpolate(axis=1, limit_direction="both").fillna(0).values
n_months = arr.shape[1] // 30
monthly = arr[:, -n_months * 30:].reshape(len(arr), n_months, 30).sum(axis=2)

# ---------------------------------------------------------------- 2. split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2, class_weight="balanced", probability=True, random_state=SEED)),
}
NAMES = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "svm": "SVM"}

# ---------------------------------------------------------------- 3. train/eval
results, cms = {}, {}
for key, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    results[key] = {
        "name": NAMES[key],
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }
    cms[key] = confusion_matrix(y_te, pred)
    joblib.dump(model, f"models/{key}.joblib")
    print(f"{NAMES[key]:<20} " + " ".join(f"{m}={results[key][m]:.3f}" for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]))

# F1 balances precision/recall on imbalanced data; ROC-AUC breaks ties
best = max(results, key=lambda k: (round(results[k]["f1"], 3), results[k]["roc_auc"]))
print("Best model by F1:", NAMES[best])

meta = {
    "data_source": source,
    "n_records": int(len(X)),
    "n_theft": int(y.sum()),
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
    "best_model": best,
    "features": FEATURES,
    "feature_medians": X.median().round(3).to_dict(),
    "results": results,
    "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
}
json.dump(meta, open("models/meta.json", "w"), indent=2)

# ---------------------------------------------------------------- 4. charts
C_NORMAL, C_THEFT, C_GREY = "#2563eb", "#dc2626", "#64748b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# 4a metric comparison
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics = ["accuracy", "precision", "recall", "f1"]
w = 0.2
for i, m in enumerate(metrics):
    vals = [results[k][m] for k in models]
    ax.bar(np.arange(len(models)) + i * w, vals, w, label=m.capitalize() if m != "f1" else "F1-score")
ax.set_xticks(np.arange(len(models)) + 1.5 * w)
ax.set_xticklabels([NAMES[k] for k in models])
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Algorithm comparison on the test set")
ax.legend(ncol=4, loc="lower center", frameon=False)
fig.tight_layout(); fig.savefig("static/charts/metrics_comparison.png", dpi=130); plt.close(fig)

# 4b confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (k, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Suspicious"]); ax.set_yticklabels(["Normal", "Suspicious"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(NAMES[k], fontsize=11)
    ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig("static/charts/confusion_matrices.png", dpi=130); plt.close(fig)

# 4c monthly pattern normal vs suspicious
fig, ax = plt.subplots(figsize=(9, 4.2))
months = np.arange(1, n_months + 1)
ax.plot(months, monthly[y == 0].mean(axis=0), color=C_NORMAL, lw=2.5, marker="o", label="Normal customers")
ax.plot(months, monthly[y == 1].mean(axis=0), color=C_THEFT, lw=2.5, marker="o", label="Theft customers")
ax.set_xlabel("Month (oldest -> newest)"); ax.set_ylabel("Average units per month")
ax.set_title("Average monthly consumption: normal vs theft"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("static/charts/consumption_pattern.png", dpi=130); plt.close(fig)

# 4d class balance + feature importance (random forest)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["Normal", "Theft"], [(y == 0).sum(), (y == 1).sum()], color=[C_NORMAL, C_THEFT])
axes[0].set_title("Class distribution (imbalanced)"); axes[0].set_ylabel("Customers")
imp = pd.Series(models["random_forest"].feature_importances_, index=FEATURES).sort_values()
axes[1].barh(imp.index, imp.values, color=C_GREY)
axes[1].set_title("Random Forest feature importance")
fig.tight_layout(); fig.savefig("static/charts/data_overview.png", dpi=130); plt.close(fig)

print("Saved models/, static/charts/")
