# Smart Electricity Theft Detection Using Machine Learning
*A Comparative Study of Classification Algorithms*

A Flask website backed by scikit-learn models that flags electricity consumption records as **Normal** or
**Potentially Suspicious**. A suspicious result is a lead for inspection, not proof of theft.

## Project structure
```
app.py              Flask website (home, predict, upload, dashboard)
train.py            trains + compares the 4 algorithms, saves models, metrics, charts
features.py         feature engineering (shared by training and website)
data_generator.py   synthetic data in SGCC format (used only if the real file is absent)
models/             saved models (*.joblib) and meta.json (metrics, medians)
static/             style.css and generated charts
templates/          HTML pages
```

## Run it
```bash
pip install -r requirements.txt
python train.py      # trains the models, creates models/ and static/charts/
python app.py        # open http://127.0.0.1:5000
```

## IMPORTANT: use the real SGCC dataset for your final submission
This project ships with **synthetic data** so it runs immediately. The reported scores in this folder
come from that synthetic data, so **do not present them as SGCC results**. To use the real data:

1. Download the SGCC dataset from the IEEE PES Data Sharing Platform (link in your brief).
2. Save the CSV as `data/data.csv`. It must have a `FLAG` column (1 = theft) and one column per day
   (it may also have `CONS_NO`; any other non-date columns should be removed).
3. Run `python train.py` again. It detects the file, prints "SGCC dataset", and rebuilds all models,
   metrics and charts. Real SGCC scores will be lower and more interesting than the synthetic ones.

## How it works
1. **Features** (`features.py`): from each customer's daily readings we compute previous-month units,
   current-month units, average monthly units, % change, current/average ratio, variability (std/mean)
   and % of zero-reading days.
2. **Training** (`train.py`): stratified 75/25 split; Logistic Regression, Decision Tree, Random Forest and
   SVM (RBF). Theft is a small minority, so every model uses `class_weight="balanced"`; LR and SVM are scaled.
3. **Selection**: best model by F1-score (ROC-AUC breaks ties).
4. **Website**: manual entry or CSV upload -> same feature pipeline -> model -> prediction, suspicion score
   and risk level. Optional fields (variability, zero days) are estimated when left blank.
5. **Dashboard**: live count of predictions made on the site, metrics table, charts.

## Suggested report outline
1. Introduction and problem statement  2. Literature / background (types of theft, SGCC dataset)
3. Methodology (features, algorithms, metrics)  4. Implementation (architecture, website)
5. Results and comparison (table + charts from the dashboard)  6. Limitations  7. Conclusion and future work

## Limitations (say these in your viva, they show understanding)
- Predictions flag patterns only; holidays, vacant houses or outages can look like theft (false positives).
- Accuracy is misleading on imbalanced data, so we focus on precision, recall and F1.
- The website uses monthly summary inputs; the offline model sees full daily history. Estimated optional
  inputs make single-record predictions less reliable than batch results on full data.
- Different algorithms can disagree on a single hand-entered record. That is normal, and it is why the
  comparison matters.

## Possible viva questions
- Why F1 instead of accuracy?  Why `class_weight="balanced"`?  Why scale features for SVM and LR but not trees?
- What does a false negative cost versus a false positive for a utility?
- How would you handle the missing values in the SGCC data?  (We interpolate along time.)
- Future work: LSTM / CNN on daily series, SMOTE, threshold tuning, user login, database of customers.
