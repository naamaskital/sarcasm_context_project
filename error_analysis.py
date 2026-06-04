import pandas as pd
from pathlib import Path

INPUT_PATH = Path("reports_backup/prediction_comparison.csv")
OUTPUT_DIR = Path("reports_backup/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Context helped: prediction without context was wrong, prediction with context was correct
context_helped = df[
    (df["prediction_without_context"] != df["label"]) &
    (df["prediction_with_context"] == df["label"])
].copy()

# Context hurt: prediction without context was correct, prediction with context was wrong
context_hurt = df[
    (df["prediction_without_context"] == df["label"]) &
    (df["prediction_with_context"] != df["label"])
].copy()

# Prediction changed, but both were either correct or wrong
changed_same_quality = df[
    (df["changed_prediction"] == True) &
    ~df.index.isin(context_helped.index) &
    ~df.index.isin(context_hurt.index)
].copy()

context_helped.to_csv(OUTPUT_DIR / "context_helped.csv", index=False)
context_hurt.to_csv(OUTPUT_DIR / "context_hurt.csv", index=False)
changed_same_quality.to_csv(OUTPUT_DIR / "changed_same_quality.csv", index=False)

summary = f"""
Error Analysis Summary
======================

Total examples: {len(df)}
Changed predictions: {df["changed_prediction"].sum()}

Context helped: {len(context_helped)}
Context hurt: {len(context_hurt)}
Changed but same quality: {len(changed_same_quality)}

Accuracy without context: {(df["prediction_without_context"] == df["label"]).mean():.4f}
Accuracy with context: {(df["prediction_with_context"] == df["label"]).mean():.4f}
"""

(OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

print(summary)
