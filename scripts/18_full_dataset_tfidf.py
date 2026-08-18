from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

REPORT_DIR = ROOT / "reports" / "full_dataset" / "tfidf"


def text_for_mode(df, mode):
    if mode == "comment_only":
        return df["comment"].astype(str)
    if mode == "context_only":
        return df["context"].astype(str)
    if mode == "context_plus_comment":
        return "Context: " + df["context"].astype(str) + " [SEP] Reply: " + df["comment"].astype(str)
    raise ValueError(mode)


def summarize(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "sarcastic_f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(
            max_iter=100,
            tol=1e-3,
            class_weight="balanced",
            solver="saga",
            random_state=42,
        )),
    ])


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_or_create_splits()

    print("Full-data split sizes:", len(train_df), len(val_df), len(test_df))
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    rows = []

    for mode in ["comment_only", "context_only", "context_plus_comment"]:
        print("\nTraining:", mode)
        model = make_model()
        model.fit(text_for_mode(train_df, mode), y_train)

        for split_name, df, y in [
            ("validation", val_df, y_val),
            ("test", test_df, y_test),
        ]:
            pred = model.predict(text_for_mode(df, mode))
            row = {
                "model": "TF-IDF + Logistic Regression",
                "input": mode,
                "split": split_name,
                "n_examples": len(df),
                **summarize(y, pred),
            }
            rows.append(row)
            print(row)

            if split_name == "test":
                out = df[["context", "comment", "label"]].copy()
                out["prediction"] = pred
                out.to_parquet(REPORT_DIR / f"{mode}_test_predictions.parquet", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT_DIR / "full_dataset_tfidf_metrics.csv", index=False)
    print("\nFINAL")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
