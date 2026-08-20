from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

SEED = 42
REPORT_DIR = ROOT / "reports" / "failure_driven" / "field_aware_tfidf"
CONTEXT_WEIGHTS = [0.0, 0.25, 0.5, 1.0]


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "sarcastic_f1": float(f1_score(y, pred, pos_label=1)),
    }


def build_vectorizer(max_features):
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_features(comment_x, context_x, context_weight):
    if context_weight == 0.0:
        return comment_x
    return hstack([comment_x, context_x * context_weight], format="csr")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_or_create_splits()

    print("Failure-driven experiment: field-aware TF-IDF")
    print("Hypothesis: naive concatenation hurts because parent-context tokens dilute reply features.")
    print("Fix: encode the two fields separately and tune only the context weight on validation.")

    comment_vec = build_vectorizer(100000)
    context_vec = build_vectorizer(60000)

    train_comment = comment_vec.fit_transform(train_df["comment"].astype(str))
    val_comment = comment_vec.transform(val_df["comment"].astype(str))
    test_comment = comment_vec.transform(test_df["comment"].astype(str))

    train_context = context_vec.fit_transform(train_df["context"].astype(str))
    val_context = context_vec.transform(val_df["context"].astype(str))
    test_context = context_vec.transform(test_df["context"].astype(str))

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    validation_rows = []
    trained = {}

    for weight in CONTEXT_WEIGHTS:
        print(f"\nTraining context weight={weight}")
        clf = LogisticRegression(
            max_iter=100,
            tol=1e-3,
            class_weight="balanced",
            solver="saga",
            random_state=SEED,
        )
        clf.fit(build_features(train_comment, train_context, weight), y_train)
        pred = clf.predict(build_features(val_comment, val_context, weight))
        row = {"context_weight": weight, **metrics(y_val, pred)}
        validation_rows.append(row)
        trained[weight] = clf
        print(row)

    validation_df = pd.DataFrame(validation_rows)
    best_weight = float(validation_df.sort_values("macro_f1", ascending=False).iloc[0]["context_weight"])
    best_clf = trained[best_weight]

    print("\nSelected on validation: context_weight =", best_weight)
    test_x = build_features(test_comment, test_context, best_weight)
    test_pred = best_clf.predict(test_x)
    test_prob = best_clf.predict_proba(test_x)[:, 1].astype(np.float32)
    test_result = {"context_weight": best_weight, **metrics(y_test, test_pred)}

    validation_df.to_csv(REPORT_DIR / "validation_context_weight_search.csv", index=False)
    pd.DataFrame([test_result]).to_csv(REPORT_DIR / "field_aware_tfidf_test_metrics.csv", index=False)

    keep = [c for c in ["subreddit", "context", "comment", "label"] if c in test_df.columns]
    out = test_df[keep].copy()
    out["prediction"] = test_pred
    out["prob_sarcastic"] = test_prob
    out.to_parquet(REPORT_DIR / "field_aware_tfidf_test_predictions.parquet", index=False)

    print("\nFINAL FIELD-AWARE TF-IDF")
    print(pd.DataFrame([test_result]).to_string(index=False))
    print("Saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
