from pathlib import Path
import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_NAME = "marcbishara/sarcasm-on-reddit"
DATASET_SPLIT = "sft_train"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_PER_CLASS = 10000
RANDOM_STATE = 42
BOOTSTRAP_RUNS = 1000
REPORT_DIR = Path("reports/hard_context_ablation")


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_data():
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    df = raw.to_pandas()

    context_col = "parent_comment" if "parent_comment" in df.columns else "context"
    needed = ["label", "comment", context_col]
    if "subreddit" in df.columns:
        needed.append("subreddit")

    df = df[needed].copy()
    df = df.rename(columns={context_col: "context"})
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"

    df["context"] = df["context"].map(clean_text)
    df["comment"] = df["comment"].map(clean_text)
    df["subreddit"] = df["subreddit"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[(df["context"] != "") & (df["comment"] != "")]

    parts = []
    for label in [0, 1]:
        class_df = df[df["label"] == label]
        n = min(TARGET_PER_CLASS, len(class_df))
        parts.append(class_df.sample(n=n, random_state=RANDOM_STATE + label))

    return (
        pd.concat(parts)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def encode(model, texts):
    return model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def random_wrong_indices(n, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    perm = rng.permutation(n)
    while np.any(perm == idx):
        perm = rng.permutation(n)
    return perm


def same_subreddit_wrong_indices(test_df, seed):
    rng = np.random.default_rng(seed)
    result = np.full(len(test_df), -1, dtype=int)

    for _, group in test_df.groupby("subreddit", sort=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        perm = rng.permutation(indices)
        if np.any(perm == indices):
            perm = np.roll(perm, 1)
        result[indices] = perm

    fallback = random_wrong_indices(len(test_df), seed + 1)
    missing = result < 0
    result[missing] = fallback[missing]
    return result


def semantic_wrong_indices(context_embeddings):
    similarities = context_embeddings @ context_embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    return np.argmax(similarities, axis=1)


def build_features(context_emb, comment_emb):
    return np.concatenate([context_emb, comment_emb], axis=1)


def make_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "sarcastic_f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def bootstrap_delta(y_true, reference_pred, alternative_pred):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    deltas = []

    for _ in range(BOOTSTRAP_RUNS):
        sample = rng.integers(0, n, size=n)
        ref = f1_score(y_true[sample], reference_pred[sample], average="macro")
        alt = f1_score(y_true[sample], alternative_pred[sample], average="macro")
        deltas.append(ref - alt)

    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(low), float(high)


def main():
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Split sizes:", len(train_df), len(test_df))
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding train contexts/comments...")
    train_context_emb = encode(model, train_df["context"])
    train_comment_emb = encode(model, train_df["comment"])

    print("Encoding test contexts/comments...")
    test_context_emb = encode(model, test_df["context"])
    test_comment_emb = encode(model, test_df["comment"])

    print("Building hard negative contexts...")
    random_idx = random_wrong_indices(len(test_df), RANDOM_STATE)
    subreddit_idx = same_subreddit_wrong_indices(test_df, RANDOM_STATE)
    semantic_idx = semantic_wrong_indices(test_context_emb)

    context_conditions = {
        "true_context": test_context_emb,
        "random_context": test_context_emb[random_idx],
        "same_subreddit_wrong_context": test_context_emb[subreddit_idx],
        "semantic_similar_wrong_context": test_context_emb[semantic_idx],
    }

    print("Training classifier on true context + comment...")
    classifier = make_classifier()
    classifier.fit(
        build_features(train_context_emb, train_comment_emb),
        train_df["label"].to_numpy(),
    )

    y_test = test_df["label"].to_numpy()
    rows = []
    predictions = test_df[["subreddit", "context", "comment", "label"]].copy()
    pred_by_condition = {}

    for name, context_emb in context_conditions.items():
        pred = classifier.predict(build_features(context_emb, test_comment_emb))
        pred_by_condition[name] = pred
        rows.append({"condition": name, **metrics(y_test, pred)})
        predictions[f"prediction_{name}"] = pred

    true_pred = pred_by_condition["true_context"]
    true_macro_f1 = metrics(y_test, true_pred)["macro_f1"]

    for row in rows:
        name = row["condition"]
        if name == "true_context":
            row["macro_f1_drop_vs_true"] = 0.0
            row["bootstrap_delta_mean"] = 0.0
            row["bootstrap_ci_low"] = 0.0
            row["bootstrap_ci_high"] = 0.0
            continue

        alt_pred = pred_by_condition[name]
        mean_delta, low, high = bootstrap_delta(y_test, true_pred, alt_pred)
        row["macro_f1_drop_vs_true"] = true_macro_f1 - row["macro_f1"]
        row["bootstrap_delta_mean"] = mean_delta
        row["bootstrap_ci_low"] = low
        row["bootstrap_ci_high"] = high

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORT_DIR / "hard_context_ablation_metrics.csv", index=False)

    predictions["random_context"] = test_df.iloc[random_idx]["context"].to_numpy()
    predictions["same_subreddit_wrong_context"] = test_df.iloc[subreddit_idx]["context"].to_numpy()
    predictions["semantic_similar_wrong_context"] = test_df.iloc[semantic_idx]["context"].to_numpy()
    predictions.to_csv(REPORT_DIR / "hard_context_ablation_predictions.csv", index=False)

    with open(REPORT_DIR / "hard_context_ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Hard Context Ablation\n")
        f.write("=" * 60 + "\n\n")
        f.write("The classifier is trained once using true context + comment.\n")
        f.write("At test time only, the context is replaced with increasingly difficult wrong contexts.\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        f.write("A positive bootstrap delta means true context has higher Macro-F1.\n")
        f.write("If the 95% CI stays above zero, the advantage is consistent under paired resampling.\n")

    print(metrics_df.to_string(index=False))
    print("Saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
