from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "behavioral_context_interpretability"
QWEN_DIR = ROOT / "reports" / "full_dataset" / "qwen"
QWEN_ABLATION_DIR = ROOT / "reports" / "full_dataset" / "qwen_context_ablation"
EMBED_CACHE = ROOT / ".cache" / "sarcasm_context_project" / "full_dataset_embeddings"

IRRELEVANT_THRESHOLD = 0.05
SENSITIVE_THRESHOLD = 0.25
EMBED_DIM = 384


def load_inputs():
    comment_path = QWEN_DIR / "comment_only_test_predictions.parquet"
    true_path = QWEN_DIR / "context_plus_comment_test_predictions.parquet"
    ablation_path = QWEN_ABLATION_DIR / "qwen_context_ablation_predictions.parquet"

    missing = [str(p) for p in [comment_path, true_path, ablation_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Qwen probability outputs are required. Missing:\n" + "\n".join(missing)
        )

    comment = pd.read_parquet(comment_path)
    true = pd.read_parquet(true_path)
    ablation = pd.read_parquet(ablation_path)

    if not (len(comment) == len(true) == len(ablation)):
        raise ValueError("Prediction files do not have identical test-set lengths.")

    for col in ["comment", "context", "label"]:
        if not comment[col].reset_index(drop=True).equals(true[col].reset_index(drop=True)):
            raise ValueError(f"Row mismatch between Qwen comment/true files for {col}")
        if not comment[col].reset_index(drop=True).equals(ablation[col].reset_index(drop=True)):
            raise ValueError(f"Row mismatch between Qwen outputs and context ablation for {col}")

    return comment.reset_index(drop=True), true.reset_index(drop=True), ablation.reset_index(drop=True)


def semantic_similarity(n):
    context_path = EMBED_CACHE / "test_context.f32"
    comment_path = EMBED_CACHE / "test_comment.f32"
    if not context_path.exists() or not comment_path.exists():
        return np.full(n, np.nan, dtype=np.float32)

    context = np.memmap(context_path, dtype="float32", mode="r", shape=(n, EMBED_DIM))
    comment = np.memmap(comment_path, dtype="float32", mode="r", shape=(n, EMBED_DIM))
    # Cached MiniLM embeddings are normalized, so row-wise dot product is cosine similarity.
    return np.einsum("ij,ij->i", context, comment).astype(np.float32)


def assign_category(df):
    comment_correct = df["prediction_comment"] == df["label"]
    true_correct = df["prediction_true"] == df["label"]
    max_context_effect = np.maximum(
        np.abs(df["delta_prob_true_minus_comment"]),
        np.abs(df["delta_prob_true_minus_random"]),
    )

    category = np.full(len(df), "other", dtype=object)
    category[(~comment_correct) & true_correct] = "context_helped"
    category[comment_correct & (~true_correct)] = "context_hurt"
    category[max_context_effect < IRRELEVANT_THRESHOLD] = "context_irrelevant"
    category[(max_context_effect >= SENSITIVE_THRESHOLD) & (category == "other")] = "context_sensitive"
    return category


def has_s_marker(text):
    text = str(text).lower()
    return int("/s" in text or "sarcasm" in text or "sarcastic" in text)


def build_analysis(comment, true, ablation):
    out = comment[[c for c in ["subreddit", "context", "comment", "label"] if c in comment.columns]].copy()
    out["prediction_comment"] = comment["prediction"].to_numpy()
    out["prob_comment"] = comment["prob_sarcastic"].to_numpy()
    out["prediction_true"] = true["prediction"].to_numpy()
    out["prob_true"] = true["prob_sarcastic"].to_numpy()
    out["prediction_random"] = ablation["prediction_random_context"].to_numpy()
    out["prob_random"] = ablation["prob_sarcastic_random_context"].to_numpy()
    out["prediction_same_subreddit"] = ablation["prediction_same_subreddit_wrong_context"].to_numpy()
    out["prob_same_subreddit"] = ablation["prob_sarcastic_same_subreddit_wrong_context"].to_numpy()

    out["delta_prob_true_minus_comment"] = out["prob_true"] - out["prob_comment"]
    out["delta_prob_true_minus_random"] = out["prob_true"] - out["prob_random"]
    out["delta_prob_true_minus_same_subreddit"] = out["prob_true"] - out["prob_same_subreddit"]
    out["abs_context_effect"] = np.abs(out["delta_prob_true_minus_comment"])
    out["abs_wrong_context_effect"] = np.abs(out["delta_prob_true_minus_random"])

    out["comment_length_chars"] = out["comment"].astype(str).str.len()
    out["context_length_chars"] = out["context"].astype(str).str.len()
    out["comment_words"] = out["comment"].astype(str).str.split().str.len()
    out["context_words"] = out["context"].astype(str).str.split().str.len()
    out["sarcasm_marker"] = out["comment"].map(has_s_marker)
    out["semantic_similarity_context_comment"] = semantic_similarity(len(out))
    out["behavior_category"] = assign_category(out)
    return out


def category_summary(df):
    summary = (
        df.groupby("behavior_category")
        .agg(
            n=("label", "size"),
            sarcastic_rate=("label", "mean"),
            mean_comment_words=("comment_words", "mean"),
            mean_context_words=("context_words", "mean"),
            mean_semantic_similarity=("semantic_similarity_context_comment", "mean"),
            sarcasm_marker_rate=("sarcasm_marker", "mean"),
            mean_abs_comment_to_true=("abs_context_effect", "mean"),
            mean_abs_true_to_random=("abs_wrong_context_effect", "mean"),
        )
        .reset_index()
    )
    summary["share"] = summary["n"] / len(df)
    return summary


def subreddit_summary(df, min_examples=100):
    if "subreddit" not in df.columns:
        return pd.DataFrame()
    group = (
        df.groupby("subreddit")
        .agg(
            n=("label", "size"),
            mean_abs_context_effect=("abs_context_effect", "mean"),
            mean_abs_wrong_context_effect=("abs_wrong_context_effect", "mean"),
            context_helped_rate=("behavior_category", lambda s: float((s == "context_helped").mean())),
            context_hurt_rate=("behavior_category", lambda s: float((s == "context_hurt").mean())),
        )
        .reset_index()
    )
    return group[group["n"] >= min_examples].sort_values("mean_abs_context_effect", ascending=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comment, true, ablation = load_inputs()
    analysis = build_analysis(comment, true, ablation)
    summary = category_summary(analysis)
    subreddits = subreddit_summary(analysis)

    analysis.to_parquet(OUT_DIR / "behavioral_context_per_example.parquet", index=False)
    summary.to_csv(OUT_DIR / "behavioral_context_category_summary.csv", index=False)
    subreddits.to_csv(OUT_DIR / "behavioral_context_by_subreddit.csv", index=False)

    top_sensitive = analysis.sort_values(
        ["abs_wrong_context_effect", "abs_context_effect"], ascending=False
    ).head(100)
    top_sensitive.to_csv(OUT_DIR / "top_100_context_sensitive_examples.csv", index=False)

    print("\nWHEN DOES CONTEXT MATTER?")
    print(summary.to_string(index=False))
    if len(subreddits):
        print("\nTOP SUBREDDITS BY CONTEXT EFFECT (min 100 examples)")
        print(subreddits.head(20).to_string(index=False))
    print("\nSaved to:", OUT_DIR)


if __name__ == "__main__":
    main()
