from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "qualitative_analysis"
SEED = 42
DEFAULT_PER_CATEGORY = 5

TFIDF_DIR = ROOT / "reports" / "full_dataset" / "tfidf"
EMBED_DIR = ROOT / "reports" / "full_dataset" / "embeddings"
QWEN_DIR = ROOT / "reports" / "full_dataset" / "qwen"
QWEN_ABLATION_DIR = ROOT / "reports" / "full_dataset" / "qwen_context_ablation"
HARD_DIR = ROOT / "reports" / "hard_context_ablation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a reproducible qualitative analysis from saved test predictions."
    )
    parser.add_argument("--examples-per-category", type=int, default=DEFAULT_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_prediction(path):
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def assert_same_rows(left, right, left_name, right_name):
    if len(left) != len(right):
        raise ValueError(f"Length mismatch: {left_name}={len(left)}, {right_name}={len(right)}")

    for col in ["label", "comment", "context"]:
        if col not in left.columns or col not in right.columns:
            raise ValueError(f"Missing {col} while aligning {left_name} and {right_name}")
        if not left[col].reset_index(drop=True).equals(right[col].reset_index(drop=True)):
            raise ValueError(
                f"Row alignment failed for {col}: {left_name} and {right_name} are not the same test set/order"
            )


def clean_for_examples(df):
    out = df.copy()
    out["comment"] = out["comment"].fillna("").astype(str)
    out["context"] = out["context"].fillna("").astype(str)

    bad_values = {"[deleted]", "[removed]", "deleted", "removed", "nan", "none", ""}
    comment_clean = out["comment"].str.strip().str.lower()
    context_clean = out["context"].str.strip().str.lower()

    mask = (
        ~comment_clean.isin(bad_values)
        & ~context_clean.isin(bad_values)
        & out["comment"].str.len().between(8, 500)
        & out["context"].str.len().between(8, 700)
    )
    return out[mask].copy()


def deterministic_sample(df, n, seed):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def add_category(selected_parts, counts, source_df, mask, category, model, n, seed, extra_cols=None):
    subset = source_df[mask].copy()
    counts.append({"category": category, "model": model, "count": int(len(subset))})
    if subset.empty:
        return

    subset = clean_for_examples(subset)
    sampled = deterministic_sample(subset, n, seed + len(counts))
    sampled.insert(0, "category", category)
    sampled.insert(1, "model", model)

    keep = ["category", "model", "context", "comment", "label"]
    if extra_cols:
        keep += [c for c in extra_cols if c in sampled.columns and c not in keep]
    selected_parts.append(sampled[keep])


def full_model_analysis(n, seed):
    counts = []
    selected = []
    merged = None

    # TF-IDF: comment-only vs naive context concatenation.
    tf_comment = read_prediction(TFIDF_DIR / "comment_only_test_predictions.parquet")
    tf_context = read_prediction(TFIDF_DIR / "context_plus_comment_test_predictions.parquet")
    if tf_comment is not None and tf_context is not None:
        assert_same_rows(tf_comment, tf_context, "TF-IDF comment", "TF-IDF context+comment")
        tf = tf_comment[["context", "comment", "label"]].copy()
        tf["tfidf_comment_pred"] = tf_comment["prediction"].to_numpy()
        tf["tfidf_context_pred"] = tf_context["prediction"].to_numpy()
        tf["tfidf_comment_correct"] = tf["tfidf_comment_pred"] == tf["label"]
        tf["tfidf_context_correct"] = tf["tfidf_context_pred"] == tf["label"]

        add_category(
            selected, counts, tf,
            (~tf["tfidf_comment_correct"]) & tf["tfidf_context_correct"],
            "tfidf_context_helped", "TF-IDF", n, seed,
            ["tfidf_comment_pred", "tfidf_context_pred"],
        )
        add_category(
            selected, counts, tf,
            tf["tfidf_comment_correct"] & (~tf["tfidf_context_correct"]),
            "tfidf_context_hurt", "TF-IDF", n, seed,
            ["tfidf_comment_pred", "tfidf_context_pred"],
        )
        merged = tf
    else:
        print("Skipping TF-IDF qualitative analysis: prediction files not found.")

    # MiniLM: comment embedding vs separate context/reply embeddings.
    emb_comment = read_prediction(EMBED_DIR / "comment_only_test_predictions.parquet")
    emb_dual = read_prediction(EMBED_DIR / "dual_embeddings_test_predictions.parquet")
    if emb_comment is not None and emb_dual is not None:
        assert_same_rows(emb_comment, emb_dual, "MiniLM comment", "MiniLM dual")
        emb = emb_comment[["context", "comment", "label"]].copy()
        emb["minilm_comment_pred"] = emb_comment["prediction"].to_numpy()
        emb["minilm_dual_pred"] = emb_dual["prediction"].to_numpy()
        emb["minilm_comment_correct"] = emb["minilm_comment_pred"] == emb["label"]
        emb["minilm_dual_correct"] = emb["minilm_dual_pred"] == emb["label"]

        add_category(
            selected, counts, emb,
            (~emb["minilm_comment_correct"]) & emb["minilm_dual_correct"],
            "minilm_context_helped", "MiniLM", n, seed,
            ["minilm_comment_pred", "minilm_dual_pred"],
        )
        add_category(
            selected, counts, emb,
            emb["minilm_comment_correct"] & (~emb["minilm_dual_correct"]),
            "minilm_context_hurt", "MiniLM", n, seed,
            ["minilm_comment_pred", "minilm_dual_pred"],
        )

        if merged is not None:
            assert_same_rows(merged, emb, "TF-IDF", "MiniLM")
            cross = merged.copy()
            for col in [
                "minilm_comment_pred", "minilm_dual_pred",
                "minilm_comment_correct", "minilm_dual_correct",
            ]:
                cross[col] = emb[col].to_numpy()

            # Most illustrative representation-design case:
            # naive TF-IDF context hurts while separate semantic context helps on the same example.
            mask = (
                cross["tfidf_comment_correct"]
                & (~cross["tfidf_context_correct"])
                & (~cross["minilm_comment_correct"])
                & cross["minilm_dual_correct"]
            )
            add_category(
                selected, counts, cross, mask,
                "representation_matters_tfidf_hurt_minilm_helped",
                "TF-IDF vs MiniLM", n, seed,
                [
                    "tfidf_comment_pred", "tfidf_context_pred",
                    "minilm_comment_pred", "minilm_dual_pred",
                ],
            )
    else:
        print("Skipping MiniLM qualitative analysis: prediction files not found.")

    # Qwen full-data comparison becomes available after GPU training.
    q_comment = read_prediction(QWEN_DIR / "comment_only_test_predictions.parquet")
    q_context = read_prediction(QWEN_DIR / "context_plus_comment_test_predictions.parquet")
    if q_comment is not None and q_context is not None:
        assert_same_rows(q_comment, q_context, "Qwen comment", "Qwen context+comment")
        q = q_comment[["context", "comment", "label"]].copy()
        q["qwen_comment_pred"] = q_comment["prediction"].to_numpy()
        q["qwen_context_pred"] = q_context["prediction"].to_numpy()
        q["qwen_comment_correct"] = q["qwen_comment_pred"] == q["label"]
        q["qwen_context_correct"] = q["qwen_context_pred"] == q["label"]

        add_category(
            selected, counts, q,
            (~q["qwen_comment_correct"]) & q["qwen_context_correct"],
            "qwen_context_helped", "Qwen + LoRA", n, seed,
            ["qwen_comment_pred", "qwen_context_pred"],
        )
        add_category(
            selected, counts, q,
            q["qwen_comment_correct"] & (~q["qwen_context_correct"]),
            "qwen_context_hurt", "Qwen + LoRA", n, seed,
            ["qwen_comment_pred", "qwen_context_pred"],
        )
    else:
        print("Qwen full-data predictions not present yet; Qwen helped/hurt categories will be added after GPU run.")

    return counts, selected


def hard_negative_analysis(n, seed, counts, selected):
    # Focused MiniLM hard-negative experiment, including semantic hard negatives.
    hard = read_prediction(HARD_DIR / "hard_context_ablation_predictions.csv")
    if hard is not None:
        required = [
            "prediction_true_context",
            "prediction_random_context",
            "prediction_same_subreddit_wrong_context",
            "prediction_semantic_similar_wrong_context",
        ]
        if all(c in hard.columns for c in required):
            true_correct = hard["prediction_true_context"] == hard["label"]
            for name, col, context_col in [
                ("random", "prediction_random_context", "random_context"),
                ("same_subreddit", "prediction_same_subreddit_wrong_context", "same_subreddit_wrong_context"),
                ("semantic_similar", "prediction_semantic_similar_wrong_context", "semantic_similar_wrong_context"),
            ]:
                alt_correct = hard[col] == hard["label"]
                temp = hard.copy()
                if context_col in temp.columns:
                    temp["wrong_context"] = temp[context_col]
                add_category(
                    selected, counts, temp,
                    true_correct & (~alt_correct),
                    f"hard_negative_{name}_true_to_wrong",
                    "MiniLM hard-context diagnostic", n, seed,
                    [
                        "wrong_context", "prediction_true_context", col,
                        "semantic_context_cosine_similarity",
                    ],
                )

            # Cases where a semantically similar *wrong* context leaves the decision unchanged.
            same_prediction = (
                hard["prediction_true_context"]
                == hard["prediction_semantic_similar_wrong_context"]
            )
            temp = hard.copy()
            temp["wrong_context"] = temp.get("semantic_similar_wrong_context", "")
            add_category(
                selected, counts, temp,
                same_prediction & true_correct,
                "semantic_wrong_context_prediction_unchanged",
                "MiniLM hard-context diagnostic", n, seed,
                [
                    "wrong_context",
                    "prediction_true_context",
                    "prediction_semantic_similar_wrong_context",
                    "semantic_context_cosine_similarity",
                ],
            )
        else:
            print("Hard-context prediction file exists but required columns are missing; skipping it.")
    else:
        print("Focused hard-context prediction file not found; skipping semantic hard-negative examples.")

    # Full-data Qwen perturbation analysis becomes available after GPU evaluation.
    qhard = read_prediction(QWEN_ABLATION_DIR / "qwen_context_ablation_predictions.parquet")
    if qhard is not None:
        true_correct = qhard["prediction_true_context"] == qhard["label"]
        for name, col, context_col in [
            ("random", "prediction_random_context", "random_context"),
            ("same_subreddit", "prediction_same_subreddit_wrong_context", "same_subreddit_wrong_context"),
        ]:
            alt_correct = qhard[col] == qhard["label"]
            temp = qhard.copy()
            temp["wrong_context"] = temp[context_col]
            add_category(
                selected, counts, temp,
                true_correct & (~alt_correct),
                f"qwen_hard_negative_{name}_true_to_wrong",
                "Qwen + LoRA hard-context", n, seed,
                ["wrong_context", "prediction_true_context", col],
            )
    else:
        print("Qwen context-ablation predictions not present yet; they will be included after GPU run.")


def write_outputs(counts, selected):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(OUT_DIR / "qualitative_category_counts.csv", index=False)

    if selected:
        examples_df = pd.concat(selected, ignore_index=True, sort=False)
    else:
        examples_df = pd.DataFrame(columns=["category", "model", "context", "comment", "label"])
    examples_df.to_csv(OUT_DIR / "selected_qualitative_examples.csv", index=False)

    lines = [
        "Qualitative Error Analysis",
        "=" * 70,
        "",
        "Examples are selected by deterministic, pre-defined outcome categories rather than manual cherry-picking.",
        "Text-length and deleted/removed-content filters are applied only to improve readability.",
        "",
        "Category counts:",
        counts_df.to_string(index=False) if len(counts_df) else "No categories available.",
        "",
        f"Selected examples saved: {len(examples_df)}",
    ]
    (OUT_DIR / "qualitative_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nCATEGORY COUNTS")
    print(counts_df.to_string(index=False) if len(counts_df) else "No categories available.")
    print("\nSelected examples:", len(examples_df))
    print("Saved to:", OUT_DIR)


def main():
    args = parse_args()
    counts, selected = full_model_analysis(args.examples_per_category, args.seed)
    hard_negative_analysis(args.examples_per_category, args.seed, counts, selected)
    write_outputs(counts, selected)


if __name__ == "__main__":
    main()
