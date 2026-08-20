from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "failure_driven" / "selective_context_routing"

MODEL_CONFIGS = {
    "tfidf": {
        "dir": ROOT / "reports" / "full_dataset" / "tfidf",
        "comment": "comment_only",
        "context": "context_plus_comment",
    },
    "minilm": {
        "dir": ROOT / "reports" / "full_dataset" / "embeddings",
        "comment": "comment_only",
        "context": "dual_embeddings",
    },
}

# If |P(sarcastic)-0.5| is below the threshold, the comment-only model is treated as uncertain
# and the context-aware prediction is used instead.
MARGINS = np.round(np.linspace(0.00, 0.45, 19), 3)


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "sarcastic_f1": float(f1_score(y, pred, pos_label=1)),
    }


def read_pair(cfg, split):
    comment = pd.read_parquet(cfg["dir"] / f"{cfg['comment']}_{split}_predictions.parquet")
    context = pd.read_parquet(cfg["dir"] / f"{cfg['context']}_{split}_predictions.parquet")

    if len(comment) != len(context):
        raise ValueError("Prediction length mismatch")
    for col in ["context", "comment", "label"]:
        if not comment[col].reset_index(drop=True).equals(context[col].reset_index(drop=True)):
            raise ValueError(f"Row alignment mismatch in column {col}")
    return comment, context


def route(comment_df, context_df, margin):
    p = comment_df["prob_sarcastic"].to_numpy()
    uncertain = np.abs(p - 0.5) <= margin
    pred = comment_df["prediction"].to_numpy().copy()
    pred[uncertain] = context_df.loc[uncertain, "prediction"].to_numpy()
    return pred, uncertain


def evaluate_model(name, cfg):
    val_comment, val_context = read_pair(cfg, "validation")
    test_comment, test_context = read_pair(cfg, "test")

    y_val = val_comment["label"].to_numpy()
    y_test = test_comment["label"].to_numpy()

    search_rows = []
    for margin in MARGINS:
        pred, uncertain = route(val_comment, val_context, float(margin))
        search_rows.append({
            "model": name,
            "margin": float(margin),
            "routed_fraction": float(uncertain.mean()),
            **metrics(y_val, pred),
        })

    search_df = pd.DataFrame(search_rows)
    best = search_df.sort_values(["macro_f1", "margin"], ascending=[False, True]).iloc[0]
    best_margin = float(best["margin"])

    test_pred, routed = route(test_comment, test_context, best_margin)
    test_result = {
        "model": name,
        "selected_margin": best_margin,
        "routed_fraction": float(routed.mean()),
        **metrics(y_test, test_pred),
    }

    baseline_comment = metrics(y_test, test_comment["prediction"].to_numpy())
    baseline_context = metrics(y_test, test_context["prediction"].to_numpy())
    test_result.update({
        "comment_only_macro_f1": baseline_comment["macro_f1"],
        "context_model_macro_f1": baseline_context["macro_f1"],
        "delta_vs_comment": test_result["macro_f1"] - baseline_comment["macro_f1"],
        "delta_vs_context_model": test_result["macro_f1"] - baseline_context["macro_f1"],
    })

    pred_out = test_comment[[c for c in ["subreddit", "context", "comment", "label"] if c in test_comment.columns]].copy()
    pred_out["comment_prediction"] = test_comment["prediction"].to_numpy()
    pred_out["context_prediction"] = test_context["prediction"].to_numpy()
    pred_out["comment_prob_sarcastic"] = test_comment["prob_sarcastic"].to_numpy()
    pred_out["used_context"] = routed
    pred_out["routed_prediction"] = test_pred

    return search_df, test_result, pred_out


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    print("Failure-driven experiment: selective context routing")
    print("Hypothesis: context is most useful when the reply-only classifier is uncertain.")
    print("The uncertainty margin is selected on validation only; test is used once for final evaluation.\n")

    for name, cfg in MODEL_CONFIGS.items():
        print("=" * 70)
        print(name)
        print("=" * 70)
        try:
            search_df, test_result, pred_out = evaluate_model(name, cfg)
        except FileNotFoundError as exc:
            print("Skipping because validation/test probability files are missing:", exc)
            continue

        search_df.to_csv(REPORT_DIR / f"{name}_validation_margin_search.csv", index=False)
        pred_out.to_parquet(REPORT_DIR / f"{name}_test_predictions.parquet", index=False)
        all_results.append(test_result)
        print(test_result)

    results = pd.DataFrame(all_results)
    results.to_csv(REPORT_DIR / "selective_context_routing_test_metrics.csv", index=False)

    print("\nFINAL SELECTIVE CONTEXT ROUTING")
    if len(results):
        print(results.to_string(index=False))
    else:
        print("No model could be evaluated. Re-run scripts 18 and 19 first to create validation probabilities.")


if __name__ == "__main__":
    main()
