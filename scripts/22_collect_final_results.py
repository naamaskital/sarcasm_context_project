from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "full_dataset"
MAIN_OUT = OUT_DIR / "final_results_full_dataset.csv"
AUX_OUT = OUT_DIR / "final_auxiliary_results.csv"


def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def add_standard_rows(rows, df, model_col="model", input_col="input", n_col="n_examples"):
    if df.empty:
        return
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    for _, r in df.iterrows():
        rows.append({
            "model": r[model_col],
            "input": r[input_col],
            "n_test": int(r[n_col]) if pd.notna(r[n_col]) else pd.NA,
            "accuracy": r.get("accuracy", pd.NA),
            "macro_f1": r.get("macro_f1", pd.NA),
            "sarcastic_f1": r.get("sarcastic_f1", pd.NA),
            "status": "complete",
        })


def main():
    main_rows = []
    auxiliary = []

    add_standard_rows(
        main_rows,
        load_csv(OUT_DIR / "tfidf" / "full_dataset_tfidf_metrics.csv"),
    )
    add_standard_rows(
        main_rows,
        load_csv(OUT_DIR / "embeddings" / "full_dataset_embedding_metrics.csv"),
    )

    for family, filename in [
        ("bert_cross_encoder", "bert_cross_encoder_metrics.csv"),
        ("roberta_cross_encoder", "roberta_cross_encoder_metrics.csv"),
    ]:
        df = load_csv(OUT_DIR / family / filename)
        if not df.empty:
            for _, r in df.iterrows():
                main_rows.append({
                    "model": r["model"],
                    "input": r["input"],
                    "n_test": int(r["test_examples"]),
                    "accuracy": r["accuracy"],
                    "macro_f1": r["macro_f1"],
                    "sarcastic_f1": r["sarcastic_f1"],
                    "status": "complete",
                })
        else:
            model_name = "bert-base-uncased cross-encoder" if family.startswith("bert") else "roberta-base cross-encoder"
            for mode in ["comment_only", "context_only", "context_plus_comment"]:
                main_rows.append({
                    "model": model_name,
                    "input": mode,
                    "n_test": 101078,
                    "accuracy": pd.NA,
                    "macro_f1": pd.NA,
                    "sarcastic_f1": pd.NA,
                    "status": "pending_gpu",
                })

    qwen = load_csv(OUT_DIR / "qwen" / "full_dataset_qwen_metrics.csv")
    if not qwen.empty:
        for _, r in qwen.iterrows():
            main_rows.append({
                "model": r["model"],
                "input": r["input"],
                "n_test": int(r["test_examples"]),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "sarcastic_f1": r["sarcastic_f1"],
                "status": "complete",
            })
    else:
        for mode in ["comment_only", "context_only", "context_plus_comment"]:
            main_rows.append({
                "model": "Qwen2.5-0.5B-Instruct + LoRA",
                "input": mode,
                "n_test": 101078,
                "accuracy": pd.NA,
                "macro_f1": pd.NA,
                "sarcastic_f1": pd.NA,
                "status": "pending_gpu",
            })

    # Failure-driven follow-ups and research extensions are kept in a second table
    # because their protocols/metrics differ from the main architecture comparison.
    auxiliary_sources = [
        ("field_aware_tfidf", REPORTS / "full_dataset" / "field_aware_tfidf" / "field_aware_tfidf_metrics.csv"),
        ("selective_context_routing", REPORTS / "full_dataset" / "selective_context_routing" / "selective_context_routing_metrics.csv"),
        ("qwen_context_ablation", REPORTS / "full_dataset" / "qwen_context_ablation" / "qwen_context_ablation_metrics.csv"),
        ("qwen_scale_context_utilization", REPORTS / "full_dataset" / "qwen_basic_controls" / "qwen_context_utilization_by_scale.csv"),
        ("unseen_subreddit", REPORTS / "subreddit_generalization" / "unseen_subreddit_metrics.csv"),
        ("unseen_subreddit_context_gain", REPORTS / "subreddit_generalization" / "unseen_subreddit_context_gain.csv"),
        ("behavioral_interpretability", REPORTS / "full_dataset" / "behavioral_interpretability" / "context_behavior_summary.csv"),
    ]

    for experiment, path in auxiliary_sources:
        df = load_csv(path)
        if df.empty:
            auxiliary.append({"experiment": experiment, "status": "pending_or_not_run", "source": str(path.relative_to(ROOT))})
        else:
            for _, r in df.iterrows():
                row = {"experiment": experiment, "status": "complete", "source": str(path.relative_to(ROOT))}
                row.update(r.to_dict())
                auxiliary.append(row)

    main_df = pd.DataFrame(main_rows)
    aux_df = pd.DataFrame(auxiliary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(MAIN_OUT, index=False)
    aux_df.to_csv(AUX_OUT, index=False)

    print("\nMAIN ARCHITECTURE RESULTS")
    print(main_df.to_string(index=False))
    print("\nAUXILIARY / HYPOTHESIS-DRIVEN RESULTS")
    print(aux_df.to_string(index=False))
    print("\nSaved:", MAIN_OUT)
    print("Saved:", AUX_OUT)


if __name__ == "__main__":
    main()
