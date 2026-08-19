from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "full_dataset" / "final_results_full_dataset.csv"


def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    rows = []

    tfidf = load_csv(REPORTS / "full_dataset" / "tfidf" / "full_dataset_tfidf_metrics.csv")
    if len(tfidf):
        tfidf = tfidf[tfidf["split"] == "test"].copy()
        for _, r in tfidf.iterrows():
            rows.append({
                "model": r["model"],
                "input": r["input"],
                "n_test": int(r["n_examples"]),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "sarcastic_f1": r["sarcastic_f1"],
                "status": "complete",
            })

    emb = load_csv(REPORTS / "full_dataset" / "embeddings" / "full_dataset_embedding_metrics.csv")
    if len(emb):
        emb = emb[emb["split"] == "test"].copy()
        for _, r in emb.iterrows():
            rows.append({
                "model": r["model"],
                "input": r["input"],
                "n_test": int(r["n_examples"]),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "sarcastic_f1": r["sarcastic_f1"],
                "status": "complete",
            })

    qwen_path = REPORTS / "full_dataset" / "qwen" / "full_dataset_qwen_metrics.csv"
    if qwen_path.exists():
        qwen = pd.read_csv(qwen_path)
        for _, r in qwen.iterrows():
            rows.append({
                "model": r["model"],
                "input": r["input"],
                "n_test": int(r["test_examples"]),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "sarcastic_f1": r["sarcastic_f1"],
                "status": "complete",
            })
    else:
        for mode in ["comment_only", "context_plus_comment"]:
            rows.append({
                "model": "Qwen2.5-0.5B-Instruct + LoRA",
                "input": mode,
                "n_test": 101078,
                "accuracy": pd.NA,
                "macro_f1": pd.NA,
                "sarcastic_f1": pd.NA,
                "status": "pending_gpu",
            })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
