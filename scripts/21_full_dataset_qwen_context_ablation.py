from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 128
BOOTSTRAP_RUNS = 1000
REPORT_DIR = ROOT / "reports" / "full_dataset" / "qwen_context_ablation"
ADAPTER_DIR = ROOT / "models" / "full_dataset_qwen" / "context_plus_comment" / "final_adapter"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-dataset Qwen evaluation.")
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Missing trained adapter at {ADAPTER_DIR}. Run scripts/20_full_dataset_qwen.py "
            "with --mode context_plus_comment first."
        )


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derangement(n, seed=SEED):
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    perm = rng.permutation(n)
    while np.any(perm == base):
        perm = rng.permutation(n)
    return perm


def same_subreddit_wrong_indices(test_df, seed=SEED):
    rng = np.random.default_rng(seed)
    result = np.full(len(test_df), -1, dtype=np.int64)
    matched = np.zeros(len(test_df), dtype=bool)

    if "subreddit" in test_df.columns:
        for subreddit, group in test_df.groupby("subreddit", sort=False):
            idx = group.index.to_numpy()
            if len(idx) < 2:
                continue
            shuffled = rng.permutation(idx)
            while np.any(shuffled == idx):
                shuffled = rng.permutation(idx)
            result[idx] = shuffled
            matched[idx] = True

    fallback = derangement(len(test_df), seed + 1)
    missing = result < 0
    result[missing] = fallback[missing]
    return result, matched


def build_perturbed_test(test_df):
    out = test_df.copy().reset_index(drop=True)

    random_idx = derangement(len(out), SEED)
    subreddit_idx, subreddit_matched = same_subreddit_wrong_indices(out, SEED)

    out["random_context"] = out.iloc[random_idx]["context"].to_numpy()
    out["same_subreddit_wrong_context"] = out.iloc[subreddit_idx]["context"].to_numpy()
    out["same_subreddit_is_matched"] = subreddit_matched
    return out


def build_text(row, context_column):
    return f"Context:\n{row[context_column]}\n\nReply:\n{row['comment']}"


def make_dataset(df, tokenizer, context_column):
    texts = [build_text(row, context_column) for _, row in df.iterrows()]
    ds = Dataset.from_dict({"text": texts, "label": df["label"].astype(int).tolist()})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def summarize(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "sarcastic_f1": float(f1_score(labels, preds, pos_label=1)),
    }


def bootstrap_delta(y_true, reference_pred, alternative_pred):
    rng = np.random.default_rng(SEED)
    deltas = np.empty(BOOTSTRAP_RUNS, dtype=np.float64)
    n = len(y_true)

    for i in range(BOOTSTRAP_RUNS):
        sample = rng.integers(0, n, size=n)
        ref = f1_score(y_true[sample], reference_pred[sample], average="macro")
        alt = f1_score(y_true[sample], alternative_pred[sample], average="macro")
        deltas[i] = ref - alt

    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(low), float(high)


def sensitivity(name, y_true, true_pred, alt_pred):
    changed = true_pred != alt_pred
    true_correct = true_pred == y_true
    alt_correct = alt_pred == y_true
    return {
        "condition": name,
        "changed_predictions": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "true_correct_to_wrong": int((changed & true_correct & ~alt_correct).sum()),
        "true_wrong_to_correct": int((changed & ~true_correct & alt_correct).sum()),
    }


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        dtype=dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    return model, tokenizer


def main():
    require_cuda()
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_df = load_or_create_splits()
    test_df = build_perturbed_test(test_df)
    print("Test examples:", len(test_df))
    print("Same-subreddit match coverage:", float(test_df["same_subreddit_is_matched"].mean()))

    model, tokenizer = load_model()
    args = TrainingArguments(
        output_dir=str(REPORT_DIR / "tmp"),
        per_device_eval_batch_size=16,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args)

    conditions = {
        "true_context": "context",
        "random_context": "random_context",
        "same_subreddit_wrong_context": "same_subreddit_wrong_context",
    }

    y = test_df["label"].to_numpy()
    preds = {}
    rows = []

    for condition, column in conditions.items():
        print("Evaluating:", condition)
        ds = make_dataset(test_df, tokenizer, column)
        output = trainer.predict(ds)
        pred = np.argmax(output.predictions, axis=-1)
        preds[condition] = pred
        rows.append({"condition": condition, **summarize(y, pred)})

    true_pred = preds["true_context"]
    true_macro = summarize(y, true_pred)["macro_f1"]
    sensitivity_rows = []

    for row in rows:
        condition = row["condition"]
        if condition == "true_context":
            row.update({
                "macro_f1_drop_vs_true": 0.0,
                "bootstrap_delta_mean": 0.0,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.0,
            })
            continue

        alt = preds[condition]
        mean_delta, low, high = bootstrap_delta(y, true_pred, alt)
        row.update({
            "macro_f1_drop_vs_true": true_macro - row["macro_f1"],
            "bootstrap_delta_mean": mean_delta,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        })
        sensitivity_rows.append(sensitivity(condition, y, true_pred, alt))

    metrics_df = pd.DataFrame(rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    metrics_df.to_csv(REPORT_DIR / "qwen_context_ablation_metrics.csv", index=False)
    sensitivity_df.to_csv(REPORT_DIR / "qwen_context_sensitivity.csv", index=False)

    pred_out = test_df[["subreddit", "context", "random_context", "same_subreddit_wrong_context", "same_subreddit_is_matched", "comment", "label"]].copy()
    for condition, pred in preds.items():
        pred_out[f"prediction_{condition}"] = pred
    pred_out.to_parquet(REPORT_DIR / "qwen_context_ablation_predictions.parquet", index=False)

    summary = {
        "test_examples": int(len(test_df)),
        "same_subreddit_coverage": float(test_df["same_subreddit_is_matched"].mean()),
    }
    (REPORT_DIR / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL METRICS")
    print(metrics_df.to_string(index=False))
    print("\nCONTEXT SENSITIVITY")
    print(sensitivity_df.to_string(index=False))


if __name__ == "__main__":
    main()
