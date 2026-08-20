from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, T5ForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

SEED = 42
MODEL_ID = "google/flan-t5-base"
MAX_LENGTH = 128
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5
ALL_MODES = ["comment_only", "context_only", "context_plus_comment"]
REPORT_DIR = ROOT / "reports" / "full_dataset" / "flan_t5_encoder_decoder"
MODEL_DIR = ROOT / "models" / "flan_t5_encoder_decoder"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=ALL_MODES + ["all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-data FLAN-T5 fine-tuning.")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(row, mode):
    if mode == "comment_only":
        return f"Classify sarcasm. Reply: {row['comment']}"
    if mode == "context_only":
        return f"Classify sarcasm from context only. Previous Reddit message: {row['context']}"
    if mode == "context_plus_comment":
        return f"Classify sarcasm. Previous Reddit message: {row['context']} Reply: {row['comment']}"
    raise ValueError(mode)


def make_dataset(df, tokenizer, mode):
    texts = [build_text(row, mode) for _, row in df.iterrows()]
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


def compute_metrics(pred):
    return summarize(pred.label_ids, np.argmax(pred.predictions, axis=-1))


def softmax_positive(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1].astype(np.float32)


def precision_flags():
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    return {"bf16": use_bf16, "fp16": not use_bf16}


def save_predictions(df, output, split_name, mode):
    pred = np.argmax(output.predictions, axis=-1)
    prob = softmax_positive(output.predictions)
    keep = [c for c in ["subreddit", "context", "comment", "label"] if c in df.columns]
    out = df[keep].copy()
    out["prediction"] = pred
    out["prob_sarcastic"] = prob
    out.to_parquet(REPORT_DIR / f"{mode}_{split_name}_predictions.parquet", index=False)
    return pred


def train_mode(train_df, val_df, test_df, mode, resume=False, skip_completed=False):
    output_dir = MODEL_DIR / mode
    final_model_dir = output_dir / "final_model"
    metrics_path = REPORT_DIR / f"{mode}_metrics.json"

    if skip_completed and final_model_dir.exists() and metrics_path.exists():
        print("Skipping completed mode:", mode)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 72)
    print("FLAN-T5 mode:", mode)
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = T5ForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    train_ds = make_dataset(train_df, tokenizer, mode)
    val_ds = make_dataset(val_df, tokenizer, mode)
    test_ds = make_dataset(test_df, tokenizer, mode)

    args = TrainingArguments(
        output_dir=str(output_dir), learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS, eval_strategy="epoch",
        save_strategy="steps", save_steps=2000, save_total_limit=2,
        load_best_model_at_end=False, seed=SEED, data_seed=SEED,
        report_to="none", logging_steps=500, dataloader_num_workers=2,
        dataloader_pin_memory=True, gradient_checkpointing=True,
        **precision_flags(),
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    checkpoint = get_last_checkpoint(str(output_dir)) if resume and output_dir.exists() else None
    if checkpoint:
        print("Resuming from:", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)

    val_output = trainer.predict(val_ds)
    test_output = trainer.predict(test_ds)
    val_preds = save_predictions(val_df, val_output, "validation", mode)
    test_preds = save_predictions(test_df, test_output, "test", mode)
    val_metrics = summarize(val_df["label"].to_numpy(), val_preds)
    test_metrics = summarize(test_df["label"].to_numpy(), test_preds)

    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    result = {
        "model": "flan-t5-base encoder-decoder classifier", "input": mode, "epochs": NUM_EPOCHS,
        "train_examples": int(len(train_df)), "validation_examples": int(len(val_df)), "test_examples": int(len(test_df)),
        "validation_accuracy": val_metrics["accuracy"], "validation_macro_f1": val_metrics["macro_f1"],
        "validation_sarcastic_f1": val_metrics["sarcastic_f1"], "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"], "sarcastic_f1": test_metrics["sarcastic_f1"],
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    del trainer, model, train_ds, val_ds, test_ds
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    require_cuda()
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0))
    train_df, val_df, test_df = load_or_create_splits()
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    modes = ALL_MODES if args.mode == "all" else [args.mode]
    for mode in modes:
        train_mode(train_df, val_df, test_df, mode, args.resume, args.skip_completed)

    completed = []
    for mode in ALL_MODES:
        path = REPORT_DIR / f"{mode}_metrics.json"
        if path.exists():
            completed.append(json.loads(path.read_text(encoding="utf-8")))
    summary = pd.DataFrame(completed)
    summary.to_csv(REPORT_DIR / "flan_t5_encoder_decoder_metrics.csv", index=False)
    print("\nFINAL FLAN-T5")
    print(summary.to_string(index=False) if len(summary) else "No completed modes.")


if __name__ == "__main__":
    main()
