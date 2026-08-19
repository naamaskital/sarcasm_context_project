from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_dataset_utils import load_or_create_splits

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 128
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
REPORT_DIR = Path("reports/full_dataset/qwen")
MODEL_DIR = Path("models/full_dataset_qwen")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Full-dataset Qwen training is intentionally disabled on CPU. "
            "Check nvidia-smi and install a CUDA-enabled PyTorch build before running this script."
        )


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text(row, mode):
    if mode == "comment_only":
        return f"Reddit reply:\n{row['comment']}"
    if mode == "context_plus_comment":
        return f"Context:\n{row['context']}\n\nReply:\n{row['comment']}"
    raise ValueError(mode)


def make_dataset(df, tokenizer, mode):
    ds = Dataset.from_pandas(
        pd.DataFrame({
            "text": df.apply(lambda row: build_text(row, mode), axis=1),
            "label": df["label"].astype(int),
        }),
        preserve_index=False,
    )

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
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return summarize(labels, preds)


def train_mode(train_df, val_df, test_df, mode):
    print("\n" + "=" * 72)
    print("Full-data Qwen mode:", mode)
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()

    train_ds = make_dataset(train_df, tokenizer, mode)
    val_ds = make_dataset(val_df, tokenizer, mode)
    test_ds = make_dataset(test_df, tokenizer, mode)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / mode),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    output = trainer.predict(test_ds)
    preds = np.argmax(output.predictions, axis=-1)
    metrics = summarize(test_df["label"].to_numpy(), preds)

    predictions = test_df[["context", "comment", "label"]].copy()
    predictions["prediction"] = preds
    predictions.to_parquet(REPORT_DIR / f"{mode}_test_predictions.parquet", index=False)

    del trainer, model, train_ds, val_ds, test_ds
    torch.cuda.empty_cache()

    return metrics


def main():
    require_cuda()
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA build:", torch.version.cuda)

    train_df, val_df, test_df = load_or_create_splits()
    print("Full-data split sizes:", len(train_df), len(val_df), len(test_df))
    print("Total examples used:", len(train_df) + len(val_df) + len(test_df))

    rows = []
    for mode in ["comment_only", "context_plus_comment"]:
        metrics = train_mode(train_df, val_df, test_df, mode)
        rows.append({
            "model": "Qwen2.5-0.5B-Instruct + LoRA",
            "input": mode,
            "epochs": NUM_EPOCHS,
            "train_examples": len(train_df),
            "validation_examples": len(val_df),
            "test_examples": len(test_df),
            **metrics,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "full_dataset_qwen_metrics.csv", index=False)

    print("\nFINAL")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
