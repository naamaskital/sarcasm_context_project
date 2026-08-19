from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

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
REPORT_DIR = ROOT / "reports" / "full_dataset" / "qwen"
MODEL_DIR = ROOT / "models" / "full_dataset_qwen"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["comment_only", "context_plus_comment", "all"],
        default="all",
        help="Train one input condition or both conditions sequentially.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest Trainer checkpoint for the selected mode when available.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a mode when its metrics JSON and final adapter already exist.",
    )
    return parser.parse_args()


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Full-dataset Qwen training is intentionally disabled on CPU. "
            "Run this script on a CUDA-capable machine with a CUDA-enabled PyTorch build."
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
    texts = [build_text(row, mode) for _, row in df.iterrows()]
    ds = Dataset.from_dict({
        "text": texts,
        "label": df["label"].astype(int).tolist(),
    })

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


def precision_flags():
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    return {"bf16": use_bf16, "fp16": not use_bf16}


def train_mode(train_df, val_df, test_df, mode, resume=False, skip_completed=False):
    output_dir = MODEL_DIR / mode
    final_adapter_dir = output_dir / "final_adapter"
    metrics_path = REPORT_DIR / f"{mode}_metrics.json"

    if skip_completed and final_adapter_dir.exists() and metrics_path.exists():
        print(f"Skipping completed mode: {mode}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 72)
    print("Full-data Qwen mode:", mode)
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        dtype=dtype,
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
        output_dir=str(output_dir),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        logging_steps=250,
        **precision_flags(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if resume and output_dir.exists() else None
    if checkpoint:
        print("Resuming from:", checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)

    validation_output = trainer.predict(val_ds)
    validation_preds = np.argmax(validation_output.predictions, axis=-1)
    validation_metrics = summarize(val_df["label"].to_numpy(), validation_preds)

    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_metrics = summarize(test_df["label"].to_numpy(), test_preds)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    trainer.save_state()

    predictions = test_df[["context", "comment", "label"]].copy()
    predictions["prediction"] = test_preds
    predictions.to_parquet(REPORT_DIR / f"{mode}_test_predictions.parquet", index=False)

    result = {
        "model": "Qwen2.5-0.5B-Instruct + LoRA",
        "input": mode,
        "epochs": NUM_EPOCHS,
        "train_examples": int(len(train_df)),
        "validation_examples": int(len(val_df)),
        "test_examples": int(len(test_df)),
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_sarcastic_f1": validation_metrics["sarcastic_f1"],
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "sarcastic_f1": test_metrics["sarcastic_f1"],
        "final_adapter": str(final_adapter_dir.relative_to(ROOT)),
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

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA build:", torch.version.cuda)

    train_df, val_df, test_df = load_or_create_splits()
    print("Full-data split sizes:", len(train_df), len(val_df), len(test_df))
    print("Total examples used:", len(train_df) + len(val_df) + len(test_df))

    modes = ["comment_only", "context_plus_comment"] if args.mode == "all" else [args.mode]
    rows = []
    for mode in modes:
        rows.append(train_mode(
            train_df,
            val_df,
            test_df,
            mode,
            resume=args.resume,
            skip_completed=args.skip_completed,
        ))

    # Rebuild the summary from every completed mode, not just modes from this invocation.
    completed = []
    for mode in ["comment_only", "context_plus_comment"]:
        path = REPORT_DIR / f"{mode}_metrics.json"
        if path.exists():
            completed.append(json.loads(path.read_text(encoding="utf-8")))

    summary = pd.DataFrame(completed)
    summary.to_csv(REPORT_DIR / "full_dataset_qwen_metrics.csv", index=False)

    print("\nFINAL")
    if len(summary):
        print(summary.to_string(index=False))
    else:
        print("No completed Qwen mode yet.")


if __name__ == "__main__":
    main()
