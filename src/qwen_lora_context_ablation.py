from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_EPOCHS = 5
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 250
TEST_PER_CLASS = 500
MAX_LENGTH = 128
REPORT_DIR = Path("reports/qwen_lora_ablation")
MODEL_DIR = Path("models/qwen_lora_ablation")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_dataframe():
    print("Loading dataset from Hugging Face: marcbishara/sarcasm-on-reddit")
    raw = load_dataset("marcbishara/sarcasm-on-reddit", split="train")
    df = raw.to_pandas()
    if "context" not in df.columns and "parent_comment" in df.columns:
        df = df.rename(columns={"parent_comment": "context"})
    df = df[["label", "comment", "context"]].dropna().copy()
    df["label"] = df["label"].astype(int)
    print(df["label"].value_counts())
    return df


def make_balanced_splits(df):
    parts = []
    for label in [0, 1]:
        class_df = df[df["label"] == label].sample(frac=1, random_state=SEED + label).reset_index(drop=True)
        needed = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
        if len(class_df) < needed:
            raise ValueError(f"Not enough examples for label {label}")
        parts.append((
            class_df.iloc[:TRAIN_PER_CLASS],
            class_df.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS],
            class_df.iloc[TRAIN_PER_CLASS + VAL_PER_CLASS:needed],
        ))

    train_df = pd.concat([p[0] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat([p[1] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat([p[2] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(test_df))
    if np.any(perm == np.arange(len(test_df))):
        perm = np.roll(perm, 1)
    test_df["random_context"] = test_df.iloc[perm]["context"].to_numpy()

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, len(split), split["label"].value_counts().to_dict())
    return train_df, val_df, test_df


def build_input(row, mode):
    if mode == "comment_only":
        return f"Reddit reply:\n{row['comment']}"
    if mode == "context_only":
        return f"Conversation context:\n{row['context']}"
    if mode == "context_plus_comment":
        return f"Context:\n{row['context']}\n\nReply:\n{row['comment']}"
    if mode == "random_context_plus_comment":
        return f"Context:\n{row['random_context']}\n\nReply:\n{row['comment']}"
    raise ValueError(mode)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "sarcastic_f1": f1_score(labels, preds, pos_label=1),
    }


def make_dataset(df, mode, tokenizer):
    tmp = pd.DataFrame({
        "text": df.apply(lambda row: build_input(row, mode), axis=1),
        "label": df["label"].astype(int),
    })
    ds = Dataset.from_pandas(tmp, preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def summarize(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "sarcastic_f1": float(f1_score(labels, preds, pos_label=1)),
    }


def train_mode(train_df, val_df, test_df, mode):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()

    train_ds = make_dataset(train_df, mode, tokenizer)
    val_ds = make_dataset(val_df, mode, tokenizer)
    test_ds = make_dataset(test_df, mode, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / mode),
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
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
    return trainer, tokenizer, metrics, preds


def main():
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_and_prepare_dataframe()
    train_df, val_df, test_df = make_balanced_splits(df)

    rows = []
    saved = {}
    for mode in ["comment_only", "context_plus_comment", "context_only"]:
        trainer, tokenizer, metrics, preds = train_mode(train_df, val_df, test_df, mode)
        saved[mode] = (trainer, tokenizer)
        rows.append({"trained_on": mode, "evaluated_on": mode, **metrics})

        pred_df = test_df[["context", "comment", "label"]].copy()
        pred_df["prediction"] = preds
        pred_df.to_csv(REPORT_DIR / f"{mode}_predictions.csv", index=False)

    trainer, tokenizer = saved["context_plus_comment"]
    random_ds = make_dataset(test_df, "random_context_plus_comment", tokenizer)
    output = trainer.predict(random_ds)
    random_preds = np.argmax(output.predictions, axis=-1)
    random_metrics = summarize(test_df["label"].to_numpy(), random_preds)
    rows.append({
        "trained_on": "context_plus_comment",
        "evaluated_on": "random_context_plus_comment",
        **random_metrics,
    })

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "qwen_lora_context_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
