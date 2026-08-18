from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

SEED = 42
DATASET_NAME = "marcbishara/sarcasm-on-reddit"
DATASET_SPLIT = "sft_train"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

NUM_EPOCHS = 5
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 250
TEST_PER_CLASS = 500
MAX_LENGTH = 128
BOOTSTRAP_RUNS = 1000

REPORT_DIR = Path("reports/qwen_hard_context_ablation")
MODEL_DIR = Path("models/qwen_hard_context_ablation")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_and_prepare_dataframe():
    print(f"Loading dataset: {DATASET_NAME} ({DATASET_SPLIT})")
    raw = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    df = raw.to_pandas()

    if "context" not in df.columns and "parent_comment" in df.columns:
        df = df.rename(columns={"parent_comment": "context"})

    needed = ["label", "comment", "context"]
    if "subreddit" in df.columns:
        needed.append("subreddit")

    df = df[needed].copy()
    if "subreddit" not in df.columns:
        df["subreddit"] = ""

    df["context"] = df["context"].map(clean_text)
    df["comment"] = df["comment"].map(clean_text)
    df["subreddit"] = df["subreddit"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[(df["context"] != "") & (df["comment"] != "")].reset_index(drop=True)

    print("Label counts after cleaning:")
    print(df["label"].value_counts())
    return df


def make_balanced_splits(df):
    parts = []
    for label in [0, 1]:
        class_df = (
            df[df["label"] == label]
            .sample(frac=1, random_state=SEED + label)
            .reset_index(drop=True)
        )
        needed = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
        if len(class_df) < needed:
            raise ValueError(f"Not enough examples for label {label}")

        parts.append((
            class_df.iloc[:TRAIN_PER_CLASS].copy(),
            class_df.iloc[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS].copy(),
            class_df.iloc[TRAIN_PER_CLASS + VAL_PER_CLASS:needed].copy(),
        ))

    train_df = pd.concat([p[0] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = pd.concat([p[1] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = pd.concat([p[2] for p in parts]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("Split sizes:")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, len(split), split["label"].value_counts().to_dict())

    return train_df, val_df, test_df


def derangement(values, rng):
    values = np.asarray(values)
    if len(values) < 2:
        raise ValueError("Need at least two values for a derangement")

    perm = rng.permutation(values)
    while np.any(perm == values):
        perm = rng.permutation(values)
    return perm


def random_wrong_indices(n, seed):
    rng = np.random.default_rng(seed)
    return derangement(np.arange(n), rng)


def same_subreddit_wrong_indices(test_df, seed):
    rng = np.random.default_rng(seed)
    result = np.full(len(test_df), -1, dtype=int)
    matched = np.zeros(len(test_df), dtype=bool)

    for subreddit, group in test_df.groupby("subreddit", sort=False):
        if not subreddit:
            continue
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        result[indices] = derangement(indices, rng)
        matched[indices] = True

    fallback = random_wrong_indices(len(test_df), seed + 1)
    missing = result < 0
    result[missing] = fallback[missing]
    return result, matched


def semantic_wrong_indices(test_df):
    print("Encoding test contexts for semantic hard negatives...")
    encoder = SentenceTransformer(SEMANTIC_MODEL_ID)
    embeddings = encoder.encode(
        test_df["context"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)

    text_to_indices = {}
    for idx, text in enumerate(test_df["context"].tolist()):
        text_to_indices.setdefault(text, []).append(idx)

    for indices in text_to_indices.values():
        if len(indices) > 1:
            similarities[np.ix_(indices, indices)] = -np.inf

    chosen = np.argmax(similarities, axis=1)
    chosen_similarity = similarities[np.arange(len(test_df)), chosen]

    if np.any(~np.isfinite(chosen_similarity)):
        raise ValueError("Could not find a distinct semantic hard negative for every test row")

    del encoder, embeddings, similarities
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return chosen, chosen_similarity


def add_hard_negative_contexts(test_df):
    test_df = test_df.copy()

    random_idx = random_wrong_indices(len(test_df), SEED)
    subreddit_idx, subreddit_matched = same_subreddit_wrong_indices(test_df, SEED)
    semantic_idx, semantic_similarity = semantic_wrong_indices(test_df)

    test_df["random_context"] = test_df.iloc[random_idx]["context"].to_numpy()
    test_df["same_subreddit_wrong_context"] = test_df.iloc[subreddit_idx]["context"].to_numpy()
    test_df["same_subreddit_is_matched"] = subreddit_matched
    test_df["semantic_similar_wrong_context"] = test_df.iloc[semantic_idx]["context"].to_numpy()
    test_df["semantic_context_cosine_similarity"] = semantic_similarity

    return test_df


def build_text(row, context_column="context"):
    return f"Context:\n{row[context_column]}\n\nReply:\n{row['comment']}"


def make_dataset(df, tokenizer, context_column="context"):
    tmp = pd.DataFrame({
        "text": df.apply(lambda row: build_text(row, context_column), axis=1),
        "label": df["label"].astype(int),
    })
    ds = Dataset.from_pandas(tmp, preserve_index=False)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    return summarize(labels, preds)


def summarize(labels, preds):
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "sarcastic_f1": float(f1_score(labels, preds, pos_label=1)),
    }


def bootstrap_delta(y_true, reference_pred, alternative_pred):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    deltas = []

    for _ in range(BOOTSTRAP_RUNS):
        sample = rng.integers(0, n, size=n)
        ref = f1_score(y_true[sample], reference_pred[sample], average="macro")
        alt = f1_score(y_true[sample], alternative_pred[sample], average="macro")
        deltas.append(ref - alt)

    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(low), float(high)


def sensitivity_row(name, y_true, reference_pred, alternative_pred):
    changed = reference_pred != alternative_pred
    ref_correct = reference_pred == y_true
    alt_correct = alternative_pred == y_true

    return {
        "condition": name,
        "changed_predictions": int(changed.sum()),
        "changed_rate": float(changed.mean()),
        "true_correct_to_wrong": int((changed & ref_correct & ~alt_correct).sum()),
        "true_wrong_to_correct": int((changed & ~ref_correct & alt_correct).sum()),
    }


def train_true_context_model(train_df, val_df):
    print("Loading tokenizer and Qwen model...")
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

    train_ds = make_dataset(train_df, tokenizer, "context")
    val_ds = make_dataset(val_df, tokenizer, "context")

    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
    return trainer, tokenizer


def predict_condition(trainer, tokenizer, test_df, context_column):
    ds = make_dataset(test_df, tokenizer, context_column)
    output = trainer.predict(ds)
    return np.argmax(output.predictions, axis=-1)


def main():
    set_seed()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_and_prepare_dataframe()
    train_df, val_df, test_df = make_balanced_splits(df)

    print("Building hard-negative test contexts before loading Qwen...")
    test_df = add_hard_negative_contexts(test_df)
    subreddit_coverage = float(test_df["same_subreddit_is_matched"].mean())
    mean_semantic_similarity = float(test_df["semantic_context_cosine_similarity"].mean())
    print("Same-subreddit match coverage:", round(subreddit_coverage, 4))
    print("Mean semantic hard-negative similarity:", round(mean_semantic_similarity, 4))

    print("Training Qwen once on TRUE context + reply...")
    trainer, tokenizer = train_true_context_model(train_df, val_df)

    conditions = {
        "true_context": "context",
        "random_context": "random_context",
        "same_subreddit_wrong_context": "same_subreddit_wrong_context",
        "semantic_similar_wrong_context": "semantic_similar_wrong_context",
    }

    y_test = test_df["label"].to_numpy()
    pred_by_condition = {}
    rows = []

    for condition, context_column in conditions.items():
        print("Evaluating:", condition)
        preds = predict_condition(trainer, tokenizer, test_df, context_column)
        pred_by_condition[condition] = preds
        rows.append({"condition": condition, **summarize(y_test, preds)})

    true_pred = pred_by_condition["true_context"]
    true_macro_f1 = summarize(y_test, true_pred)["macro_f1"]

    sensitivity_rows = []
    for row in rows:
        condition = row["condition"]
        if condition == "true_context":
            row["macro_f1_drop_vs_true"] = 0.0
            row["bootstrap_delta_mean"] = 0.0
            row["bootstrap_ci_low"] = 0.0
            row["bootstrap_ci_high"] = 0.0
            continue

        alt_pred = pred_by_condition[condition]
        mean_delta, low, high = bootstrap_delta(y_test, true_pred, alt_pred)
        row["macro_f1_drop_vs_true"] = true_macro_f1 - row["macro_f1"]
        row["bootstrap_delta_mean"] = mean_delta
        row["bootstrap_ci_low"] = low
        row["bootstrap_ci_high"] = high
        sensitivity_rows.append(sensitivity_row(condition, y_test, true_pred, alt_pred))

    metrics_df = pd.DataFrame(rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    metrics_df.to_csv(REPORT_DIR / "qwen_hard_context_metrics.csv", index=False)
    sensitivity_df.to_csv(REPORT_DIR / "qwen_hard_context_sensitivity.csv", index=False)

    predictions = test_df[
        [
            "subreddit",
            "context",
            "random_context",
            "same_subreddit_wrong_context",
            "same_subreddit_is_matched",
            "semantic_similar_wrong_context",
            "semantic_context_cosine_similarity",
            "comment",
            "label",
        ]
    ].copy()

    for condition, preds in pred_by_condition.items():
        predictions[f"prediction_{condition}"] = preds

    predictions.to_csv(REPORT_DIR / "qwen_hard_context_predictions.csv", index=False)

    semantic_changed = predictions[
        predictions["prediction_true_context"]
        != predictions["prediction_semantic_similar_wrong_context"]
    ].copy()
    semantic_changed = semantic_changed.sort_values(
        "semantic_context_cosine_similarity",
        ascending=False,
    )
    semantic_changed.to_csv(
        REPORT_DIR / "semantic_hard_negative_changed_examples.csv",
        index=False,
    )

    with open(REPORT_DIR / "qwen_hard_context_summary.txt", "w", encoding="utf-8") as f:
        f.write("Qwen Hard Context Ablation\n")
        f.write("=" * 70 + "\n\n")
        f.write("Qwen is trained ONCE on true context + reply.\n")
        f.write("Only the context is replaced at test time.\n\n")
        f.write(f"Same-subreddit match coverage: {subreddit_coverage:.4f}\n")
        f.write(f"Mean semantic-hard-negative cosine similarity: {mean_semantic_similarity:.4f}\n\n")
        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nContext sensitivity:\n")
        f.write(sensitivity_df.to_string(index=False))
        f.write("\n\n")
        f.write("A positive bootstrap delta means true context has higher Macro-F1.\n")
        f.write("If the paired 95% CI stays above zero, the true-context advantage is consistent.\n")

    print("\nFINAL METRICS")
    print(metrics_df.to_string(index=False))
    print("\nCONTEXT SENSITIVITY")
    print(sensitivity_df.to_string(index=False))
    print("\nSaved to:", REPORT_DIR)


if __name__ == "__main__":
    main()
