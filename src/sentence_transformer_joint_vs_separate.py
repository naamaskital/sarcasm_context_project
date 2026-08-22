import os
import time
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"
REPORTS_DIR = "reports"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
TEST_SIZE = 0.20
BATCH_SIZE = 64

os.makedirs(REPORTS_DIR, exist_ok=True)


def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def build_joint_text(row):
    context = clean_text(row["context"])
    comment = clean_text(row["comment"])

    return f"Context: {context}\nComment: {comment}"


def train_and_evaluate(name, x_train, x_test, y_train, y_test):
    clf = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    sarcastic_f1 = f1_score(y_test, preds, pos_label=1)

    report = classification_report(
        y_test,
        preds,
        target_names=["not_sarcastic", "sarcastic"],
        digits=4
    )

    cm = confusion_matrix(y_test, preds)

    print(f"\n{name}")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Sarcastic F1: {sarcastic_f1:.4f}")
    print(report)
    print("Conf1:.4f}")
    print(f"Sarcastic F1: {sarcastic_f1:.4f}")
    print(report)
    print("Confusion matrix:")
    print(cm)

    return {
        "method": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "sarcastic_f1": sarcastic_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": preds
    }


def main():
    start_time = time.time()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run prepare_balanced_sarcasm_data.py first."
        )

    print("CUDA available:", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print(f"Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"context", "comment", "label"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[["context", "comment", "label"]].copy()
    df["context"] = df["context"].apply(clean_text)
    df["comment"] = df["comment"].apply(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print("\nTrain size:", len(train_df))
    print(train_df["label"].value_counts().sort_index())

    print("\nTest size:", len(test_df))
    print(test_df["label"].value_counts().sort_index())

    print(f"\nLoading SentenceTransformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    results = []

    # ------------------------------------------------------------
    # Experiment 1: joint / together embedding
    # ------------------------------------------------------------
    print("\nEncoding joint context+comment texts...")

    train_joint_texts = train_df.apply(build_joint_text, axis=1).tolist()
    test_joint_texts = test_df.apply(build_joint_text, axis=1).tolist()

    x_train_joint = model.encode(
        train_joint_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    x_test_joint = model.encode(
        test_joint_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    joint_result = train_and_evaluate(
        "joint_context_comment_embedding",
        x_train_joint,
        x_test_joint,
        y_train,
        y_test
    )

    results.append(joint_result)

    # ------------------------------------------------------------
    # Experiment 2: separate embeddings + concatenation
    # ------------------------------------------------------------
    print("\nEncoding context separately...")

    train_context_texts = train_df["context"].tolist()
    test_context_texts = test_df["context"].tolist()

    x_train_context = model.encode(
        train_context_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    x_test_context = model.encode(
        test_context_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("\nEncoding comment separately...")

    train_comment_texts = train_df["comment"].tolist()
    test_comment_texts = test_df["comment"].tolist()

    x_train_comment = model.encode(
        train_comment_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    x_test_comment = model.encode(
        test_comment_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    x_train_separate = np.concatenate(
        [x_train_context, x_train_comment],
        axis=1
    )

    x_test_separate = np.concatenate(
        [x_test_context, x_test_comment],
        axis=1
    )

    separate_result = train_and_evaluate(
        "separate_context_comment_embeddings_concat",
        x_train_separate,
        x_test_separate,
        y_train,
        y_test
    )

    results.append(separate_result)

    rows = []

    for result in results:
        rows.append({
            "method": result["method"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "sarcastic_f1": result["sarcastic_f1"]
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        "reports/sentence_transformer_joint_vs_separate_metrics.csv",
        index=False
    )

    pred_df = test_df.copy()
    pred_df["true_label"] = y_test
    pred_df["joint_prediction"] = joint_result["predictions"]
    pred_df["separate_prediction"] = separate_result["predictions"]

    pred_df.to_csv(
        "reports/sentence_transformer_joint_vs_separate_predictions.csv",
        index=False
    )

    elapsed_minutes = (time.time() - start_time) / 60

    summary_lines = []
    summary_lines.append("Sentence Transformer: joint vs separate embeddings")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Model: {MODEL_NAME}")
    summary_lines.append(f"Input file: {INPUT_PATH}")
    summary_lines.append(f"Train size: {len(train_df)}")
    summary_lines.append(f"Test size: {len(test_df)}")
    summary_lines.append("")
    summary_lines.append(metrics_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append(f"Elapsed minutes: {elapsed_minutes:.2f}")
    summary_lines.append("")
    summary_lines.append("Interpretation:")
    summary_lines.append(
        "joint_context_comment_embedding means the context and comment were "
        "combined into one text before embedding."
    )
    summary_lines.append(
        "separate_context_comment_embeddings_concat means the context and "
        "comment were embedded separately, and the two vectors were concatenated "
        "before training the classifier."
    )

    summary_text = "\n".join(summary_lines)

    with open(
        "reports/sentence_transformer_joint_vs_separate_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(summary_text)

    print("\n" + summary_text)

    print("\nSaved:")
    print("reports/sentence_transformer_joint_vs_separate_metrics.csv")
    print("reports/sentence_transformer_joint_vs_separate_predictions.csv")
    print("reports/sentence_transformer_joint_vs_separate_summary.txt")


if __name__ == "__main__":
    main()
