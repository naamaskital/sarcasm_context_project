import os
import ast
import numpy as np
import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


DATASET_NAME = "marcbishara/sarcasm-on-reddit"
DATASET_SPLIT = "sft_train"
TARGET_PER_CLASS = 10000
RANDOM_STATE = 42

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("reports", exist_ok=True)


def value_to_text(value):
    """Convert dataset values into clean strings."""
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(item) for item in value)

    if isinstance(value, str):
        text = value.strip()

        # Some context columns are saved as a string representation of a list.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return " ".join(str(item) for item in parsed)
            except Exception:
                pass

        return text

    return str(value)


def normalize_label(value):
    """Convert labels into 0/1, where 1 means sarcastic."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    text = str(value).strip().lower()

    if text in {"1", "true", "sarcastic", "sarcasm", "yes"}:
        return 1

    if text in {"0", "false", "not_sarcastic", "non_sarcastic", "no"}:
        return 0

    raise ValueError(f"Unknown label value: {value}")


def find_column(df, candidates):
    """Find the first existing column from a list of possible names."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_and_prepare_data():
    """Load the sarcasm dataset and create a balanced binary sample."""
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    df = pd.DataFrame(dataset)

    context_col = find_column(
        df,
        ["context", "parent_comment", "parent", "post", "submission_title", "title"],
    )

    comment_col = find_column(
        df,
        ["comment", "response", "reply", "body", "text"],
    )

    label_col = find_column(
        df,
        ["label", "is_sarcastic", "sarcastic", "target"],
    )

    if context_col is None:
        raise ValueError(f"Could not find context column. Columns: {list(df.columns)}")

    if comment_col is None:
        raise ValueError(f"Could not find comment column. Columns: {list(df.columns)}")

    if label_col is None:
        raise ValueError(f"Could not find label column. Columns: {list(df.columns)}")

    df = df[[context_col, comment_col, label_col]].copy()
    df.columns = ["context", "comment", "label"]

    df["context"] = df["context"].apply(value_to_text)
    df["comment"] = df["comment"].apply(value_to_text)
    df["label"] = df["label"].apply(normalize_label)

    df = df[(df["context"].str.len() > 0) & (df["comment"].str.len() > 0)]

    # Balanced sample: same number of sarcastic and non-sarcastic examples.
    parts = []
    for label in [0, 1]:
        class_df = df[df["label"] == label]
        n = min(TARGET_PER_CLASS, len(class_df))
        parts.append(class_df.sample(n=n, random_state=RANDOM_STATE))

    df = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


def build_contrast_features(model, contexts, comments):
    """
    Build features:
    1. context embedding
    2. comment embedding
    3. absolute difference between embeddings
    4. cosine similarity between context and comment
    """
    context_emb = model.encode(
        contexts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    comment_emb = model.encode(
        comments,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    abs_diff = np.abs(context_emb - comment_emb)

    cosine_similarity = np.sum(context_emb * comment_emb, axis=1).reshape(-1, 1)

    features = np.concatenate(
        [
            context_emb,
            comment_emb,
            abs_diff,
            cosine_similarity,
        ],
        axis=1,
    )

    return features


def evaluate_model(model, X, y, split_name):
    """Evaluate a trained classifier and return metrics."""
    predictions = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "sarcastic_f1": f1_score(y, predictions, pos_label=1),
    }

    report = classification_report(
        y,
        predictions,
        target_names=["not_sarcastic", "sarcastic"],
        digits=4,
    )

    matrix = confusion_matrix(y, predictions)

    return metrics, report, matrix, predictions


def main():
    print("Loading data...")
    df = load_and_prepare_data()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Building train features...")
    X_train = build_contrast_features(
        embedding_model,
        train_df["context"].tolist(),
        train_df["comment"].tolist(),
    )
    y_train = train_df["label"].values

    print("Building test features...")
    X_test = build_contrast_features(
        embedding_model,
        test_df["context"].tolist(),
        test_df["comment"].tolist(),
    )
    y_test = test_df["label"].values

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Training classifier...")
    classifier.fit(X_train, y_train)

    print("Evaluating...")
    train_metrics, train_report, train_matrix, train_predictions = evaluate_model(
        classifier, X_train, y_train, "train"
    )

    test_metrics, test_report, test_matrix, test_predictions = evaluate_model(
        classifier, X_test, y_test, "test"
    )

    metrics_df = pd.DataFrame([train_metrics, test_metrics])
    metrics_df.to_csv("reports/contrast_features_metrics.csv", index=False)

    predictions_df = test_df.copy()
    predictions_df["prediction"] = test_predictions
    predictions_df.to_csv("reports/contrast_features_predictions.csv", index=False)

    with open("reports/contrast_features_summary.txt", "w", encoding="utf-8") as f:
        f.write("Contrast Features Experiment\n")
        f.write("=" * 50 + "\n\n")

        f.write("Goal:\n")
        f.write(
            "Represent context and comment separately, then add explicit contrast features between them.\n\n"
        )

        f.write("Features:\n")
        f.write("- v_context\n")
        f.write("- v_comment\n")
        f.write("- |v_context - v_comment|\n")
        f.write("- cosine_similarity(v_context, v_comment)\n\n")

        f.write("Embedding model:\n")
        f.write(f"{EMBEDDING_MODEL}\n\n")

        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("Train classification report:\n")
        f.write(train_report)
        f.write("\nTrain confusion matrix:\n")
        f.write(str(train_matrix))
        f.write("\n\n")

        f.write("Test classification report:\n")
        f.write(test_report)
        f.write("\nTest confusion matrix:\n")
        f.write(str(test_matrix))
        f.write("\n")

    print("Done.")
    print(metrics_df)
    print()
    print("Files created:")
    print("reports/contrast_features_metrics.csv")
    print("reports/contrast_features_predictions.csv")
    print("reports/contrast_features_summary.txt")


if __name__ == "__main__":
    main()
