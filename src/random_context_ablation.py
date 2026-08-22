import os
import time
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"
REPORTS_DIR = "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def build_comment_only(df):
    return df["comment"].apply(clean_text).tolist()


def build_context_plus_comment(df, context_col="context"):
    texts = []

    for _, row in df.iterrows():
        context = clean_text(row[context_col])
        comment = clean_text(row["comment"])

        texts.append(f"Context: {context}\nComment: {comment}")

    return texts


def add_random_context(df, seed):
    out = df.copy().reset_index(drop=True)

    shuffled_contexts = out["context"].sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    # cyclic shift to reduce chance that a row keeps its own context
    shuffled_contexts = shuffled_contexts.shift(1).fillna(shuffled_contexts.iloc[-1])

    out["random_context"] = shuffled_contexts

    return out


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "sarcastic_f1": f1_score(y_true, y_pred, pos_label=1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["not_sarcastic", "sarcastic"],
            digits=4
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


def run_tfidf_experiment(name, train_texts, test_texts, y_train, y_test):
    print(f"\nRunning TF-IDF experiment: {name}")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=2
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    model.fit(train_texts, y_train)
    preds = model.predict(test_texts)

    metrics = evaluate_predictions(y_test, preds)

    return metrics, preds


def encode_texts(model, texts):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def run_sentence_transformer_joint(name, st_model, train_texts, test_texts, y_train, y_test):
    print(f"\nRunning Sentence Transformer joint experiment: {name}")

    x_train = encode_texts(st_model, train_texts)
    x_test = encode_texts(st_model, test_texts)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    metrics = evaluate_predictions(y_test, preds)

    return metrics, preds


def run_sentence_transformer_separate(name, st_model, train_df, test_df, y_train, y_test, context_col):
    print(f"\nRunning Sentence Transformer separate experiment: {name}")

    train_contexts = train_df[context_col].apply(clean_text).tolist()
    test_contexts = test_df[context_col].apply(clean_text).tolist()

    train_comments = train_df["comment"].apply(clean_text).tolist()
    test_comments = test_df["comment"].apply(clean_text).tolist()

    print("Encoding contexts...")
    x_train_context = encode_texts(st_model, train_contexts)
    x_test_context = encode_texts(st_model, test_contexts)

    print("Encoding comments...")
    x_train_comment = encode_texts(st_model, train_comments)
    x_test_comment = encode_texts(st_model, test_comments)

    x_train = np.concatenate([x_train_context, x_train_comment], axis=1)
    x_test = np.concatenate([x_test_context, x_test_comment], axis=1)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    metrics = evaluate_predictions(y_test, preds)

    return metrics, preds


def add_result(rows, method, setting, metrics):
    rows.append({
        "method": method,
        "setting": setting,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "sarcastic_f1": metrics["sarcastic_f1"]
    })


def main():
    start_time = time.time()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run prepare_balanced_sarcasm_data.py first."
        )

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

    train_df = add_random_context(train_df, RANDOM_STATE)
    test_df = add_random_context(test_df, RANDOM_STATE + 1)

    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print("\nTrain size:", len(train_df))
    print(train_df["label"].value_counts().sort_index())

    print("\nTest size:", len(test_df))
    print(test_df["label"].value_counts().sort_index())

    rows = []
    prediction_df = test_df.copy()
    prediction_df["true_label"] = y_test

    # ------------------------------------------------------------
    # TF-IDF baselines
    # ------------------------------------------------------------
    tfidf_conditions = {
        "comment_only": (
            build_comment_only(train_df),
            build_comment_only(test_df)
        ),
        "true_context_plus_comment": (
            build_context_plus_comment(train_df, "context"),
            build_context_plus_comment(test_df, "context")
        ),
        "random_context_plus_comment": (
            build_context_plus_comment(train_df, "random_context"),
            build_context_plus_comment(test_df, "random_context")
        )
    }

    detailed_text = []

    for setting, (train_texts, test_texts) in tfidf_conditions.items():
        metrics, preds = run_tfidf_experiment(
            setting,
            train_texts,
            test_texts,
            y_train,
            y_test
        )

        add_result(rows, "tfidf_logistic_regression", setting, metrics)
        prediction_df[f"tfidf_{setting}_pred"] = preds

        detailed_text.append(f"\nTF-IDF / {setting}")
        detailed_text.append("=" * 60)
        detailed_text.append(f"Accuracy: {metrics['accuracy']:.4f}")
        detailed_text.append(f"Macro F1: {metrics['macro_f1']:.4f}")
        detailed_text.append(f"Sarcastic F1: {metrics['sarcastic_f1']:.4f}")
        detailed_text.append(metrics["classification_report"])
        detailed_text.append("Confusion matrix:")
        detailed_text.append(str(metrics["confusion_matrix"]))

    # ------------------------------------------------------------
    # Sentence Transformer ablations
    # ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nCUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print(f"Loading Sentence Transformer: {ST_MODEL_NAME}")

    st_model = SentenceTransformer(ST_MODEL_NAME, device=device)

    st_joint_conditions = {
        "true_context_plus_comment_joint": (
            build_context_plus_comment(train_df, "context"),
            build_context_plus_comment(test_df, "context")
        ),
        "random_context_plus_comment_joint": (
            build_context_plus_comment(train_df, "random_context"),
            build_context_plus_comment(test_df, "random_context")
        )
    }

    for setting, (train_texts, test_texts) in st_joint_conditions.items():
        metrics, preds = run_sentence_transformer_joint(
            setting,
            st_model,
            train_texts,
            test_texts,
            y_train,
            y_test
        )

        add_result(rows, "sentence_transformer_joint", setting, metrics)
        prediction_df[f"st_{setting}_pred"] = preds

        detailed_text.append(f"\nSentence Transformer joint / {setting}")
        detailed_text.append("=" * 60)
        detailed_text.append(f"Accuracy: {metrics['accuracy']:.4f}")
        detailed_text.append(f"Macro F1: {metrics['macro_f1']:.4f}")
        detailed_text.append(f"Sarcastic F1: {metrics['sarcastic_f1']:.4f}")
        detailed_text.append(metrics["classification_report"])
        detailed_text.append("Confusion matrix:")
        detailed_text.append(str(metrics["confusion_matrix"]))

    st_separate_conditions = {
        "true_context_plus_comment_separate": "context",
        "random_context_plus_comment_separate": "random_context"
    }

    for setting, context_col in st_separate_conditions.items():
        metrics, preds = run_sentence_transformer_separate(
            setting,
            st_model,
            train_df,
            test_df,
            y_train,
            y_test,
            context_col
        )

        add_result(rows, "sentence_transformer_separate_concat", setting, metrics)
        prediction_df[f"st_{setting}_pred"] = preds

        detailed_text.append(f"\nSentence Transformer separate / {setting}")
        detailed_text.append("=" * 60)
        detailed_text.append(f"Accuracy: {metrics['accuracy']:.4f}")
        detailed_text.append(f"Macro F1: {metrics['macro_f1']:.4f}")
        detailed_text.append(f"Sarcastic F1: {metrics['sarcastic_f1']:.4f}")
        detailed_text.append(metrics["classification_report"])
        detailed_text.append("Confusion matrix:")
        detailed_text.append(str(metrics["confusion_matrix"]))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv("reports/random_context_ablation_metrics.csv", index=False)
    prediction_df.to_csv("reports/random_context_ablation_predictions.csv", index=False)

    elapsed_minutes = (time.time() - start_time) / 60

    summary_lines = []
    summary_lines.append("Random Context Ablation")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Input file: {INPUT_PATH}")
    summary_lines.append(f"Train size: {len(train_df)}")
    summary_lines.append(f"Test size: {len(test_df)}")
    summary_lines.append("")
    summary_lines.append(metrics_df.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Interpretation guide:")
    summary_lines.append(
        "If true_context_plus_comment is better than random_context_plus_comment, "
        "then the specific conversational context adds useful information."
    )
    summary_lines.append(
        "If random_context_plus_comment is similar to true_context_plus_comment, "
        "then the model may not be using the specific context, and may rely mostly "
        "on the comment or on superficial extra text."
    )
    summary_lines.append("")
    summary_lines.append(f"Elapsed minutes: {elapsed_minutes:.2f}")
    summary_lines.append("")
    summary_lines.extend(detailed_text)

    summary_text = "\n".join(summary_lines)

    with open("reports/random_context_ablation_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + metrics_df.to_string(index=False))
    print(f"\nElapsed minutes: {elapsed_minutes:.2f}")

    print("\nSaved:")
    print("reports/random_context_ablation_metrics.csv")
    print("reports/random_context_ablation_predictions.csv")
    print("reports/random_context_ablation_summary.txt")


if __name__ == "__main__":
    main()
