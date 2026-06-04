import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


INPUT_PATH = "data/reddit_sarcasm_context_sample.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42

os.makedirs("reports", exist_ok=True)


def evaluate_setting(model, setting_name, train_texts, test_texts, y_train, y_test):
    print(f"\nEncoding: {setting_name}")

    train_embeddings = model.encode(
        train_texts.tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    test_embeddings = model.encode(
        test_texts.tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    classifier.fit(train_embeddings, y_train)
    predictions = classifier.predict(test_embeddings)

    result = {
        "method": "sentence_transformer_embeddings",
        "setting": setting_name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "sarcastic_f1": f1_score(y_test, predictions, pos_label=1),
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["not_sarcastic", "sarcastic"]
    )

    return result, predictions, report


def main():
    df = pd.read_csv(INPUT_PATH)

    df["context_only"] = df["context"].astype(str)
    df["comment_only"] = df["comment"].astype(str)
    df["context_plus_comment"] = (
        "Previous message: " + df["context"].astype(str) +
        "\nReddit comment: " + df["comment"].astype(str)
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    settings = [
        ("context_only", "context_only"),
        ("comment_only", "comment_only"),
        ("context_plus_comment", "context_plus_comment"),
    ]

    results = []
    comparison_df = test_df[["context", "comment", "label"]].copy()
    reports = {}

    for setting_name, column_name in settings:
        result, predictions, report = evaluate_setting(
            model,
            setting_name,
            train_df[column_name],
            test_df[column_name],
            y_train,
            y_test
        )

        results.append(result)
        comparison_df[f"prediction_{setting_name}"] = predictions
        reports[setting_name] = report

    metrics_df = pd.DataFrame(results)

    metrics_df.to_csv("reports/sentence_transformer_metrics.csv", index=False)
    comparison_df.to_csv("reports/sentence_transformer_predictions.csv", index=False)

    with open("reports/sentence_transformer_summary.txt", "w", encoding="utf-8") as f:
        f.write("Sentence Transformer experiment\n")
        f.write("=" * 45 + "\n\n")

        f.write("Model:\n")
        f.write(f"{MODEL_NAME}\n\n")

        f.write("Goal:\n")
        f.write(
            "Compare context only, comment only, and context + comment "
            "using pretrained sentence embeddings instead of TF-IDF.\n\n"
        )

        f.write("Metrics:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        for setting_name, report in reports.items():
            f.write(f"Classification report - {setting_name}:\n")
            f.write(report)
            f.write("\n\n")

        best = metrics_df.sort_values("macro_f1", ascending=False).iloc[0]
        f.write("Best setting by Macro-F1:\n")
        f.write(f"{best['setting']} with Macro-F1 = {best['macro_f1']:.4f}\n")

    print("\nDone.")
    print(metrics_df)

    print("\nFiles created:")
    print("reports/sentence_transformer_metrics.csv")
    print("reports/sentence_transformer_predictions.csv")
    print("reports/sentence_transformer_summary.txt")


if __name__ == "__main__":
    main()
